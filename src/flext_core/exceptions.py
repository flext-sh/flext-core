"""Structured exception hierarchy for dispatcher-aware foundations.

Layer 1 defines typed errors with correlation metadata so dispatcher pipelines
and services can rely on FlextResult without losing context. The module keeps
logging integration and error codes centralized while respecting the clean
architecture boundary from infrastructure into domain/application flows.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Callable, Mapping
from typing import ClassVar, cast

from flext_core._models.base import FlextModelsBase
from flext_core.constants import FlextConstants
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


class FlextExceptions:
    """Foundation error types with correlation metadata for CQRS flows.

    Exceptions in this namespace enrich failures with error codes and optional
    metadata so dispatcher-driven handlers can surface structured details
    through FlextResult or structured logging without bespoke wrappers.
    """

    class Configuration:
        """Hierarchical exception handling configuration."""

        _global_failure_level: ClassVar[
            FlextConstants.Exceptions.FailureLevel | None
        ] = None
        # Business Rule: Mutable ClassVar dicts for runtime state - must use dict, not Mapping
        # These are mutable storage structures for singleton pattern and registry
        # Type annotation uses dict because they are mutable storage, not read-only Mapping
        _library_exception_levels: ClassVar[dict[str, dict[type, str]]] = {}
        _container_exception_levels: ClassVar[dict[str, str]] = {}
        _call_level_context: ClassVar[
            contextvars.ContextVar[FlextConstants.Exceptions.FailureLevel | None]
        ] = contextvars.ContextVar("exception_mode", default=None)

        @classmethod
        def set_global_level(
            cls,
            level: FlextConstants.Exceptions.FailureLevel,
        ) -> None:
            """Set the global failure level for exception handling."""
            cls._global_failure_level = level

        @classmethod
        def _get_global_failure_level(cls) -> FlextConstants.Exceptions.FailureLevel:
            """Get the global failure level, initializing from config if needed."""
            if cls._global_failure_level is not None:
                return cls._global_failure_level
            try:
                # Business Rule: Access global config singleton for exception failure level
                # Lazy import to avoid circular dependency: config.py imports exceptions.py
                # This is a genuine circular import case - use lazy import per FLEXT standards
                from flext_core.config import FlextConfig  # noqa: PLC0415

                config = FlextConfig.get_global_instance()
                # exception_failure_level is now a FailureLevel StrEnum directly
                level = config.exception_failure_level
                cls._global_failure_level = level
                return level
            except (AttributeError, ValueError, TypeError):
                strict_level = FlextConstants.Exceptions.FailureLevel.STRICT
                cls._global_failure_level = strict_level
                return strict_level

        @classmethod
        def register_library_exception_level(
            cls,
            library_name: str,
            exception_type: type,
            level: str,
        ) -> None:
            """Register exception level for a specific library and exception type."""
            if library_name not in cls._library_exception_levels:
                cls._library_exception_levels[library_name] = {}
            cls._library_exception_levels[library_name][exception_type] = level

        @classmethod
        def set_container_level(cls, container_id: str, level: str) -> None:
            """Set exception level for a specific container."""
            cls._container_exception_levels[container_id] = level

        @classmethod
        def get_effective_level(
            cls,
            library_name: str | None = None,
            container_id: str | None = None,
            exception_type: type | None = None,
        ) -> FlextConstants.Exceptions.FailureLevel:
            """Get the effective failure level for the given context."""
            call_level = cls._call_level_context.get()
            if call_level is not None:
                return call_level
            if container_id and container_id in cls._container_exception_levels:
                return FlextConstants.Exceptions.FailureLevel(
                    cls._container_exception_levels[container_id],
                )
            if (
                library_name
                and exception_type
                and library_name in cls._library_exception_levels
                and exception_type in cls._library_exception_levels[library_name]
            ):
                return FlextConstants.Exceptions.FailureLevel(
                    cls._library_exception_levels[library_name][exception_type],
                )
            return cls._get_global_failure_level()

    class BaseError(Exception):
        """Base exception with correlation metadata and error codes.

        All FLEXT exceptions inherit from this to ensure consistent error
        handling, logging, and correlation tracking across the ecosystem.
        """

        # NOTE: Use FlextRuntime.normalize_to_metadata_value() directly - no wrapper needed

        def __init__(  # noqa: PLR0913
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.UNKNOWN_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            metadata: FlextModelsBase.Metadata
            | Mapping[str, FlextTypes.MetadataAttributeValue]
            | FlextTypes.GeneralValueType
            | None = None,
            correlation_id: str | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
            merged_kwargs: dict[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize base error with message and optional metadata.

            Business Rule: Initializes exception with message, error code, and optional
            metadata. Generates correlation ID automatically if auto_correlation=True.
            Logs exception automatically if auto_log=True. Merges context, metadata,
            and extra_kwargs into unified metadata structure. Uses FlextRuntime for
            safe metadata normalization.

            Audit Implication: Exception initialization ensures audit trail completeness
            by capturing error context, correlation IDs, and metadata. All exceptions
            are logged with full context for audit trail reconstruction. Used throughout
            FLEXT for structured error handling.

            Args:
                message: Error message
                error_code: Optional error code
                context: Optional context mapping
                metadata: Optional metadata (FlextModelsBase.Metadata, dict, or GeneralValueType)
                correlation_id: Optional correlation ID
                auto_correlation: Auto-generate correlation ID if not provided
                auto_log: Auto-log error on creation
                merged_kwargs: Additional metadata attributes to merge

            """
            super().__init__(message)
            self.message = message
            self.error_code = error_code

            # Merge context and extra_kwargs into final_kwargs
            final_kwargs = dict(merged_kwargs) if merged_kwargs else {}
            if context:
                for k, v in context.items():
                    final_kwargs[k] = v
            for k, v in extra_kwargs.items():
                final_kwargs[k] = FlextRuntime.normalize_to_metadata_value(v)

            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if auto_correlation and not correlation_id
                else correlation_id
            )
            # Convert metadata to proper type for _normalize_metadata
            # _normalize_metadata expects FlextModelsBase.Metadata | Mapping[str, FlextTypes.MetadataAttributeValue] | GeneralValueType | None
            metadata_for_normalize: (
                FlextModelsBase.Metadata
                | Mapping[str, FlextTypes.MetadataAttributeValue]
                | FlextTypes.GeneralValueType
                | None
            ) = metadata
            self.metadata = FlextExceptions.BaseError._normalize_metadata(
                metadata_for_normalize,
                final_kwargs,
            )
            self.timestamp = time.time()
            self.auto_log = auto_log

            # auto_log is stored for caller to check via log_via_context()
            # Auto-logging via context propagation - no direct logging dependency
            # The dispatcher/service layer handles logging when exceptions are caught
            # This maintains clean architecture and avoids circular imports
            _ = auto_log  # Used by log_via_context caller to decide on logging

        def log_via_context(self) -> None:
            """Log this exception via FlextLogger - call this when context is available.

            Business Rule: Logs exception using FlextRuntime.get_logger() with failure
            level-based logging. Uses get_effective_level() to determine log level
            (STRICT logs as error with exc_info, WARN logs as warning). This method
            should be called by dispatcher/service layer when catching FlextExceptions.

            Audit Implication: Exception logging ensures audit trail completeness by
            logging all exceptions with full context. All exceptions are logged with
            appropriate log levels based on failure level configuration. Used throughout
            FLEXT for structured exception logging.

            This method should be called by the dispatcher/service layer when catching
            FlextExceptions, ensuring proper contextual logging without circular dependencies.
            """
            # Import here to avoid circular dependency during module initialization
            from flext_core.runtime import FlextRuntime  # noqa: PLC0415

            failure_level = FlextExceptions.Configuration.get_effective_level(
                library_name=None,
                container_id=None,
                exception_type=type(self),
            )

            # Use FlextRuntime for logging - maintains architectural boundaries
            logger = FlextRuntime.get_logger(__name__)
            if failure_level == FlextConstants.Exceptions.FailureLevel.STRICT:
                logger.error(str(self), exc_info=self)
            elif failure_level == FlextConstants.Exceptions.FailureLevel.WARN:
                logger.warning(str(self))

        def __str__(self) -> str:
            """Return string representation with error code if present."""
            if self.error_code:
                return f"[{self.error_code}] {self.message}"
            return self.message

        def to_dict(self) -> dict[str, FlextTypes.MetadataAttributeValue]:
            """Convert exception to dictionary representation.

            Business Rule: Converts exception to dictionary with error_type, message,
            error_code, correlation_id, timestamp, and metadata attributes. Merges
            metadata attributes into result dictionary without overriding existing keys.
            Used for serialization and audit trail storage.

            Audit Implication: Dictionary representation ensures audit trail completeness
            by providing structured exception data for storage and transmission. All
            exception data is serialized consistently for audit systems.

            Returns:
                Dictionary with error_type, message, error_code, and other fields.

            """
            result: dict[str, FlextTypes.MetadataAttributeValue] = {
                "error_type": type(self).__name__,
                "message": self.message,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
            }
            # Add metadata attributes
            if self.metadata and self.metadata.attributes:
                for k, v in self.metadata.attributes.items():
                    if k not in result:  # Don't override existing keys
                        result[k] = v
            return result

        @staticmethod
        def _normalize_metadata(
            metadata: FlextModelsBase.Metadata
            | Mapping[str, FlextTypes.MetadataAttributeValue]
            | FlextTypes.GeneralValueType
            | None,
            merged_kwargs: dict[str, FlextTypes.MetadataAttributeValue],
        ) -> FlextModelsBase.Metadata:
            """Normalize metadata from various input types to FlextModelsBase.Metadata model.

            Business Rule: Normalizes metadata from various input types (Metadata model,
            dict-like objects, GeneralValueType, or None) to FlextModelsBase.Metadata.
            Uses FlextRuntime.normalize_to_metadata_value() for safe normalization.
            Merges merged_kwargs into final metadata attributes. Fallback converts
            non-dict values to string representation.

            Audit Implication: Metadata normalization ensures audit trail completeness
            by converting all metadata types to consistent Metadata model format. All
            metadata is normalized before being stored in audit trails.

            Args:
                metadata: FlextModelsBase.Metadata instance, dict-like object, or None
                merged_kwargs: Additional attributes to merge

            Returns:
                Normalized FlextModelsBase.Metadata instance

            """
            if metadata is None:
                # Normalize attributes to FlextTypes.MetadataAttributeValue
                normalized_attrs: dict[str, FlextTypes.MetadataAttributeValue] = {}
                for k, v in (merged_kwargs or {}).items():
                    # Always normalize - normalize_to_metadata_value handles all GeneralValueType
                    normalized_attrs[k] = FlextRuntime.normalize_to_metadata_value(v)
                return FlextModelsBase.Metadata(attributes=normalized_attrs)

            if isinstance(metadata, FlextModelsBase.Metadata):
                if merged_kwargs:
                    existing_attrs = metadata.attributes
                    new_attrs: dict[str, FlextTypes.MetadataAttributeValue] = {}
                    for k, v in {**existing_attrs, **merged_kwargs}.items():
                        # Always normalize - normalize_to_metadata_value handles all GeneralValueType
                        new_attrs[k] = FlextRuntime.normalize_to_metadata_value(v)
                    return FlextModelsBase.Metadata(attributes=new_attrs)
                return metadata

            if FlextRuntime.is_dict_like(metadata):
                # After is_dict_like type guard, metadata is Mapping[str, GeneralValueType]
                # Normalize attributes to FlextTypes.MetadataAttributeValue
                merged_attrs: dict[str, FlextTypes.MetadataAttributeValue] = {}

                # Type guard ensures metadata is Mapping[str, GeneralValueType]
                metadata_mapping: Mapping[str, FlextTypes.GeneralValueType] = metadata
                for meta_key, meta_val in metadata_mapping.items():
                    # meta_val is GeneralValueType - normalize to FlextTypes.MetadataAttributeValue
                    merged_attrs[meta_key] = FlextRuntime.normalize_to_metadata_value(
                        meta_val
                    )

                # Normalize merged_kwargs values
                if merged_kwargs:
                    for kwarg_key, kwarg_val in merged_kwargs.items():
                        merged_attrs[kwarg_key] = (
                            FlextRuntime.normalize_to_metadata_value(kwarg_val)
                        )

                return FlextModelsBase.Metadata(attributes=merged_attrs)

            # Fallback: convert to FlextModelsBase.Metadata with string value
            return FlextModelsBase.Metadata(attributes={"value": str(metadata)})

    # Specific exception classes with minimal code
    class ValidationError(BaseError):
        """Exception raised for input validation failures."""

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: FlextTypes.MetadataAttributeValue | None = None,
            error_code: str = FlextConstants.Errors.VALIDATION_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize validation error with field and value information."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            validation_context: dict[str, FlextTypes.MetadataAttributeValue] = {}

            if context is not None:
                for k, v in context.items():
                    validation_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if field is not None:
                validation_context["field"] = field
            if value is not None:
                validation_context["value"] = FlextRuntime.normalize_to_metadata_value(
                    value
                )
            for k, v in extra_kwargs.items():
                validation_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=validation_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.field = field
            self.value = value

    class ConfigurationError(BaseError):
        """Exception raised for configuration-related errors."""

        def __init__(
            self,
            message: str,
            *,
            config_key: str | None = None,
            config_source: str | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize configuration error with config context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            config_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
            if context is not None:
                config_context.update(context)
            if config_key is not None:
                config_context["config_key"] = config_key
            if config_source is not None:
                config_context["config_source"] = config_source
            for k, v in extra_kwargs.items():
                config_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=config_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.config_key = config_key
            self.config_source = config_source

    class ConnectionError(BaseError):
        """Exception raised for network and connection failures."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.CONNECTION_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize connection error with network context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            conn_context: dict[str, FlextTypes.MetadataAttributeValue] = {}

            if context is not None:
                for k, v in context.items():
                    conn_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                conn_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=conn_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.host = conn_context.get("host")
            self.port = conn_context.get("port")
            self.timeout = conn_context.get("timeout")

    class TimeoutError(BaseError):
        """Exception raised for operation timeout errors."""

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            operation: str | None = None,
            error_code: str = FlextConstants.Errors.TIMEOUT_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize timeout error with timeout context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            timeout_context: dict[str, FlextTypes.MetadataAttributeValue] = {}

            if context is not None:
                for k, v in context.items():
                    timeout_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if timeout_seconds is not None:
                timeout_context["timeout_seconds"] = timeout_seconds
            if operation is not None:
                timeout_context["operation"] = operation
            for k, v in extra_kwargs.items():
                timeout_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=timeout_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.timeout_seconds = timeout_seconds
            self.operation = operation

    class AuthenticationError(BaseError):
        """Exception raised for authentication failures."""

        def __init__(
            self,
            message: str,
            *,
            auth_method: str | None = None,
            user_id: str | None = None,
            error_code: str = FlextConstants.Errors.AUTHENTICATION_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize authentication error with auth context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            auth_context: dict[str, FlextTypes.MetadataAttributeValue] = {}

            if context is not None:
                for k, v in context.items():
                    auth_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if auth_method is not None:
                auth_context["auth_method"] = auth_method
            if user_id is not None:
                auth_context["user_id"] = user_id
            for k, v in extra_kwargs.items():
                auth_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=auth_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.auth_method = auth_method
            self.user_id = user_id

    class AuthorizationError(BaseError):
        """Exception raised for permission and authorization failures."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.AUTHORIZATION_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize authorization error with permission context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            authz_context: dict[str, FlextTypes.MetadataAttributeValue] = {}

            if context is not None:
                for k, v in context.items():
                    authz_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                authz_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=authz_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.user_id = authz_context.get("user_id")
            self.resource = authz_context.get("resource")
            self.permission = authz_context.get("permission")

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        @staticmethod
        def _extract_context_values(
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None,
        ) -> tuple[str | None, FlextModelsBase.Metadata | None, bool, bool]:
            """Extract context values from mapping.

            Returns:
                Tuple of (correlation_id, metadata, auto_log, auto_correlation)

            """
            if context is None:
                return (None, None, False, False)

            corr_id = context.get("correlation_id")
            correlation_id_val = corr_id if isinstance(corr_id, str) else None

            metadata_obj = context.get("metadata")
            metadata_val = (
                metadata_obj
                if isinstance(metadata_obj, FlextModelsBase.Metadata)
                else None
            )

            auto_log_obj = context.get("auto_log")
            auto_log_val = auto_log_obj if isinstance(auto_log_obj, bool) else False

            auto_corr_obj = context.get("auto_correlation")
            auto_correlation_val = (
                auto_corr_obj if isinstance(auto_corr_obj, bool) else False
            )

            return (
                correlation_id_val,
                metadata_val,
                auto_log_val,
                auto_correlation_val,
            )

        @staticmethod
        def _build_notfound_kwargs(
            resource_type: str | None,
            resource_id: str | None,
            extra_kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None,
        ) -> dict[str, FlextTypes.MetadataAttributeValue]:
            """Build notfound-specific kwargs from fields and context.

            Returns:
                Dictionary of notfound kwargs

            """
            notfound_kwargs: dict[str, FlextTypes.MetadataAttributeValue] = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }

            # Convert extra_kwargs to FlextTypes.MetadataAttributeValue and normalize values
            valid_extra: dict[str, FlextTypes.MetadataAttributeValue] = {
                k: FlextRuntime.normalize_to_metadata_value(v)
                for k, v in extra_kwargs.items()
                if isinstance(v, (str, int, float, bool, list, Mapping, type(None)))
            }
            notfound_kwargs.update(valid_extra)

            # Add context items (excluding reserved keys)
            if context is not None:
                excluded_keys = {
                    "correlation_id",
                    "metadata",
                    "auto_log",
                    "auto_correlation",
                }
                valid_context = {
                    k: v
                    for k, v in context.items()
                    if k not in excluded_keys
                    and isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                notfound_kwargs.update(valid_context)

            return notfound_kwargs

        def __init__(  # noqa: PLR0913
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            error_code: str = FlextConstants.Errors.NOT_FOUND_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            metadata: FlextModelsBase.Metadata
            | Mapping[str, FlextTypes.MetadataAttributeValue]
            | FlextTypes.GeneralValueType
            | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize not found error with resource context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Use explicit params or preserved from extra_kwargs
            final_metadata = metadata if metadata is not None else preserved_metadata
            final_corr_id = (
                correlation_id
                if correlation_id is not None
                else (preserved_corr_id if isinstance(preserved_corr_id, str) else None)
            )

            # Build notfound context
            notfound_context: dict[str, FlextTypes.MetadataAttributeValue] = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }

            # Add extra_kwargs to context
            for k, v in extra_kwargs.items():
                notfound_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            # Add context items (excluding reserved keys)
            if context is not None:
                excluded_keys = {"correlation_id", "metadata"}
                for k, v in context.items():
                    if k not in excluded_keys:
                        notfound_context[k] = FlextRuntime.normalize_to_metadata_value(
                            v
                        )

            super().__init__(
                message,
                error_code=error_code,
                context=notfound_context,
                metadata=final_metadata,
                correlation_id=final_corr_id,
            )
            self.resource_type = resource_type
            self.resource_id = resource_id

    class ConflictError(BaseError):
        """Exception raised for resource conflicts."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.ALREADY_EXISTS,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize conflict error with resource context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            conflict_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
            if context is not None:
                conflict_context.update(context)
            for k, v in extra_kwargs.items():
                conflict_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=conflict_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.resource_type = conflict_context.get("resource_type")
            self.resource_id = conflict_context.get("resource_id")
            self.conflict_reason = conflict_context.get("conflict_reason")

    class RateLimitError(BaseError):
        """Exception raised when rate limits are exceeded."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.OPERATION_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize rate limit error with limit context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            rate_limit_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
            if context is not None:
                rate_limit_context.update(context)
            for k, v in extra_kwargs.items():
                rate_limit_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=rate_limit_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            limit_val = rate_limit_context.get("limit")
            self.limit = limit_val if isinstance(limit_val, int) else None
            window_seconds_val = rate_limit_context.get("window_seconds")
            self.window_seconds = (
                window_seconds_val if isinstance(window_seconds_val, int) else None
            )
            retry_after_val = rate_limit_context.get("retry_after")
            self.retry_after = (
                retry_after_val if isinstance(retry_after_val, (int, float)) else None
            )

    class CircuitBreakerError(BaseError):
        """Exception raised when circuit breaker is open."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.EXTERNAL_SERVICE_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize circuit breaker error with service context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            cb_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
            if context is not None:
                cb_context.update(context)
            for k, v in extra_kwargs.items():
                cb_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=cb_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            service_name_val = cb_context.get("service_name")
            self.service_name = (
                service_name_val if isinstance(service_name_val, str) else None
            )
            failure_count_val = cb_context.get("failure_count")
            self.failure_count = (
                failure_count_val if isinstance(failure_count_val, int) else None
            )
            reset_timeout_val = cb_context.get("reset_timeout")
            self.reset_timeout = (
                reset_timeout_val
                if isinstance(reset_timeout_val, (int, float))
                else None
            )

    class TypeError(BaseError):
        """Exception raised for type mismatch errors."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.TYPE_ERROR,
            expected_type: type | None = None,
            actual_type: type | None = None,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize type error with type information."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            type_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
            if context is not None:
                for k, v in context.items():
                    type_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            type_context["expected_type"] = (
                expected_type.__qualname__ if expected_type else None
            )
            type_context["actual_type"] = (
                actual_type.__qualname__ if actual_type else None
            )
            for k, v in extra_kwargs.items():
                type_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=type_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.expected_type = expected_type
            self.actual_type = actual_type

    class OperationError(BaseError):
        """Exception raised for general operation failures."""

        def __init__(
            self,
            message: str,
            *,
            operation: str | None = None,
            reason: str | None = None,
            error_code: str = FlextConstants.Errors.OPERATION_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize operation error with operation context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            op_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
            if context is not None:
                for k, v in context.items():
                    op_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if operation is not None:
                op_context["operation"] = operation
            if reason is not None:
                op_context["reason"] = reason
            for k, v in extra_kwargs.items():
                op_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=op_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.operation = operation
            self.reason = reason

    class AttributeAccessError(BaseError):
        """Exception raised for attribute access errors."""

        def __init__(
            self,
            message: str,
            *,
            attribute_name: str | None = None,
            attribute_context: str | None = None,
            error_code: str = FlextConstants.Errors.ATTRIBUTE_ERROR,
            context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
            **extra_kwargs: FlextTypes.MetadataAttributeValue,
        ) -> None:
            """Initialize attribute access error with attribute context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            attr_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
            if context is not None:
                attr_context.update(context)
            if attribute_name is not None:
                attr_context["attribute_name"] = attribute_name
            if attribute_context is not None:
                attr_context["attribute_context"] = attribute_context
            for k, v in extra_kwargs.items():
                attr_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=attr_context or None,
                metadata=preserved_metadata,
                correlation_id=preserved_corr_id
                if isinstance(preserved_corr_id, str)
                else None,
            )
            self.attribute_name = attribute_name
            self.attribute_context = attribute_context

    @staticmethod
    def prepare_exception_kwargs(
        kwargs: dict[str, FlextTypes.MetadataAttributeValue],
        specific_params: dict[str, FlextTypes.MetadataAttributeValue] | None = None,
    ) -> tuple[
        str | None,
        FlextTypes.MetadataAttributeValue,
        bool,
        bool,
        FlextTypes.MetadataAttributeValue,
        dict[str, FlextTypes.MetadataAttributeValue],
    ]:
        """Prepare exception kwargs by extracting common parameters."""
        if specific_params:
            for key, value in specific_params.items():
                if value is not None:
                    _ = kwargs.setdefault(key, value)
        extra_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "correlation_id",
                "metadata",
                "auto_log",
                "auto_correlation",
                "config",
            }
        }
        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id: str | None = (
            str(correlation_id_raw)
            if correlation_id_raw is not None and isinstance(correlation_id_raw, str)
            else None
        )
        return (
            correlation_id,
            kwargs.get("metadata"),
            bool(kwargs.get("auto_log")),
            bool(kwargs.get("auto_correlation")),
            kwargs.get("config"),
            extra_kwargs,
        )

    @staticmethod
    def extract_common_kwargs(
        kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
    ) -> tuple[
        str | None,
        FlextModelsBase.Metadata
        | Mapping[str, FlextTypes.MetadataAttributeValue]
        | None,
    ]:
        """Extract correlation_id and metadata from kwargs.

        Returns typed values: correlation_id as str | None, metadata as FlextModelsBase.Metadata | Mapping | None.
        """
        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id: str | None = (
            str(correlation_id_raw)
            if correlation_id_raw is not None and isinstance(correlation_id_raw, str)
            else None
        )
        metadata_raw = kwargs.get("metadata")
        # Return metadata as-is if it's Metadata or dict-like, otherwise None
        metadata: (
            FlextModelsBase.Metadata
            | Mapping[str, FlextTypes.MetadataAttributeValue]
            | None
        ) = None
        if metadata_raw is not None:
            if isinstance(metadata_raw, FlextModelsBase.Metadata):
                metadata = metadata_raw
            elif isinstance(metadata_raw, dict) or FlextRuntime.is_dict_like(
                metadata_raw
            ):
                # Convert dict or dict-like values to MetadataAttributeValue
                converted_dict: dict[str, FlextTypes.MetadataAttributeValue] = {}
                for k, v in metadata_raw.items():
                    converted_dict[k] = FlextRuntime.normalize_to_metadata_value(v)
                metadata = converted_dict
        return (correlation_id, metadata)

    @staticmethod
    def create_error(error_type: str, message: str) -> FlextExceptions.BaseError:
        """Create an exception instance based on error type."""
        error_classes: dict[str, type[FlextExceptions.BaseError]] = {
            "ValidationError": FlextExceptions.ValidationError,
            "ConfigurationError": FlextExceptions.ConfigurationError,
            "ConnectionError": FlextExceptions.ConnectionError,
            "TimeoutError": FlextExceptions.TimeoutError,
            "AuthenticationError": FlextExceptions.AuthenticationError,
            "AuthorizationError": FlextExceptions.AuthorizationError,
            "NotFoundError": FlextExceptions.NotFoundError,
            "ConflictError": FlextExceptions.ConflictError,
            "RateLimitError": FlextExceptions.RateLimitError,
            "CircuitBreakerError": FlextExceptions.CircuitBreakerError,
            "TypeError": FlextExceptions.TypeError,
            "OperationError": FlextExceptions.OperationError,
            "AttributeError": FlextExceptions.AttributeAccessError,
        }
        error_class = error_classes.get(error_type)
        if not error_class:
            msg = f"Unknown error type: {error_type}"
            raise ValueError(msg)
        return error_class(message)

    @staticmethod
    def _determine_error_type(
        kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
    ) -> str | None:
        """Determine error type from kwargs using pattern matching.

        Returns:
            Error type string or None if no match

        """
        # Error type detection patterns - ordered by specificity
        # Each pattern checks if any key in the set is present
        error_patterns: list[tuple[list[str], str]] = [
            (["field", "value"], "validation"),
            (["config_key", "config_source"], "configuration"),
            (["operation"], "operation"),
            (["host", "port"], "connection"),
            (["timeout_seconds"], "timeout"),
            (["user_id", "permission"], "authorization"),  # Both required
            (["auth_method"], "authentication"),
            (["resource_id"], "not_found"),
            (["attribute_name"], "attribute_access"),
        ]

        for keys, error_type in error_patterns:
            # Special case: authorization requires both keys
            if error_type == "authorization":
                if "user_id" in kwargs and "permission" in kwargs:
                    return error_type
            elif any(key in kwargs for key in keys):
                return error_type

        return None

    @staticmethod
    def _prepare_metadata_value(
        meta: FlextModelsBase.Metadata | None,
    ) -> FlextTypes.MetadataAttributeValue | None:
        """Prepare metadata value for error creation."""
        return (
            cast("FlextTypes.MetadataAttributeValue", meta.attributes)
            if meta is not None
            else None
        )

    @staticmethod
    def _get_str_from_kwargs(
        kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
        key: str,
    ) -> str | None:
        """Extract and convert value from kwargs to string."""
        val = kwargs.get(key)
        return str(val) if val is not None else None

    @staticmethod
    def _get_error_creator(
        error_type: str,
    ) -> (
        Callable[
            [
                str,
                str | None,
                Mapping[str, FlextTypes.MetadataAttributeValue],
                str | None,
                FlextModelsBase.Metadata | None,
            ],
            BaseError,
        ]
        | None
    ):
        def _create_validation_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.ValidationError:
            field = FlextExceptions._get_str_from_kwargs(kwargs, "field")
            value = kwargs.get("value")
            metadata_value = (
                cast("FlextTypes.MetadataAttributeValue", meta.attributes)
                if meta is not None
                else None
            )
            return FlextExceptions.ValidationError(
                msg,
                error_code=code or FlextConstants.Errors.VALIDATION_ERROR,
                field=field,
                value=value,
                correlation_id=cid,
                metadata=metadata_value,
            )

        def _create_configuration_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.ConfigurationError:
            config_key = FlextExceptions._get_str_from_kwargs(kwargs, "config_key")
            config_source = FlextExceptions._get_str_from_kwargs(
                kwargs, "config_source"
            )
            metadata_value = FlextExceptions._prepare_metadata_value(meta)
            return FlextExceptions.ConfigurationError(
                msg,
                error_code=code or FlextConstants.Errors.CONFIGURATION_ERROR,
                config_key=config_key,
                config_source=config_source,
                correlation_id=cid,
                metadata=metadata_value,
            )

        def _create_operation_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.OperationError:
            operation = FlextExceptions._get_str_from_kwargs(kwargs, "operation")
            reason = FlextExceptions._get_str_from_kwargs(kwargs, "reason")
            metadata_value = FlextExceptions._prepare_metadata_value(meta)
            return FlextExceptions.OperationError(
                msg,
                error_code=code or FlextConstants.Errors.OPERATION_ERROR,
                operation=operation,
                reason=reason,
                correlation_id=cid,
                metadata=metadata_value,
            )

        def _create_connection_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.ConnectionError:
            metadata_value = FlextExceptions._prepare_metadata_value(meta)
            return FlextExceptions.ConnectionError(
                msg,
                error_code=code or FlextConstants.Errors.CONNECTION_ERROR,
                host=kwargs.get("host"),
                port=kwargs.get("port"),
                timeout=kwargs.get("timeout"),
                correlation_id=cid,
                metadata=metadata_value,
            )

        def _create_timeout_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.TimeoutError:
            timeout_seconds_val = kwargs.get("timeout_seconds")
            timeout_seconds: float | None = (
                float(timeout_seconds_val)
                if timeout_seconds_val is not None
                and isinstance(timeout_seconds_val, (int, float))
                else (
                    float(str(timeout_seconds_val))
                    if timeout_seconds_val is not None
                    else None
                )
            )
            operation = FlextExceptions._get_str_from_kwargs(kwargs, "operation")
            metadata_value = FlextExceptions._prepare_metadata_value(meta)
            return FlextExceptions.TimeoutError(
                msg,
                error_code=code or FlextConstants.Errors.TIMEOUT_ERROR,
                timeout_seconds=timeout_seconds,
                operation=operation,
                correlation_id=cid,
                metadata=metadata_value,
            )

        def _create_authorization_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.AuthorizationError:
            metadata_value = FlextExceptions._prepare_metadata_value(meta)
            return FlextExceptions.AuthorizationError(
                msg,
                error_code=code or FlextConstants.Errors.AUTHORIZATION_ERROR,
                user_id=kwargs.get("user_id"),
                resource=kwargs.get("resource"),
                permission=kwargs.get("permission"),
                correlation_id=cid,
                metadata=metadata_value,
            )

        def _create_authentication_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.AuthenticationError:
            auth_method = FlextExceptions._get_str_from_kwargs(kwargs, "auth_method")
            user_id = FlextExceptions._get_str_from_kwargs(kwargs, "user_id")
            metadata_value = FlextExceptions._prepare_metadata_value(meta)
            return FlextExceptions.AuthenticationError(
                msg,
                error_code=code or FlextConstants.Errors.AUTHENTICATION_ERROR,
                auth_method=auth_method,
                user_id=user_id,
                correlation_id=cid,
                metadata=metadata_value,
            )

        def _create_not_found_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.NotFoundError:
            resource_type = FlextExceptions._get_str_from_kwargs(
                kwargs, "resource_type"
            )
            resource_id = FlextExceptions._get_str_from_kwargs(kwargs, "resource_id")
            metadata_for_error: (
                FlextModelsBase.Metadata
                | Mapping[str, FlextTypes.MetadataAttributeValue]
                | FlextTypes.GeneralValueType
                | None
            ) = (
                (meta.attributes if hasattr(meta, "attributes") else meta)
                if meta is not None
                else None
            )
            return FlextExceptions.NotFoundError(
                msg,
                error_code=code or FlextConstants.Errors.NOT_FOUND_ERROR,
                resource_type=resource_type,
                resource_id=resource_id,
                correlation_id=cid,
                metadata=metadata_for_error,
            )

        def _create_attribute_access_error(
            msg: str,
            code: str | None,
            kwargs: Mapping[str, FlextTypes.MetadataAttributeValue],
            cid: str | None,
            meta: FlextModelsBase.Metadata | None,
        ) -> FlextExceptions.AttributeAccessError:
            attribute_name = FlextExceptions._get_str_from_kwargs(
                kwargs, "attribute_name"
            )
            attribute_context = FlextExceptions._get_str_from_kwargs(
                kwargs, "attribute_context"
            )
            metadata_value = FlextExceptions._prepare_metadata_value(meta)
            return FlextExceptions.AttributeAccessError(
                msg,
                error_code=code or FlextConstants.Errors.ATTRIBUTE_ERROR,
                attribute_name=attribute_name,
                attribute_context=attribute_context,
                correlation_id=cid,
                metadata=metadata_value,
            )

        creators = {
            "validation": _create_validation_error,
            "configuration": _create_configuration_error,
            "operation": _create_operation_error,
            "connection": _create_connection_error,
            "timeout": _create_timeout_error,
            "authorization": _create_authorization_error,
            "authentication": _create_authentication_error,
            "not_found": _create_not_found_error,
            "attribute_access": _create_attribute_access_error,
        }
        return creators.get(error_type)

    @staticmethod
    def _create_error_by_type(
        error_type: str | None,
        message: str,
        error_code: str | None,
        context: Mapping[str, FlextTypes.MetadataAttributeValue] | None = None,
    ) -> FlextExceptions.BaseError:
        """Create error by type using context dict."""
        # Build context with error_code
        error_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
        if context is not None:
            error_context.update(context)
        if error_code is not None:
            error_context["error_code"] = error_code

        # Create appropriate error class based on type
        error_classes: dict[str, type[FlextExceptions.BaseError]] = {
            "validation": FlextExceptions.ValidationError,
            "configuration": FlextExceptions.ConfigurationError,
            "connection": FlextExceptions.ConnectionError,
            "timeout": FlextExceptions.TimeoutError,
            "authentication": FlextExceptions.AuthenticationError,
            "authorization": FlextExceptions.AuthorizationError,
            "not_found": FlextExceptions.NotFoundError,
            "operation": FlextExceptions.OperationError,
            "attribute_access": FlextExceptions.AttributeAccessError,
        }

        error_class = error_classes.get(error_type or "") if error_type else None
        if error_class:
            return error_class(
                message,
                error_code=error_code or FlextConstants.Errors.UNKNOWN_ERROR,
                context=error_context or None,
            )

        return FlextExceptions.BaseError(
            message,
            error_code=error_code or FlextConstants.Errors.UNKNOWN_ERROR,
            context=error_context or None,
        )

    @staticmethod
    def create(
        message: str,
        error_code: str | None = None,
        **kwargs: FlextTypes.MetadataAttributeValue,
    ) -> FlextExceptions.BaseError:
        """Create an appropriate exception instance based on kwargs context."""
        correlation_id_obj, metadata_obj = FlextExceptions.extract_common_kwargs(kwargs)
        error_type = FlextExceptions._determine_error_type(kwargs)
        # correlation_id_obj is already str | None from extract_common_kwargs
        correlation_id: str | None = correlation_id_obj
        # Build context dict
        error_context: dict[str, FlextTypes.MetadataAttributeValue] = {}
        if correlation_id is not None:
            error_context["correlation_id"] = correlation_id
        # Handle metadata_obj - can be FlextModelsBase.Metadata, dict, or None
        if metadata_obj is not None:
            if isinstance(metadata_obj, FlextModelsBase.Metadata):
                # Extract attributes from FlextModelsBase.Metadata model
                error_context.update(dict(metadata_obj.attributes.items()))
            elif isinstance(metadata_obj, dict):
                # Direct dict - normalize values and update
                for k, v in metadata_obj.items():
                    error_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            elif FlextRuntime.is_dict_like(metadata_obj):
                # Dict-like object - normalize and update
                for k, v in metadata_obj.items():
                    error_context[k] = FlextRuntime.normalize_to_metadata_value(v)
        for k, v in kwargs.items():
            if k not in {"correlation_id", "metadata"}:
                error_context[k] = FlextRuntime.normalize_to_metadata_value(v)

        return FlextExceptions._create_error_by_type(
            error_type,
            message,
            error_code,
            context=error_context or None,
        )

    # Mutable ClassVar dict for runtime metrics - must use dict, not Mapping
    _exception_counts: ClassVar[dict[type, int]] = {}

    @classmethod
    def record_exception(cls, exception_type: type) -> None:
        """Record an exception occurrence for metrics tracking."""
        if exception_type not in cls._exception_counts:
            cls._exception_counts[exception_type] = 0
        cls._exception_counts[exception_type] += 1

    @classmethod
    def get_metrics(cls) -> FlextTypes.Types.ErrorTypeMapping:
        """Get exception metrics and statistics."""
        total = sum(cls._exception_counts.values(), 0)
        # Serialize exception counts as a single string for compatibility with ErrorTypeMapping
        exception_counts_list = [
            f"{exc_type.__qualname__ if hasattr(exc_type, '__qualname__') else str(exc_type)}:{count}"
            for exc_type, count in cls._exception_counts.items()
        ]
        exception_counts_str = ";".join(exception_counts_list)
        # Build exception_counts dict for test compatibility
        # Use wider type compatible with ErrorTypeMapping value type
        exception_counts_dict: dict[
            str,
            str
            | int
            | float
            | bool
            | list[str | int | float | bool | None]
            | dict[str, str | int | float | bool | None]
            | None,
        ] = {}
        for exc_type, count in cls._exception_counts.items():
            exc_name = (
                exc_type.__qualname__
                if hasattr(exc_type, "__qualname__")
                else str(exc_type)
            )
            exception_counts_dict[exc_name] = count
        # Build result dict matching ErrorTypeMapping type
        result: FlextTypes.Types.ErrorTypeMapping = {
            "total_exceptions": total,
            "exception_counts": exception_counts_dict,
            "exception_counts_summary": exception_counts_str,  # String format for summary
            "unique_exception_types": len(cls._exception_counts),
        }
        return result

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._exception_counts.clear()

    def __call__(
        self,
        message: str,
        error_code: str | None = None,
        **kwargs: FlextTypes.Types.ExceptionKwargsType,
    ) -> FlextExceptions.BaseError:
        """Create exception by calling the class instance."""
        # Normalize ExceptionKwargsType to FlextTypes.MetadataAttributeValue

        normalized_kwargs: dict[str, FlextTypes.MetadataAttributeValue] = {}
        for k, v in kwargs.items():
            normalized_kwargs[k] = FlextRuntime.normalize_to_metadata_value(v)
        return self.create(message, error_code, **normalized_kwargs)


__all__ = ["FlextExceptions"]
