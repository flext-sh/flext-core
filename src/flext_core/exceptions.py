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
from typing import ClassVar, Self, cast

from flext_core._models.metadata import Metadata, MetadataAttributeValue
from flext_core.config import FlextConfig
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
        _library_exception_levels: ClassVar[
            FlextTypes.Types.NestedExceptionLevelMapping
        ] = {}
        _container_exception_levels: ClassVar[
            FlextTypes.Types.ExceptionLevelMapping
        ] = {}
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
                config = FlextConfig.get_global_instance()
                # exception_failure_level is now a FailureLevel StrEnum directly
                cls._global_failure_level = config.exception_failure_level
                return cls._global_failure_level
            except (AttributeError, ValueError, TypeError):
                cls._global_failure_level = (
                    FlextConstants.Exceptions.FailureLevel.STRICT
                )
                return cls._global_failure_level

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
        """Base exception class with structured logging."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.UNKNOWN_ERROR,
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize base exception with structured metadata and correlation.

            Tracks correlation IDs and structured metadata for distributed tracing.

            Args:
                message: Human-readable error message
                error_code: Error code from FlextConstants.Errors
                context: Optional dict with correlation_id, metadata, auto_log, auto_correlation
                **extra_kwargs: Additional metadata attributes

            """
            super().__init__(message)
            self.message = message
            self.error_code = error_code

            # Extract context values
            if context is not None:
                correlation_id = context.get("correlation_id")
                metadata = context.get("metadata")
                auto_log = context.get("auto_log", False)
                auto_correlation = context.get("auto_correlation", False)
                # Merge context extra_kwargs with provided extra_kwargs
                context_kwargs = {
                    k: v
                    for k, v in context.items()
                    if k
                    not in {
                        "correlation_id",
                        "metadata",
                        "auto_log",
                        "auto_correlation",
                    }
                }
                merged_kwargs = {**context_kwargs, **extra_kwargs}
            else:
                correlation_id = None
                metadata = None
                auto_log = False
                auto_correlation = False
                merged_kwargs = extra_kwargs

            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if auto_correlation and not correlation_id
                else correlation_id
            )
            if metadata is None:
                # Normalize attributes to MetadataAttributeValue
                normalized_attrs: dict[
                    str, MetadataAttributeValue
                ] = {}  # Mutable dict for construction

                for k, v in (merged_kwargs or {}).items():
                    if isinstance(
                        v, (str, int, float, bool, list, Mapping, type(None))
                    ):
                        normalized_attrs[k] = FlextRuntime.normalize_to_metadata_value(
                            v
                        )
                self.metadata = Metadata(attributes=normalized_attrs)
            else:
                self.metadata = cast("Metadata", metadata)
            if FlextRuntime.is_dict_like(metadata):
                # Normalize attributes to MetadataAttributeValue
                merged_attrs: dict[
                    str, MetadataAttributeValue
                ] = {}  # Mutable dict for construction
                # Convert Mapping to dict for merging, then wrap in Metadata
                metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
                merged_kwargs_dict = dict(merged_kwargs) if merged_kwargs else {}
                merged_dict = {**metadata_dict, **merged_kwargs_dict}

                for k, v in merged_dict.items():
                    if isinstance(
                        v, (str, int, float, bool, list, Mapping, type(None))
                    ):
                        merged_attrs[k] = FlextRuntime.normalize_to_metadata_value(v)
                self.metadata = Metadata(attributes=merged_attrs)
            elif merged_kwargs:
                existing_attrs = self.metadata.attributes
                new_attrs = {
                    k: v
                    for k, v in {**existing_attrs, **merged_kwargs}.items()
                    if isinstance(v, (str, int, float, bool, list, Mapping, type(None)))
                }
                self.metadata = Metadata(attributes=new_attrs)
            self.timestamp = time.time()
            self.auto_log = auto_log
            if auto_log:
                self._log_exception()

        def _log_exception(self) -> None:
            """Log the exception using structured logging."""
            try:
                # Get logger - BindableLogger protocol has methods at runtime
                logger = FlextRuntime.get_logger()
                # Type narrowing: logger from structlog.get_logger() has error/debug/warning methods
                logger.error(
                    "exception_raised",
                    event_type="exception",
                    error_type=self.__class__.__name__,
                    error_code=self.error_code,
                    message=self.message,
                    correlation_id=self.correlation_id,
                    timestamp=self.timestamp,
                    metadata=self.metadata,
                )
            except Exception as e:
                FlextRuntime.get_logger(__name__).debug(
                    "Logging failed in exception handler",
                    exc_info=e,
                )

        def __str__(self) -> str:
            """Return string representation with error code and correlation ID."""
            parts = []
            if self.error_code:
                parts.append(f"[{self.error_code}]")
            if self.correlation_id:
                parts.append(f"(ID: {self.correlation_id})")
            parts.append(self.message)
            return " ".join(parts)

        @staticmethod
        def _normalize_list_value(
            items: list[object],
        ) -> list[str | int | float | bool | None]:
            """Normalize list items to ErrorTypeMapping compatible types."""
            normalized: list[str | int | float | bool | None] = []
            for item in items:
                if isinstance(item, (str, int, float, bool, type(None))):
                    normalized.append(item)
                else:
                    normalized.append(str(item))
            return normalized

        @staticmethod
        def _normalize_nested_value(
            dv: FlextTypes.GeneralValueType,
        ) -> (
            str
            | int
            | float
            | bool
            | list[str | int | float | bool | None]
            | Mapping[str, str | int | float | bool | None]
            | None
        ):
            """Normalize nested dict value to ErrorTypeMapping compatible type."""
            if isinstance(dv, (str, int, float, bool, type(None))):
                return dv
            if isinstance(dv, list):
                return FlextExceptions.BaseError._normalize_list_value(dv)
            return str(dv)  # Flatten nested dict to string

        @staticmethod
        def _normalize_dict_value(
            d: Mapping[str, FlextTypes.GeneralValueType],
        ) -> dict[
            str,
            str
            | int
            | float
            | bool
            | list[str | int | float | bool | None]
            | Mapping[str, str | int | float | bool | None]
            | None,
        ]:
            """Normalize Mapping value to ErrorTypeMapping nested dict compatible type."""
            normalized_dict: dict[
                str,
                str
                | int
                | float
                | bool
                | list[str | int | float | bool | None]
                | Mapping[str, str | int | float | bool | None]
                | None,
            ] = {}
            for dk, dv in d.items():
                if isinstance(dk, str):
                    normalized_dict[dk] = (
                        FlextExceptions.BaseError._normalize_nested_value(dv)
                    )
            return normalized_dict

        @staticmethod
        def _normalize_metadata_value(
            v: FlextTypes.GeneralValueType,
        ) -> (
            str
            | int
            | float
            | dict[
                str,
                str
                | int
                | float
                | bool
                | list[str | int | float | bool | None]
                | Mapping[str, str | int | float | bool | None]
                | None,
            ]
            | None
        ):
            """Normalize metadata value to ErrorTypeMapping compatible type."""
            if isinstance(v, (str, int, float, type(None))):
                return v
            if isinstance(v, bool):
                return str(v)  # Convert bool to string (not allowed at top level)
            if isinstance(v, list):
                return str(v)  # Convert list to string (not allowed at top level)
            if isinstance(v, Mapping):
                return FlextExceptions.BaseError._normalize_dict_value(v)
            return str(v)  # Default: convert to string

        def to_dict(self) -> FlextTypes.Types.ErrorTypeMapping:
            """Convert exception to dictionary representation."""
            # Convert metadata.attributes to compatible type
            # ErrorTypeMapping allows: dict[str, str | int | float | Mapping[str, ...] | None]
            # Mutable dict needed for construction
            metadata_dict: dict[
                str,
                str
                | int
                | float
                | Mapping[
                    str,
                    str
                    | int
                    | float
                    | bool
                    | list[str | int | float | bool | None]
                    | Mapping[str, str | int | float | bool | None]
                    | None,
                ]
                | None,
            ] = {}
            for k, v in self.metadata.attributes.items():
                normalized = self._normalize_metadata_value(v)
                metadata_dict[k] = normalized
            result: FlextTypes.Types.ErrorTypeMapping = cast(
                "FlextTypes.Types.ErrorTypeMapping",
                {
                    "error_type": self.__class__.__name__,
                    "message": self.message,
                    "error_code": self.error_code,
                    "correlation_id": self.correlation_id or None,
                    "timestamp": self.timestamp,
                    "metadata": metadata_dict,
                },
            )
            return result

        def with_context(self, **context: MetadataAttributeValue) -> Self:
            """Add context information to the exception metadata."""
            existing_attrs = self.metadata.attributes
            new_attrs = {
                k: v
                for k, v in {**existing_attrs, **context}.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            self.metadata = Metadata(attributes=new_attrs)
            return self

        def chain_from(self, cause: Exception) -> Self:
            """Chain this exception from another exception."""
            self.__cause__ = cause
            if parent_correlation_id := getattr(cause, "correlation_id", None):
                existing_attrs = self.metadata.attributes
                new_attrs = {
                    **existing_attrs,
                    "parent_correlation_id": parent_correlation_id,
                }
                self.metadata = Metadata(attributes=new_attrs)
            try:
                logger = FlextRuntime.get_logger()
                logger.warning(
                    "exception_chained",
                    event_type="exception_chain",
                    child_error=self.__class__.__name__,
                    parent_error=cause.__class__.__name__,
                    correlation_id=self.correlation_id,
                )
            except Exception as e:
                FlextRuntime.get_logger(__name__).debug(
                    "Logging failed in exception handler",
                    exc_info=e,
                )
            return self

    # Specific exception classes with minimal code
    class ValidationError(BaseError):
        """Exception raised for input validation failures."""

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: MetadataAttributeValue | None = None,
            error_code: str = FlextConstants.Errors.VALIDATION_ERROR,
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize validation error with field and value information."""
            validation_context: dict[str, MetadataAttributeValue] = {}

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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize configuration error with config context."""
            config_context: dict[str, MetadataAttributeValue] = {}
            if context is not None:
                config_context.update(context)
            config_context.update({
                "config_key": config_key,
                "config_source": config_source,
                **extra_kwargs,
            })
            super().__init__(
                message,
                error_code=error_code,
                context=config_context or None,
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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize connection error with network context."""
            conn_context: dict[str, MetadataAttributeValue] = {}

            if context is not None:
                for k, v in context.items():
                    conn_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                conn_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            super().__init__(
                message,
                error_code=error_code,
                context=conn_context or None,
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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize timeout error with timeout context."""
            timeout_context: dict[str, MetadataAttributeValue] = {}

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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize authentication error with auth context."""
            auth_context: dict[str, MetadataAttributeValue] = {}

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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize authorization error with permission context."""
            authz_context: dict[str, MetadataAttributeValue] = {}

            if context is not None:
                for k, v in context.items():
                    authz_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                authz_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            super().__init__(
                message,
                error_code=error_code,
                context=authz_context or None,
            )
            self.user_id = authz_context.get("user_id")
            self.resource = authz_context.get("resource")
            self.permission = authz_context.get("permission")

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        @staticmethod
        def _extract_context_values(
            context: Mapping[str, MetadataAttributeValue] | None,
        ) -> tuple[str | None, Metadata | None, bool, bool]:
            """Extract context values from mapping.

            Returns:
                Tuple of (correlation_id, metadata, auto_log, auto_correlation)

            """
            if context is None:
                return (None, None, False, False)

            corr_id = context.get("correlation_id")
            correlation_id_val = corr_id if isinstance(corr_id, str) else None

            metadata_obj = context.get("metadata")
            metadata_val = metadata_obj if isinstance(metadata_obj, Metadata) else None

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
            extra_kwargs: Mapping[str, MetadataAttributeValue],
            context: Mapping[str, MetadataAttributeValue] | None,
        ) -> dict[str, MetadataAttributeValue]:
            """Build notfound-specific kwargs from fields and context.

            Returns:
                Dictionary of notfound kwargs

            """
            notfound_kwargs: dict[str, MetadataAttributeValue] = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }

            # Convert extra_kwargs to MetadataAttributeValue
            valid_extra = {
                k: v
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

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            error_code: str = FlextConstants.Errors.NOT_FOUND_ERROR,
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize not found error with resource context."""
            # Extract context values
            (
                correlation_id_val,
                metadata_val,
                auto_log_val,
                auto_correlation_val,
            ) = FlextExceptions.NotFoundError._extract_context_values(context)

            # Build notfound kwargs
            notfound_kwargs = FlextExceptions.NotFoundError._build_notfound_kwargs(
                resource_type,
                resource_id,
                extra_kwargs,
                context,
            )

            # Build context dict with normalized values
            notfound_context: dict[str, MetadataAttributeValue] = {}
            if auto_log_val:
                notfound_context["auto_log"] = auto_log_val
            if auto_correlation_val:
                notfound_context["auto_correlation"] = auto_correlation_val
            if correlation_id_val is not None:
                notfound_context["correlation_id"] = correlation_id_val
            if metadata_val is not None:
                # Extract attributes from Metadata model
                notfound_context.update(dict(metadata_val.attributes.items()))

            # Add notfound_kwargs to context

            for k, v in notfound_kwargs.items():
                notfound_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=notfound_context or None,
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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize conflict error with resource context."""
            conflict_context: dict[str, MetadataAttributeValue] = {}
            if context is not None:
                conflict_context.update(context)
            conflict_context.update(extra_kwargs)
            super().__init__(
                message,
                error_code=error_code,
                context=conflict_context or None,
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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize rate limit error with limit context."""
            rate_limit_context: dict[str, MetadataAttributeValue] = {}
            if context is not None:
                rate_limit_context.update(context)
            rate_limit_context.update(extra_kwargs)
            super().__init__(
                message,
                error_code=error_code,
                context=rate_limit_context or None,
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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize circuit breaker error with service context."""
            cb_context: dict[str, MetadataAttributeValue] = {}
            if context is not None:
                cb_context.update(context)
            cb_context.update(extra_kwargs)
            super().__init__(
                message,
                error_code=error_code,
                context=cb_context or None,
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
            expected_type: type | None = None,
            actual_type: type | None = None,
            error_code: str = FlextConstants.Errors.TYPE_ERROR,
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize type error with type information."""
            type_context: dict[str, MetadataAttributeValue] = {}
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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize operation error with operation context."""
            op_context: dict[str, MetadataAttributeValue] = {}
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
            context: Mapping[str, MetadataAttributeValue] | None = None,
            **extra_kwargs: MetadataAttributeValue,
        ) -> None:
            """Initialize attribute access error with attribute context."""
            attr_context: dict[str, MetadataAttributeValue] = {}
            if context is not None:
                attr_context.update(context)
            attr_context.update({
                "attribute_name": attribute_name,
                "attribute_context": attribute_context,
                **extra_kwargs,
            })
            super().__init__(
                message,
                error_code=error_code,
                context=attr_context or None,
            )
            self.attribute_name = attribute_name
            self.attribute_context = attribute_context

    @staticmethod
    def prepare_exception_kwargs(
        kwargs: dict[str, MetadataAttributeValue],
        specific_params: dict[str, MetadataAttributeValue] | None = None,
    ) -> tuple[
        str | None,
        MetadataAttributeValue,
        bool,
        bool,
        MetadataAttributeValue,
        dict[str, MetadataAttributeValue],
    ]:
        """Prepare exception kwargs by extracting common parameters."""
        if specific_params:
            for key, value in specific_params.items():
                if value is not None:
                    kwargs.setdefault(key, value)
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
        kwargs: Mapping[str, MetadataAttributeValue],
    ) -> tuple[str | None, Metadata | None]:
        """Extract correlation_id and metadata from kwargs.

        Returns typed values: correlation_id as str | None, metadata as Metadata | None.
        """
        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id: str | None = (
            str(correlation_id_raw)
            if correlation_id_raw is not None and isinstance(correlation_id_raw, str)
            else None
        )
        metadata_raw = kwargs.get("metadata")
        metadata: Metadata | None = (
            metadata_raw if isinstance(metadata_raw, Metadata) else None
        )
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
        kwargs: Mapping[str, MetadataAttributeValue],
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
    def _get_error_creator(
        error_type: str,
    ) -> (
        Callable[
            [
                str,
                str | None,
                Mapping[str, MetadataAttributeValue],
                str | None,
                Metadata | None,
            ],
            BaseError,
        ]
        | None
    ):
        creators = {
            "validation": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.ValidationError(
                msg,
                error_code=code or FlextConstants.Errors.VALIDATION_ERROR,
                field=kwargs.get("field"),
                value=kwargs.get("value"),
                correlation_id=cid,
                metadata=meta,
            ),
            "configuration": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.ConfigurationError(
                msg,
                error_code=code or FlextConstants.Errors.CONFIGURATION_ERROR,
                config_key=kwargs.get("config_key"),
                config_source=kwargs.get("config_source"),
                correlation_id=cid,
                metadata=meta,
            ),
            "operation": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.OperationError(
                msg,
                error_code=code or FlextConstants.Errors.OPERATION_ERROR,
                operation=kwargs.get("operation"),
                reason=kwargs.get("reason"),
                correlation_id=cid,
                metadata=meta,
            ),
            "connection": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.ConnectionError(
                msg,
                error_code=code or FlextConstants.Errors.CONNECTION_ERROR,
                host=kwargs.get("host"),
                port=kwargs.get("port"),
                timeout=kwargs.get("timeout"),
                correlation_id=cid,
                metadata=meta,
            ),
            "timeout": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.TimeoutError(
                msg,
                error_code=code or FlextConstants.Errors.TIMEOUT_ERROR,
                timeout_seconds=kwargs.get("timeout_seconds"),
                operation=kwargs.get("operation"),
                correlation_id=cid,
                metadata=meta,
            ),
            "authorization": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.AuthorizationError(
                msg,
                error_code=code or FlextConstants.Errors.AUTHORIZATION_ERROR,
                user_id=kwargs.get("user_id"),
                resource=kwargs.get("resource"),
                permission=kwargs.get("permission"),
                correlation_id=cid,
                metadata=meta,
            ),
            "authentication": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.AuthenticationError(
                msg,
                error_code=code or FlextConstants.Errors.AUTHENTICATION_ERROR,
                auth_method=kwargs.get("auth_method"),
                user_id=kwargs.get("user_id"),
                correlation_id=cid,
                metadata=meta,
            ),
            "not_found": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.NotFoundError(
                msg,
                error_code=code or FlextConstants.Errors.NOT_FOUND_ERROR,
                resource_type=kwargs.get("resource_type"),
                resource_id=kwargs.get("resource_id"),
                correlation_id=cid,
                metadata=meta,
            ),
            "attribute_access": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.AttributeAccessError(
                msg,
                error_code=code or FlextConstants.Errors.ATTRIBUTE_ERROR,
                attribute_name=kwargs.get("attribute_name"),
                attribute_context=kwargs.get("attribute_context"),
                correlation_id=cid,
                metadata=meta,
            ),
        }
        return creators.get(error_type)

    @staticmethod
    def _create_error_by_type(
        error_type: str | None,
        message: str,
        error_code: str | None,
        context: Mapping[str, MetadataAttributeValue] | None = None,
    ) -> FlextExceptions.BaseError:
        """Create error by type using context dict."""
        # Build context with error_code
        error_context: dict[str, MetadataAttributeValue] = {}
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
        **kwargs: MetadataAttributeValue,
    ) -> FlextExceptions.BaseError:
        """Create an appropriate exception instance based on kwargs context."""
        correlation_id_obj, metadata_obj = FlextExceptions.extract_common_kwargs(kwargs)
        error_type = FlextExceptions._determine_error_type(kwargs)
        # Convert correlation_id_obj to str | None
        correlation_id: str | None = (
            str(correlation_id_obj)
            if correlation_id_obj is not None and isinstance(correlation_id_obj, str)
            else None
        )
        # Convert metadata_obj to dict | None (only pass dicts to _create_error_by_type)
        if isinstance(metadata_obj, dict):
            pass
        # Note: if metadata_obj is dict-like but not a dict (Mapping interface),
        # we don't convert it since Metadata expects concrete dict with MetadataAttributeValue values
        # Build context dict

        error_context: dict[str, MetadataAttributeValue] = {}
        if correlation_id is not None:
            error_context["correlation_id"] = correlation_id
        if metadata_obj is not None and isinstance(metadata_obj, Metadata):
            # Extract attributes from Metadata model
            error_context.update(dict(metadata_obj.attributes.items()))
        for k, v in kwargs.items():
            if k not in {"correlation_id", "metadata"}:
                error_context[k] = FlextRuntime.normalize_to_metadata_value(v)

        return FlextExceptions._create_error_by_type(
            error_type,
            message,
            error_code,
            context=error_context or None,
        )

    _exception_counts: ClassVar[FlextTypes.Types.ExceptionMetricsMapping] = {}

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
        exception_counts_dict: dict[str, int] = {}
        for exc_type, count in cls._exception_counts.items():
            exc_name = (
                exc_type.__qualname__
                if hasattr(exc_type, "__qualname__")
                else str(exc_type)
            )
            exception_counts_dict[exc_name] = count
        # Build result dict matching ErrorTypeMapping type
        # ErrorTypeMapping allows dict[str, int] for exception_counts
        # Cast to satisfy type checker - dict[str, int] is compatible with dict[str, str | int | ...]
        result: FlextTypes.Types.ErrorTypeMapping = {
            "total_exceptions": total,
            "exception_counts": cast(
                "dict[str, str | int | float | bool | list[str | int | float | bool | None] | dict[str, str | int | float | bool | None] | None]",
                exception_counts_dict,
            ),
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
        # Normalize ExceptionKwargsType to MetadataAttributeValue

        normalized_kwargs: dict[str, MetadataAttributeValue] = {}
        for k, v in kwargs.items():
            normalized_kwargs[k] = FlextRuntime.normalize_to_metadata_value(v)
        return self.create(message, error_code, **normalized_kwargs)


__all__ = ["FlextExceptions"]
