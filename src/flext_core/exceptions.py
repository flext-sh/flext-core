"""Exception hierarchy with correlation metadata.

Provides structured exceptions with error codes and correlation tracking
for consistent error handling across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import ClassVar, TypeGuard

from flext_core.constants import c
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

# Use FlextRuntime.Metadata to avoid importing from _models.base
# This maintains proper architecture layering (exceptions.py is Tier 1)
_Metadata = FlextRuntime.Metadata


def _is_metadata_protocol(obj: object) -> TypeGuard[p.Log.Metadata]:
    """TypeGuard to check if object implements the Metadata protocol."""
    return (
        hasattr(obj, "created_at")
        and hasattr(obj, "updated_at")
        and hasattr(obj, "version")
        and hasattr(obj, "attributes")
    )


class FlextExceptions:
    """Exception types with correlation metadata.

    Provides structured exceptions with error codes and correlation tracking
    for consistent error handling and logging.
    """

    class BaseError(Exception):
        """Base exception with correlation metadata and error codes.

        All FLEXT exceptions inherit from this to ensure consistent error
        handling, logging, and correlation tracking across the ecosystem.
        """

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.UNKNOWN_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            metadata: p.Log.Metadata
            | Mapping[str, t.MetadataAttributeValue]
            | t.GeneralValueType
            | None = None,
            correlation_id: str | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
            merged_kwargs: dict[str, t.MetadataAttributeValue] | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize base error with message and optional metadata.

            Args:
                message: Error message
                error_code: Optional error code
                context: Optional context mapping
                metadata: Optional metadata (_Metadata, dict, or t.GeneralValueType)
                correlation_id: Optional correlation ID
                auto_correlation: Auto-generate correlation ID if not provided
                auto_log: Auto-log error on creation
                merged_kwargs: Additional metadata attributes to merge

            """
            super().__init__(message)
            self.message = message
            self.error_code = error_code

            # Merge context and extra_kwargs into final_kwargs
            final_kwargs: dict[str, t.MetadataAttributeValue] = (
                dict(merged_kwargs) if merged_kwargs else {}
            )

            # Merge context and normalize extra_kwargs using FlextRuntime
            if context:
                final_kwargs.update(dict(context))
            if extra_kwargs:
                normalized_extra = {
                    k: FlextRuntime.normalize_to_metadata_value(v)
                    for k, v in extra_kwargs.items()
                }
                final_kwargs.update(normalized_extra)

            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if auto_correlation and not correlation_id
                else correlation_id
            )
            # Convert metadata to proper type for _normalize_metadata
            # _normalize_metadata expects _Metadata | Mapping[str, t.MetadataAttributeValue] | t.GeneralValueType | None
            metadata_for_normalize: (
                p.Log.Metadata
                | Mapping[str, t.MetadataAttributeValue]
                | t.GeneralValueType
                | None
            ) = metadata
            self.metadata = e.BaseError._normalize_metadata(
                metadata_for_normalize,
                final_kwargs,
            )
            self.timestamp = time.time()
            self.auto_log = auto_log

        def __str__(self) -> str:
            """Return string representation with error code if present."""
            if self.error_code:
                return f"[{self.error_code}] {self.message}"
            return self.message

        def to_dict(self) -> dict[str, t.GeneralValueType]:
            """Convert exception to dictionary representation.

            Returns:
                Dictionary with error_type, message, error_code, and other fields.

            """
            result: dict[str, t.GeneralValueType] = {
                "error_type": type(self).__name__,
                "message": self.message,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
            }

            # Add metadata attributes (only keys not in result)
            if self.metadata and self.metadata.attributes:
                filtered_attrs = {
                    k: v for k, v in self.metadata.attributes.items() if k not in result
                }
                result.update(filtered_attrs)
            return result

        @staticmethod
        def _normalize_metadata_from_dict(
            metadata_dict: Mapping[str, t.GeneralValueType],
            merged_kwargs: dict[str, t.MetadataAttributeValue],
        ) -> _Metadata:
            """Normalize metadata from dict-like object."""
            # Use MetadataAttributeDict - normalize_to_metadata_value returns MetadataAttributeValue
            merged_attrs: dict[str, t.MetadataAttributeValue] = {}

            # Normalize metadata_dict values
            for k, v in metadata_dict.items():
                merged_attrs[k] = FlextRuntime.normalize_to_metadata_value(v)

            # Normalize merged_kwargs values
            if merged_kwargs:
                for k, v in merged_kwargs.items():
                    merged_attrs[k] = FlextRuntime.normalize_to_metadata_value(v)

            return _Metadata(attributes=merged_attrs)

        @staticmethod
        def _normalize_metadata(
            metadata: p.Log.Metadata
            | Mapping[str, t.MetadataAttributeValue]
            | t.GeneralValueType
            | None,
            merged_kwargs: dict[str, t.MetadataAttributeValue],
        ) -> _Metadata:
            """Normalize metadata from various input types to _Metadata model.

            Args:
                metadata: _Metadata instance, dict-like object, or None
                merged_kwargs: Additional attributes to merge

            Returns:
                Normalized _Metadata instance

            """
            if metadata is None:
                if merged_kwargs:
                    normalized_attrs = {
                        k: FlextRuntime.normalize_to_metadata_value(v)
                        for k, v in merged_kwargs.items()
                    }
                    return _Metadata(attributes=normalized_attrs)
                return _Metadata(attributes={})

            if isinstance(metadata, _Metadata):
                if merged_kwargs:
                    existing_attrs = metadata.attributes
                    # Use MetadataAttributeDict - normalize_to_metadata_value returns MetadataAttributeValue
                    new_attrs: dict[str, t.MetadataAttributeValue] = {}

                    # Combine attributes - use ConfigurationDict for combining then normalize
                    combined_attrs: dict[str, t.GeneralValueType] = dict(existing_attrs)
                    # merged_kwargs values are MetadataAttributeValue (subset of GeneralValueType)
                    # Add merged_kwargs with explicit type handling
                    combined_attrs.update(dict(merged_kwargs.items()))
                    # Normalize all values - result is MetadataAttributeValue
                    for k, v_combined in combined_attrs.items():
                        # normalize_to_metadata_value returns MetadataAttributeValue
                        new_attrs[k] = FlextRuntime.normalize_to_metadata_value(
                            v_combined,
                        )
                    return _Metadata(attributes=new_attrs)
                return metadata

            # Check if it's a Mapping (covers both dict-like objects and protocol instances)
            if isinstance(metadata, Mapping):
                # Convert to dict for type safety
                metadata_dict = dict(metadata.items())
                return e.BaseError._normalize_metadata_from_dict(
                    metadata_dict,
                    merged_kwargs,
                )

            # Fallback: convert to _Metadata with string value
            return _Metadata(attributes={"value": str(metadata)})

    # Specific exception classes with minimal code
    class ValidationError(BaseError):
        """Exception raised for input validation failures."""

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: t.MetadataAttributeValue | None = None,
            error_code: str = c.Errors.VALIDATION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            _correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize validation error with field and value information."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Build validation context
            validation_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    validation_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                validation_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if field is not None:
                validation_context["field"] = field
            if value is not None:
                validation_context["value"] = FlextRuntime.normalize_to_metadata_value(
                    value,
                )

            super().__init__(
                message,
                error_code=error_code,
                context=validation_context or None,
                metadata=preserved_metadata,
                correlation_id=str(preserved_corr_id)
                if preserved_corr_id is not None
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
            error_code: str = c.Errors.CONFIGURATION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            _correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize configuration error with config context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Build config context
            config_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    config_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                config_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if config_key is not None:
                config_context["config_key"] = config_key
            if config_source is not None:
                config_context["config_source"] = config_source

            super().__init__(
                message,
                error_code=error_code,
                context=config_context or None,
                metadata=preserved_metadata,
                correlation_id=str(preserved_corr_id)
                if preserved_corr_id is not None
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
            error_code: str = c.Errors.CONNECTION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            _correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize connection error with network context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Build connection context
            conn_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    conn_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                conn_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=conn_context or None,
                metadata=preserved_metadata,
                correlation_id=str(preserved_corr_id)
                if preserved_corr_id is not None
                else None,
            )
            self.host = conn_context.get("host") if conn_context else None
            self.port = conn_context.get("port") if conn_context else None
            self.timeout = conn_context.get("timeout") if conn_context else None

    class TimeoutError(BaseError):
        """Exception raised for operation timeout errors."""

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            operation: str | None = None,
            error_code: str = c.Errors.TIMEOUT_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            _correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize timeout error with timeout context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Build timeout context
            timeout_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    timeout_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                timeout_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if timeout_seconds is not None:
                timeout_context["timeout_seconds"] = timeout_seconds
            if operation is not None:
                timeout_context["operation"] = operation

            super().__init__(
                message,
                error_code=error_code,
                context=timeout_context or None,
                metadata=preserved_metadata,
                correlation_id=str(preserved_corr_id)
                if preserved_corr_id is not None
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
            error_code: str = c.Errors.AUTHENTICATION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            _correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize authentication error with auth context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Build auth context
            auth_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    auth_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                auth_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if auth_method is not None:
                auth_context["auth_method"] = auth_method
            if user_id is not None:
                auth_context["user_id"] = user_id

            super().__init__(
                message,
                error_code=error_code,
                context=auth_context or None,
                metadata=preserved_metadata,
                correlation_id=str(preserved_corr_id)
                if preserved_corr_id is not None
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
            error_code: str = c.Errors.AUTHORIZATION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize authorization error with permission context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Build authorization context
            authz_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    authz_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                authz_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=authz_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else (
                    str(preserved_corr_id) if preserved_corr_id is not None else None
                ),
            )
            self.user_id = authz_context.get("user_id") if authz_context else None
            self.resource = authz_context.get("resource") if authz_context else None
            self.permission = authz_context.get("permission") if authz_context else None

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        @staticmethod
        def _extract_context_values(
            context: Mapping[str, t.MetadataAttributeValue] | None,
        ) -> tuple[str | None, p.Log.Metadata | None, bool, bool]:
            """Extract context values from mapping.

            Returns:
                Tuple of (correlation_id, metadata, auto_log, auto_correlation)

            """
            if context is None:
                return (None, None, False, False)

            corr_id = context.get("correlation_id")
            correlation_id_val: str | None = (
                corr_id if isinstance(corr_id, str) else None
            )

            metadata_obj = context.get("metadata")
            # Use module-level TypeGuard for protocol compliance
            metadata_val: p.Log.Metadata | None = (
                metadata_obj if _is_metadata_protocol(metadata_obj) else None
            )

            auto_log_obj = context.get("auto_log")
            auto_log_val: bool = (
                auto_log_obj if isinstance(auto_log_obj, bool) else False
            )

            auto_corr_obj = context.get("auto_correlation")
            auto_correlation_val: bool = (
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
            extra_kwargs: Mapping[str, t.MetadataAttributeValue],
            context: Mapping[str, t.MetadataAttributeValue] | None,
        ) -> dict[str, t.MetadataAttributeValue]:
            """Build notfound-specific kwargs from fields and context.

            Returns:
                Dictionary of notfound kwargs

            """
            notfound_kwargs: dict[str, t.MetadataAttributeValue] = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }

            # Convert extra_kwargs to t.MetadataAttributeValue and normalize values
            for k, v in extra_kwargs.items():
                if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                    notfound_kwargs[k] = FlextRuntime.normalize_to_metadata_value(v)

            # Add context items (excluding reserved keys)
            if context is not None:
                excluded_keys = {
                    "correlation_id",
                    "metadata",
                    "auto_log",
                    "auto_correlation",
                }
                notfound_kwargs.update({
                    k: v
                    for k, v in context.items()
                    if k not in excluded_keys
                    and isinstance(
                        v,
                        (str, int, float, bool, type(None), list, dict),
                    )
                })

            return notfound_kwargs

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            error_code: str = c.Errors.NOT_FOUND_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            metadata: p.Log.Metadata
            | Mapping[str, t.MetadataAttributeValue]
            | t.GeneralValueType
            | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize not found error with resource context."""
            # Preserve metadata from extra_kwargs (correlation_id is
            # consumed by the explicit parameter and never reaches **extra_kwargs)
            preserved_metadata = extra_kwargs.pop("metadata", None)
            extra_kwargs.pop("correlation_id", None)

            # Use explicit params or preserved from extra_kwargs
            final_metadata = metadata if metadata is not None else preserved_metadata
            final_corr_id = correlation_id

            # Build notfound context
            notfound_context: dict[str, t.MetadataAttributeValue] = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }

            # Normalize extra_kwargs
            if extra_kwargs:
                for k, v in extra_kwargs.items():
                    notfound_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            # Add context items (excluding reserved keys)
            if context is not None:
                excluded_keys = {"correlation_id", "metadata"}
                for k, v in context.items():
                    if k not in excluded_keys:
                        notfound_context[k] = FlextRuntime.normalize_to_metadata_value(
                            v,
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
            error_code: str = c.Errors.ALREADY_EXISTS,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize conflict error with resource context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            conflict_context: dict[str, t.MetadataAttributeValue] = {}
            if context is not None:
                for k, v in context.items():
                    conflict_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            # Normalize extra_kwargs
            if extra_kwargs:
                for k, v in extra_kwargs.items():
                    conflict_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=conflict_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else (
                    str(preserved_corr_id) if preserved_corr_id is not None else None
                ),
            )
            self.resource_type = (
                conflict_context.get("resource_type") if conflict_context else None
            )
            self.resource_id = (
                conflict_context.get("resource_id") if conflict_context else None
            )
            self.conflict_reason = (
                conflict_context.get("conflict_reason") if conflict_context else None
            )

    class RateLimitError(BaseError):
        """Exception raised when rate limits are exceeded."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.OPERATION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize rate limit error with limit context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            rate_limit_context: dict[str, t.MetadataAttributeValue] = {}
            if context is not None:
                for k, v in context.items():
                    rate_limit_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            # Normalize extra_kwargs
            if extra_kwargs:
                for k, v in extra_kwargs.items():
                    rate_limit_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=rate_limit_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else (
                    str(preserved_corr_id) if preserved_corr_id is not None else None
                ),
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
            error_code: str = c.Errors.EXTERNAL_SERVICE_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize circuit breaker error with service context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            cb_context: dict[str, t.MetadataAttributeValue] = {}
            if context is not None:
                for k, v in context.items():
                    cb_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            # Normalize extra_kwargs
            if extra_kwargs:
                for k, v in extra_kwargs.items():
                    cb_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            super().__init__(
                message,
                error_code=error_code,
                context=cb_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else (
                    str(preserved_corr_id) if preserved_corr_id is not None else None
                ),
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

        @staticmethod
        def _get_type_map() -> dict[str, type]:
            """Get mapping of type names to actual types."""
            return {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "bytes": bytes,
            }

        @staticmethod
        def _normalize_type(
            type_value: type | str | None,
            type_map: dict[str, type],
            extra_kwargs: dict[str, t.MetadataAttributeValue],
            key: str,
        ) -> type | None:
            """Normalize type value from various sources."""
            # Extract from extra_kwargs if not provided as named arg
            if type_value is None and key in extra_kwargs:
                type_raw = extra_kwargs.pop(key)
                if isinstance(type_raw, str):
                    # Use dict.get directly for type-safe lookup
                    return type_map.get(type_raw)
                if isinstance(type_raw, type):
                    return type_raw
            # Handle case where type is passed as string in named arg
            if isinstance(type_value, str):
                # Use dict.get directly for type-safe lookup
                return type_map.get(type_value)
            # type_value is already type | None at this point (str case handled above)
            if isinstance(type_value, type):
                return type_value
            return None

        @staticmethod
        def _build_type_context(
            expected_type: type | str | None,
            actual_type: type | str | None,
            context: Mapping[str, t.MetadataAttributeValue] | None,
            extra_kwargs: dict[str, t.MetadataAttributeValue],
        ) -> dict[str, t.MetadataAttributeValue]:
            """Build type context dictionary."""
            # Build context from context and extra_kwargs
            type_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    type_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                type_context[k] = FlextRuntime.normalize_to_metadata_value(v)

            # Handle both type objects and string representations
            if expected_type is not None:
                if isinstance(expected_type, str):
                    type_context["expected_type"] = expected_type
                else:
                    type_context["expected_type"] = expected_type.__qualname__
            else:
                type_context["expected_type"] = None
            if actual_type is not None:
                if isinstance(actual_type, str):
                    type_context["actual_type"] = actual_type
                else:
                    type_context["actual_type"] = actual_type.__qualname__
            else:
                type_context["actual_type"] = None

            return type_context

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.TYPE_ERROR,
            expected_type: type | None = None,
            actual_type: type | None = None,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize type error with type information."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            type_map = self._get_type_map()

            # Normalize types from various sources
            expected_type = self._normalize_type(
                expected_type,
                type_map,
                extra_kwargs,
                "expected_type",
            )
            actual_type = self._normalize_type(
                actual_type,
                type_map,
                extra_kwargs,
                "actual_type",
            )

            # Build type context
            type_context = self._build_type_context(
                expected_type,
                actual_type,
                context,
                extra_kwargs,
            )

            super().__init__(
                message,
                error_code=error_code,
                context=type_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else (
                    str(preserved_corr_id) if preserved_corr_id is not None else None
                ),
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
            error_code: str = c.Errors.OPERATION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize operation error with operation context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Build operation context
            op_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    op_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                op_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if operation is not None:
                op_context["operation"] = operation
            if reason is not None:
                op_context["reason"] = reason

            super().__init__(
                message,
                error_code=error_code,
                context=op_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else (
                    str(preserved_corr_id) if preserved_corr_id is not None else None
                ),
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
            error_code: str = c.Errors.ATTRIBUTE_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize attribute access error with attribute context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Build attribute context
            attr_context: dict[str, t.MetadataAttributeValue] = {}
            if context:
                for k, v in context.items():
                    attr_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            for k, v in extra_kwargs.items():
                attr_context[k] = FlextRuntime.normalize_to_metadata_value(v)
            if attribute_name is not None:
                attr_context["attribute_name"] = attribute_name
            if attribute_context is not None:
                attr_context["attribute_context"] = attribute_context

            super().__init__(
                message,
                error_code=error_code,
                context=attr_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else (
                    str(preserved_corr_id) if preserved_corr_id is not None else None
                ),
            )
            self.attribute_name = attribute_name
            self.attribute_context = attribute_context

    @staticmethod
    def prepare_exception_kwargs(
        kwargs: dict[str, t.MetadataAttributeValue],
        specific_params: dict[str, t.MetadataAttributeValue] | None = None,
    ) -> tuple[
        str | None,
        t.MetadataAttributeValue,
        bool,
        bool,
        t.MetadataAttributeValue,
        dict[str, t.MetadataAttributeValue],
    ]:
        """Prepare exception kwargs by extracting common parameters."""
        if specific_params:
            # Filter out None values - specific_params is already MetadataAttributeDict (dict)
            filtered_params = {
                k: v for k, v in specific_params.items() if v is not None
            }
            kwargs.update(filtered_params)
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
        kwargs: Mapping[str, t.MetadataAttributeValue],
    ) -> tuple[
        str | None,
        p.Log.Metadata | Mapping[str, t.MetadataAttributeValue] | None,
    ]:
        """Extract correlation_id and metadata from kwargs.

        Returns typed values: correlation_id as str | None, metadata as _Metadata | Mapping | None.
        """
        correlation_id_raw = kwargs.get("correlation_id")
        # Use isinstance for proper type narrowing
        correlation_id: str | None = (
            correlation_id_raw if isinstance(correlation_id_raw, str) else None
        )
        metadata_raw = kwargs.get("metadata")
        metadata: p.Log.Metadata | Mapping[str, t.MetadataAttributeValue] | None = None
        if metadata_raw is not None and _is_metadata_protocol(metadata_raw):
            metadata = metadata_raw
        elif isinstance(metadata_raw, Mapping):
            metadata = dict(metadata_raw.items())
        return (correlation_id, metadata)

    @staticmethod
    def create_error(error_type: str, message: str) -> e.BaseError:
        """Create an exception instance based on error type."""
        error_classes: dict[str, type[e.BaseError]] = {
            "ValidationError": e.ValidationError,
            "ConfigurationError": e.ConfigurationError,
            "ConnectionError": e.ConnectionError,
            "TimeoutError": e.TimeoutError,
            "AuthenticationError": e.AuthenticationError,
            "AuthorizationError": e.AuthorizationError,
            "NotFoundError": e.NotFoundError,
            "ConflictError": e.ConflictError,
            "RateLimitError": e.RateLimitError,
            "CircuitBreakerError": e.CircuitBreakerError,
            "TypeError": e.TypeError,
            "OperationError": e.OperationError,
            "AttributeError": e.AttributeAccessError,
        }
        error_class = error_classes.get(error_type)
        if error_class is None:
            msg = f"Unknown error type: {error_type}"
            raise ValueError(msg)
        return error_class(message)

    @staticmethod
    def _determine_error_type(
        kwargs: Mapping[str, t.MetadataAttributeValue],
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
        meta: p.Log.Metadata | None,
    ) -> t.ConfigMap | None:
        """Prepare metadata value for error creation."""
        return meta.attributes if meta is not None else None

    @staticmethod
    def _get_str_from_kwargs(
        kwargs: Mapping[str, t.MetadataAttributeValue],
        key: str,
    ) -> str | None:
        """Extract and convert value from kwargs to string."""
        val = kwargs.get(key)
        return str(val) if val is not None else None

    @staticmethod
    def _create_error_by_type(
        error_type: str | None,
        message: str,
        error_code: str | None,
        context: Mapping[str, t.MetadataAttributeValue] | None = None,
    ) -> e.BaseError:
        """Create error by type using context dict."""
        # Build context with error_code
        error_context: dict[str, t.MetadataAttributeValue] = {}
        if context is not None:
            error_context.update(context)
        if error_code is not None:
            error_context["error_code"] = error_code

        # Create appropriate error class based on type
        error_classes: dict[str, type[e.BaseError]] = {
            "validation": e.ValidationError,
            "configuration": e.ConfigurationError,
            "connection": e.ConnectionError,
            "timeout": e.TimeoutError,
            "authentication": e.AuthenticationError,
            "authorization": e.AuthorizationError,
            "not_found": e.NotFoundError,
            "operation": e.OperationError,
            "attribute_access": e.AttributeAccessError,
        }

        error_class = error_classes.get(error_type) if error_type else None
        # Extract correlation_id if present in context
        correlation_id = None
        if error_context and "correlation_id" in error_context:
            # We can pass it as kwarg because all error classes accept **extra_kwargs
            # and they look for 'correlation_id' in extra_kwargs
            correlation_id = str(error_context["correlation_id"])

        if error_class is not None:
            # Type narrowing: error_class is type[e.BaseError] after None check
            return error_class(
                message,
                error_code=error_code or c.Errors.UNKNOWN_ERROR,
                context=error_context or None,
                correlation_id=correlation_id,
            )

        return e.BaseError(
            message,
            error_code=error_code or c.Errors.UNKNOWN_ERROR,
            context=error_context or None,
            correlation_id=correlation_id,
        )

    @staticmethod
    def _merge_metadata_into_context(
        context: dict[str, t.MetadataAttributeValue],
        metadata_obj: p.Log.Metadata | Mapping[str, t.MetadataAttributeValue] | None,
    ) -> None:
        """Merge metadata object into context dictionary."""
        if metadata_obj is None:
            return

        if isinstance(metadata_obj, _Metadata):
            # Extract attributes from _Metadata model
            attrs = metadata_obj.attributes
            if isinstance(attrs, t.ConfigMap):
                for k, v in attrs.root.items():
                    normalized: t.MetadataAttributeValue = (
                        FlextRuntime.normalize_to_metadata_value(v)
                    )
                    context[k] = normalized
            elif isinstance(attrs, Mapping):
                for k, v in attrs.items():
                    normalized: t.MetadataAttributeValue = (
                        FlextRuntime.normalize_to_metadata_value(v)
                    )
                    context[k] = normalized
        elif isinstance(metadata_obj, dict):
            # Direct dict - normalize values and update
            for k, v in metadata_obj.items():
                normalized_dict: t.MetadataAttributeValue = (
                    FlextRuntime.normalize_to_metadata_value(v)
                )
                context[k] = normalized_dict
        elif isinstance(metadata_obj, Mapping):
            # Mapping object - normalize and update
            for k, v in metadata_obj.items():
                normalized_mapping: t.MetadataAttributeValue = (
                    FlextRuntime.normalize_to_metadata_value(v)
                )
                context[k] = normalized_mapping

    @staticmethod
    def _build_error_context(
        correlation_id: str | None,
        metadata_obj: p.Log.Metadata | Mapping[str, t.MetadataAttributeValue] | None,
        kwargs: dict[str, t.MetadataAttributeValue],
    ) -> dict[str, t.MetadataAttributeValue]:
        """Build error context dictionary."""
        error_context: dict[str, t.MetadataAttributeValue] = {}
        if correlation_id is not None:
            error_context["correlation_id"] = correlation_id

        e._merge_metadata_into_context(error_context, metadata_obj)

        for k, v in kwargs.items():
            if k not in {"correlation_id", "metadata"}:
                error_context[k] = FlextRuntime.normalize_to_metadata_value(v)
        return error_context

    @staticmethod
    def create(
        message: str,
        error_code: str | None = None,
        **kwargs: t.MetadataAttributeValue,
    ) -> e.BaseError:
        """Create an appropriate exception instance based on kwargs context."""
        correlation_id_obj, metadata_obj = e.extract_common_kwargs(kwargs)
        error_type = e._determine_error_type(kwargs)
        # correlation_id_obj is already str | None from extract_common_kwargs
        correlation_id: str | None = correlation_id_obj

        error_context = e._build_error_context(correlation_id, metadata_obj, kwargs)

        return e._create_error_by_type(
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
    def get_metrics(cls) -> dict[str, str | int | dict[str, int]]:
        """Get exception metrics and statistics."""
        total = sum(cls._exception_counts.values(), 0)
        # Serialize exception counts as a single string for compatibility with ErrorTypeMapping
        exception_counts_list = [
            f"{exc_type.__qualname__ if hasattr(exc_type, '__qualname__') else str(exc_type)}:{count}"
            for exc_type, count in cls._exception_counts.items()
        ]
        exception_counts_str = ";".join(exception_counts_list)
        # Build exception_counts dict for test compatibility
        # Use type compatible with ErrorTypeMapping value type
        exception_counts_dict: dict[str, int] = {}
        for exc_type, count in cls._exception_counts.items():
            exc_name = (
                exc_type.__qualname__
                if hasattr(exc_type, "__qualname__")
                else str(exc_type)
            )
            exception_counts_dict[exc_name] = count
        # Build result dict matching ErrorTypeMapping type
        # Values are int | str | dict[str, int] - all compatible with ErrorTypeMapping
        result_dict: dict[str, str | int | dict[str, int]] = {
            "total_exceptions": total,
            "exception_counts": exception_counts_dict,
            "exception_counts_summary": exception_counts_str,
            "unique_exception_types": len(cls._exception_counts),
        }
        return result_dict

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._exception_counts.clear()

    def __call__(
        self,
        message: str,
        error_code: str | None = None,
        **kwargs: t.MetadataAttributeValue,
    ) -> e.BaseError:
        """Create exception by calling the class instance."""
        # Normalize ExceptionKwargsType to t.MetadataAttributeValue
        # normalize_to_metadata_value already returns t.MetadataAttributeValue
        normalized_kwargs: dict[str, t.MetadataAttributeValue] = {}
        for k, v in kwargs.items():
            normalized_kwargs[k] = FlextRuntime.normalize_to_metadata_value(v)
        return self.create(message, error_code, **normalized_kwargs)


e = FlextExceptions

__all__ = ["FlextExceptions", "e"]
