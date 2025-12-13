"""Structured exception hierarchy for dispatcher-aware foundations.

Layer 1 defines typed errors with correlation metadata so dispatcher pipelines
and services can rely on FlextResult without losing context. The module keeps
logging integration and error codes centralized while respecting the clean
architecture boundary from infrastructure into domain/application flows.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from typing import ClassVar, cast

from flext_core._models.base import FlextModelsBase
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

# Type alias matching facade naming - same class as m.Metadata
_Metadata = FlextModelsBase.Metadata


# Helper functions to replace u.mapper() calls and break circular dependency
# NOTE: These functions avoid importing utilities to prevent circular dependencies
def _to_dict(
    value: t.GeneralValueType | Mapping[str, t.GeneralValueType] | None,
) -> dict[str, t.GeneralValueType]:
    """Convert value to dict - replaces u.mapper().to_dict()."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    return {}


def _transform_values(
    data: Mapping[str, t.GeneralValueType],
    transform: Callable[[t.GeneralValueType], t.GeneralValueType],
) -> dict[str, t.GeneralValueType]:
    """Transform dict values - replaces u.mapper().transform_values()."""
    return {k: transform(v) for k, v in data.items()}


def _get_from_dict(
    data: Mapping[str, t.GeneralValueType],
    key: str,
    default: t.GeneralValueType | None = None,
) -> t.GeneralValueType | None:
    """Get value from dict - replaces u.mapper().get()."""
    return data.get(key, default) if isinstance(data, dict) else default


def _filter_dict(
    data: Mapping[str, t.GeneralValueType],
    predicate: Callable[[str, t.GeneralValueType], bool],
) -> dict[str, t.GeneralValueType]:
    """Filter dict items - replaces u.mapper().filter_dict()."""
    return {k: v for k, v in data.items() if predicate(k, v)}


def _get(
    data: Mapping[str, t.GeneralValueType] | t.GeneralValueType | Mapping[str, object],
    key: str,
    default: t.GeneralValueType | None = None,
) -> t.GeneralValueType | None:
    """Get value from dict-like object - replaces u.mapper().get()."""
    # Cast to t.GeneralValueType for is_dict_like check
    data_typed = data
    if not FlextRuntime.is_dict_like(data_typed):
        return default
    # Type narrowing: is_dict_like ensures data is Mapping-like
    # Cast to allow both t.GeneralValueType and object values (for type maps)
    data_mapping = cast("Mapping[str, t.GeneralValueType | object]", data)
    value = data_mapping.get(key, default)
    # Cast result back to t.GeneralValueType | None
    return cast("t.GeneralValueType | None", value)


class FlextExceptions:
    """Foundation error types with correlation metadata for CQRS flows.

    Exceptions in this namespace enrich failures with error codes and optional
    metadata so dispatcher-driven handlers can surface structured details
    through FlextResult or structured logging without bespoke wrappers.
    """

    class BaseError(Exception):
        """Base exception with correlation metadata and error codes.

        All FLEXT exceptions inherit from this to ensure consistent error
        handling, logging, and correlation tracking across the ecosystem.
        """

        # NOTE: Use FlextRuntime.normalize_to_metadata_value() directly - no wrapper needed

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
            merged_kwargs: t.MetadataAttributeDict | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
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
            final_kwargs: t.MetadataAttributeDict = (
                dict(merged_kwargs) if merged_kwargs else {}
            )
            # Use helper functions to merge context and normalize extra_kwargs
            if context:
                context_dict = _to_dict(context)
                # Type narrowing: context_dict is dict[str, t.GeneralValueType]
                # MetadataAttributeDict accepts t.GeneralValueType values
                final_kwargs.update(context_dict)
            if extra_kwargs:
                extra_kwargs_dict = _to_dict(extra_kwargs)
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                # Type narrowing: normalized_extra is dict[str, MetadataAttributeValue]
                # MetadataAttributeDict accepts MetadataAttributeValue values
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

        def to_dict(self) -> t.ConfigurationDict:
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
            result: t.ConfigurationDict = {
                "error_type": type(self).__name__,
                "message": self.message,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
            }
            # Add metadata attributes using mapper (only keys not in result)
            if self.metadata and self.metadata.attributes:
                attrs_dict = (
                    _to_dict(self.metadata.attributes)
                    if not isinstance(self.metadata.attributes, dict)
                    else self.metadata.attributes
                )
                filtered_attrs = _filter_dict(attrs_dict, lambda k, _v: k not in result)
                result.update(filtered_attrs)
            return result

        @staticmethod
        def _normalize_metadata_from_dict(
            metadata_dict: t.ConfigurationMapping,
            merged_kwargs: t.MetadataAttributeDict,
        ) -> _Metadata:
            """Normalize metadata from dict-like object."""
            # Normalize attributes to t.GeneralValueType for Metadata
            merged_attrs: t.ConfigurationDict = {}

            # Use mapper to normalize metadata_dict values
            metadata_dict_typed = (
                _to_dict(metadata_dict)
                if not isinstance(metadata_dict, dict)
                else metadata_dict
            )
            normalized_metadata = _transform_values(
                metadata_dict_typed,
                FlextRuntime.normalize_to_metadata_value,
            )
            merged_attrs.update(normalized_metadata)

            # Normalize merged_kwargs values using mapper
            if merged_kwargs:
                merged_kwargs_typed = (
                    _to_dict(merged_kwargs)
                    if not isinstance(merged_kwargs, dict)
                    else merged_kwargs
                )
                normalized_kwargs = _transform_values(
                    merged_kwargs_typed,
                    FlextRuntime.normalize_to_metadata_value,
                )
                merged_attrs.update(normalized_kwargs)

            return _Metadata(attributes=merged_attrs)

        @staticmethod
        def _normalize_metadata(
            metadata: p.Log.Metadata
            | Mapping[str, t.MetadataAttributeValue]
            | t.GeneralValueType
            | None,
            merged_kwargs: t.MetadataAttributeDict,
        ) -> _Metadata:
            """Normalize metadata from various input types to _Metadata model.

            Business Rule: Normalizes metadata from various input types (Metadata model,
            dict-like objects, t.GeneralValueType, or None) to _Metadata.
            Uses FlextRuntime.normalize_to_metadata_value() for safe normalization.
            Merges merged_kwargs into final metadata attributes. Fallback converts
            non-dict values to string representation.

            Audit Implication: Metadata normalization ensures audit trail completeness
            by converting all metadata types to consistent Metadata model format. All
            metadata is normalized before being stored in audit trails.

            Args:
                metadata: _Metadata instance, dict-like object, or None
                merged_kwargs: Additional attributes to merge

            Returns:
                Normalized _Metadata instance

            """
            if metadata is None:
                # Normalize attributes to t.GeneralValueType for Metadata
                # Use mapper to normalize merged_kwargs
                if merged_kwargs:
                    merged_kwargs_dict = (
                        _to_dict(merged_kwargs)
                        if not isinstance(merged_kwargs, dict)
                        else merged_kwargs
                    )
                    normalized_attrs = _transform_values(
                        merged_kwargs_dict,
                        FlextRuntime.normalize_to_metadata_value,
                    )
                    return _Metadata(attributes=normalized_attrs)
                return _Metadata(attributes={})

            if isinstance(metadata, _Metadata):
                if merged_kwargs:
                    existing_attrs = metadata.attributes
                    new_attrs: t.ConfigurationDict = {}
                    # Combine attributes - both are t.GeneralValueType compatible
                    # Type alias cannot be called - use dict() constructor directly
                    combined_attrs: t.ConfigurationDict = dict(existing_attrs)
                    # merged_kwargs values are MetadataAttributeValue (subset of t.GeneralValueType)
                    # Add merged_kwargs with explicit type handling
                    for k, v_merged in merged_kwargs.items():
                        combined_attrs[k] = v_merged
                    # Normalize all values with explicit type handling
                    for k, v_combined in combined_attrs.items():
                        # Always normalize - normalize_to_metadata_value handles all t.GeneralValueType
                        normalized_value = FlextRuntime.normalize_to_metadata_value(
                            v_combined,
                        )
                        # normalize_to_metadata_value returns t.GeneralValueType
                        new_attrs[k] = normalized_value
                    # new_attrs is already t.ConfigurationDict from loop
                    return _Metadata(attributes=new_attrs)
                return metadata

            if FlextRuntime.is_dict_like(metadata):
                # After is_dict_like type guard, metadata is Mapping[str, t.GeneralValueType]
                metadata_mapping = cast("t.ConfigurationMapping", metadata)
                return e.BaseError._normalize_metadata_from_dict(
                    metadata_mapping,
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

            validation_context: t.MetadataAttributeDict = {}

            # Use mapper to normalize context values
            if context is not None:
                context_dict = (
                    _to_dict(context) if not isinstance(context, dict) else context
                )
                normalized_context = _transform_values(
                    context_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                validation_context.update(normalized_context)
            if field is not None:
                validation_context["field"] = field
            if value is not None:
                validation_context["value"] = FlextRuntime.normalize_to_metadata_value(
                    value,
                )
            # Use mapper to normalize extra_kwargs
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                validation_context.update(normalized_extra)

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

            config_context: t.MetadataAttributeDict = {}
            if context is not None:
                config_context.update(context)
            if config_key is not None:
                config_context["config_key"] = config_key
            if config_source is not None:
                config_context["config_source"] = config_source
            # Use mapper to normalize extra_kwargs
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                config_context.update(normalized_extra)

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

            conn_context: t.MetadataAttributeDict = {}

            # Use mapper to normalize context and extra_kwargs
            if context is not None:
                context_dict = (
                    _to_dict(context) if not isinstance(context, dict) else context
                )
                normalized_context = _transform_values(
                    context_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                conn_context.update(normalized_context)
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                conn_context.update(normalized_extra)

            super().__init__(
                message,
                error_code=error_code,
                context=conn_context or None,
                metadata=preserved_metadata,
                correlation_id=str(preserved_corr_id)
                if preserved_corr_id is not None
                else None,
            )
            self.host = _get(conn_context, "host")
            self.port = _get(conn_context, "port")
            self.timeout = _get(conn_context, "timeout")

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

            timeout_context: t.MetadataAttributeDict = {}

            # Use mapper to normalize context and extra_kwargs
            if context is not None:
                context_dict = (
                    _to_dict(context) if not isinstance(context, dict) else context
                )
                normalized_context = _transform_values(
                    context_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                timeout_context.update(normalized_context)
            if timeout_seconds is not None:
                timeout_context["timeout_seconds"] = timeout_seconds
            if operation is not None:
                timeout_context["operation"] = operation
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                timeout_context.update(normalized_extra)

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

            auth_context: t.MetadataAttributeDict = {}

            # Use mapper to normalize context and extra_kwargs
            if context is not None:
                context_dict = (
                    _to_dict(context) if not isinstance(context, dict) else context
                )
                normalized_context = _transform_values(
                    context_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                auth_context.update(normalized_context)
            if auth_method is not None:
                auth_context["auth_method"] = auth_method
            if user_id is not None:
                auth_context["user_id"] = user_id
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                auth_context.update(normalized_extra)

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

            authz_context: t.MetadataAttributeDict = {}

            # Use mapper to normalize context and extra_kwargs
            if context is not None:
                context_dict = (
                    _to_dict(context) if not isinstance(context, dict) else context
                )
                normalized_context = _transform_values(
                    context_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                authz_context.update(normalized_context)
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                authz_context.update(normalized_extra)

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
            self.user_id = _get(authz_context, "user_id")
            self.resource = _get(authz_context, "resource")
            self.permission = _get(authz_context, "permission")

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        @staticmethod
        def _extract_context_values(
            context: Mapping[str, t.MetadataAttributeValue] | None,
        ) -> tuple[str | None, _Metadata | None, bool, bool]:
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
            metadata_val = metadata_obj if isinstance(metadata_obj, _Metadata) else None

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
        ) -> t.MetadataAttributeDict:
            """Build notfound-specific kwargs from fields and context.

            Returns:
                Dictionary of notfound kwargs

            """
            notfound_kwargs: t.MetadataAttributeDict = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }

            # Convert extra_kwargs to t.MetadataAttributeValue and normalize values
            valid_extra: t.MetadataAttributeDict = {
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
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            # Use explicit params or preserved from extra_kwargs
            final_metadata = metadata if metadata is not None else preserved_metadata
            # Type-narrow preserved_corr_id with isinstance for mypy
            final_corr_id: str | None
            if correlation_id is not None:
                final_corr_id = correlation_id
            elif isinstance(preserved_corr_id, str):
                final_corr_id = preserved_corr_id
            else:
                final_corr_id = None

            # Build notfound context
            notfound_context: t.MetadataAttributeDict = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }

            # Use mapper to normalize extra_kwargs
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                notfound_context.update(normalized_extra)

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

            conflict_context: t.MetadataAttributeDict = {}
            if context is not None:
                conflict_context.update(context)
            # Use mapper to normalize extra_kwargs
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                conflict_context.update(normalized_extra)

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
            self.resource_type = _get(conflict_context, "resource_type")
            self.resource_id = _get(conflict_context, "resource_id")
            self.conflict_reason = _get(conflict_context, "conflict_reason")

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

            rate_limit_context: t.MetadataAttributeDict = {}
            if context is not None:
                rate_limit_context.update(context)
            # Use mapper to normalize extra_kwargs
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                rate_limit_context.update(normalized_extra)

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
            self.limit = (
                limit_val if FlextUtilitiesGuards.is_type(limit_val, int) else None
            )
            window_seconds_val = rate_limit_context.get("window_seconds")
            self.window_seconds = (
                window_seconds_val
                if FlextUtilitiesGuards.is_type(window_seconds_val, int)
                else None
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

            cb_context: t.MetadataAttributeDict = {}
            if context is not None:
                cb_context.update(context)
            # Use mapper to normalize extra_kwargs
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                cb_context.update(normalized_extra)

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
                service_name_val
                if FlextUtilitiesGuards.is_type(service_name_val, str)
                else None
            )
            failure_count_val = cb_context.get("failure_count")
            self.failure_count = (
                failure_count_val
                if FlextUtilitiesGuards.is_type(failure_count_val, int)
                else None
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
        def _get_type_map() -> t.StringTypeDict:
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
            type_map: t.StringTypeDict,
            extra_kwargs: t.MetadataAttributeDict,
            key: str,
        ) -> type | None:
            """Normalize type value from various sources.

            Business Rule: extra_kwargs can contain type objects at runtime even though
            t.MetadataAttributeValue doesn't include type in the type system.
            This method handles both string type names and actual type objects.
            """
            # Extract from extra_kwargs if not provided as named arg
            if type_value is None and key in extra_kwargs:
                type_raw = extra_kwargs.pop(key)
                if isinstance(type_raw, str):
                    result = _get(type_map, type_raw)
                    return cast("type[object] | None", result)
                # Runtime safety: type_raw is t.MetadataAttributeValue per type system,
                # but at runtime can be type object (defensive programming)
                if isinstance(type_raw, type):
                    return type_raw
                # type_raw is not a type, return None
                return None
            # Handle case where type is passed as string in named arg
            if isinstance(type_value, str):
                # type_map.get returns type | None
                result = _get(type_map, type_value)
                return cast("type[object] | None", result)
            # type_value is already type | None at this point (str case handled above)
            if isinstance(type_value, type):
                return type_value
            return None

        @staticmethod
        def _build_type_context(
            expected_type: type | str | None,
            actual_type: type | str | None,
            context: Mapping[str, t.MetadataAttributeValue] | None,
            extra_kwargs: t.MetadataAttributeDict,
        ) -> t.MetadataAttributeDict:
            """Build type context dictionary."""
            type_context: t.MetadataAttributeDict = {}
            # Use mapper to normalize context
            if context is not None:
                context_dict = (
                    _to_dict(context) if not isinstance(context, dict) else context
                )
                normalized_context = _transform_values(
                    context_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                type_context.update(normalized_context)
            # Handle both type objects and string representations
            if expected_type is not None:
                # Handle both type objects and string representations
                if isinstance(expected_type, str):
                    type_context["expected_type"] = expected_type
                else:
                    type_context["expected_type"] = expected_type.__qualname__
            else:
                type_context["expected_type"] = None
            if actual_type is not None:
                # Handle both type objects and string representations
                if isinstance(actual_type, str):
                    type_context["actual_type"] = actual_type
                else:
                    type_context["actual_type"] = actual_type.__qualname__
            else:
                type_context["actual_type"] = None
            # Use mapper to normalize extra_kwargs
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                type_context.update(normalized_extra)
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

            op_context: t.MetadataAttributeDict = {}
            # Use mapper to normalize context and extra_kwargs
            if context is not None:
                context_dict = (
                    _to_dict(context) if not isinstance(context, dict) else context
                )
                normalized_context = _transform_values(
                    context_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                op_context.update(normalized_context)
            if operation is not None:
                op_context["operation"] = operation
            if reason is not None:
                op_context["reason"] = reason
            if extra_kwargs:
                extra_kwargs_dict = (
                    _to_dict(extra_kwargs)
                    if not isinstance(extra_kwargs, dict)
                    else extra_kwargs
                )
                normalized_extra = _transform_values(
                    extra_kwargs_dict,
                    FlextRuntime.normalize_to_metadata_value,
                )
                op_context.update(normalized_extra)

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

            attr_context: t.MetadataAttributeDict = {}
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
        kwargs: t.MetadataAttributeDict,
        specific_params: t.MetadataAttributeDict | None = None,
    ) -> tuple[
        str | None,
        t.MetadataAttributeValue,
        bool,
        bool,
        t.MetadataAttributeValue,
        t.MetadataAttributeDict,
    ]:
        """Prepare exception kwargs by extracting common parameters."""
        if specific_params:
            # Use mapper to filter out None values
            filtered_params = _filter_dict(
                specific_params if isinstance(specific_params, dict) else {},
                lambda _k, v: v is not None,
            )
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
        correlation_id_raw = _get(kwargs, "correlation_id")
        correlation_id: str | None = (
            str(correlation_id_raw)
            if correlation_id_raw is not None
            and FlextUtilitiesGuards.is_type(correlation_id_raw, str)
            else None
        )
        return (
            correlation_id,
            _get(kwargs, "metadata"),
            bool(_get(kwargs, "auto_log")),
            bool(_get(kwargs, "auto_correlation")),
            _get(kwargs, "config"),
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
        correlation_id_raw = _get(kwargs, "correlation_id")
        # Use isinstance for proper type narrowing
        correlation_id: str | None = (
            correlation_id_raw if isinstance(correlation_id_raw, str) else None
        )
        metadata_raw = _get(kwargs, "metadata")
        # Return metadata as-is if it's Metadata or dict-like, otherwise None
        metadata: p.Log.Metadata | Mapping[str, t.MetadataAttributeValue] | None = None
        if metadata_raw is not None:
            if isinstance(metadata_raw, _Metadata):
                metadata = metadata_raw
            elif isinstance(metadata_raw, dict):
                # Convert dict values to MetadataAttributeValue
                converted_dict: t.MetadataAttributeDict = {}
                for k, v in metadata_raw.items():
                    converted_dict[k] = FlextRuntime.normalize_to_metadata_value(v)
                metadata = converted_dict
            elif isinstance(metadata_raw, Mapping):
                # Convert Mapping values to MetadataAttributeValue
                converted_dict_mapping: t.MetadataAttributeDict = {}
                for k, v in metadata_raw.items():
                    converted_dict_mapping[k] = (
                        FlextRuntime.normalize_to_metadata_value(
                            v,
                        )
                    )
                metadata = converted_dict_mapping
        return (correlation_id, metadata)

    @staticmethod
    def create_error(error_type: str, message: str) -> e.BaseError:
        """Create an exception instance based on error type."""
        error_classes: t.StringFlextExceptionTypeDict = {
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
        if not error_class:
            msg = f"Unknown error type: {error_type}"
            raise ValueError(msg)
        return cast("type[e.BaseError]", error_class)(message)

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
    ) -> t.MetadataAttributeValue | None:
        """Prepare metadata value for error creation."""
        return (
            cast("t.MetadataAttributeValue", meta.attributes)
            if meta is not None
            else None
        )

    @staticmethod
    def _get_str_from_kwargs(
        kwargs: Mapping[str, t.MetadataAttributeValue],
        key: str,
    ) -> str | None:
        """Extract and convert value from kwargs to string."""
        val = _get(kwargs, key)
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
        error_context: t.MetadataAttributeDict = {}
        if context is not None:
            error_context.update(context)
        if error_code is not None:
            error_context["error_code"] = error_code

        # Create appropriate error class based on type
        error_classes: t.StringFlextExceptionTypeDict = {
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

        error_class = _get(error_classes, error_type or "") if error_type else None
        # Extract correlation_id if present in context
        correlation_id = None
        if error_context and "correlation_id" in error_context:
            # We can pass it as kwarg because all error classes accept **extra_kwargs
            # and they look for 'correlation_id' in extra_kwargs
            correlation_id = str(error_context["correlation_id"])

        if error_class:
            return cast("type[e.BaseError]", error_class)(
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
        context: t.MetadataAttributeDict,
        metadata_obj: p.Log.Metadata
        | Mapping[str, t.MetadataAttributeValue]
        | t.GeneralValueType
        | None,
    ) -> None:
        """Merge metadata object into context dictionary."""
        if metadata_obj is None:
            return

        if isinstance(metadata_obj, _Metadata):
            # Extract attributes from _Metadata model
            for k, v in metadata_obj.attributes.items():
                context[k] = FlextRuntime.normalize_to_metadata_value(v)
        elif isinstance(metadata_obj, dict):
            # Direct dict - normalize values and update
            for k, v in metadata_obj.items():
                context[k] = FlextRuntime.normalize_to_metadata_value(v)
        elif isinstance(metadata_obj, Mapping):
            # Mapping object - normalize and update
            for k, v in metadata_obj.items():
                context[k] = FlextRuntime.normalize_to_metadata_value(v)

    @staticmethod
    def _build_error_context(
        correlation_id: str | None,
        metadata_obj: p.Log.Metadata
        | Mapping[str, t.MetadataAttributeValue]
        | t.GeneralValueType
        | None,
        kwargs: t.MetadataAttributeDict,
    ) -> t.MetadataAttributeDict:
        """Build error context dictionary."""
        error_context: t.MetadataAttributeDict = {}
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
    def get_metrics(cls) -> t.ErrorTypeMapping:
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
        exception_counts_dict: t.StringIntDict = {}
        for exc_type, count in cls._exception_counts.items():
            exc_name = (
                exc_type.__qualname__
                if hasattr(exc_type, "__qualname__")
                else str(exc_type)
            )
            exception_counts_dict[exc_name] = count
        # Build result dict matching ErrorTypeMapping type
        # int is compatible with str | int | float | Mapping | None in ErrorTypeMapping
        # Use cast to satisfy type checker (int values are valid in ErrorTypeMapping)
        result: t.ErrorTypeMapping = cast(
            "t.ErrorTypeMapping",
            {
                "total_exceptions": total,
                "exception_counts": exception_counts_dict,
                "exception_counts_summary": exception_counts_str,  # String format for summary
                "unique_exception_types": len(cls._exception_counts),
            },
        )
        return result

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._exception_counts.clear()

    def __call__(
        self,
        message: str,
        error_code: str | None = None,
        **kwargs: t.ExceptionKwargsType,
    ) -> e.BaseError:
        """Create exception by calling the class instance."""
        # Normalize ExceptionKwargsType to t.MetadataAttributeValue
        # normalize_to_metadata_value already returns t.MetadataAttributeValue
        normalized_kwargs: t.MetadataAttributeDict = {}
        for k, v in kwargs.items():
            normalized_kwargs[k] = FlextRuntime.normalize_to_metadata_value(v)
        return self.create(message, error_code, **normalized_kwargs)


e = FlextExceptions

__all__ = ["FlextExceptions", "e"]
