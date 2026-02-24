"""Exception hierarchy with correlation metadata.

Provides structured exceptions with error codes and correlation tracking
for consistent error handling across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, MutableMapping
from typing import ClassVar

from pydantic import ConfigDict, Field, ValidationError as PydanticValidationError

from flext_core.constants import c
from flext_core._models.base import FlextModelsBase
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

# FlextRuntime.Metadata used directly (no alias) per runtime-alias-only policy


class FlextExceptions:
    """Exception types with correlation metadata.

    Provides structured exceptions with error codes and correlation tracking
    for consistent error handling and logging.
    """

    class _ParamsModel(FlextModelsBase.ArbitraryTypesModel):
        """Shared strict params model for exception helpers."""

        model_config = ConfigDict(
            extra=c.ModelConfig.EXTRA_FORBID,
            strict=True,
            validate_assignment=True,
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class _StrictStringValue(_ParamsModel):
        """Strict string extractor for kwargs/context parsing."""

        value: str = Field(strict=True)

    class _StrictBooleanValue(_ParamsModel):
        """Strict boolean extractor for kwargs/context parsing."""

        value: bool = Field(strict=True)

    class _StrictIntValue(_ParamsModel):
        """Strict integer extractor for kwargs/context parsing."""

        value: int = Field(strict=True)

    class _StrictNumberValue(_ParamsModel):
        """Strict numeric extractor for kwargs/context parsing."""

        value: int | float = Field()

    class ValidationErrorParams(_ParamsModel):
        """Validated params for ValidationError."""

        field: str | None = Field(default=None, strict=True)
        value: t.MetadataAttributeValue | None = None

    class ConfigurationErrorParams(_ParamsModel):
        """Validated params for ConfigurationError."""

        config_key: str | None = Field(default=None, strict=True)
        config_source: str | None = Field(default=None, strict=True)

    class ConnectionErrorParams(_ParamsModel):
        """Validated params for ConnectionError."""

        host: str | None = Field(default=None, strict=True)
        port: int | None = Field(default=None, strict=True)
        timeout: int | float | None = Field(default=None)

    class TimeoutErrorParams(_ParamsModel):
        """Validated params for TimeoutError."""

        timeout_seconds: int | float | None = Field(default=None)
        operation: str | None = Field(default=None, strict=True)

    class AuthenticationErrorParams(_ParamsModel):
        """Validated params for AuthenticationError."""

        auth_method: str | None = Field(default=None, strict=True)
        user_id: str | None = Field(default=None, strict=True)

    class AuthorizationErrorParams(_ParamsModel):
        """Validated params for AuthorizationError."""

        user_id: str | None = Field(default=None, strict=True)
        resource: str | None = Field(default=None, strict=True)
        permission: str | None = Field(default=None, strict=True)

    class NotFoundErrorParams(_ParamsModel):
        """Validated params for NotFoundError."""

        resource_type: str | None = Field(default=None, strict=True)
        resource_id: str | None = Field(default=None, strict=True)

    class ConflictErrorParams(_ParamsModel):
        """Validated params for ConflictError."""

        resource_type: str | None = Field(default=None, strict=True)
        resource_id: str | None = Field(default=None, strict=True)
        conflict_reason: str | None = Field(default=None, strict=True)

    class RateLimitErrorParams(_ParamsModel):
        """Validated params for RateLimitError."""

        limit: int | None = Field(default=None, strict=True)
        window_seconds: int | None = Field(default=None, strict=True)
        retry_after: int | float | None = Field(default=None)

    class CircuitBreakerErrorParams(_ParamsModel):
        """Validated params for CircuitBreakerError."""

        service_name: str | None = Field(default=None, strict=True)
        failure_count: int | None = Field(default=None, strict=True)
        reset_timeout: int | float | None = Field(default=None)

    class TypeErrorParams(_ParamsModel):
        """Validated params for TypeError."""

        expected_type: str | None = Field(default=None, strict=True)
        actual_type: str | None = Field(default=None, strict=True)

    class OperationErrorParams(_ParamsModel):
        """Validated params for OperationError."""

        operation: str | None = Field(default=None, strict=True)
        reason: str | None = Field(default=None, strict=True)

    class AttributeAccessErrorParams(_ParamsModel):
        """Validated params for AttributeAccessError."""

        attribute_name: str | None = Field(default=None, strict=True)
        attribute_context: str | None = Field(default=None, strict=True)

    @staticmethod
    def _safe_optional_str(
        value: t.MetadataAttributeValue | type | None,
    ) -> str | None:
        """Extract optional strict string from dynamic values."""
        if value is None:
            return None
        try:
            return e._StrictStringValue.model_validate({"value": value}).value
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_bool(value: t.MetadataAttributeValue | None, *, default: bool) -> bool:
        """Extract strict bool from dynamic values with default fallback."""
        if value is None:
            return default
        try:
            return e._StrictBooleanValue.model_validate({"value": value}).value
        except PydanticValidationError:
            return default

    @staticmethod
    def _safe_int(value: t.MetadataAttributeValue | None) -> int | None:
        """Extract optional strict integer from dynamic values."""
        if value is None:
            return None
        try:
            return e._StrictIntValue.model_validate({"value": value}).value
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_number(value: t.MetadataAttributeValue | None) -> int | float | None:
        """Extract optional strict numeric value from dynamic values."""
        if value is None:
            return None
        try:
            return e._StrictNumberValue.model_validate({"value": value}).value
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_config_map(
        value: FlextRuntime.Metadata
        | Mapping[str, t.ConfigMapValue]
        | m.ConfigMap
        | t.ConfigMapValue
        | None,
    ) -> m.ConfigMap | None:
        """Extract ConfigMap when value is mapping-compatible."""
        if value is None:
            return None
        try:
            return m.ConfigMap.model_validate(value)
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_metadata(
        value: FlextRuntime.Metadata
        | Mapping[str, t.ConfigMapValue]
        | m.ConfigMap
        | t.ConfigMapValue
        | None,
    ) -> FlextRuntime.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model."""
        if value is None:
            return None

        try:
            return FlextRuntime.Metadata.model_validate(value)
        except PydanticValidationError:
            pass

        dumped_map: m.ConfigMap | None = None
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped_candidate = model_dump()
            try:
                dumped_map = m.ConfigMap.model_validate(dumped_candidate)
            except PydanticValidationError:
                dumped_map = None

        if dumped_map is not None:
            attrs_raw = dumped_map.get("attributes")
            attrs_map = e._safe_config_map(attrs_raw)
            if attrs_map is not None:
                attrs = {
                    k: FlextRuntime.normalize_to_metadata_value(v)
                    for k, v in attrs_map.items()
                }
                return FlextRuntime.Metadata(
                    attributes=attrs,
                )

        attrs_map = e._safe_config_map(value)
        if attrs_map is not None:
            attrs = {
                k: FlextRuntime.normalize_to_metadata_value(v)
                for k, v in attrs_map.items()
            }
            return FlextRuntime.Metadata(
                attributes=attrs,
            )

        return None

    @staticmethod
    def _build_context_map(
        context: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap | None,
        extra_kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> m.ConfigMap:
        """Build normalized context map from context and kwargs."""
        excluded = excluded_keys or frozenset()
        context_map: m.ConfigMap = m.ConfigMap()
        if context:
            context_map.update({
                k: FlextRuntime.normalize_to_metadata_value(v)
                for k, v in context.items()
                if k not in excluded
            })
        if extra_kwargs:
            context_map.update({
                k: FlextRuntime.normalize_to_metadata_value(v)
                for k, v in extra_kwargs.items()
                if k not in excluded
            })
        return context_map

    @staticmethod
    def _build_param_map(
        context: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap | None,
        extra_kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
        keys: set[str] | frozenset[str],
    ) -> m.ConfigMap:
        """Build unnormalized parameter map for strict params validation."""
        param_map: m.ConfigMap = m.ConfigMap()
        if context:
            param_map.update({k: v for k, v in context.items() if k in keys})
        if extra_kwargs:
            param_map.update({k: v for k, v in extra_kwargs.items() if k in keys})
        return param_map

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
            context: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap | None = None,
            metadata: FlextRuntime.Metadata
            | m.ConfigMap
            | t.MetadataAttributeValue
            | None = None,
            correlation_id: str | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
            merged_kwargs: Mapping[str, t.MetadataAttributeValue]
            | m.ConfigMap
            | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize base error with message and optional metadata.

            Args:
                message: Error message
                error_code: Optional error code
                context: Optional context mapping
                metadata: Optional metadata (FlextRuntime.Metadata, dict, or payload types)
                correlation_id: Optional correlation ID
                auto_correlation: Auto-generate correlation ID if not provided
                auto_log: Auto-log error on creation
                merged_kwargs: Additional metadata attributes to merge

            """
            super().__init__(message)
            self.message = message
            self.error_code = error_code

            # Merge context and extra_kwargs into final_kwargs
            final_kwargs: m.ConfigMap = (
                m.ConfigMap(root=dict(merged_kwargs))
                if merged_kwargs
                else m.ConfigMap()
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
            self.metadata = e.BaseError._normalize_metadata(
                metadata,
                final_kwargs,
            )
            self.timestamp = time.time()
            self.auto_log = auto_log

        def __str__(self) -> str:
            """Return string representation with error code if present."""
            if self.error_code:
                return f"[{self.error_code}] {self.message}"
            return self.message

        def to_dict(
            self,
        ) -> m.ConfigMap:
            """Convert exception to dictionary representation.

            Returns:
                Dictionary with error_type, message, error_code, and other fields.

            """
            result: m.ConfigMap = m.ConfigMap(
                root={
                    "error_type": type(self).__name__,
                    "message": self.message,
                    "error_code": self.error_code,
                    "correlation_id": self.correlation_id,
                    "timestamp": self.timestamp,
                },
            )

            # Add metadata attributes (only keys not in result)
            if self.metadata and self.metadata.attributes:
                filtered_attrs = {
                    k: v for k, v in self.metadata.attributes.items() if k not in result
                }
                result.update(filtered_attrs)
            return result

        @staticmethod
        def _normalize_metadata_from_dict(
            metadata_dict: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
            merged_kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
        ) -> FlextRuntime.Metadata:
            """Normalize metadata from dict-like object."""
            # Use MetadataAttributeDict - normalize_to_metadata_value returns MetadataAttributeValue
            merged_attrs = {}

            # Normalize metadata_dict values
            for k, v in metadata_dict.items():
                merged_attrs[k] = FlextRuntime.normalize_to_metadata_value(v)

            # Normalize merged_kwargs values
            if merged_kwargs:
                for k, v in merged_kwargs.items():
                    merged_attrs[k] = FlextRuntime.normalize_to_metadata_value(v)

            return FlextRuntime.Metadata(
                attributes={
                    k: FlextRuntime.normalize_to_metadata_value(v)
                    for k, v in merged_attrs.items()
                },
            )

        @staticmethod
        def _normalize_metadata(
            metadata: FlextRuntime.Metadata
            | m.ConfigMap
            | t.MetadataAttributeValue
            | None,
            merged_kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
        ) -> FlextRuntime.Metadata:
            """Normalize metadata from various input types to FlextRuntime.Metadata model.

            Args:
                metadata: FlextRuntime.Metadata instance, dict-like object, or None
                merged_kwargs: Additional attributes to merge

            Returns:
                Normalized FlextRuntime.Metadata instance

            """
            if metadata is None:
                if merged_kwargs:
                    normalized_attrs = {
                        k: FlextRuntime.normalize_to_metadata_value(v)
                        for k, v in merged_kwargs.items()
                    }
                    return FlextRuntime.Metadata(attributes=normalized_attrs)
                return FlextRuntime.Metadata(attributes={})

            metadata_model = e._safe_metadata(metadata)
            if metadata_model is not None:
                if not merged_kwargs:
                    return metadata_model

                merged_attrs = {
                    k: FlextRuntime.normalize_to_metadata_value(v)
                    for k, v in metadata_model.attributes.items()
                }
                for k, v in merged_kwargs.items():
                    merged_attrs[k] = FlextRuntime.normalize_to_metadata_value(v)
                return FlextRuntime.Metadata(attributes=merged_attrs)

            metadata_dict = e._safe_config_map(metadata)
            if metadata_dict is not None:
                return e.BaseError._normalize_metadata_from_dict(
                    metadata_dict,
                    merged_kwargs,
                )

            # Fallback: convert to FlextRuntime.Metadata with string value
            return FlextRuntime.Metadata(attributes={"value": str(metadata)})

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
            params: e.ValidationErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize validation error with field and value information."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"field", "value"},
            )
            if field is not None:
                param_values["field"] = field
            if value is not None:
                param_values["value"] = value
            resolved_params = (
                params
                if params is not None
                else e.ValidationErrorParams.model_validate(dict(param_values))
            )

            validation_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.field is not None:
                validation_context["field"] = resolved_params.field
            if resolved_params.value is not None:
                validation_context["value"] = FlextRuntime.normalize_to_metadata_value(
                    resolved_params.value,
                )

            super().__init__(
                message,
                error_code=error_code,
                context=validation_context or None,
                metadata=preserved_metadata,
                correlation_id=e._safe_optional_str(preserved_corr_id),
            )
            self.field = resolved_params.field
            self.value = resolved_params.value

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
            params: e.ConfigurationErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize configuration error with config context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"config_key", "config_source"},
            )
            if config_key is not None:
                param_values["config_key"] = config_key
            if config_source is not None:
                param_values["config_source"] = config_source
            resolved_params = (
                params
                if params is not None
                else e.ConfigurationErrorParams.model_validate(dict(param_values))
            )

            config_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.config_key is not None:
                config_context["config_key"] = resolved_params.config_key
            if resolved_params.config_source is not None:
                config_context["config_source"] = resolved_params.config_source

            super().__init__(
                message,
                error_code=error_code,
                context=config_context or None,
                metadata=preserved_metadata,
                correlation_id=e._safe_optional_str(preserved_corr_id),
            )
            self.config_key = resolved_params.config_key
            self.config_source = resolved_params.config_source

    class ConnectionError(BaseError):
        """Exception raised for network and connection failures."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.CONNECTION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            _correlation_id: str | None = None,
            params: e.ConnectionErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize connection error with network context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"host", "port", "timeout"},
            )
            resolved_params = (
                params
                if params is not None
                else e.ConnectionErrorParams.model_validate(dict(param_values))
            )

            conn_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.host is not None:
                conn_context["host"] = resolved_params.host
            if resolved_params.port is not None:
                conn_context["port"] = resolved_params.port
            if resolved_params.timeout is not None:
                conn_context["timeout"] = resolved_params.timeout

            super().__init__(
                message,
                error_code=error_code,
                context=conn_context or None,
                metadata=preserved_metadata,
                correlation_id=e._safe_optional_str(preserved_corr_id),
            )
            self.host = resolved_params.host
            self.port = resolved_params.port
            self.timeout = resolved_params.timeout

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
            params: e.TimeoutErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize timeout error with timeout context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"timeout_seconds", "operation"},
            )
            if timeout_seconds is not None:
                param_values["timeout_seconds"] = timeout_seconds
            if operation is not None:
                param_values["operation"] = operation
            resolved_params = (
                params
                if params is not None
                else e.TimeoutErrorParams.model_validate(dict(param_values))
            )

            timeout_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.timeout_seconds is not None:
                timeout_context["timeout_seconds"] = resolved_params.timeout_seconds
            if resolved_params.operation is not None:
                timeout_context["operation"] = resolved_params.operation

            super().__init__(
                message,
                error_code=error_code,
                context=timeout_context or None,
                metadata=preserved_metadata,
                correlation_id=e._safe_optional_str(preserved_corr_id),
            )
            self.timeout_seconds = resolved_params.timeout_seconds
            self.operation = resolved_params.operation

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
            params: e.AuthenticationErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize authentication error with auth context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"auth_method", "user_id"},
            )
            if auth_method is not None:
                param_values["auth_method"] = auth_method
            if user_id is not None:
                param_values["user_id"] = user_id
            resolved_params = (
                params
                if params is not None
                else e.AuthenticationErrorParams.model_validate(dict(param_values))
            )

            auth_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.auth_method is not None:
                auth_context["auth_method"] = resolved_params.auth_method
            if resolved_params.user_id is not None:
                auth_context["user_id"] = resolved_params.user_id

            super().__init__(
                message,
                error_code=error_code,
                context=auth_context or None,
                metadata=preserved_metadata,
                correlation_id=e._safe_optional_str(preserved_corr_id),
            )
            self.auth_method = resolved_params.auth_method
            self.user_id = resolved_params.user_id

    class AuthorizationError(BaseError):
        """Exception raised for permission and authorization failures."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.AUTHORIZATION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            params: e.AuthorizationErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize authorization error with permission context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"user_id", "resource", "permission"},
            )
            resolved_params = (
                params
                if params is not None
                else e.AuthorizationErrorParams.model_validate(dict(param_values))
            )

            authz_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.user_id is not None:
                authz_context["user_id"] = resolved_params.user_id
            if resolved_params.resource is not None:
                authz_context["resource"] = resolved_params.resource
            if resolved_params.permission is not None:
                authz_context["permission"] = resolved_params.permission

            super().__init__(
                message,
                error_code=error_code,
                context=authz_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else e._safe_optional_str(preserved_corr_id),
            )
            self.user_id = resolved_params.user_id
            self.resource = resolved_params.resource
            self.permission = resolved_params.permission

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        @staticmethod
        def _extract_context_values(
            context: Mapping[str, t.MetadataAttributeValue] | None,
        ) -> tuple[str | None, FlextRuntime.Metadata | None, bool, bool]:
            """Extract context values from mapping.

            Returns:
                Tuple of (correlation_id, metadata, auto_log, auto_correlation)

            """
            if context is None:
                return (None, None, False, False)

            correlation_id_val = e._safe_optional_str(context.get("correlation_id"))
            metadata_val = e._safe_metadata(context.get("metadata"))
            auto_log_val = e._safe_bool(context.get("auto_log"), default=False)
            auto_correlation_val = e._safe_bool(
                context.get("auto_correlation"),
                default=False,
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
            extra_kwargs: MutableMapping[str, t.MetadataAttributeValue],
            context: Mapping[str, t.MetadataAttributeValue] | None,
        ) -> m.ConfigMap:
            """Build notfound-specific kwargs from fields and context.

            Returns:
                Dictionary of notfound kwargs

            """
            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"resource_type", "resource_id"},
            )
            if resource_type is not None:
                param_values["resource_type"] = resource_type
            if resource_id is not None:
                param_values["resource_id"] = resource_id

            notfound_params = e.NotFoundErrorParams.model_validate(dict(param_values))
            notfound_kwargs: m.ConfigMap = m.ConfigMap(
                root=notfound_params.model_dump(exclude_none=True),
            )

            notfound_kwargs.update(
                dict(
                    e._build_context_map(
                        context,
                        extra_kwargs,
                        excluded_keys={
                            "correlation_id",
                            "metadata",
                            "auto_log",
                            "auto_correlation",
                        },
                    ).items(),
                ),
            )

            return notfound_kwargs

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            error_code: str = c.Errors.NOT_FOUND_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            metadata: FlextRuntime.Metadata
            | m.ConfigMap
            | t.MetadataAttributeValue
            | None = None,
            correlation_id: str | None = None,
            params: e.NotFoundErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize not found error with resource context."""
            # Preserve metadata from extra_kwargs (correlation_id is
            # consumed by the explicit parameter and never reaches **extra_kwargs)
            preserved_metadata = extra_kwargs.pop("metadata", None)
            _ = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"resource_type", "resource_id"},
            )
            if resource_type is not None:
                param_values["resource_type"] = resource_type
            if resource_id is not None:
                param_values["resource_id"] = resource_id
            resolved_params = (
                params
                if params is not None
                else e.NotFoundErrorParams.model_validate(dict(param_values))
            )

            notfound_context = e._build_context_map(
                context,
                extra_kwargs,
                excluded_keys={"correlation_id", "metadata"},
            )
            if resolved_params.resource_type is not None:
                notfound_context["resource_type"] = resolved_params.resource_type
            if resolved_params.resource_id is not None:
                notfound_context["resource_id"] = resolved_params.resource_id

            metadata_input = metadata if metadata is not None else preserved_metadata

            super().__init__(
                message,
                error_code=error_code,
                context=notfound_context,
                metadata=metadata_input,
                correlation_id=correlation_id,
            )
            self.resource_type = resolved_params.resource_type
            self.resource_id = resolved_params.resource_id

    class ConflictError(BaseError):
        """Exception raised for resource conflicts."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.ALREADY_EXISTS,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            params: e.ConflictErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize conflict error with resource context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"resource_type", "resource_id", "conflict_reason"},
            )
            resolved_params = (
                params
                if params is not None
                else e.ConflictErrorParams.model_validate(dict(param_values))
            )

            conflict_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.resource_type is not None:
                conflict_context["resource_type"] = resolved_params.resource_type
            if resolved_params.resource_id is not None:
                conflict_context["resource_id"] = resolved_params.resource_id
            if resolved_params.conflict_reason is not None:
                conflict_context["conflict_reason"] = resolved_params.conflict_reason

            super().__init__(
                message,
                error_code=error_code,
                context=conflict_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else e._safe_optional_str(preserved_corr_id),
            )
            self.resource_type = resolved_params.resource_type
            self.resource_id = resolved_params.resource_id
            self.conflict_reason = resolved_params.conflict_reason

    class RateLimitError(BaseError):
        """Exception raised when rate limits are exceeded."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.OPERATION_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            params: e.RateLimitErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize rate limit error with limit context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"limit", "window_seconds", "retry_after"},
            )
            resolved_params = (
                params
                if params is not None
                else e.RateLimitErrorParams.model_validate(dict(param_values))
            )

            rate_limit_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.limit is not None:
                rate_limit_context["limit"] = resolved_params.limit
            if resolved_params.window_seconds is not None:
                rate_limit_context["window_seconds"] = resolved_params.window_seconds
            if resolved_params.retry_after is not None:
                rate_limit_context["retry_after"] = resolved_params.retry_after

            super().__init__(
                message,
                error_code=error_code,
                context=rate_limit_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else e._safe_optional_str(preserved_corr_id),
            )
            self.limit = resolved_params.limit
            self.window_seconds = resolved_params.window_seconds
            self.retry_after = resolved_params.retry_after

    class CircuitBreakerError(BaseError):
        """Exception raised when circuit breaker is open."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.EXTERNAL_SERVICE_ERROR,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            params: e.CircuitBreakerErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize circuit breaker error with service context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"service_name", "failure_count", "reset_timeout"},
            )
            resolved_params = (
                params
                if params is not None
                else e.CircuitBreakerErrorParams.model_validate(dict(param_values))
            )

            cb_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.service_name is not None:
                cb_context["service_name"] = resolved_params.service_name
            if resolved_params.failure_count is not None:
                cb_context["failure_count"] = resolved_params.failure_count
            if resolved_params.reset_timeout is not None:
                cb_context["reset_timeout"] = resolved_params.reset_timeout

            super().__init__(
                message,
                error_code=error_code,
                context=cb_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else e._safe_optional_str(preserved_corr_id),
            )
            self.service_name = resolved_params.service_name
            self.failure_count = resolved_params.failure_count
            self.reset_timeout = resolved_params.reset_timeout

    class TypeError(BaseError):
        """Exception raised for type mismatch errors."""

        @staticmethod
        def _get_type_map() -> Mapping[str, type]:
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
            type_map: Mapping[str, type],
            extra_kwargs: MutableMapping[str, t.MetadataAttributeValue],
            key: str,
        ) -> type | None:
            """Normalize type value from various sources."""
            source_value: type | str | t.MetadataAttributeValue | None = type_value
            if source_value is None and key in extra_kwargs:
                source_value = extra_kwargs.pop(key)

            type_name = e.TypeError._resolve_type_name(source_value)
            if type_name is None:
                return None
            return type_map.get(type_name)

        @staticmethod
        def _resolve_type_name(
            type_value: type | str | t.MetadataAttributeValue | None,
        ) -> str | None:
            """Resolve type-like input to canonical string name."""
            if type_value is None:
                return None

            string_value = e._safe_optional_str(type_value)
            if string_value is not None:
                return string_value

            qualname_value = getattr(type_value, "__qualname__", None)
            return e._safe_optional_str(qualname_value)
            return None

        @staticmethod
        def _build_type_context(
            expected_type: type | str | None,
            actual_type: type | str | None,
            context: Mapping[str, t.MetadataAttributeValue] | None,
            extra_kwargs: Mapping[str, t.MetadataAttributeValue],
        ) -> m.ConfigMap:
            """Build type context dictionary."""
            type_context = e._build_context_map(context, extra_kwargs)
            type_context["expected_type"] = e.TypeError._resolve_type_name(
                expected_type
            )
            type_context["actual_type"] = e.TypeError._resolve_type_name(actual_type)

            return type_context

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.Errors.TYPE_ERROR,
            expected_type: type | str | None = None,
            actual_type: type | str | None = None,
            context: Mapping[str, t.MetadataAttributeValue] | None = None,
            correlation_id: str | None = None,
            params: e.TypeErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize type error with type information."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            type_map = self._get_type_map()

            # Normalize types from various sources
            normalized_expected_type = self._normalize_type(
                expected_type,
                type_map,
                extra_kwargs,
                "expected_type",
            )
            normalized_actual_type = self._normalize_type(
                actual_type,
                type_map,
                extra_kwargs,
                "actual_type",
            )

            param_values = {
                "expected_type": (
                    normalized_expected_type.__qualname__
                    if normalized_expected_type is not None
                    else None
                ),
                "actual_type": (
                    normalized_actual_type.__qualname__
                    if normalized_actual_type is not None
                    else None
                ),
            }
            resolved_params = (
                params
                if params is not None
                else e.TypeErrorParams.model_validate(param_values)
            )

            # Build type context
            type_context = self._build_type_context(
                resolved_params.expected_type,
                resolved_params.actual_type,
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
                else e._safe_optional_str(preserved_corr_id),
            )
            self.expected_type = normalized_expected_type
            self.actual_type = normalized_actual_type

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
            params: e.OperationErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize operation error with operation context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"operation", "reason"},
            )
            if operation is not None:
                param_values["operation"] = operation
            if reason is not None:
                param_values["reason"] = reason
            resolved_params = (
                params
                if params is not None
                else e.OperationErrorParams.model_validate(dict(param_values))
            )

            op_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.operation is not None:
                op_context["operation"] = resolved_params.operation
            if resolved_params.reason is not None:
                op_context["reason"] = resolved_params.reason

            super().__init__(
                message,
                error_code=error_code,
                context=op_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else e._safe_optional_str(preserved_corr_id),
            )
            self.operation = resolved_params.operation
            self.reason = resolved_params.reason

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
            params: e.AttributeAccessErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize attribute access error with attribute context."""
            # Preserve metadata and correlation_id from extra_kwargs
            preserved_metadata = extra_kwargs.pop("metadata", None)
            preserved_corr_id = extra_kwargs.pop("correlation_id", None)

            param_values = e._build_param_map(
                context,
                extra_kwargs,
                keys={"attribute_name", "attribute_context"},
            )
            if attribute_name is not None:
                param_values["attribute_name"] = attribute_name
            if attribute_context is not None:
                param_values["attribute_context"] = attribute_context
            resolved_params = (
                params
                if params is not None
                else e.AttributeAccessErrorParams.model_validate(dict(param_values))
            )

            attr_context = e._build_context_map(context, extra_kwargs)
            if resolved_params.attribute_name is not None:
                attr_context["attribute_name"] = resolved_params.attribute_name
            if resolved_params.attribute_context is not None:
                attr_context["attribute_context"] = resolved_params.attribute_context

            super().__init__(
                message,
                error_code=error_code,
                context=attr_context or None,
                metadata=preserved_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else e._safe_optional_str(preserved_corr_id),
            )
            self.attribute_name = resolved_params.attribute_name
            self.attribute_context = resolved_params.attribute_context

    @staticmethod
    def prepare_exception_kwargs(
        kwargs: MutableMapping[str, t.MetadataAttributeValue],
        specific_params: Mapping[str, t.MetadataAttributeValue] | None = None,
    ) -> tuple[
        str | None,
        t.MetadataAttributeValue,
        bool,
        bool,
        t.MetadataAttributeValue,
        m.ConfigMap,
    ]:
        """Prepare exception kwargs by extracting common parameters."""
        if specific_params:
            # Filter out None values - specific_params is already MetadataAttributeDict (dict)
            filtered_params = {
                k: v for k, v in specific_params.items() if v is not None
            }
            kwargs.update(filtered_params)
        extra_kwargs: m.ConfigMap = m.ConfigMap(
            root={
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
            },
        )
        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id = e._safe_optional_str(correlation_id_raw)
        return (
            correlation_id,
            kwargs.get("metadata"),
            e._safe_bool(kwargs.get("auto_log"), default=False),
            e._safe_bool(kwargs.get("auto_correlation"), default=False),
            kwargs.get("config"),
            extra_kwargs,
        )

    @staticmethod
    def extract_common_kwargs(
        kwargs: Mapping[str, t.MetadataAttributeValue],
    ) -> tuple[
        str | None,
        FlextRuntime.Metadata | m.ConfigMap | None,
    ]:
        """Extract correlation_id and metadata from kwargs.

        Returns typed values: correlation_id as str | None, metadata as FlextRuntime.Metadata | Mapping | None.
        """
        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id = e._safe_optional_str(correlation_id_raw)
        metadata_raw = kwargs.get("metadata")
        metadata = e._safe_metadata(metadata_raw)
        if metadata is None:
            metadata = e._safe_config_map(metadata_raw)
        return (correlation_id, metadata)

    @staticmethod
    def create_error(error_type: str, message: str) -> e.BaseError:
        """Create an exception instance based on error type."""
        error_classes = {
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
        meta: FlextRuntime.Metadata | None,
    ) -> m.ConfigMap | None:
        """Prepare metadata value for error creation."""
        if meta is None:
            return None
        return m.ConfigMap.model_validate(meta.attributes)

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
        context: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap | None = None,
    ) -> e.BaseError:
        """Create error by type using context dict."""
        # Build context with error_code
        error_context: m.ConfigMap = m.ConfigMap()
        if context is not None:
            error_context.update(dict(context.items()))
        if error_code is not None:
            error_context["error_code"] = error_code

        # Create appropriate error class based on type
        error_classes = {
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
        context: m.ConfigMap,
        metadata_obj: FlextRuntime.Metadata | m.ConfigMap | None,
    ) -> None:
        """Merge metadata object into context dictionary."""
        if metadata_obj is None:
            return

        metadata_model = e._safe_metadata(metadata_obj)
        if metadata_model is not None:
            for k, v in metadata_model.attributes.items():
                context[k] = FlextRuntime.normalize_to_metadata_value(v)
            return

        metadata_map = e._safe_config_map(metadata_obj)
        if metadata_map is not None:
            for k, v in metadata_map.items():
                context[k] = FlextRuntime.normalize_to_metadata_value(v)

    @staticmethod
    def _build_error_context(
        correlation_id: str | None,
        metadata_obj: FlextRuntime.Metadata | m.ConfigMap | None,
        kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
    ) -> m.ConfigMap:
        """Build error context dictionary."""
        error_context: m.ConfigMap = m.ConfigMap()
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

    # Mutable ClassVar for runtime metrics
    _exception_counts: ClassVar[MutableMapping[type, int]] = {}

    @classmethod
    def record_exception(cls, exception_type: type) -> None:
        """Record an exception occurrence for metrics tracking."""
        if exception_type not in cls._exception_counts:
            cls._exception_counts[exception_type] = 0
        cls._exception_counts[exception_type] += 1

    @classmethod
    def get_metrics(cls) -> m.ErrorMap:
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
        exception_counts_dict = {}
        for exc_type, count in cls._exception_counts.items():
            exc_name = (
                exc_type.__qualname__
                if hasattr(exc_type, "__qualname__")
                else str(exc_type)
            )
            exception_counts_dict[exc_name] = count
        # Build result dict matching ErrorTypeMapping type
        # Values are int | str | dict[str, int] - all compatible with ErrorTypeMapping
        result_dict: m.ErrorMap = m.ErrorMap(
            root={
                "total_exceptions": total,
                "exception_counts": exception_counts_dict,
                "exception_counts_summary": exception_counts_str,
                "unique_exception_types": len(cls._exception_counts),
            },
        )
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
        normalized_kwargs: m.ConfigMap = m.ConfigMap()
        for k, v in kwargs.items():
            normalized_kwargs[k] = FlextRuntime.normalize_to_metadata_value(v)
        return self.create(
            message,
            error_code,
            **{
                k: FlextRuntime.normalize_to_metadata_value(v)
                for k, v in normalized_kwargs.items()
            },
        )


e = FlextExceptions

__all__ = ["FlextExceptions", "e"]
