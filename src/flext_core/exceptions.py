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
from typing import ClassVar, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError as PydanticValidationError,
)

from flext_core.constants import c
from flext_core.models import m
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

# FlextRuntime.Metadata used directly (no alias) per runtime-alias-only policy


class MetadataProtocol(Protocol):
    @property
    def attributes(self) -> Mapping[str, t.MetadataAttributeValue]: ...


class FlextExceptions:
    """Exception types with correlation metadata.

    Provides structured exceptions with error codes and correlation tracking
    for consistent error handling and logging.
    """

    _metadata_map_adapter: ClassVar[
        TypeAdapter[dict[str, t.MetadataAttributeValue]]
    ] = TypeAdapter(dict[str, t.MetadataAttributeValue])

    class _ParamsModel(m.ArbitraryTypesModel):
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
        attribute_context: t.MetadataAttributeValue | None = Field(default=None)

    @staticmethod
    def _safe_optional_str(
        value: t.MetadataAttributeValue | type | None,
    ) -> str | None:
        """Extract optional strict string from dynamic values."""
        if value is None:
            return None
        try:
            return e._StrictStringValue(value=value).value
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_bool(value: t.MetadataAttributeValue | None, *, default: bool) -> bool:
        """Extract strict bool from dynamic values with default fallback."""
        if value is None:
            return default
        try:
            return e._StrictBooleanValue(value=value).value
        except PydanticValidationError:
            return default

    @staticmethod
    def _safe_int(value: t.MetadataAttributeValue | None) -> int | None:
        """Extract optional strict integer from dynamic values."""
        if value is None:
            return None
        try:
            return e._StrictIntValue(value=value).value
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_number(value: t.MetadataAttributeValue | None) -> int | float | None:
        """Extract optional strict numeric value from dynamic values."""
        if value is None:
            return None
        try:
            return e._StrictNumberValue(value=value).value
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_config_map(
        value: MetadataProtocol
        | FlextRuntime.Metadata
        | Mapping[str, t.ConfigMapValue]
        | m.ConfigMap
        | t.ConfigMapValue
        | None,
    ) -> Mapping[str, t.MetadataAttributeValue] | None:
        """Extract ConfigMap when value is mapping-compatible."""
        if value is None:
            return None
        try:
            return e._metadata_map_adapter.validate_python(value)
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_metadata(
        value: MetadataProtocol
        | FlextRuntime.Metadata
        | Mapping[str, t.ConfigMapValue]
        | m.ConfigMap
        | t.ConfigMapValue
        | None,
    ) -> MetadataProtocol | None:
        """Normalize supported metadata inputs to runtime metadata model."""
        if value is None:
            return None

        try:
            return m.Metadata.model_validate(value)
        except PydanticValidationError:
            pass

        dumped_map: Mapping[str, t.MetadataAttributeValue] | None = None
        model_dump = value.model_dump if hasattr(value, "model_dump") else None
        if callable(model_dump):
            dumped_candidate = model_dump()
            try:
                dumped_map = e._metadata_map_adapter.validate_python(dumped_candidate)
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
                return m.Metadata(attributes=attrs)

        attrs_map = e._safe_config_map(value)
        if attrs_map is not None:
            attrs = {
                k: FlextRuntime.normalize_to_metadata_value(v)
                for k, v in attrs_map.items()
            }
            return m.Metadata(attributes=attrs)

        return None

    @staticmethod
    def _build_context_map(
        context: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap | None,
        extra_kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> m.ConfigMap:
        """Build normalized context map from context and kwargs."""
        excluded = excluded_keys or frozenset()
        context_map: m.ConfigMap = m.ConfigMap(root={})
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
        param_map: m.ConfigMap = m.ConfigMap(root={})
        if context:
            param_map.update({k: v for k, v in context.items() if k in keys})
        if extra_kwargs:
            param_map.update({k: v for k, v in extra_kwargs.items() if k in keys})
        return param_map

    @staticmethod
    def _init_error_params[TParams: BaseModel](
        context: Mapping[str, t.MetadataAttributeValue] | None,
        extra_kwargs: dict[str, t.MetadataAttributeValue],
        named_params: dict[str, t.MetadataAttributeValue | None],
        params_cls: type[TParams],
        existing_params: TParams | None,
        param_keys: set[str] | frozenset[str],
        *,
        excluded_context_keys: set[str] | frozenset[str] | None = None,
    ) -> tuple[
        TParams,
        m.ConfigMap | None,
        t.MetadataAttributeValue | None,
        str | None,
    ]:
        """Extract, resolve and build error parameters from kwargs.

        Shared init boilerplate for all typed error subclasses.

        Args:
            context: Optional context mapping
            extra_kwargs: Additional kwargs (metadata/correlation_id popped)
            named_params: Explicitly named params (override if not None)
            params_cls: Pydantic model class for params
            existing_params: Pre-built params (skip validation if provided)
            param_keys: Set of param field names
            excluded_context_keys: Keys to exclude from context map

        Returns:
            Tuple of (resolved_params, error_context, metadata, correlation_id)

        """
        preserved_metadata = extra_kwargs.pop("metadata", None)
        correlation_id_raw = extra_kwargs.pop("correlation_id", None)
        correlation_id_str = e._safe_optional_str(correlation_id_raw)

        param_values = e._build_param_map(context, extra_kwargs, keys=param_keys)
        for key, val in named_params.items():
            if val is not None:
                param_values[key] = val

        resolved = (
            existing_params
            if existing_params is not None
            else params_cls.model_validate(dict(param_values))
        )

        error_context = e._build_context_map(
            context,
            extra_kwargs,
            excluded_keys=excluded_context_keys,
        )
        for key in param_keys:
            attr_val = getattr(resolved, key, None)
            if attr_val is not None:
                error_context[key] = FlextRuntime.normalize_to_metadata_value(attr_val)

        return resolved, error_context or None, preserved_metadata, correlation_id_str

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
            | MetadataProtocol
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
                else m.ConfigMap(root={})
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
        ) -> Mapping[str, t.MetadataAttributeValue]:
            """Convert exception to dictionary representation.

            Returns:
                Dictionary with error_type, message, error_code, and other fields.

            """
            result: dict[str, t.MetadataAttributeValue] = {
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
            metadata_dict: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
            merged_kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
        ) -> MetadataProtocol:
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

            return m.Metadata(
                attributes={
                    k: FlextRuntime.normalize_to_metadata_value(v)
                    for k, v in merged_attrs.items()
                },
            )

        @staticmethod
        def _normalize_metadata(
            metadata: FlextRuntime.Metadata
            | MetadataProtocol
            | m.ConfigMap
            | t.MetadataAttributeValue
            | None,
            merged_kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
        ) -> MetadataProtocol:
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
                    return m.Metadata(attributes=normalized_attrs)
                return m.Metadata(attributes={})

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
                return m.Metadata(attributes=merged_attrs)

            metadata_dict = e._safe_config_map(metadata)
            if metadata_dict is not None:
                return e.BaseError._normalize_metadata_from_dict(
                    metadata_dict,
                    merged_kwargs,
                )

            # Fallback: convert to FlextRuntime.Metadata with string value
            return m.Metadata(attributes={"value": str(metadata)})

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {"field": field, "value": value},
                e.ValidationErrorParams,
                params,
                {"field", "value"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=corr,
            )
            self.field = resolved.field
            self.value = resolved.value

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {"config_key": config_key, "config_source": config_source},
                e.ConfigurationErrorParams,
                params,
                {"config_key", "config_source"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=corr,
            )
            self.config_key = resolved.config_key
            self.config_source = resolved.config_source

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {},
                e.ConnectionErrorParams,
                params,
                {"host", "port", "timeout"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=corr,
            )
            self.host = resolved.host
            self.port = resolved.port
            self.timeout = resolved.timeout

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {"timeout_seconds": timeout_seconds, "operation": operation},
                e.TimeoutErrorParams,
                params,
                {"timeout_seconds", "operation"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=corr,
            )
            self.timeout_seconds = resolved.timeout_seconds
            self.operation = resolved.operation

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {"auth_method": auth_method, "user_id": user_id},
                e.AuthenticationErrorParams,
                params,
                {"auth_method", "user_id"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=corr,
            )
            self.auth_method = resolved.auth_method
            self.user_id = resolved.user_id

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {},
                e.AuthorizationErrorParams,
                params,
                {"user_id", "resource", "permission"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=correlation_id if correlation_id is not None else corr,
            )
            self.user_id = resolved.user_id
            self.resource = resolved.resource
            self.permission = resolved.permission

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        @staticmethod
        def _extract_context_values(
            context: Mapping[str, t.MetadataAttributeValue] | None,
        ) -> tuple[
            str | None,
            MetadataProtocol | None,
            bool,
            bool,
        ]:
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
            | MetadataProtocol
            | m.ConfigMap
            | t.MetadataAttributeValue
            | None = None,
            correlation_id: str | None = None,
            params: e.NotFoundErrorParams | None = None,
            **extra_kwargs: t.MetadataAttributeValue,
        ) -> None:
            """Initialize not found error with resource context."""
            resolved, ctx, meta, _ = e._init_error_params(
                context,
                extra_kwargs,
                {"resource_type": resource_type, "resource_id": resource_id},
                e.NotFoundErrorParams,
                params,
                {"resource_type", "resource_id"},
                excluded_context_keys={"correlation_id", "metadata"},
            )
            metadata_input = metadata if metadata is not None else meta
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=metadata_input,
                correlation_id=correlation_id,
            )
            self.resource_type = resolved.resource_type
            self.resource_id = resolved.resource_id

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {},
                e.ConflictErrorParams,
                params,
                {"resource_type", "resource_id", "conflict_reason"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=correlation_id if correlation_id is not None else corr,
            )
            self.resource_type = resolved.resource_type
            self.resource_id = resolved.resource_id
            self.conflict_reason = resolved.conflict_reason

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {},
                e.RateLimitErrorParams,
                params,
                {"limit", "window_seconds", "retry_after"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=correlation_id if correlation_id is not None else corr,
            )
            self.limit = resolved.limit
            self.window_seconds = resolved.window_seconds
            self.retry_after = resolved.retry_after

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {},
                e.CircuitBreakerErrorParams,
                params,
                {"service_name", "failure_count", "reset_timeout"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=correlation_id if correlation_id is not None else corr,
            )
            self.service_name = resolved.service_name
            self.failure_count = resolved.failure_count
            self.reset_timeout = resolved.reset_timeout

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

            qualname_value = (
                type_value.__qualname__ if hasattr(type_value, "__qualname__") else None
            )
            return e._safe_optional_str(qualname_value)

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
                expected_type,
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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {"operation": operation, "reason": reason},
                e.OperationErrorParams,
                params,
                {"operation", "reason"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=correlation_id if correlation_id is not None else corr,
            )
            self.operation = resolved.operation
            self.reason = resolved.reason

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
            resolved, ctx, meta, corr = e._init_error_params(
                context,
                extra_kwargs,
                {
                    "attribute_name": attribute_name,
                    "attribute_context": attribute_context,
                },
                e.AttributeAccessErrorParams,
                params,
                {"attribute_name", "attribute_context"},
            )
            super().__init__(
                message,
                error_code=error_code,
                context=ctx,
                metadata=meta,
                correlation_id=correlation_id if correlation_id is not None else corr,
            )
            self.attribute_name = resolved.attribute_name
            self.attribute_context = resolved.attribute_context

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
        Mapping[str, t.MetadataAttributeValue],
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
        MetadataProtocol | Mapping[str, t.MetadataAttributeValue] | None,
    ]:
        """Extract correlation_id and metadata from kwargs.

        Returns typed values: correlation_id as str | None, metadata as FlextRuntime.Metadata | Mapping | None.
        """
        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id = e._safe_optional_str(correlation_id_raw)
        metadata_raw = kwargs.get("metadata")
        metadata: MetadataProtocol | Mapping[str, t.MetadataAttributeValue] | None = (
            None
        )
        model_dump = (
            metadata_raw.model_dump if hasattr(metadata_raw, "model_dump") else None
        )
        if callable(model_dump):
            metadata = e._safe_metadata(metadata_raw)
        if metadata is None:
            metadata = e._safe_config_map(metadata_raw)
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
        meta: MetadataProtocol | Mapping[str, t.MetadataAttributeValue] | None,
    ) -> Mapping[str, t.MetadataAttributeValue] | None:
        """Prepare metadata value for error creation."""
        if meta is None:
            return None
        if isinstance(meta, Mapping):
            return {
                key: FlextRuntime.normalize_to_metadata_value(value)
                for key, value in meta.items()
            }
        return dict(meta.attributes)

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
        error_context: m.ConfigMap = m.ConfigMap(root={})
        if context is not None:
            error_context.update(dict(context.items()))
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

        context_payload: Mapping[str, t.MetadataAttributeValue] | None = None
        if error_context:
            context_payload = {
                key: FlextRuntime.normalize_to_metadata_value(value)
                for key, value in error_context.items()
            }

        if error_class is not None:
            # Type narrowing: error_class is type[e.BaseError] after None check
            return error_class(
                message,
                error_code=error_code or c.Errors.UNKNOWN_ERROR,
                context=context_payload,
                correlation_id=correlation_id,
            )

        return e.BaseError(
            message,
            error_code=error_code or c.Errors.UNKNOWN_ERROR,
            context=context_payload,
            correlation_id=correlation_id,
        )

    @staticmethod
    def _merge_metadata_into_context(
        context: m.ConfigMap,
        metadata_obj: (
            MetadataProtocol | Mapping[str, t.MetadataAttributeValue] | None
        ),
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
            for k, metadata_value in metadata_map.items():
                context[k] = FlextRuntime.normalize_to_metadata_value(metadata_value)

    @staticmethod
    def _build_error_context(
        correlation_id: str | None,
        metadata_obj: (
            MetadataProtocol | Mapping[str, t.MetadataAttributeValue] | None
        ),
        kwargs: Mapping[str, t.MetadataAttributeValue] | m.ConfigMap,
    ) -> m.ConfigMap:
        """Build error context dictionary."""
        error_context: m.ConfigMap = m.ConfigMap(root={})
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
        # Normalize kwargs to t.MetadataAttributeValue
        # normalize_to_metadata_value already returns t.MetadataAttributeValue
        normalized_kwargs: m.ConfigMap = m.ConfigMap(root={})
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

_Metadata = m.Metadata

__all__ = ["FlextExceptions", "_Metadata", "e"]
