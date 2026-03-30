"""Exception hierarchy with correlation metadata.

Provides structured exceptions with error codes and correlation tracking
for consistent error handling across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from typing import ClassVar, override

from pydantic import (
    BaseModel,
    ValidationError as PydanticValidationError,
)

from flext_core import (
    FlextRuntime,
    FlextUtilitiesGuardsTypeCore,
    c,
    m,
    t,
)


class FlextExceptions:
    """Exception types with correlation metadata.

    Provides structured exceptions with error codes and correlation tracking
    for consistent error handling and logging.
    """

    @staticmethod
    def _build_context_map(
        context: Mapping[str, t.MetadataValue] | t.ConfigMap | None,
        extra_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap,
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> t.ConfigMap:
        """Build normalized context map from context and kwargs."""
        excluded = excluded_keys or frozenset()
        context_map: t.ConfigMap = t.ConfigMap(root={})
        if context:
            context_map.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in context.items()
                if k not in excluded
            })
        if extra_kwargs:
            context_map.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in extra_kwargs.items()
                if k not in excluded
            })
        return context_map

    @staticmethod
    def _build_param_map(
        context: Mapping[str, t.MetadataValue] | t.ConfigMap | None,
        extra_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap,
        keys: set[str] | frozenset[str],
    ) -> t.ConfigMap:
        """Build unnormalized parameter map for strict params validation."""
        param_map: t.ConfigMap = t.ConfigMap(root={})
        if context:
            param_map.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in context.items()
                if k in keys
            })
        if extra_kwargs:
            param_map.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in extra_kwargs.items()
                if k in keys
            })
        return param_map

    @staticmethod
    def _init_error_params[TParams: BaseModel](
        context: Mapping[str, t.MetadataValue] | None,
        extra_kwargs: t.FlatContainerMapping,
        named_params: Mapping[str, t.RuntimeData | None],
        params_cls: type[TParams],
        existing_params: TParams | None,
        param_keys: set[str] | frozenset[str],
        *,
        excluded_context_keys: set[str] | frozenset[str] | None = None,
    ) -> tuple[TParams, t.ConfigMap | None, t.MetadataValue | None, str | None]:
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
        mutable_extra: t.MutableFlatContainerMapping = dict(extra_kwargs)
        preserved_metadata_raw = mutable_extra.pop(c.FIELD_METADATA, None)
        preserved_metadata = (
            FlextRuntime.normalize_to_metadata(preserved_metadata_raw)
            if preserved_metadata_raw is not None
            else None
        )
        correlation_id_raw = mutable_extra.pop(c.KEY_CORRELATION_ID, None)
        correlation_id_str = (
            e._safe_optional_str(correlation_id_raw)
            if FlextUtilitiesGuardsTypeCore.is_scalar(correlation_id_raw)
            else None
        )
        normalized_extra_kwargs: Mapping[str, t.MetadataValue] = {
            key: FlextRuntime.normalize_to_metadata(value)
            for key, value in mutable_extra.items()
        }
        param_values: MutableMapping[str, t.ValueOrModel] = dict(
            e._build_param_map(context, normalized_extra_kwargs, keys=param_keys),
        )
        for key, val in named_params.items():
            if val is not None:
                normalized_val = FlextRuntime.normalize_to_metadata(val)

                def to_normalized(value: t.MetadataValue) -> t.NormalizedValue:
                    if FlextUtilitiesGuardsTypeCore.is_scalar(value):
                        return value
                    if isinstance(value, Mapping):
                        return {
                            str(inner_key): to_normalized(
                                FlextRuntime.normalize_to_metadata(inner_val),
                            )
                            for inner_key, inner_val in value.items()
                        }
                    if isinstance(value, list):
                        return [
                            to_normalized(FlextRuntime.normalize_to_metadata(inner_val))
                            for inner_val in value
                        ]
                    return str(value)

                param_values[key] = to_normalized(normalized_val)
        resolved: TParams = (
            existing_params
            if existing_params is not None
            else params_cls.model_validate(dict(param_values))
        )
        error_context = e._build_context_map(
            context,
            normalized_extra_kwargs,
            excluded_keys=excluded_context_keys,
        )
        for key in param_keys:
            attr_val = getattr(resolved, key, None)
            if attr_val is not None:
                error_context[key] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(attr_val),
                )
        return (resolved, error_context or None, preserved_metadata, correlation_id_str)

    @staticmethod
    def _safe_bool(value: t.Scalar | None, *, default: bool) -> bool:
        """Extract strict bool from dynamic values with default fallback."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return default

    @staticmethod
    def _safe_config_map(
        value: m.Metadata
        | t.ConfigMap
        | Mapping[str, t.MetadataOrValue | None]
        | t.MetadataValue
        | t.NormalizedValue
        | None,
    ) -> Mapping[str, t.MetadataOrValue | None] | None:
        """Extract ConfigMap when value is mapping-compatible."""
        if value is None:
            return None
        try:
            return m.Validators.dict_str_metadata_adapter().validate_python(value)
        except PydanticValidationError:
            return None

    @staticmethod
    def _safe_int(value: t.Scalar | None) -> int | None:
        """Extract optional strict integer from dynamic values."""
        if value is None:
            return None
        if isinstance(value, int) and (not isinstance(value, bool)):
            return value
        return None

    @staticmethod
    def _safe_metadata(
        value: m.Metadata
        | t.ConfigMap
        | Mapping[str, t.MetadataOrValue | None]
        | t.MetadataValue
        | t.NormalizedValue
        | None,
    ) -> m.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model."""
        if value is None:
            return None
        try:
            return m.Metadata.model_validate(value, from_attributes=True)
        except (PydanticValidationError, TypeError):
            pass
        dumped_map: Mapping[str, t.MetadataOrValue | None] | None = None
        if isinstance(value, BaseModel):
            dumped_candidate = value.model_dump()
            try:
                dumped_map = m.Validators.dict_str_metadata_adapter().validate_python(
                    dumped_candidate,
                )
            except PydanticValidationError:
                dumped_map = None
        if dumped_map is not None:
            attrs_raw = dumped_map.get(c.FIELD_ATTRIBUTES)
            attrs_map = e._safe_config_map(attrs_raw)
            if attrs_map is not None:
                attrs = {
                    k: FlextRuntime.normalize_to_metadata(v)
                    for k, v in attrs_map.items()
                }
                return m.Metadata.model_validate({c.FIELD_ATTRIBUTES: attrs})
        attrs_map = e._safe_config_map(value)
        if attrs_map is not None:
            attrs = {
                k: FlextRuntime.normalize_to_metadata(v) for k, v in attrs_map.items()
            }
            return m.Metadata.model_validate({c.FIELD_ATTRIBUTES: attrs})
        return None

    @staticmethod
    def _safe_number(value: t.Scalar | None) -> t.Numeric | None:
        """Extract optional strict numeric value from dynamic values."""
        if value is None:
            return None
        if isinstance(value, (int, float)) and (not isinstance(value, bool)):
            return value
        return None

    @staticmethod
    def _safe_optional_str(value: t.Container | type | None) -> str | None:
        """Extract optional strict string from dynamic values."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return None

    class BaseError(Exception):
        """Base exception with correlation metadata and error codes.

        All FLEXT exceptions inherit from this to ensure consistent error
        handling, logging, and correlation tracking across the ecosystem.
        """

        _params_cls: ClassVar[type[BaseModel] | None] = None
        _param_keys: ClassVar[frozenset[str]] = frozenset()
        _excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = None

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.UNKNOWN_ERROR,
            context: Mapping[str, t.MetadataValue] | t.ConfigMap | None = None,
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None = None,
            correlation_id: str | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
            merged_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize base error with message and optional metadata.

            Args:
                message: Error message
                error_code: Optional error code
                context: Optional context mapping
                metadata: Optional metadata (m.Metadata, dict, or payload types)
                correlation_id: Optional correlation ID
                auto_correlation: Auto-generate correlation ID if not provided
                auto_log: Auto-log error on creation
                merged_kwargs: Additional metadata attributes to merge

            """
            super().__init__(message)
            self.message = message
            self.error_code = error_code
            final_kwargs: t.ConfigMap = t.ConfigMap(root={})
            if merged_kwargs:
                final_kwargs.update({
                    k: FlextRuntime.normalize_to_container(
                        FlextRuntime.normalize_to_metadata(v),
                    )
                    for k, v in merged_kwargs.items()
                })
            if context:
                final_kwargs.update({
                    k: FlextRuntime.normalize_to_container(
                        FlextRuntime.normalize_to_metadata(v),
                    )
                    for k, v in context.items()
                })
            if extra_kwargs:
                final_kwargs.update({
                    k: FlextRuntime.normalize_to_container(
                        FlextRuntime.normalize_to_metadata(v),
                    )
                    for k, v in extra_kwargs.items()
                })
            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if auto_correlation and (not correlation_id)
                else correlation_id
            )
            self.metadata = e.BaseError._normalize_metadata(metadata, final_kwargs)
            self.timestamp = time.time()
            self.auto_log = auto_log

        @classmethod
        def __init_subclass__(cls, **kwargs: t.Container) -> None:
            """Auto-generate __init__ for declarative typed error subclasses.

            Subclasses that declare _params_cls, _default_error_code, and
            _param_keys as ClassVar but omit __init__ get a standard
            __init__ auto-injected. This eliminates boilerplate for exceptions
            whose domain params come entirely from **extra_kwargs.
            """
            super().__init_subclass__(**kwargs)
            params_cls = cls.__dict__.get("_params_cls")
            if params_cls is None or "__init__" in cls.__dict__:
                return
            default_code = cls.__dict__.get("_default_error_code", c.UNKNOWN_ERROR)

            def _auto_init(
                self: e.BaseError,
                message: str,
                *,
                error_code: str = default_code,
                context: Mapping[str, t.MetadataValue] | None = None,
                correlation_id: str | None = None,
                params: BaseModel | None = None,
                **extra_kwargs: t.Container,
            ) -> None:
                self._init_declared_error(
                    message,
                    error_code=error_code,
                    context=context,
                    params=params,
                    correlation_id=correlation_id,
                    extra_kwargs=extra_kwargs,
                )

            cls.__init__ = _auto_init  # type: ignore[assignment]

        @override
        def __str__(self) -> str:
            """Return string representation with error code if present."""
            if self.error_code:
                return f"[{self.error_code}] {self.message}"
            return self.message

        @staticmethod
        def _normalize_metadata(
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None,
            merged_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap,
        ) -> m.Metadata:
            """Normalize metadata from various input types to m.Metadata model.

            Args:
                metadata: m.Metadata instance, dict-like t.NormalizedValue, or None
                merged_kwargs: Additional attributes to merge

            Returns:
                Normalized m.Metadata instance

            """
            if metadata is None:
                if merged_kwargs:
                    normalized_attrs = {
                        k: FlextRuntime.normalize_to_metadata(v)
                        for k, v in merged_kwargs.items()
                    }
                    return m.Metadata.model_validate({
                        c.FIELD_ATTRIBUTES: normalized_attrs,
                    })
                return m.Metadata.model_validate({c.FIELD_ATTRIBUTES: {}})
            metadata_model = e._safe_metadata(metadata)
            if metadata_model is not None:
                if not merged_kwargs:
                    return metadata_model
                merged_attrs = {
                    k: FlextRuntime.normalize_to_metadata(v)
                    for k, v in metadata_model.attributes.items()
                }
                for k, v in merged_kwargs.items():
                    merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
                return m.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: merged_attrs,
                })
            metadata_dict = e._safe_config_map(metadata)
            if metadata_dict is not None:
                return e.BaseError._normalize_metadata_from_dict(
                    metadata_dict,
                    merged_kwargs,
                )
            return m.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: {"value": str(metadata)},
            })

        @staticmethod
        def _normalize_metadata_from_dict(
            metadata_dict: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap,
            merged_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap,
        ) -> m.Metadata:
            """Normalize metadata from dict-like t.NormalizedValue."""
            merged_attrs: MutableMapping[str, t.MetadataValue | None] = {}
            for k, v in metadata_dict.items():
                merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
            if merged_kwargs:
                for k, v in merged_kwargs.items():
                    merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
            return m.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: {
                    k: FlextRuntime.normalize_to_metadata(v)
                    for k, v in merged_attrs.items()
                },
            })

        def to_dict(self) -> Mapping[str, t.MetadataValue | None]:
            """Convert exception to dictionary representation.

            Returns:
                Dictionary with error_type, message, error_code, and other fields.

            """
            result: MutableMapping[str, t.MetadataValue | None] = {
                "error_type": type(self).__name__,
                "message": self.message,
                "error_code": self.error_code,
                c.KEY_CORRELATION_ID: self.correlation_id,
                "timestamp": self.timestamp,
            }
            if self.metadata and self.metadata.attributes:
                filtered_attrs = {
                    k: v for k, v in self.metadata.attributes.items() if k not in result
                }
                result.update(filtered_attrs)
            return result

        def _init_from_params[TParams: BaseModel](
            self,
            message: str,
            *,
            error_code: str,
            context: Mapping[str, t.MetadataValue] | None,
            extra_kwargs: t.FlatContainerMapping,
            named_params: Mapping[str, t.RuntimeData | None],
            params_cls: type[TParams],
            existing_params: TParams | None,
            param_keys: set[str] | frozenset[str],
            correlation_id: str | None = None,
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None = None,
            excluded_context_keys: set[str] | frozenset[str] | None = None,
        ) -> TParams:
            """Resolve params, call super().__init__, return resolved params model."""
            result: tuple[
                TParams, t.ConfigMap | None, t.MetadataValue | None, str | None
            ] = e._init_error_params(
                context,
                extra_kwargs,
                named_params,
                params_cls,
                existing_params,
                param_keys,
                excluded_context_keys=excluded_context_keys,
            )
            resolved, ctx, meta, corr = result
            final_meta = metadata if metadata is not None else meta
            final_corr = correlation_id if correlation_id is not None else corr
            e.BaseError.__init__(
                self,
                message,
                error_code=error_code,
                context=ctx,
                metadata=final_meta,
                correlation_id=final_corr,
            )
            return resolved

        def _init_typed_error[TParams: BaseModel](
            self,
            message: str,
            *,
            error_code: str,
            context: Mapping[str, t.MetadataValue] | None,
            extra_kwargs: t.FlatContainerMapping,
            named_params: Mapping[str, t.RuntimeData | None],
            params_cls: type[TParams],
            existing_params: TParams | None,
            param_keys: frozenset[str],
            correlation_id: str | None = None,
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None = None,
            excluded_context_keys: set[str] | frozenset[str] | None = None,
        ) -> None:
            """Resolve params via _init_from_params and auto-assign to self."""
            resolved = self._init_from_params(
                message,
                error_code=error_code,
                context=context,
                extra_kwargs=extra_kwargs,
                named_params=named_params,
                params_cls=params_cls,
                existing_params=existing_params,
                param_keys=param_keys,
                correlation_id=correlation_id,
                metadata=metadata,
                excluded_context_keys=excluded_context_keys,
            )
            for key in param_keys:
                setattr(self, key, getattr(resolved, key))

        def _init_declared_error(
            self,
            message: str,
            *,
            error_code: str,
            context: Mapping[str, t.MetadataValue] | None,
            params: BaseModel | None,
            named_params: Mapping[str, t.RuntimeData | None] | None = None,
            extra_kwargs: t.FlatContainerMapping | None = None,
            param_keys: frozenset[str] | None = None,
            correlation_id: str | None = None,
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None = None,
        ) -> None:
            """Initialize a typed error from class-declared params metadata."""
            declared_params_cls = type(self)._params_cls
            if declared_params_cls is None:
                msg = f"{type(self).__qualname__} is missing _params_cls"
                raise ValueError(msg)
            declared_param_keys = (
                param_keys if param_keys is not None else type(self)._param_keys
            )
            remaining_extra_kwargs: t.MutableFlatContainerMapping = dict(
                extra_kwargs or {}
            )
            resolved_named_params: MutableMapping[str, t.RuntimeData | None] = dict(
                named_params or {},
            )
            for key in declared_param_keys:
                resolved_named_params.setdefault(
                    key,
                    remaining_extra_kwargs.pop(key, None),
                )
            self._init_typed_error(
                message,
                error_code=error_code,
                context=context,
                extra_kwargs=remaining_extra_kwargs,
                named_params=resolved_named_params,
                params_cls=declared_params_cls,
                existing_params=params,
                param_keys=declared_param_keys,
                correlation_id=correlation_id,
                metadata=metadata,
                excluded_context_keys=type(self)._excluded_context_keys,
            )

    class ValidationError(BaseError):
        """Exception raised for input validation failures."""

        field: str | None = None
        value: t.Scalar | None = None
        _params_cls: ClassVar[type[BaseModel] | None] = m.ValidationErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({"field", "value"})

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: t.Scalar | None = None,
            error_code: str = c.VALIDATION_ERROR,
            context: Mapping[str, t.MetadataValue] | None = None,
            correlation_id: str | None = None,
            params: m.ValidationErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize validation error with field and value information."""
            self._init_declared_error(
                message,
                error_code=error_code,
                context=context,
                params=params,
                named_params={"field": field, "value": value},
                correlation_id=correlation_id,
                extra_kwargs=extra_kwargs,
            )

    class ConfigurationError(BaseError):
        """Exception raised for configuration-related errors."""

        config_key: str | None = None
        config_source: str | None = None
        _params_cls: ClassVar[type[BaseModel] | None] = m.ConfigurationErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            "config_key",
            "config_source",
        })

        def __init__(
            self,
            message: str,
            *,
            config_key: str | None = None,
            config_source: str | None = None,
            error_code: str = c.CONFIGURATION_ERROR,
            context: Mapping[str, t.MetadataValue] | None = None,
            correlation_id: str | None = None,
            params: m.ConfigurationErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize configuration error with config context."""
            self._init_declared_error(
                message,
                error_code=error_code,
                context=context,
                params=params,
                named_params={
                    "config_key": config_key,
                    "config_source": config_source,
                },
                correlation_id=correlation_id,
                extra_kwargs=extra_kwargs,
            )

    class ConnectionError(BaseError):
        """Exception raised for network and connection failures."""

        host: str | None = None
        port: int | None = None
        timeout: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.CONNECTION_ERROR
        _params_cls: ClassVar[type[BaseModel] | None] = m.ConnectionErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            "host",
            "port",
            "timeout",
        })

    class TimeoutError(BaseError):
        """Exception raised for operation timeout errors."""

        timeout_seconds: t.Numeric | None = None
        operation: str | None = None
        _params_cls: ClassVar[type[BaseModel] | None] = m.TimeoutErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            "timeout_seconds",
            "operation",
        })

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            operation: str | None = None,
            error_code: str = c.TIMEOUT_ERROR,
            context: Mapping[str, t.MetadataValue] | None = None,
            correlation_id: str | None = None,
            params: m.TimeoutErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize timeout error with timeout context."""
            self._init_declared_error(
                message,
                error_code=error_code,
                context=context,
                params=params,
                named_params={
                    "timeout_seconds": timeout_seconds,
                    "operation": operation,
                },
                correlation_id=correlation_id,
                extra_kwargs=extra_kwargs,
            )

    class AuthenticationError(BaseError):
        """Exception raised for authentication failures."""

        auth_method: str | None = None
        user_id: str | None = None
        _params_cls: ClassVar[type[BaseModel] | None] = m.AuthenticationErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            "auth_method",
            c.KEY_USER_ID,
        })

        def __init__(
            self,
            message: str,
            *,
            auth_method: str | None = None,
            user_id: str | None = None,
            error_code: str = c.AUTHENTICATION_ERROR,
            context: Mapping[str, t.MetadataValue] | None = None,
            correlation_id: str | None = None,
            params: m.AuthenticationErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize authentication error with auth context."""
            self._init_declared_error(
                message,
                error_code=error_code,
                context=context,
                params=params,
                named_params={
                    "auth_method": auth_method,
                    c.KEY_USER_ID: user_id,
                },
                correlation_id=correlation_id,
                extra_kwargs=extra_kwargs,
            )

    class AuthorizationError(BaseError):
        """Exception raised for permission and authorization failures."""

        user_id: str | None = None
        resource: str | None = None
        permission: str | None = None
        _default_error_code: ClassVar[str] = c.AUTHORIZATION_ERROR
        _params_cls: ClassVar[type[BaseModel] | None] = m.AuthorizationErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            c.KEY_USER_ID,
            "resource",
            "permission",
        })

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        resource_type: str | None = None
        resource_id: str | None = None
        _params_cls: ClassVar[type[BaseModel] | None] = m.NotFoundErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            "resource_type",
            "resource_id",
        })
        _excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = frozenset({
            c.KEY_CORRELATION_ID,
            c.FIELD_METADATA,
        })

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            error_code: str = c.NOT_FOUND_ERROR,
            context: Mapping[str, t.MetadataValue] | None = None,
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None = None,
            correlation_id: str | None = None,
            params: m.NotFoundErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize not found error with resource context."""
            self._init_declared_error(
                message,
                error_code=error_code,
                context=context,
                params=params,
                named_params={
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                },
                correlation_id=correlation_id,
                metadata=metadata,
                extra_kwargs=extra_kwargs,
            )

    class ConflictError(BaseError):
        """Exception raised for resource conflicts."""

        resource_type: str | None = None
        resource_id: str | None = None
        conflict_reason: str | None = None
        _default_error_code: ClassVar[str] = c.ALREADY_EXISTS
        _params_cls: ClassVar[type[BaseModel] | None] = m.ConflictErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            "resource_type",
            "resource_id",
            "conflict_reason",
        })

    class RateLimitError(BaseError):
        """Exception raised when rate limits are exceeded."""

        limit: int | None = None
        window_seconds: int | None = None
        retry_after: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.OPERATION_ERROR
        _params_cls: ClassVar[type[BaseModel] | None] = m.RateLimitErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            "limit",
            "window_seconds",
            "retry_after",
        })

    class CircuitBreakerError(BaseError):
        """Exception raised when circuit breaker is open."""

        service_name: str | None = None
        failure_count: int | None = None
        reset_timeout: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.EXTERNAL_SERVICE_ERROR
        _params_cls: ClassVar[type[BaseModel] | None] = m.CircuitBreakerErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            c.KEY_SERVICE_NAME,
            "failure_count",
            "reset_timeout",
        })

        def __init__(
            self,
            message: str,
            *,
            service_name: str | None = None,
            service: str | None = None,
            failure_count: int | None = None,
            reset_timeout: t.Numeric | None = None,
            error_code: str = c.EXTERNAL_SERVICE_ERROR,
            context: Mapping[str, t.MetadataValue] | None = None,
            correlation_id: str | None = None,
            params: m.CircuitBreakerErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize circuit breaker error with canonical service metadata."""
            resolved_service_name = (
                service_name if service_name is not None else service
            )
            self._init_declared_error(
                message,
                error_code=error_code,
                context=context,
                params=params,
                named_params={
                    c.KEY_SERVICE_NAME: resolved_service_name,
                    "failure_count": failure_count,
                    "reset_timeout": reset_timeout,
                },
                correlation_id=correlation_id,
                extra_kwargs=extra_kwargs,
            )

    class TypeError(BaseError):
        """Exception raised for type mismatch errors."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.TYPE_ERROR,
            expected_type: type | str | None = None,
            actual_type: type | str | None = None,
            context: Mapping[str, t.MetadataValue] | None = None,
            correlation_id: str | None = None,
            params: m.TypeErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize type error with type information."""
            preserved_metadata = extra_kwargs.pop(c.FIELD_METADATA, None)
            normalized_metadata = (
                FlextRuntime.normalize_to_metadata(preserved_metadata)
                if preserved_metadata is not None
                else None
            )
            preserved_corr_id = extra_kwargs.pop(c.KEY_CORRELATION_ID, None)
            type_map = self._get_type_map()
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
            param_values: dict[str, str | None] = {
                "expected_type": normalized_expected_type.__qualname__
                if normalized_expected_type is not None
                else None,
                "actual_type": normalized_actual_type.__qualname__
                if normalized_actual_type is not None
                else None,
            }
            resolved_params: m.TypeErrorParams = (
                params
                if params is not None
                else m.TypeErrorParams.model_validate(param_values)
            )
            normalized_extra_kwargs: Mapping[str, t.MetadataValue] = {
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in extra_kwargs.items()
            }
            type_context = self._build_type_context(
                resolved_params.expected_type,
                resolved_params.actual_type,
                context,
                normalized_extra_kwargs,
            )
            super().__init__(
                message,
                error_code=error_code,
                context=type_context or None,
                metadata=normalized_metadata,
                correlation_id=correlation_id
                if correlation_id is not None
                else e._safe_optional_str(preserved_corr_id),
            )
            self.expected_type = normalized_expected_type
            self.actual_type = normalized_actual_type

        @staticmethod
        def _build_type_context(
            expected_type: type | str | None,
            actual_type: type | str | None,
            context: Mapping[str, t.MetadataValue] | None,
            extra_kwargs: Mapping[str, t.MetadataValue],
        ) -> t.ConfigMap:
            """Build type context dictionary."""
            type_context = e._build_context_map(context, extra_kwargs)
            resolved_expected_type = e.TypeError._resolve_type_name(expected_type)
            resolved_actual_type = e.TypeError._resolve_type_name(actual_type)
            if resolved_expected_type is not None:
                type_context["expected_type"] = resolved_expected_type
            if resolved_actual_type is not None:
                type_context["actual_type"] = resolved_actual_type
            return type_context

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
            extra_kwargs: t.MutableFlatContainerMapping,
            key: str,
        ) -> type | None:
            """Normalize type value from various sources."""
            source_value: type | str | t.Container | None = type_value
            if source_value is None and key in extra_kwargs:
                source_value = extra_kwargs.pop(key)
            type_name = e.TypeError._resolve_type_name(source_value)
            if type_name is None:
                return None
            return type_map.get(type_name)

        @staticmethod
        def _resolve_type_name(
            type_value: type | str | t.Container | None,
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

    class OperationError(BaseError):
        """Exception raised for general operation failures."""

        operation: str | None
        reason: str | None

        _params_cls: ClassVar[type[BaseModel] | None] = m.OperationErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({"operation", "reason"})

        def __init__(
            self,
            message: str,
            *,
            operation: str | None = None,
            reason: str | None = None,
            error_code: str = c.OPERATION_ERROR,
            context: Mapping[str, t.MetadataValue] | None = None,
            correlation_id: str | None = None,
            params: m.OperationErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize operation error with operation context."""
            self._init_declared_error(
                message,
                error_code=error_code,
                context=context,
                params=params,
                named_params={"operation": operation, "reason": reason},
                correlation_id=correlation_id,
                extra_kwargs=extra_kwargs,
            )

    class AttributeAccessError(BaseError):
        """Exception raised for attribute access errors."""

        attribute_name: str | None
        attribute_context: t.MetadataValue | None

        _params_cls: ClassVar[type[BaseModel] | None] = m.AttributeAccessErrorParams
        _param_keys: ClassVar[frozenset[str]] = frozenset({
            "attribute_name",
            "attribute_context",
        })

        def __init__(
            self,
            message: str,
            *,
            attribute_name: str | None = None,
            attribute_context: t.MetadataValue | None = None,
            error_code: str = c.ATTRIBUTE_ERROR,
            context: Mapping[str, t.MetadataValue] | None = None,
            correlation_id: str | None = None,
            params: m.AttributeAccessErrorParams | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize attribute access error with attribute context."""
            self._init_declared_error(
                message,
                error_code=error_code,
                context=context,
                params=params,
                named_params={
                    "attribute_name": attribute_name,
                    "attribute_context": attribute_context,
                },
                correlation_id=correlation_id,
                extra_kwargs=extra_kwargs,
            )

    @staticmethod
    def _build_error_context(
        correlation_id: str | None,
        metadata_obj: m.Metadata | Mapping[str, t.MetadataOrValue | None] | None,
        kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap,
    ) -> t.ConfigMap:
        """Build error context dictionary."""
        error_context: t.ConfigMap = t.ConfigMap(root={})
        if correlation_id is not None:
            error_context[c.KEY_CORRELATION_ID] = correlation_id
        e._merge_metadata_into_context(error_context, metadata_obj)
        for k, v in kwargs.items():
            if k not in {c.KEY_CORRELATION_ID, c.FIELD_METADATA}:
                error_context[k] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
        return error_context

    @staticmethod
    def _create_error_by_type(
        error_type: str | None,
        message: str,
        error_code: str | None,
        context: Mapping[str, t.MetadataValue] | t.ConfigMap | None = None,
    ) -> e.BaseError:
        """Create error by type using context dict."""
        error_context: t.ConfigMap = t.ConfigMap(root={})
        if context is not None:
            error_context.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in context.items()
            })
        if error_code is not None:
            error_context["error_code"] = error_code
        error_classes: Mapping[str, type[e.BaseError]] = {
            "validation": e.ValidationError,
            "configuration": e.ConfigurationError,
            "connection": e.ConnectionError,
            "timeout": e.TimeoutError,
            "authentication": e.AuthenticationError,
            "authorization": e.AuthorizationError,
            "not_found": e.NotFoundError,
            "conflict": e.ConflictError,
            "rate_limit": e.RateLimitError,
            "circuit_breaker": e.CircuitBreakerError,
            "type": e.TypeError,
            "operation": e.OperationError,
            "attribute_access": e.AttributeAccessError,
        }
        error_class: type[e.BaseError] | None = (
            error_classes.get(error_type) if error_type else None
        )
        correlation_id = None
        if error_context and c.KEY_CORRELATION_ID in error_context:
            correlation_id = str(error_context[c.KEY_CORRELATION_ID])
        context_payload: Mapping[str, t.MetadataValue] | None = None
        if error_context:
            context_payload = {
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in error_context.items()
            }
        corr_str: str = correlation_id if correlation_id is not None else ""
        if error_class is not None:
            return error_class(
                message,
                error_code=error_code or c.UNKNOWN_ERROR,
                context=context_payload,
                correlation_id=corr_str or "",
            )
        return e.BaseError(
            message,
            error_code=error_code or c.UNKNOWN_ERROR,
            context=context_payload,
            correlation_id=correlation_id,
        )

    @staticmethod
    def _determine_error_type(kwargs: Mapping[str, t.MetadataValue]) -> str | None:
        """Determine error type from kwargs using pattern matching.

        Returns:
            Error type string or None if no match

        """
        error_patterns: Sequence[tuple[t.StrSequence, str]] = [
            (["field", "value"], "validation"),
            (["config_key", "config_source"], "configuration"),
            ([c.HandlerType.OPERATION], "operation"),
            (["host", "port"], "connection"),
            (["timeout_seconds"], "timeout"),
            ([c.KEY_USER_ID, "permission"], "authorization"),
            (["auth_method"], "authentication"),
            (["resource_id"], "not_found"),
            (["attribute_name"], "attribute_access"),
        ]
        for keys, error_type in error_patterns:
            if error_type == "authorization":
                if c.KEY_USER_ID in kwargs and "permission" in kwargs:
                    return error_type
            elif any(key in kwargs for key in keys):
                return error_type
        return None

    @staticmethod
    def _extract_common_kwargs(
        kwargs: Mapping[str, t.MetadataValue],
    ) -> tuple[str | None, m.Metadata | Mapping[str, t.MetadataOrValue] | None]:
        """Extract correlation_id and metadata from kwargs.

        Returns typed values: correlation_id as str | None, metadata as m.Metadata | Mapping | None.
        """
        correlation_id_raw = kwargs.get(c.KEY_CORRELATION_ID)
        correlation_id = (
            e._safe_optional_str(correlation_id_raw)
            if FlextUtilitiesGuardsTypeCore.is_scalar(correlation_id_raw)
            else None
        )
        metadata_raw = kwargs.get(c.FIELD_METADATA)
        metadata: m.Metadata | Mapping[str, t.MetadataOrValue | None] | None = None
        model_dump = getattr(metadata_raw, "model_dump", None)
        if callable(model_dump):
            metadata = e._safe_metadata(metadata_raw)
        if metadata is None:
            metadata = e._safe_config_map(metadata_raw)
        if metadata is None:
            attrs_raw = getattr(metadata_raw, c.FIELD_ATTRIBUTES, None)
            attrs_map = e._safe_config_map(attrs_raw)
            if attrs_map is not None:
                metadata = m.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: dict(attrs_map.items()),
                })
        return (correlation_id, metadata)

    @staticmethod
    def _merge_metadata_into_context(
        context: t.ConfigMap,
        metadata_obj: m.Metadata | Mapping[str, t.MetadataOrValue | None] | None,
    ) -> None:
        """Merge metadata t.NormalizedValue into context dictionary."""
        if metadata_obj is None:
            return
        metadata_model = e._safe_metadata(metadata_obj)
        if metadata_model is not None:
            for k, v in metadata_model.attributes.items():
                context[k] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
            return
        metadata_map = e._safe_config_map(metadata_obj)
        if metadata_map is not None:
            for k, metadata_value in metadata_map.items():
                context[k] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(metadata_value),
                )

    @staticmethod
    def _prepare_exception_kwargs(
        kwargs: Mapping[str, t.MetadataValue],
        specific_params: Mapping[str, t.MetadataValue | None] | None = None,
    ) -> tuple[
        str | None,
        t.MetadataValue | None,
        bool,
        bool,
        t.MetadataValue | None,
        Mapping[str, t.MetadataValue],
    ]:
        """Prepare exception kwargs by extracting common parameters."""
        merged_kwargs: MutableMapping[str, t.MetadataValue] = dict(kwargs)
        if specific_params:
            for k, v in specific_params.items():
                if v is None:
                    continue
                merged_kwargs[k] = v
        field_metadata = c.FIELD_METADATA
        field_auto_log = getattr(c, "FIELD_AUTO_LOG", "auto_log")
        field_auto_correlation = c.FIELD_AUTO_CORRELATION
        field_config = c.DIR_CONFIG
        extra_kwargs = {
            k: v
            for k, v in merged_kwargs.items()
            if k
            not in {
                c.KEY_CORRELATION_ID,
                field_metadata,
                field_auto_log,
                field_auto_correlation,
                field_config,
            }
        }
        correlation_id_raw = merged_kwargs.get(c.KEY_CORRELATION_ID)
        correlation_id = (
            e._safe_optional_str(correlation_id_raw)
            if FlextUtilitiesGuardsTypeCore.is_scalar(correlation_id_raw)
            else None
        )
        auto_log_raw = merged_kwargs.get(field_auto_log)
        auto_correlation_raw = merged_kwargs.get(field_auto_correlation)
        return (
            correlation_id,
            merged_kwargs.get(field_metadata),
            e._safe_bool(
                auto_log_raw
                if FlextUtilitiesGuardsTypeCore.is_scalar(auto_log_raw)
                else None,
                default=False,
            ),
            e._safe_bool(
                auto_correlation_raw
                if FlextUtilitiesGuardsTypeCore.is_scalar(auto_correlation_raw)
                else None,
                default=False,
            ),
            merged_kwargs.get(field_config),
            extra_kwargs,
        )

    @staticmethod
    def create(
        message: str,
        error_code: str | None = None,
        **kwargs: t.MetadataValue,
    ) -> e.BaseError:
        """Create an appropriate exception instance based on kwargs context."""
        legacy_type_map: t.StrMapping = {
            "ValidationError": "validation",
            "ConfigurationError": "configuration",
            "ConnectionError": "connection",
            "TimeoutError": "timeout",
            "AuthenticationError": "authentication",
            "AuthorizationError": "authorization",
            "NotFoundError": "not_found",
            "ConflictError": "conflict",
            "RateLimitError": "rate_limit",
            "CircuitBreakerError": "circuit_breaker",
            "TypeError": "type",
            "OperationError": "operation",
            "AttributeAccessError": "attribute_access",
            "AttributeError": "attribute_access",
        }
        explicit_error_type = legacy_type_map.get(message)
        resolved_message = message
        resolved_error_code = error_code
        if (
            explicit_error_type is None
            and error_code is not None
            and (not kwargs)
            and (message.endswith("Error") or "_" in message)
        ):
            msg = f"Unknown error type: {message}"
            raise ValueError(msg)
        if explicit_error_type is not None and error_code is not None:
            resolved_message = error_code
            resolved_error_code = None
        correlation_id_obj, metadata_obj = e._extract_common_kwargs(kwargs)
        error_type = explicit_error_type or e._determine_error_type(kwargs)
        correlation_id: str | None = correlation_id_obj
        error_context = e._build_error_context(correlation_id, metadata_obj, kwargs)
        return e._create_error_by_type(
            error_type,
            resolved_message,
            resolved_error_code,
            context=error_context or None,
        )

    @staticmethod
    def extract_common_kwargs(
        kwargs: Mapping[str, t.MetadataValue],
    ) -> tuple[str | None, m.Metadata | Mapping[str, t.MetadataOrValue] | None]:
        """Extract common correlation and metadata fields from kwargs."""
        return e._extract_common_kwargs(kwargs)

    @staticmethod
    def prepare_exception_kwargs(
        kwargs: Mapping[str, t.MetadataValue],
        specific_params: Mapping[str, t.MetadataValue | None] | None = None,
    ) -> tuple[
        str | None,
        t.MetadataValue | None,
        bool,
        bool,
        t.MetadataValue | None,
        Mapping[str, t.MetadataValue],
    ]:
        """Prepare normalized kwargs payload for exception construction."""
        return e._prepare_exception_kwargs(kwargs, specific_params)

    _exception_counts: ClassVar[MutableMapping[type, int]] = {}

    def __call__(
        self,
        message: str,
        error_code: str | None = None,
        **kwargs: t.MetadataValue,
    ) -> e.BaseError:
        """Create exception by calling the class instance."""
        normalized_kwargs: t.ConfigMap = t.ConfigMap(root={})
        for k, v in kwargs.items():
            normalized_kwargs[k] = FlextRuntime.normalize_to_container(
                FlextRuntime.normalize_to_metadata(v),
            )
        return self.create(
            message,
            error_code,
            **{
                k: FlextRuntime.normalize_to_metadata(v)
                for k, v in normalized_kwargs.items()
            },
        )

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._exception_counts.clear()

    @classmethod
    def get_metrics(cls) -> t.ConfigMap:
        """Get exception metrics and statistics."""
        total = sum(cls._exception_counts.values(), 0)
        exception_counts_list = [
            f"{(exc_type.__qualname__ if hasattr(exc_type, '__qualname__') else str(exc_type))}:{count}"
            for exc_type, count in cls._exception_counts.items()
        ]
        exception_counts_str = ";".join(exception_counts_list)
        exception_counts_dict: MutableMapping[str, int] = {}
        for exc_type, count in cls._exception_counts.items():
            exc_name = (
                exc_type.__qualname__
                if hasattr(exc_type, "__qualname__")
                else str(exc_type)
            )
            exception_counts_dict[exc_name] = count
        exception_counts_payload = t.ConfigMap.model_validate(exception_counts_dict)
        result_dict: t.ConfigMap = t.ConfigMap(
            root={
                "total_exceptions": total,
                "exception_counts": exception_counts_payload,
                "exception_counts_summary": exception_counts_str,
                "unique_exception_types": len(cls._exception_counts),
            },
        )
        return result_dict

    @classmethod
    def record_exception(cls, exception_type: type) -> None:
        """Record an exception occurrence for metrics tracking."""
        if exception_type not in cls._exception_counts:
            cls._exception_counts[exception_type] = 0
        cls._exception_counts[exception_type] += 1


e = FlextExceptions
__all__ = ["FlextExceptions", "e"]
