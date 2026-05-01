"""Typed exception subclasses — all named exception types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import ValidationError as _PydanticValidationError

from flext_core import (
    FlextConstants as c,
    FlextExceptionsBase,
    FlextExceptionsHelpers,
    FlextModelsExceptionParams as m,
    FlextModelsPydantic as mp,
    FlextRuntime,
    FlextTypes as t,
)


class FlextExceptionsTypes(FlextExceptionsBase):
    """All typed FLEXT exception subclasses."""

    PydanticValidationError: type[_PydanticValidationError] = _PydanticValidationError

    class ValidationError(FlextExceptionsBase.BaseError):
        """Exception raised for input validation failures."""

        field: str | None = None
        value: t.Scalar | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.VALIDATION_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.ValidationErrorParams
        )

    class ConfigurationError(FlextExceptionsBase.BaseError):
        """Exception raised for configuration-related errors."""

        config_key: str | None = None
        config_source: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.CONFIGURATION_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.ConfigurationErrorParams
        )

    class ConnectionError(FlextExceptionsBase.BaseError):
        """Exception raised for network and connection failures."""

        host: str | None = None
        port: int | None = None
        timeout: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.CONNECTION_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.ConnectionErrorParams
        )

    class TimeoutError(FlextExceptionsBase.BaseError):
        """Exception raised for operation timeout errors."""

        timeout_seconds: t.Numeric | None = None
        operation: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.TIMEOUT_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = m.TimeoutErrorParams

    class AuthenticationError(FlextExceptionsBase.BaseError):
        """Exception raised for authentication failures."""

        auth_method: str | None = None
        user_id: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.AUTHENTICATION_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.AuthenticationErrorParams
        )

    class AuthorizationError(FlextExceptionsBase.BaseError):
        """Exception raised for permission and authorization failures."""

        user_id: str | None = None
        resource: str | None = None
        permission: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.AUTHORIZATION_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.AuthorizationErrorParams
        )

    class NotFoundError(FlextExceptionsBase.BaseError):
        """Exception raised when a resource is not found."""

        resource_type: str | None = None
        resource_id: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.NOT_FOUND_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = m.NotFoundErrorParams
        _excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = frozenset({
            c.ContextKey.CORRELATION_ID,
            c.FIELD_METADATA,
        })

    class ConflictError(FlextExceptionsBase.BaseError):
        """Exception raised for resource conflicts."""

        resource_type: str | None = None
        resource_id: str | None = None
        conflict_reason: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.ALREADY_EXISTS
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = m.ConflictErrorParams

    class RateLimitError(FlextExceptionsBase.BaseError):
        """Exception raised when rate limits are exceeded."""

        limit: int | None = None
        window_seconds: int | None = None
        retry_after: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.OPERATION_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.RateLimitErrorParams
        )

    class CircuitBreakerError(FlextExceptionsBase.BaseError):
        """Exception raised when circuit breaker is open."""

        service_name: str | None = None
        failure_count: int | None = None
        reset_timeout: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.EXTERNAL_SERVICE_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.CircuitBreakerErrorParams
        )

        def __init__(
            self,
            message: str,
            *,
            service_name: str | None = None,
            service: str | None = None,
            failure_count: int | None = None,
            reset_timeout: t.Numeric | None = None,
            error_code: str = c.ErrorCode.EXTERNAL_SERVICE_ERROR,
            context: t.MappingKV[str, t.JsonValue] | None = None,
            correlation_id: str | None = None,
            params: mp.BaseModel | None = None,
            **extra_kwargs: t.JsonValue,
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
                    c.ContextKey.SERVICE_NAME: resolved_service_name,
                    "failure_count": failure_count,
                    "reset_timeout": reset_timeout,
                },
                correlation_id=correlation_id,
                extra_kwargs=extra_kwargs,
            )

    class TypeError(FlextExceptionsBase.BaseError):
        """Exception raised for type mismatch errors."""

        expected_type: type | None = None
        actual_type: type | None = None

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.ErrorCode.TYPE_ERROR,
            expected_type: type | str | None = None,
            actual_type: type | str | None = None,
            context: t.MappingKV[str, t.JsonValue] | None = None,
            correlation_id: str | None = None,
            params: mp.BaseModel | None = None,
            **extra_kwargs: t.JsonValue,
        ) -> None:
            """Initialize type error with type information."""
            preserved_metadata = extra_kwargs.pop(c.FIELD_METADATA, None)
            normalized_metadata = (
                FlextRuntime.normalize_to_metadata(preserved_metadata)
                if preserved_metadata is not None
                else None
            )
            preserved_corr_id = extra_kwargs.pop(c.ContextKey.CORRELATION_ID, None)
            type_map = FlextExceptionsTypes.TypeError._get_type_map()
            normalized_expected_type = FlextExceptionsTypes.TypeError._normalize_type(
                expected_type,
                type_map,
                extra_kwargs,
                "expected_type",
            )
            normalized_actual_type = FlextExceptionsTypes.TypeError._normalize_type(
                actual_type,
                type_map,
                extra_kwargs,
                "actual_type",
            )
            param_values: t.MappingKV[str, str | None] = {
                "expected_type": normalized_expected_type.__qualname__
                if normalized_expected_type is not None
                else None,
                "actual_type": normalized_actual_type.__qualname__
                if normalized_actual_type is not None
                else None,
            }
            resolved_params = (
                m.TypeErrorParams.model_validate(params.model_dump())
                if params is not None
                else m.TypeErrorParams.model_validate(param_values)
            )
            normalized_extra_kwargs: t.MappingKV[str, t.JsonValue] = {
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in extra_kwargs.items()
            }
            type_context = FlextExceptionsTypes.TypeError._build_type_context(
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
                else FlextExceptionsHelpers.safe_optional_str(preserved_corr_id),
            )
            self.expected_type = normalized_expected_type
            self.actual_type = normalized_actual_type

        @staticmethod
        def _build_type_context(
            expected_type: type | str | None,
            actual_type: type | str | None,
            context: t.MappingKV[str, t.JsonValue] | None,
            extra_kwargs: t.MappingKV[str, t.JsonValue],
        ) -> t.MappingKV[str, t.JsonValue]:
            """Build type context dictionary."""
            type_context = FlextExceptionsHelpers.build_context_map(
                context,
                extra_kwargs,
            )
            resolved_expected_type = FlextExceptionsTypes.TypeError._resolve_type_name(
                expected_type,
            )
            resolved_actual_type = FlextExceptionsTypes.TypeError._resolve_type_name(
                actual_type,
            )
            if resolved_expected_type is not None:
                type_context["expected_type"] = resolved_expected_type
            if resolved_actual_type is not None:
                type_context["actual_type"] = resolved_actual_type
            return type_context

        @staticmethod
        def _get_type_map() -> t.MappingKV[str, type]:
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
            type_map: t.MappingKV[str, type],
            extra_kwargs: t.MutableJsonMapping,
            key: str,
        ) -> type | None:
            """Normalize type value from various sources."""
            source_value: type | str | t.JsonValue | None = type_value
            if source_value is None and key in extra_kwargs:
                source_value = extra_kwargs.pop(key)
            type_name = FlextExceptionsTypes.TypeError._resolve_type_name(source_value)
            if type_name is None:
                return None
            return type_map.get(type_name)

        @staticmethod
        def _resolve_type_name(
            type_value: type | str | t.JsonValue | None,
        ) -> str | None:
            """Resolve type-like input to canonical string name."""
            if type_value is None:
                return None
            string_value = FlextExceptionsHelpers.safe_optional_str(type_value)
            if string_value is not None:
                return string_value
            qualname_value = (
                type_value.__qualname__ if hasattr(type_value, "__qualname__") else None
            )
            return FlextExceptionsHelpers.safe_optional_str(qualname_value)

    class OperationError(FlextExceptionsBase.BaseError):
        """Exception raised for general operation failures."""

        operation: str | None
        reason: str | None
        _default_error_code: ClassVar[str] = c.ErrorCode.OPERATION_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.OperationErrorParams
        )

    class AttributeAccessError(FlextExceptionsBase.BaseError):
        """Exception raised for attribute access errors."""

        attribute_name: str | None
        attribute_context: t.JsonValue | None
        _default_error_code: ClassVar[str] = c.ErrorCode.ATTRIBUTE_ERROR
        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = (
            m.AttributeAccessErrorParams
        )


__all__: t.SequenceOf[str] = ["FlextExceptionsTypes"]
