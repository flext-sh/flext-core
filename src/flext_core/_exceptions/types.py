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
    FlextModelsExceptionParams as m,
    FlextModelsPydantic as mp,
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

    class TypeError(FlextExceptionsBase.BaseError):
        """Exception raised for type mismatch errors."""

        expected_type: type | None = None
        actual_type: type | None = None

        _TYPE_MAP: ClassVar[dict[str, type]] = {
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

        def __init__(
            self,
            message: str,
            *,
            expected_type: type | str | None = None,
            actual_type: type | str | None = None,
            error_code: str = c.ErrorCode.TYPE_ERROR,
            context: t.MappingKV[str, t.JsonValue] | None = None,
            correlation_id: str | None = None,
        ) -> None:
            """Initialize type error with type information."""
            cls = FlextExceptionsTypes.TypeError
            super().__init__(
                message,
                error_code=error_code,
                expected_type=cls._to_type_name(expected_type),
                actual_type=cls._to_type_name(actual_type),
                context=context,
                correlation_id=correlation_id,
            )
            self.expected_type = cls._from_type_name(expected_type)
            self.actual_type = cls._from_type_name(actual_type)

        @staticmethod
        def _to_type_name(v: type | str | None) -> str | None:
            """Convert type object or string to canonical qualified name."""
            return v.__qualname__ if isinstance(v, type) else v

        @staticmethod
        def _from_type_name(v: type | str | None) -> type | None:
            """Resolve type name string or type object to actual type."""
            if isinstance(v, type):
                return v
            return (
                FlextExceptionsTypes.TypeError._TYPE_MAP.get(v)
                if isinstance(v, str)
                else None
            )

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
