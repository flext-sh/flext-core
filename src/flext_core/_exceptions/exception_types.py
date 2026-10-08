"""Typed exception subclasses — all named exception types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar, override

from pydantic import ValidationError as _PydanticValidationError

from flext_core import c, m, t
from flext_core._exceptions.base import FlextExceptionsBase


class FlextExceptionsTypes(FlextExceptionsBase):
    """All typed FLEXT exception subclasses."""

    PydanticValidationError: type[_PydanticValidationError] = _PydanticValidationError

    class ValidationError(FlextExceptionsBase.BaseError):
        """Exception raised for input validation failures."""

        field: str | None = None
        value: t.Scalar | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.VALIDATION_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = m.ValidationErrorParams

    class ConfigurationError(FlextExceptionsBase.BaseError):
        """Exception raised for configuration-related errors."""

        config_key: str | None = None
        config_source: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.CONFIGURATION_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = (
            m.ConfigurationErrorParams
        )

    class FlextConnectionError(FlextExceptionsBase.BaseError):
        """Exception raised for network and connection failures."""

        host: str | None = None
        port: int | None = None
        timeout: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.CONNECTION_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = m.ConnectionErrorParams

    class FlextTimeoutError(FlextExceptionsBase.BaseError):
        """Exception raised for operation timeout errors."""

        timeout_seconds: t.Numeric | None = None
        operation: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.TIMEOUT_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = m.TimeoutErrorParams

    class AuthenticationError(FlextExceptionsBase.BaseError):
        """Exception raised for authentication failures."""

        auth_method: str | None = None
        user_id: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.AUTHENTICATION_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = (
            m.AuthenticationErrorParams
        )

    class AuthorizationError(FlextExceptionsBase.BaseError):
        """Exception raised for permission and authorization failures."""

        user_id: str | None = None
        resource: str | None = None
        permission: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.AUTHORIZATION_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = (
            m.AuthorizationErrorParams
        )

    class NotFoundError(FlextExceptionsBase.BaseError):
        """Exception raised when a resource is not found."""

        resource_type: str | None = None
        resource_id: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.NOT_FOUND_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = m.NotFoundErrorParams
        excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = frozenset({
            c.ContextKey.CORRELATION_ID,
            c.FIELD_METADATA,
        })

    class ConflictError(FlextExceptionsBase.BaseError):
        """Exception raised for resource conflicts."""

        resource_type: str | None = None
        resource_id: str | None = None
        conflict_reason: str | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.ALREADY_EXISTS
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = m.ConflictErrorParams

    class RateLimitError(FlextExceptionsBase.BaseError):
        """Exception raised when rate limits are exceeded."""

        limit: int | None = None
        window_seconds: int | None = None
        retry_after: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.OPERATION_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = m.RateLimitErrorParams

    class CircuitBreakerError(FlextExceptionsBase.BaseError):
        """Exception raised when circuit breaker is open."""

        service_name: str | None = None
        failure_count: int | None = None
        reset_timeout: t.Numeric | None = None
        _default_error_code: ClassVar[str] = c.ErrorCode.EXTERNAL_SERVICE_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = (
            m.CircuitBreakerErrorParams
        )

    class FlextTypeError(FlextExceptionsBase.BaseError):
        """Exception raised for type mismatch errors."""

        expected_type: type | None = None
        actual_type: type | None = None

        TYPE_MAP: ClassVar[dict[str, type]] = {
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

        @override
        def __init__(
            self,
            message: str,
            *format_args: t.JsonPayload,
            expected_type: type | str | None = None,
            actual_type: type | str | None = None,
            options: m.ExceptionInitOptions | None = None,
            params: m.BaseModel | None = None,
            **extra_kwargs: t.JsonValue,
        ) -> None:
            """Initialize type error with type information."""
            # Why: the public keyword contract defaults this error family to
            # ``TYPE_ERROR`` (a type mismatch IS a validation-domain defect);
            # an explicit ``error_code`` keyword or ``options`` object wins.
            if options is None:
                extra_kwargs.setdefault("error_code", c.ErrorCode.TYPE_ERROR)
            cls = FlextExceptionsTypes.FlextTypeError
            super().__init__(
                message,
                *format_args,
                options=options,
                params=params,
                expected_type=cls._to_type_name(expected_type),
                actual_type=cls._to_type_name(actual_type),
                **extra_kwargs,
            )
            self.expected_type = cls._from_type_name(expected_type)
            self.actual_type = cls._from_type_name(actual_type)

        @staticmethod
        def _to_type_name(v: type | str | None) -> str | None:
            """Convert type object or string to canonical qualified name.

            Returns:
                The resulting ``str | None``.

            """
            return v.__qualname__ if isinstance(v, type) else v

        @staticmethod
        def _from_type_name(v: type | str | None) -> type | None:
            """Resolve type name string or type object to actual type.

            Returns:
                The resulting ``type | None``.

            """
            if isinstance(v, type):
                return v
            return (
                FlextExceptionsTypes.FlextTypeError.TYPE_MAP.get(v)
                if isinstance(v, str)
                else None
            )

    class OperationError(FlextExceptionsBase.BaseError):
        """Exception raised for general operation failures."""

        operation: str | None
        reason: str | None
        _default_error_code: ClassVar[str] = c.ErrorCode.OPERATION_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = m.OperationErrorParams

    class AttributeAccessError(FlextExceptionsBase.BaseError):
        """Exception raised for attribute access errors."""

        attribute_name: str | None
        attribute_context: t.JsonValue | None
        _default_error_code: ClassVar[str] = c.ErrorCode.ATTRIBUTE_ERROR
        params_cls: ClassVar[t.ModelClass[m.BaseModel] | None] = (
            m.AttributeAccessErrorParams
        )


__all__: t.SequenceOf[str] = ["FlextExceptionsTypes"]
