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
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from typing import ClassVar, Self

import structlog

from flext_core._models.metadata import Metadata
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

# Type alias for GeneralValueType (PEP 695)
type GeneralValueType = FlextTypes.GeneralValueType


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
            cls, level: FlextConstants.Exceptions.FailureLevel
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
            cls, library_name: str, exception_type: type, level: str
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
                    cls._container_exception_levels[container_id]
                )
            if (
                library_name
                and exception_type
                and library_name in cls._library_exception_levels
                and exception_type in cls._library_exception_levels[library_name]
            ):
                return FlextConstants.Exceptions.FailureLevel(
                    cls._library_exception_levels[library_name][exception_type]
                )
            return cls._get_global_failure_level()

    class BaseError(Exception):
        """Base exception class with structured logging."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.UNKNOWN_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **extra_kwargs: GeneralValueType,
        ) -> None:
            """Initialize base exception with structured metadata and correlation.

            Tracks correlation IDs and structured metadata for distributed tracing.

            Args:
                message: Human-readable error message
                error_code: Error code from FlextConstants.Errors
                correlation_id: Correlation ID for distributed tracing
                metadata: Additional metadata for error context
                auto_log: Whether to automatically log the exception
                auto_correlation: Whether to auto-generate correlation ID
                **extra_kwargs: Additional metadata attributes

            """
            super().__init__(message)
            self.message = message
            self.error_code = error_code
            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if auto_correlation and not correlation_id
                else correlation_id
            )
            self.metadata = (
                Metadata(
                    attributes={
                        k: v
                        for k, v in (extra_kwargs or {}).items()
                        if isinstance(
                            v, (str, int, float, bool, list, dict, type(None))
                        )
                    }
                )
                if metadata is None
                else metadata
            )
            if FlextRuntime.is_dict_like(metadata):
                merged_attrs = {
                    k: v
                    for k, v in (
                        {**metadata, **extra_kwargs} if extra_kwargs else metadata
                    ).items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                self.metadata = Metadata(attributes=merged_attrs)
            elif extra_kwargs:
                existing_attrs = self.metadata.attributes
                new_attrs = {
                    k: v
                    for k, v in {**existing_attrs, **extra_kwargs}.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                self.metadata = Metadata(attributes=new_attrs)
            self.timestamp = time.time()
            self.auto_log = auto_log
            if auto_log:
                self._log_exception()

        def _log_exception(self) -> None:
            """Log the exception using structured logging."""
            try:
                logger = structlog.get_logger()
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
                logging.getLogger(__name__).debug(
                    "Logging failed in exception handler: %s", e
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

        def to_dict(self) -> FlextTypes.Types.ErrorTypeMapping:
            """Convert exception to dictionary representation."""
            return {
                "error_type": self.__class__.__name__,
                "message": self.message,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
                "metadata": self.metadata.attributes,
            }

        def with_context(self, **context: GeneralValueType) -> Self:
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
                logger = structlog.get_logger()
                logger.warning(
                    "exception_chained",
                    event_type="exception_chain",
                    child_error=self.__class__.__name__,
                    parent_error=cause.__class__.__name__,
                    correlation_id=self.correlation_id,
                )
            except Exception as e:
                logging.getLogger(__name__).debug(
                    "Logging failed in exception handler: %s", e
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
            value: GeneralValueType | None = None,
            error_code: str = FlextConstants.Errors.VALIDATION_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize validation error with field and value information."""
            kwargs.update({"field": field, "value": value})
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
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
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize configuration error with config context."""
            kwargs.update({"config_key": config_key, "config_source": config_source})
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.config_key = config_key
            self.config_source = config_source

    class ConnectionError(BaseError):
        """Exception raised for network and connection failures."""

        def __init__(
            self,
            message: str,
            *,
            host: str | None = None,
            port: int | None = None,
            timeout: float | None = None,
            error_code: str = FlextConstants.Errors.CONNECTION_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize connection error with network context."""
            kwargs.update({"host": host, "port": port, "timeout": timeout})
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.host = host
            self.port = port
            self.timeout = timeout

    class TimeoutError(BaseError):
        """Exception raised for operation timeout errors."""

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            operation: str | None = None,
            error_code: str = FlextConstants.Errors.TIMEOUT_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize timeout error with timeout context."""
            kwargs.update({"timeout_seconds": timeout_seconds, "operation": operation})
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
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
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize authentication error with auth context."""
            kwargs.update({"auth_method": auth_method, "user_id": user_id})
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.auth_method = auth_method
            self.user_id = user_id

    class AuthorizationError(BaseError):
        """Exception raised for permission and authorization failures."""

        def __init__(
            self,
            message: str,
            *,
            user_id: str | None = None,
            resource: str | None = None,
            permission: str | None = None,
            error_code: str = FlextConstants.Errors.AUTHORIZATION_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize authorization error with permission context."""
            kwargs.update({
                "user_id": user_id,
                "resource": resource,
                "permission": permission,
            })
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.user_id = user_id
            self.resource = resource
            self.permission = permission

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            error_code: str = FlextConstants.Errors.NOT_FOUND_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize not found error with resource context."""
            kwargs.update({"resource_type": resource_type, "resource_id": resource_id})
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.resource_type = resource_type
            self.resource_id = resource_id

    class ConflictError(BaseError):
        """Exception raised for resource conflicts."""

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            conflict_reason: str | None = None,
            error_code: str = FlextConstants.Errors.ALREADY_EXISTS,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize conflict error with resource context."""
            kwargs.update({
                "resource_type": resource_type,
                "resource_id": resource_id,
                "conflict_reason": conflict_reason,
            })
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.resource_type = resource_type
            self.resource_id = resource_id
            self.conflict_reason = conflict_reason

    class RateLimitError(BaseError):
        """Exception raised when rate limits are exceeded."""

        def __init__(
            self,
            message: str,
            *,
            limit: int | None = None,
            window_seconds: int | None = None,
            retry_after: float | None = None,
            error_code: str = FlextConstants.Errors.OPERATION_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize rate limit error with limit context."""
            kwargs.update({
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after,
            })
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.limit = limit
            self.window_seconds = window_seconds
            self.retry_after = retry_after

    class CircuitBreakerError(BaseError):
        """Exception raised when circuit breaker is open."""

        def __init__(
            self,
            message: str,
            *,
            service_name: str | None = None,
            failure_count: int | None = None,
            reset_timeout: float | None = None,
            error_code: str = FlextConstants.Errors.EXTERNAL_SERVICE_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize circuit breaker error with service context."""
            kwargs.update({
                "service_name": service_name,
                "failure_count": failure_count,
                "reset_timeout": reset_timeout,
            })
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.service_name = service_name
            self.failure_count = failure_count
            self.reset_timeout = reset_timeout

    class TypeError(BaseError):
        """Exception raised for type mismatch errors."""

        def __init__(
            self,
            message: str,
            *,
            expected_type: type | None = None,
            actual_type: type | None = None,
            error_code: str = FlextConstants.Errors.TYPE_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize type error with type information."""
            # Convert type objects to their qualified names (GeneralValueType compatible)
            type_kwargs: dict[str, GeneralValueType] = {
                "expected_type": expected_type.__qualname__ if expected_type else None,
                "actual_type": actual_type.__qualname__ if actual_type else None,
            }
            kwargs_updated = {**kwargs, **type_kwargs}
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs_updated,
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
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize operation error with operation context."""
            kwargs.update({"operation": operation, "reason": reason})
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
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
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: GeneralValueType,
        ) -> None:
            """Initialize attribute access error with attribute context."""
            kwargs.update({
                "attribute_name": attribute_name,
                "attribute_context": attribute_context,
            })
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
            )
            self.attribute_name = attribute_name
            self.attribute_context = attribute_context

    @staticmethod
    def prepare_exception_kwargs(
        kwargs: dict[str, GeneralValueType],
        specific_params: dict[str, GeneralValueType] | None = None,
    ) -> tuple[
        str | None,
        GeneralValueType,
        bool,
        bool,
        GeneralValueType,
        dict[str, GeneralValueType],
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
        kwargs: Mapping[str, GeneralValueType],
    ) -> tuple[GeneralValueType | None, GeneralValueType | None]:
        """Extract correlation_id and metadata from kwargs.

        Returns raw values without narrowing types, as they could be any GeneralValueType.
        Callers should validate types as needed.
        """
        return (kwargs.get("correlation_id"), kwargs.get("metadata"))

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
        kwargs: Mapping[str, GeneralValueType],
    ) -> str | None:
        if "field" in kwargs or "value" in kwargs:
            return "validation"
        if "config_key" in kwargs or "config_source" in kwargs:
            return "configuration"
        if "operation" in kwargs:
            return "operation"
        if "host" in kwargs or "port" in kwargs:
            return "connection"
        if "timeout_seconds" in kwargs:
            return "timeout"
        if "user_id" in kwargs and "permission" in kwargs:
            return "authorization"
        if "auth_method" in kwargs:
            return "authentication"
        if "resource_id" in kwargs:
            return "not_found"
        if "attribute_name" in kwargs:
            return "attribute_access"
        return None

    @staticmethod
    def _get_error_creator(
        error_type: str,
    ) -> (
        Callable[
            [
                str,
                str | None,
                Mapping[str, GeneralValueType],
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
        kwargs: Mapping[str, GeneralValueType],
        correlation_id: str | None,
        metadata: dict[str, GeneralValueType] | None,
    ) -> BaseError:
        creator = FlextExceptions._get_error_creator(error_type) if error_type else None
        normalized_metadata = None
        if metadata:
            # Filter to metadata-compatible types before passing to Metadata constructor
            filtered_attrs = {
                k: v
                for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            normalized_metadata = Metadata(attributes=filtered_attrs)
        if creator:
            # creator expects object as last parameter
            # normalized_metadata is Metadata | None which is compatible
            return creator(
                message,
                error_code,
                kwargs,
                correlation_id,
                normalized_metadata,
            )
        return FlextExceptions.BaseError(
            message,
            error_code=error_code or "UNKNOWN",
            metadata=normalized_metadata,
            correlation_id=correlation_id,
        )

    @staticmethod
    def create(
        message: str,
        error_code: str | None = None,
        **kwargs: GeneralValueType,
    ) -> BaseError:
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
        metadata_dict: dict[str, GeneralValueType] | None = None
        if isinstance(metadata_obj, dict):
            metadata_dict = metadata_obj
        # Note: if metadata_obj is dict-like but not a dict (Mapping interface),
        # we don't convert it since Metadata expects concrete dict with GeneralValueType values
        return FlextExceptions._create_error_by_type(
            error_type, message, error_code, kwargs, correlation_id, metadata_dict
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
            f"{exc_type.__qualname__}:{count}"
            for exc_type, count in cls._exception_counts.items()
        ]
        exception_counts_str = ";".join(exception_counts_list)
        return {
            "total_exceptions": total,
            "exception_counts_summary": exception_counts_str,
            "unique_exception_types": len(cls._exception_counts),
        }

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._exception_counts.clear()

    def __call__(
        self,
        message: str,
        error_code: str | None = None,
        **kwargs: FlextTypes.Types.ExceptionKwargsType,
    ) -> BaseError:
        """Create exception by calling the class instance."""
        return self.create(message, error_code, **kwargs)


__all__ = ["FlextExceptions"]
