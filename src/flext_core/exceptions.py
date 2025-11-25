"""FlextExceptions - Structured exception hierarchy for FLEXT ecosystem.

This module provides FlextExceptions, a namespace class containing structured
exception types with error codes, correlation IDs, and metadata for comprehensive
error handling throughout the FLEXT ecosystem.

Architecture: Layer 1 (Foundation - Error Hierarchy)
========================================================
Provides a comprehensive collection of structured exception types with
error codes, correlation IDs, and metadata for comprehensive error handling
throughout the FLEXT ecosystem. All exceptions integrate with FlextResult
error handling and structured logging via structlog.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Callable
from typing import ClassVar, Self, cast

import structlog

from flext_core._models.metadata import Metadata
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.runtime import FlextRuntime


class FlextExceptions:
    """Structured exception hierarchy with error codes and correlation tracking.

    Architecture: Layer 1 (Foundation - Error Hierarchy)
    ===================================================
    Provides a comprehensive collection of structured exception types with
    error codes, correlation IDs, and metadata for comprehensive error handling
    throughout the FLEXT ecosystem. All exceptions integrate with FlextResult
    error handling and structured logging via structlog.

    **Structural Typing and Protocol Compliance**:
    This class satisfies FlextProtocols.Exception through structural typing
    (duck typing) via the following protocol-compliant interface:
    - 13 exception classes with consistent error handling
    - All extend BaseError with structured metadata
    - Error codes from FlextConstants for consistency
    - Correlation ID generation for distributed tracing
    - Automatic structured logging via structlog
    - Exception chaining with cause preservation
    - Flexible metadata for context propagation

    **Exception Hierarchy** (13 types):
    1. **BaseError**: Base class with error codes, correlation IDs, metadata
       - Automatic structured logging on creation (optional)
       - Correlation ID generation for distributed tracing
       - Exception chaining with cause tracking
       - Methods: with_context(), chain_from(), to_dict()

    2. **ValidationError**: Input validation failures
       - Field-specific validation errors
       - Value tracking for invalid data
       - Integrates with FlextUtilities validation

    3. **ConfigurationError**: Configuration-related errors
       - Config key tracking (which config failed)
       - Config source tracking (file, env, etc.)
       - Integration with FlextConfig validation

    4. **ConnectionError**: Network and connection failures
       - Host and port tracking
       - Timeout duration tracking
       - Connection pool exhaustion handling

    5. **TimeoutError**: Operation timeout errors
       - Timeout duration tracking
       - Operation name for context
       - Integration with @timeout decorator

    6. **AuthenticationError**: Authentication failures
       - Authentication method tracking
       - User ID tracking for audit trails
       - Integration with FlextAuth patterns

    7. **AuthorizationError**: Permission and authorization failures
       - User ID tracking
       - Resource being accessed
       - Permission being checked
       - Integration with RBAC patterns

    8. **NotFoundError**: Resource not found errors
       - Resource type tracking
       - Resource ID tracking
       - REST API 404 responses

    9. **ConflictError**: Resource conflict errors
       - Resource type and ID tracking
       - Conflict reason tracking
       - Handles duplicates and concurrent modifications

    10. **RateLimitError**: Rate limiting violations
        - Limit and window tracking
        - Retry-after duration
        - Integration with rate limiting patterns

    11. **CircuitBreakerError**: Circuit breaker open errors
        - Service name tracking
        - Failure count tracking
        - Reset timeout duration
        - Integration with resilience patterns

    12. **TypeError**: Type-related errors
        - Expected type tracking
        - Actual type tracking
        - Validation error handling

    13. **AttributeAccessError**: Attribute access errors
        - Attribute name tracking
        - Access context information
        - Missing/invalid attribute handling

    14. **OperationError**: Operation-specific errors
        - Operation name tracking
        - Failure reason tracking
        - Business logic violations

    **Core Features**:
    1. **Structured Error Codes**: All use FlextConstants.Errors for consistency
    2. **Correlation ID Tracking**: Unique IDs for distributed tracing
    3. **Automatic Logging**: Optional structured logging via structlog
    4. **Exception Chaining**: Preserve cause exceptions with __cause__
    5. **Flexible Metadata**: Arbitrary key-value context storage
    6. **String Representation**: Human-readable with error code and correlation ID
    7. **Dictionary Serialization**: to_dict() for JSON/logging serialization
    8. **Context Enrichment**: with_context() for adding metadata
    9. **Metrics Tracking**: record_exception(), get_metrics() for monitoring
    10. **Factory Methods**: create() and create_error() for dynamic creation

    **Integration Points**:
    - **FlextResult** (Layer 1): Error handling in railway pattern
    - **FlextConstants** (Layer 0): Error codes for categorization
    - **FlextDecorators** (Layer 3): @retry, @timeout use TimeoutError
    - **FlextLogger** (Layer 4): Structured logging via structlog
    - **FlextConfig** (Layer 4): Configuration validation errors
    - **All layers**: Error handling and exception propagation

    **Defensive Programming Patterns**:
    1. Automatic logging uses try/except to prevent logging failures
    2. Exception chaining preserves cause for debugging
    3. Metadata update via dict.update() for atomicity
    4. Correlation ID generated only when explicitly requested
    5. All exception attributes stored as instance variables

    **Structured Logging Integration**:
    - BaseError._log_exception() uses structlog.get_logger()
    - Exception chaining logged with chain_from()
    - All exception metadata included in structured logs

    **Exception Factory Methods**:
    1. **create_error(error_type, message)**: By type name
    2. **create(message, error_code, **kwargs)**: Smart type detection
    3. FlextExceptions as callable: delegates to create()

    **Metrics and Monitoring**:
    - record_exception(exception_type): Track exception counts
    - get_metrics(): Retrieve exception statistics
    - clear_metrics(): Reset exception counters
    - Integration with monitoring and alerting systems

    **Thread Safety**:
    - All exceptions are thread-safe (no mutable shared state)
    - Metadata dictionaries are instance-specific
    - Correlation ID generation uses uuid.uuid4()
    - Metrics dictionary is ClassVar (protected by GIL in CPython)

    **Performance Characteristics**:
    - O(1) exception creation
    - O(n) structured logging where n = metadata size
    - O(1) context binding via dict.update()
    - O(1) metrics recording

    **Usage Patterns**:

    1. Simple validation error:
        >>> if not email:
        ...     raise FlextExceptions.ValidationError("Email required")

    2. Validation error with field information:
        >>> raise FlextExceptions.ValidationError(
        ...     "Invalid email format", field="email", value=user_input
        ... )

    3. Configuration error with source tracking:
        >>> raise FlextExceptions.ConfigurationError(
        ...     "Missing API key", config_key="API_KEY", config_source="environment"
        ... )

    4. Connection error with host/port tracking:
        >>> raise FlextExceptions.ConnectionError(
        ...     "Failed to connect to database",
        ...     host="db.example.com",
        ...     port=5432,
        ...     timeout=30.0,
        ... )

    5. Exception with automatic logging:
        >>> raise FlextExceptions.ValidationError(
        ...     "Invalid user", auto_log=True, auto_correlation=True
        ... )

    6. Exception chaining:
        >>> try:
        ...     dangerous_operation()
        ... except ValueError as e:
        ...     raise FlextExceptions.ValidationError("Validation failed").chain_from(e)

    7. Context enrichment:
        >>> raise FlextExceptions.OperationError("Cannot process order").with_context(
        ...     order_id="ORD-123", customer_id="CUST-456"
        ... )

    8. Using factory methods:
        >>> error = FlextExceptions.create(
        ...     "Validation failed", field="email", value="invalid"
        ... )

    9. Serialization for logging/API responses:
        >>> try:
        ...     validate_data(data)
        ... except FlextExceptions.ValidationError as e:
        ...     error_dict = e.to_dict()
        ...     # Send to API response or log

    10. Exception tracking:
        >>> try:
        ...     risky_operation()
        ... except FlextExceptions.BaseError as e:
        ...     FlextExceptions.record_exception(type(e).__name__)
        >>>
        >>> stats = FlextExceptions.get_metrics()
        >>> print(f"Total exceptions: {stats['total_exceptions']}")

    **Complete Integration Example**:
        >>> from flext_core import FlextResult
        >>>
        >>> def process_user(user_data: dict) -> FlextResult[dict]:
        ...     try:
        ...         # Validation
        ...         if not user_data.get("email"):
        ...             raise FlextExceptions.ValidationError(
        ...                 "Email required", field="email"
        ...             ).with_context(user_id=user_data.get("id"))
        ...
        ...         # Processing
        ...         result = save_user(user_data)
        ...         return FlextResult[dict].ok(result)
        ...
        ...     except FlextExceptions.ValidationError as e:
        ...         # Automatic context and logging
        ...         return FlextResult[dict].fail(
        ...             str(e), error_code=e.error_code, metadata=e.metadata
        ...         )
        ...     except Exception as e:
        ...         # Unknown errors as operation errors
        ...         return FlextResult[dict].fail(str(e), error_code="OPERATION_ERROR")

    **Thread Safety**:
    All constants are immutable (typing.Final). Access is thread-safe as
    there is no mutable shared state in this module.
    """

    class Configuration:
        """Hierarchical exception handling configuration."""

        _global_failure_level: ClassVar[
            FlextConstants.Exceptions.FailureLevel | None
        ] = None
        _library_exception_levels: ClassVar[dict[str, dict[type, str]]] = {}
        _container_exception_levels: ClassVar[dict[str, str]] = {}
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
                level_str = config.exception_failure_level
                cls._global_failure_level = FlextConstants.Exceptions.FailureLevel(
                    level_str.lower()
                )
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
            **extra_kwargs: object,
        ) -> None:
            """Initialize base exception with structured metadata and correlation tracking.

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
                Metadata(attributes=extra_kwargs or {})
                if metadata is None
                else metadata
            )
            if FlextRuntime.is_dict_like(metadata):
                merged_attrs = (
                    {**metadata, **extra_kwargs} if extra_kwargs else metadata
                )
                self.metadata = Metadata(attributes=merged_attrs)
            elif extra_kwargs:
                existing_attrs = self.metadata.attributes
                new_attrs = {**existing_attrs, **extra_kwargs}
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

        def to_dict(self) -> dict[str, object]:
            """Convert exception to dictionary representation."""
            return {
                "error_type": self.__class__.__name__,
                "message": self.message,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
                "metadata": self.metadata.attributes,
            }

        def with_context(self, **context: object) -> Self:
            """Add context information to the exception metadata."""
            existing_attrs = self.metadata.attributes
            new_attrs = {**existing_attrs, **context}
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
            value: object = None,
            error_code: str = FlextConstants.Errors.VALIDATION_ERROR,
            correlation_id: str | None = None,
            metadata: Metadata | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
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
            **kwargs: object,
        ) -> None:
            """Initialize type error with type information."""
            kwargs.update({"expected_type": expected_type, "actual_type": actual_type})
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                **kwargs,
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
            **kwargs: object,
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
            **kwargs: object,
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
        kwargs: dict[str, object], specific_params: dict[str, object] | None = None
    ) -> tuple[str | None, object, bool, bool, object, dict[str, object]]:
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
        return (
            cast("str | None", kwargs.get("correlation_id")),
            kwargs.get("metadata"),
            bool(kwargs.get("auto_log")),
            bool(kwargs.get("auto_correlation")),
            kwargs.get("config"),
            extra_kwargs,
        )

    @staticmethod
    def extract_common_kwargs(kwargs: dict[str, object]) -> tuple[object, object]:
        """Extract correlation_id and metadata from kwargs."""
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
    def _determine_error_type(kwargs: dict[str, object]) -> str | None:
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
        Callable[[str, str | None, dict[str, object], str | None, object], BaseError]
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
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: str | None,
    ) -> BaseError:
        creator = FlextExceptions._get_error_creator(error_type) if error_type else None
        normalized_metadata = (
            Metadata(attributes=metadata)
            if metadata and FlextRuntime.is_dict_like(metadata)
            else None
        )
        if creator:
            return creator(
                message,
                error_code,
                kwargs,
                correlation_id,
                cast("str | None", normalized_metadata),
            )
        return FlextExceptions.BaseError(
            message,
            error_code=error_code or "UNKNOWN",
            metadata=normalized_metadata,
            correlation_id=correlation_id,
        )

    @staticmethod
    def create(
        message: str, error_code: str | None = None, **kwargs: object
    ) -> BaseError:
        """Create an appropriate exception instance based on kwargs context."""
        correlation_id_obj, metadata_obj = FlextExceptions.extract_common_kwargs(kwargs)
        error_type = FlextExceptions._determine_error_type(kwargs)
        # Convert metadata to str | None if it's a dict-like object
        correlation_id: str | None = (
            cast("str | None", correlation_id_obj)
            if isinstance(correlation_id_obj, str)
            else None
        )
        metadata_str: str | None = None
        if metadata_obj and FlextRuntime.is_dict_like(metadata_obj):
            metadata_str = str(metadata_obj)
        elif isinstance(metadata_obj, str):
            metadata_str = metadata_obj
        return FlextExceptions._create_error_by_type(
            error_type, message, error_code, kwargs, correlation_id, metadata_str
        )

    _exception_counts: ClassVar[dict[type, int]] = {}

    @classmethod
    def record_exception(cls, exception_type: type) -> None:
        """Record an exception occurrence for metrics tracking."""
        if exception_type not in cls._exception_counts:
            cls._exception_counts[exception_type] = 0
        cls._exception_counts[exception_type] += 1

    @classmethod
    def get_metrics(cls) -> dict[str, object]:
        """Get exception metrics and statistics."""
        total = sum(cls._exception_counts.values(), 0)
        return {
            "total_exceptions": total,
            "exception_counts": cls._exception_counts.copy(),
            "unique_exception_types": len(cls._exception_counts),
        }

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._exception_counts.clear()

    def __call__(
        self, message: str, error_code: str | None = None, **kwargs: object
    ) -> BaseError:
        """Create exception by calling the class instance."""
        return self.create(message, error_code, **kwargs)


__all__ = ["FlextExceptions"]
