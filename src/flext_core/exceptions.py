"""Namespace class for FLEXT exception hierarchy.

This module provides the FlextExceptions namespace class containing
structured exception types with error codes and correlation tracking
for the entire FLEXT ecosystem.

Namespace Class Pattern:
- FlextExceptions (namespace class)
  - BaseError (base exception class)
  - ValidationError (validation failures)
  - ConfigurationError (configuration issues)
  - ConnectionError (network/connection failures)
  - TimeoutError (operation timeouts)
  - AuthenticationError (auth failures)
  - AuthorizationError (permission failures)
  - NotFoundError (resource not found)
  - ConflictError (resource conflicts)
  - RateLimitError (rate limiting)
  - CircuitBreakerError (circuit breaker open)

All exceptions include:
- Error codes for categorization
- Correlation IDs for tracking
- Structured metadata and context
- Proper inheritance hierarchy

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes


class FlextExceptions:
    """Namespace class for FLEXT exception hierarchy.

    Provides structured exception types with error codes, correlation tracking,
    and consistent error handling across the entire FLEXT ecosystem.

    Usage:
        ```python
        from flext_core import FlextExceptions

        # Raise structured exceptions
        raise FlextExceptions.ValidationError(
            "Invalid email format", error_code="VAL_EMAIL", field="email"
        )

        # Catch specific exception types
        try:
            validate_data(data)
        except FlextExceptions.ConfigurationError as e:
            logger.error(f"Config error: {e.error_code}")
        ```
    """

    class BaseError(Exception):
        """Base exception class for all FLEXT exceptions.

        Provides common functionality including error codes, correlation IDs,
        and structured metadata for all FLEXT exceptions.

        Attributes:
            error_code: Error code for categorization
            correlation_id: Correlation ID for tracking
            metadata: Additional structured metadata
            timestamp: When the error occurred

        """

        def __init__(
            self,
            message: str,
            *,
            error_code: str | None = None,
            correlation_id: str | None = None,
            metadata: FlextTypes.Dict | None = None,
            **extra_kwargs: object,
        ) -> None:
            """Initialize base error with structured information.

            Args:
                message: Error message
                error_code: Optional error code for categorization
                correlation_id: Optional correlation ID for tracking
                metadata: Optional additional metadata
                **extra_kwargs: Additional keyword arguments stored in metadata

            """
            super().__init__(message)
            self.message = message
            self.error_code = error_code
            self.correlation_id = correlation_id
            self.metadata = metadata or {}
            self.metadata.update(extra_kwargs)
            self.timestamp = time.time()

        def __str__(self) -> str:
            """String representation with error code."""
            if self.error_code:
                return f"[{self.error_code}] {self.message}"
            return self.message

        def to_dict(self) -> FlextTypes.Dict:
            """Convert exception to dictionary representation."""
            return {
                "error_type": self.__class__.__name__,
                "message": self.message,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
                "metadata": self.metadata,
            }

    class ValidationError(BaseError):
        """Exception raised for validation failures.

        Used when input data fails validation rules, including field-level
        validation errors with specific field information.
        """

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: object | None = None,
            error_code: str | None = None,
            correlation_id: str | None = None,
            metadata: FlextTypes.Dict | None = None,
        ) -> None:
            """Initialize validation error.

            Args:
                message: Validation error message
                field: Optional field name that failed validation
                value: Optional invalid value
                error_code: Optional error code override
                correlation_id: Optional correlation ID
                metadata: Optional metadata dictionary

            """
            super().__init__(
                message,
                error_code=error_code or FlextConstants.Errors.VALIDATION_ERROR,
                correlation_id=correlation_id,
                metadata=metadata,
            )
            self.field = field
            self.value = value

    class ConfigurationError(BaseError):
        """Exception raised for configuration-related errors.

        Used when configuration loading, parsing, or validation fails,
        including missing required configuration or invalid values.
        """

        def __init__(
            self,
            message: str,
            *,
            config_key: str | None = None,
            config_source: str | None = None,
        ) -> None:
            """Initialize configuration error.

            Args:
                message: Configuration error message
                config_key: Optional configuration key that caused the error
                config_source: Optional configuration source (file, env, etc.)
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
                config_key=config_key,
                config_source=config_source,
            )
            self.config_key = config_key
            self.config_source = config_source

    class ConnectionError(BaseError):
        """Exception raised for connection-related failures.

        Used when network connections, database connections, or service
        connections fail, including timeouts and unreachable endpoints.
        """

        def __init__(
            self,
            message: str,
            *,
            host: str | None = None,
            port: int | None = None,
            timeout: float | None = None,
        ) -> None:
            """Initialize connection error.

            Args:
                message: Connection error message
                host: Optional host that failed to connect
                port: Optional port that failed to connect
                timeout: Optional timeout value
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.CONNECTION_ERROR,
                host=host,
                port=port,
                timeout=timeout,
            )
            self.host = host
            self.port = port
            self.timeout = timeout

    class TimeoutError(BaseError):
        """Exception raised for operation timeouts.

        Used when operations exceed configured timeout limits,
        including network timeouts, processing timeouts, and I/O timeouts.
        """

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            operation: str | None = None,
        ) -> None:
            """Initialize timeout error.

            Args:
                message: Timeout error message
                timeout_seconds: Optional timeout duration in seconds
                operation: Optional operation that timed out
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.TIMEOUT_ERROR,
                timeout_seconds=timeout_seconds,
                operation=operation,
            )
            self.timeout_seconds = timeout_seconds
            self.operation = operation

    class AuthenticationError(BaseError):
        """Exception raised for authentication failures.

        Used when authentication credentials are invalid, expired,
        or authentication services are unavailable.
        """

        def __init__(
            self,
            message: str,
            *,
            auth_method: str | None = None,
            user_id: str | None = None,
        ) -> None:
            """Initialize authentication error.

            Args:
                message: Authentication error message
                auth_method: Optional authentication method used
                user_id: Optional user ID that failed authentication
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.AUTHENTICATION_ERROR,
                auth_method=auth_method,
                user_id=user_id,
            )
            self.auth_method = auth_method
            self.user_id = user_id

    class AuthorizationError(BaseError):
        """Exception raised for authorization failures.

        Used when authenticated users lack permissions for requested operations,
        including role-based and resource-based authorization failures.
        """

        def __init__(
            self,
            message: str,
            *,
            user_id: str | None = None,
            resource: str | None = None,
            permission: str | None = None,
        ) -> None:
            """Initialize authorization error.

            Args:
                message: Authorization error message
                user_id: Optional user ID that was denied access
                resource: Optional resource being accessed
                permission: Optional permission being checked
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.PERMISSION_ERROR,
                user_id=user_id,
                resource=resource,
                permission=permission,
            )
            self.user_id = user_id
            self.resource = resource
            self.permission = permission

    class NotFoundError(BaseError):
        """Exception raised when requested resources are not found.

        Used when database records, files, endpoints, or other resources
        cannot be located or do not exist.
        """

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
        ) -> None:
            """Initialize not found error.

            Args:
                message: Not found error message
                resource_type: Optional type of resource not found
                resource_id: Optional ID of resource not found
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
                resource_type=resource_type,
                resource_id=resource_id,
            )
            self.resource_type = resource_type
            self.resource_id = resource_id

    class ConflictError(BaseError):
        """Exception raised for resource conflicts.

        Used when operations conflict with existing state, including
        concurrent modifications, duplicate resources, and constraint violations.
        """

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            conflict_reason: str | None = None,
        ) -> None:
            """Initialize conflict error.

            Args:
                message: Conflict error message
                resource_type: Optional type of resource in conflict
                resource_id: Optional ID of resource in conflict
                conflict_reason: Optional reason for the conflict
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.ALREADY_EXISTS,
                resource_type=resource_type,
                resource_id=resource_id,
                conflict_reason=conflict_reason,
            )
            self.resource_type = resource_type
            self.resource_id = resource_id
            self.conflict_reason = conflict_reason

    class RateLimitError(BaseError):
        """Exception raised when rate limits are exceeded.

        Used when operations exceed configured rate limits,
        including API rate limits and resource usage limits.
        """

        def __init__(
            self,
            message: str,
            *,
            limit: int | None = None,
            window_seconds: int | None = None,
            retry_after: int | None = None,
        ) -> None:
            """Initialize rate limit error.

            Args:
                message: Rate limit error message
                limit: Optional rate limit that was exceeded
                window_seconds: Optional time window for rate limiting
                retry_after: Optional seconds to wait before retrying
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.OPERATION_ERROR,
                limit=limit,
                window_seconds=window_seconds,
                retry_after=retry_after,
            )
            self.limit = limit
            self.window_seconds = window_seconds
            self.retry_after = retry_after

    class CircuitBreakerError(BaseError):
        """Exception raised when circuit breaker is open.

        Used when circuit breakers are open due to repeated failures,
        preventing further operations until the circuit resets.
        """

        def __init__(
            self,
            message: str,
            *,
            service_name: str | None = None,
            failure_count: int | None = None,
            reset_timeout: int | None = None,
        ) -> None:
            """Initialize circuit breaker error.

            Args:
                message: Circuit breaker error message
                service_name: Optional name of service with open circuit
                failure_count: Optional number of failures that opened circuit
                reset_timeout: Optional seconds until circuit resets
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.EXTERNAL_SERVICE_ERROR,
                service_name=service_name,
                failure_count=failure_count,
                reset_timeout=reset_timeout,
            )
            self.service_name = service_name
            self.failure_count = failure_count
            self.reset_timeout = reset_timeout

    class TypeError(BaseError):
        """Exception raised for type-related errors.

        Used when type validation, conversion, or type mismatches occur,
        including invalid type arguments and type constraint violations.
        """

        def __init__(
            self,
            message: str,
            *,
            expected_type: str | None = None,
            actual_type: str | None = None,
            error_code: str | None = None,
            correlation_id: str | None = None,
            metadata: FlextTypes.Dict | None = None,
            **extra_kwargs: object,
        ) -> None:
            """Initialize type error.

            Args:
                message: Type error message
                expected_type: Optional expected type name
                actual_type: Optional actual type name
                error_code: Optional error code override
                correlation_id: Optional correlation ID
                metadata: Optional metadata dictionary
                **extra_kwargs: Additional keyword arguments stored in metadata

            """
            super().__init__(
                message,
                error_code=error_code or FlextConstants.Errors.TYPE_ERROR,
                correlation_id=correlation_id,
                metadata=metadata,
                **extra_kwargs,
            )
            self.expected_type = expected_type
            self.actual_type = actual_type

    class OperationError(BaseError):
        """Exception raised for operation-related errors.

        Used when operations fail due to business logic, invalid states,
        or operation-specific constraints.
        """

        def __init__(
            self,
            message: str,
            *,
            operation: str | None = None,
            reason: str | None = None,
        ) -> None:
            """Initialize operation error.

            Args:
                message: Operation error message
                operation: Optional operation name that failed
                reason: Optional reason for the failure
            : Additional metadata

            """
            super().__init__(
                message,
                error_code=FlextConstants.Errors.OPERATION_ERROR,
                operation=operation,
                reason=reason,
            )
            self.operation = operation
            self.reason = reason

    # =========================================================================
    # UTILITY METHODS - Convenience functions for error creation
    # =========================================================================

    @staticmethod
    def create_error(
        error_type: str,
        message: str,
    ) -> BaseError:
        """Create an error instance by type name.

        Args:
            error_type: Name of the error class (e.g., 'ValidationError')
            message: Error message

        Returns:
            Error instance

        Raises:
            ValueError: If error type is not recognized

        """
        error_classes = {
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
        }

        error_class = error_classes.get(error_type)
        if not error_class:
            msg = f"Unknown error type: {error_type}"
            raise ValueError(msg)

        return error_class(message)


__all__ = [
    "FlextExceptions",
]
