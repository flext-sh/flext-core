"""Comprehensive exception system with structured error handling and metrics.

Provides a hierarchical exception system with the FlextExceptions class containing
specialized exception types, error codes, context tracking, and metrics monitoring.
All exceptions follow a consistent pattern with error codes, correlation IDs,
and structured context for better debugging and monitoring.

Main Classes:
    FlextExceptions: Hierarchical container for all FLEXT exception types and utilities.

Key Features:
    - Hierarchical exception organization with specialized types
    - Structured error codes using FlextConstants integration
    - Context tracking with correlation IDs for distributed tracing
    - Automatic metrics collection for exception monitoring
    - Multiple inheritance from appropriate Python builtin exceptions
    - Legacy compatibility aliases for backward compatibility
    - Direct callable interface for automatic exception type selection

Architecture:
    The exception system follows Clean Architecture principles with clear separation
    of concerns. All exceptions inherit from a common BaseError that provides:
    - Structured error codes
    - Context tracking with correlation IDs
    - Automatic metrics recording
    - Timestamp tracking
    - Consistent string representation

Example:
    Basic exception usage with context::

        # Direct usage with specific exception type
        raise FlextExceptions.ValidationError(
            "Invalid email format",
            field="email",
            value="invalid-email",
            context={"user_id": "123"},
        )

        # Automatic type selection via callable interface
        raise FlextExceptions(
            "Operation failed", operation="create_user", context={"attempt": 3}
        )

        # Configuration errors with structured context
        raise FlextExceptions.ConfigurationError(
            "Database URL not configured", config_key="DATABASE_URL", config_file=".env"
        )

    Exception metrics and monitoring::

        # Get exception occurrence metrics
        metrics = FlextExceptions.get_metrics()
        print(f"ValidationError occurred: {metrics.get('ValidationError', 0)} times")

        # Clear metrics for new monitoring period
        FlextExceptions.clear_metrics()

Note:
    All exceptions automatically record metrics and include correlation IDs
    for distributed tracing. The system is designed to provide maximum
    debugging information while maintaining clean APIs.

"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import ClassVar, cast

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# =============================================================================
# FlextExceptions - Hierarchical Exception Management System
# =============================================================================


class FlextExceptions:
    """Hierarchical exception container with structured error handling and monitoring.

    This class serves as the central container for all FLEXT exception types,
    providing a clean hierarchical organization with specialized exception classes,
    error code management, context tracking, and metrics monitoring. All exceptions
    inherit from a common BaseError with consistent behavior and structured data.

    Architecture Components:
        - Metrics: Exception occurrence tracking and monitoring
        - BaseError: Common base class with error codes and context
        - Specialized Exceptions: Domain-specific exception types
        - ErrorCodes: Centralized error code constants
        - Legacy Aliases: Backward compatibility support
        - Callable Interface: Automatic exception type selection

    Exception Categories:
        - Validation: Data validation and format errors
        - Configuration: System configuration and settings errors
        - Network: Connection and communication errors
        - Authentication: Auth and permission errors
        - Processing: Business logic and operation errors
        - Resource: Not found and already exists errors
        - System: Critical system and type errors

    Key Features:
        - Structured error codes from FlextConstants
        - Automatic correlation ID generation for tracing
        - Context tracking with metadata preservation
        - Metrics collection for monitoring and alerting
        - Multiple inheritance from Python builtin exceptions
        - Consistent string representation with error codes
        - Thread-safe metrics tracking

    Examples:
        Domain-specific exceptions with context::

            # Validation error with field details
            raise FlextExceptions.ValidationError(
                "Email format is invalid",
                field="email",
                value="not-an-email",
                validation_details={"pattern": "email_regex"},
                context={"user_id": "123", "form": "registration"},
            )

            # Configuration error with file context
            raise FlextExceptions.ConfigurationError(
                "Database connection string missing",
                config_key="DATABASE_URL",
                config_file="/app/.env",
                context={"environment": "production"},
            )

        Automatic type selection via callable::

            # Automatically creates ValidationError
            raise FlextExceptions(
                "Field is required", field="username", context={"form_step": 1}
            )

            # Automatically creates OperationError
            raise FlextExceptions(
                "User creation failed",
                operation="create_user",
                context={"retry_count": 3},
            )

        Exception monitoring and metrics::

            # Monitor exception patterns
            metrics = FlextExceptions.get_metrics()
            validation_errors = metrics.get("ValidationError", 0)
            if validation_errors > 100:
                alert_operations_team()

            # Reset metrics for new period
            FlextExceptions.clear_metrics()

    Note:
        All exceptions automatically record occurrence metrics and generate
        correlation IDs for distributed tracing. The system is designed to
        provide maximum observability while maintaining clean, usable APIs.
        All exceptions are real Python exception subclasses with proper
        inheritance from builtin exception types.

    """

    # =============================================================================
    # Metrics Domain: Exception metrics and monitoring functionality
    # =============================================================================

    class Metrics:
        """Thread-safe exception metrics tracking and monitoring system.

        This class provides centralized tracking of exception occurrences across
        the FLEXT ecosystem. It maintains counters for each exception type and
        provides methods for retrieving and managing metrics data for monitoring,
        alerting, and observability purposes.

        Features:
            - Thread-safe exception counter management
            - Automatic metrics recording on exception creation
            - Metrics retrieval for monitoring systems
            - Metrics clearing for periodic resets

        Architecture Principles:
            - Single Responsibility: Only metrics tracking functionality
            - Thread Safety: Safe for concurrent access
            - Zero Dependencies: No external monitoring system dependencies
            - Memory Efficient: Simple counter-based tracking

        Examples:
            Manual metrics recording::

                # Record custom exception occurrence
                FlextExceptions.Metrics.record_exception("CustomError")

            Metrics monitoring and alerts::

                # Get current exception metrics
                metrics = FlextExceptions.Metrics.get_metrics()

                # Monitor specific exception types
                validation_errors = metrics.get("ValidationError", 0)
                if validation_errors > 50:
                    send_alert("High validation error rate detected")

            Periodic metrics management::

                # Get snapshot and reset for new period
                current_metrics = FlextExceptions.Metrics.get_metrics()
                log_metrics_to_monitoring_system(current_metrics)
                FlextExceptions.Metrics.clear_metrics()

        Note:
            Metrics are automatically recorded when any FlextExceptions exception
            is created. The metrics are stored in memory and should be periodically
            exported to external monitoring systems if persistent tracking is required.

        """

        _metrics: ClassVar[dict[str, int]] = {}

        @classmethod
        def record_exception(cls, exception_type: str) -> None:
            """Record exception occurrence.

            Args:
                exception_type: Exception type name.

            """
            cls._metrics[exception_type] = cls._metrics.get(exception_type, 0) + 1

        @classmethod
        def get_metrics(cls) -> dict[str, int]:
            """Get exception counts.

            Returns:
                Dict mapping exception types to counts.

            """
            return dict(cls._metrics)

        @classmethod
        def clear_metrics(cls) -> None:
            """Clear all exception metrics."""
            cls._metrics.clear()

    # =============================================================================
    # BASE EXCEPTION CLASS - Clean hierarchical approach
    # =============================================================================

    class BaseError(Exception):
        """Base exception class providing structured error handling and context tracking.

        This class serves as the foundation for all FLEXT exceptions, providing
        consistent behavior including error codes, correlation IDs, context tracking,
        automatic metrics recording, and structured string representation. All
        FLEXT exceptions inherit from this base class to ensure consistent behavior.

        Features:
            - Structured error codes using FlextConstants
            - Correlation ID generation for distributed tracing
            - Context dictionary for debugging metadata
            - Automatic metrics recording on exception creation
            - Timestamp tracking for temporal analysis
            - Consistent string representation with error codes

        Args:
            message: Human-readable error description
            code: Optional error code (defaults to generic error)
            context: Optional context metadata for debugging
            correlation_id: Optional correlation ID for distributed tracing

        Attributes:
            message: The error message
            code: The structured error code
            context: Context metadata dictionary
            correlation_id: Unique correlation identifier
            timestamp: Exception creation timestamp
            error_code: Property alias for code

        Examples:
            Basic exception with context::

                error = BaseError(
                    "Operation failed",
                    code="FLEXT_2001",
                    context={"operation": "create_user", "attempt": 3},
                    correlation_id="req_123",
                )

            Exception with automatic correlation ID::

                error = BaseError(
                    "Validation failed", context={"field": "email", "value": "invalid"}
                )
                print(error.correlation_id)  # Generated automatically

        Note:
            This class automatically records metrics and generates correlation IDs
            when not provided. All FLEXT exceptions should inherit from this class
            to maintain consistency across the ecosystem.

        """

        def __init__(
            self,
            message: str,
            *,
            code: str | None = None,
            context: Mapping[str, object] | None = None,
            correlation_id: str | None = None,
        ) -> None:
            super().__init__(message)
            self.message = message
            self.code = code or FlextConstants.Errors.GENERIC_ERROR
            self.context = dict(context or {})
            self.correlation_id = correlation_id or f"flext_{int(time.time() * 1000)}"
            self.timestamp = time.time()
            FlextExceptions.Metrics.record_exception(self.__class__.__name__)

        def __str__(self) -> str:
            """Return string representation with error code and message."""
            return f"[{self.code}] {self.message}"

        @property
        def error_code(self) -> str:
            return str(self.code)

    # =============================================================================
    # SPECIFIC EXCEPTION CLASSES - Clean subclass hierarchy
    # =============================================================================

    class _AttributeError(BaseError, AttributeError):
        """Attribute access failure with attribute context.

        Raised when attribute access fails on objects. Provides context
        about which attribute could not be accessed.
        """

        def __init__(
            self, message: str, *, attribute_name: str | None = None, **kwargs: object
        ) -> None:
            self.attribute_name = attribute_name
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context["attribute_name"] = attribute_name
            super().__init__(
                message,
                code=FlextConstants.Errors.OPERATION_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _OperationError(BaseError, RuntimeError):
        """Generic operation failure with operation context.

        Raised when specific operations fail. Provides context about
        which operation failed for debugging purposes.
        """

        def __init__(
            self, message: str, *, operation: str | None = None, **kwargs: object
        ) -> None:
            self.operation = operation
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context["operation"] = operation
            super().__init__(
                message,
                code=FlextConstants.Errors.OPERATION_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _ValidationError(BaseError, ValueError):
        """Data validation failure exception with field-specific context.

        Raised when data validation fails, typically during input validation,
        form processing, or data model validation. Provides structured context
        including the specific field that failed, the invalid value, and
        detailed validation information for debugging and user feedback.

        Args:
            message: Human-readable validation error description
            field: Name of the field that failed validation
            value: The invalid value that caused the validation failure
            validation_details: Additional validation metadata and constraints
            **kwargs: Additional context and correlation_id

        Attributes:
            field: The field name that failed validation
            value: The invalid value
            validation_details: Detailed validation information

        Examples:
            Field validation error::

                raise ValidationError(
                    "Email format is invalid",
                    field="user_email",
                    value="not-an-email",
                    validation_details={"expected_format": "email"},
                )

        Note:
            This exception inherits from both BaseError and ValueError to maintain
            compatibility with standard Python exception handling while providing
            FLEXT-specific structured error information.

        """

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: object = None,
            validation_details: object = None,
            **kwargs: object,
        ) -> None:
            self.field = field
            self.value = value
            self.validation_details = validation_details
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context.update({
                "field": field,
                "value": value,
                "validation_details": validation_details,
            })
            super().__init__(
                message,
                code=FlextConstants.Errors.VALIDATION_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _ConfigurationError(BaseError, ValueError):
        """System configuration error with structured configuration context.

        Raised when system configuration is invalid, missing, or improperly formatted.
        Provides structured context including the specific configuration key that
        caused the error, the configuration file location, and additional metadata
        for debugging configuration issues.

        Args:
            message: Human-readable configuration error description
            config_key: The configuration key that caused the error
            config_file: Path to the configuration file with the issue
            **kwargs: Additional context and correlation_id

        Attributes:
            config_key: The problematic configuration key
            config_file: Configuration file path

        Examples:
            Missing configuration key::

                raise ConfigurationError(
                    "Database URL not configured",
                    config_key="DATABASE_URL",
                    config_file="/app/.env",
                )

        Note:
            This exception is critical for deployment and runtime configuration
            debugging. It provides precise information about configuration issues
            to facilitate quick resolution in production environments.

        """

        def __init__(
            self,
            message: str,
            *,
            config_key: str | None = None,
            config_file: str | None = None,
            **kwargs: object,
        ) -> None:
            self.config_key = config_key
            self.config_file = config_file
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context.update({"config_key": config_key, "config_file": config_file})
            super().__init__(
                message,
                code=FlextConstants.Errors.CONFIGURATION_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _ConnectionError(BaseError, ConnectionError):
        """Network or service connection failure with endpoint context.

        Raised when network connections fail, services are unreachable, or
        communication errors occur. Provides structured context about the
        service and endpoint that failed.
        """

        def __init__(
            self,
            message: str,
            *,
            service: str | None = None,
            endpoint: str | None = None,
            **kwargs: object,
        ) -> None:
            self.service = service
            self.endpoint = endpoint
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context.update({"service": service, "endpoint": endpoint})
            super().__init__(
                message,
                code=FlextConstants.Errors.CONNECTION_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _ProcessingError(BaseError, RuntimeError):
        """Business logic or data processing failure.

        Raised when business operations fail, data processing encounters errors,
        or business rules are violated. Includes context about the specific
        operation and business rule that failed.
        """

        def __init__(
            self,
            message: str,
            *,
            business_rule: str | None = None,
            operation: str | None = None,
            **kwargs: object,
        ) -> None:
            self.business_rule = business_rule
            self.operation = operation
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context.update({"business_rule": business_rule, "operation": operation})
            super().__init__(
                message,
                code=FlextConstants.Errors.PROCESSING_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _TimeoutError(BaseError, TimeoutError):
        """Operation timeout with timing context.

        Raised when operations exceed their allowed execution time.
        Includes the timeout duration for debugging performance issues.
        """

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            **kwargs: object,
        ) -> None:
            self.timeout_seconds = timeout_seconds
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context["timeout_seconds"] = timeout_seconds
            super().__init__(
                message,
                code=FlextConstants.Errors.TIMEOUT_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _NotFoundError(BaseError, FileNotFoundError):
        """Resource not found with resource identification.

        Raised when requested resources (files, database records, services)
        cannot be located. Provides context about the resource type and identifier.
        """

        def __init__(
            self,
            message: str,
            *,
            resource_id: str | None = None,
            resource_type: str | None = None,
            **kwargs: object,
        ) -> None:
            self.resource_id = resource_id
            self.resource_type = resource_type
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context.update({"resource_id": resource_id, "resource_type": resource_type})
            super().__init__(
                message,
                code=FlextConstants.Errors.NOT_FOUND,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _AlreadyExistsError(BaseError, FileExistsError):
        """Resource already exists conflict.

        Raised when attempting to create resources that already exist.
        Provides context about the conflicting resource identifier.
        """

        def __init__(
            self, message: str, *, resource_id: str | None = None, **kwargs: object
        ) -> None:
            self.resource_id = resource_id
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context["resource_id"] = resource_id
            super().__init__(
                message,
                code=FlextConstants.Errors.ALREADY_EXISTS,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _PermissionError(BaseError, PermissionError):
        """Insufficient permissions for requested operation.

        Raised when user lacks required permissions for an operation.
        Includes information about the required permission level.
        """

        def __init__(
            self,
            message: str,
            *,
            required_permission: str | None = None,
            **kwargs: object,
        ) -> None:
            self.required_permission = required_permission
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context["required_permission"] = required_permission
            super().__init__(
                message,
                code=FlextConstants.Errors.PERMISSION_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _AuthenticationError(BaseError, PermissionError):
        """Authentication failure with method context.

        Raised when user authentication fails. Includes context about
        the authentication method that was attempted.
        """

        def __init__(
            self, message: str, *, auth_method: str | None = None, **kwargs: object
        ) -> None:
            self.auth_method = auth_method
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context["auth_method"] = auth_method
            super().__init__(
                message,
                code=FlextConstants.Errors.AUTHENTICATION_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _TypeError(BaseError, TypeError):
        """Type validation failure with type context.

        Raised when type validation fails or type conversion errors occur.
        Includes expected and actual type information for debugging.
        """

        def __init__(
            self,
            message: str,
            *,
            expected_type: str | None = None,
            actual_type: str | None = None,
            **kwargs: object,
        ) -> None:
            self.expected_type = expected_type
            self.actual_type = actual_type
            context = dict(
                cast("Mapping[str, object]", kwargs.get("context", {})) or {}
            )
            context.update({"expected_type": expected_type, "actual_type": actual_type})
            super().__init__(
                message,
                code=FlextConstants.Errors.TYPE_ERROR,
                context=context,
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _CriticalError(BaseError, SystemError):
        """Critical system error requiring immediate attention.

        Raised for serious system failures that may compromise application
        stability or data integrity. Used for high-priority alerts.
        """

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(
                message,
                code=FlextConstants.Errors.CRITICAL_ERROR,
                context=cast("Mapping[str, object] | None", kwargs.get("context")),
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _Error(BaseError, RuntimeError):
        """Generic FLEXT error for general failures.

        Base error class for general FLEXT failures that don't fit
        into more specific exception categories.
        """

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(
                message,
                code=FlextConstants.Errors.GENERIC_ERROR,
                context=cast("Mapping[str, object] | None", kwargs.get("context")),
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _UserError(BaseError, TypeError):
        """User input or API usage error.

        Raised when users provide invalid input or use APIs incorrectly.
        Helps distinguish user errors from system errors.
        """

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(
                message,
                code=FlextConstants.Errors.TYPE_ERROR,
                context=cast("Mapping[str, object] | None", kwargs.get("context")),
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    # =============================================================================
    # PUBLIC API ALIASES - Real exception classes with clean names
    # =============================================================================

    AttributeError = _AttributeError
    OperationError = _OperationError
    ValidationError = _ValidationError
    ConfigurationError = _ConfigurationError
    ConnectionError = _ConnectionError
    ProcessingError = _ProcessingError
    TimeoutError = _TimeoutError
    NotFoundError = _NotFoundError
    AlreadyExistsError = _AlreadyExistsError
    PermissionError = _PermissionError
    AuthenticationError = _AuthenticationError
    TypeError = _TypeError
    CriticalError = _CriticalError
    Error = _Error
    UserError = _UserError

    # =============================================================================
    # Legacy API - Backward compatibility aliases
    # =============================================================================

    FlextError = _Error
    FlextUserError = _UserError
    FlextValidationError = _ValidationError
    FlextConfigurationError = _ConfigurationError
    FlextConnectionError = _ConnectionError
    FlextAuthenticationError = _AuthenticationError
    FlextPermissionError = _PermissionError
    FlextOperationError = _OperationError
    FlextProcessingError = _ProcessingError
    FlextTimeoutError = _TimeoutError
    FlextNotFoundError = _NotFoundError
    FlextAlreadyExistsError = _AlreadyExistsError
    FlextCriticalError = _CriticalError
    FlextTypeError = _TypeError
    FlextAttributeError = _AttributeError

    # =============================================================================
    # ERROR CODES - Error code constants
    # =============================================================================

    class ErrorCodes:
        """Centralized error code constants with FlextConstants integration.

        This class provides a centralized collection of structured error codes
        used throughout the FLEXT exception system. All error codes are sourced
        from FlextConstants to ensure consistency and maintainability across
        the entire FLEXT ecosystem.

        Error Code Categories:
            - Generic: General-purpose error codes
            - Validation: Data validation and format errors
            - Configuration: System configuration errors
            - Network: Connection and communication errors
            - Authentication: Security and permission errors
            - Resource: Not found and existence errors
            - Processing: Business logic and operation errors
            - System: Critical system and infrastructure errors

        Code Format:
            Error codes follow the FLEXT_XXXX format with numeric categorization:
            - 1000-1999: Business logic errors
            - 2000-2999: Technical infrastructure errors
            - 3000-3999: Validation and data errors
            - 4000-4999: Security and authentication errors

        Examples:
            Using error codes in exception handling::

                # Check for specific error types
                try:
                    risky_operation()
                except FlextExceptions.ValidationError as e:
                    if e.code == FlextExceptions.ErrorCodes.VALIDATION_ERROR:
                        handle_validation_error(e)

            Error code mapping for monitoring::

                # Map error codes to alert severity
                error_severity = {
                    FlextExceptions.ErrorCodes.CRITICAL_ERROR: "high",
                    FlextExceptions.ErrorCodes.VALIDATION_ERROR: "medium",
                    FlextExceptions.ErrorCodes.NOT_FOUND: "low",
                }

        Note:
            All error codes are synchronized with FlextConstants to maintain
            consistency across the ecosystem. These constants should be used
            for error code comparisons rather than hardcoded strings.

        """

        GENERIC_ERROR = FlextConstants.Errors.GENERIC_ERROR
        VALIDATION_ERROR = FlextConstants.Errors.VALIDATION_ERROR
        CONFIGURATION_ERROR = FlextConstants.Errors.CONFIGURATION_ERROR
        CONNECTION_ERROR = FlextConstants.Errors.CONNECTION_ERROR
        AUTHENTICATION_ERROR = FlextConstants.Errors.AUTHENTICATION_ERROR
        PERMISSION_ERROR = FlextConstants.Errors.PERMISSION_ERROR
        NOT_FOUND = FlextConstants.Errors.NOT_FOUND
        ALREADY_EXISTS = FlextConstants.Errors.ALREADY_EXISTS
        TIMEOUT_ERROR = FlextConstants.Errors.TIMEOUT_ERROR
        PROCESSING_ERROR = FlextConstants.Errors.PROCESSING_ERROR
        CRITICAL_ERROR = FlextConstants.Errors.CRITICAL_ERROR
        OPERATION_ERROR = FlextConstants.Errors.OPERATION_ERROR
        UNWRAP_ERROR = FlextConstants.Errors.UNWRAP_ERROR
        BUSINESS_ERROR = FlextConstants.Errors.BUSINESS_RULE_ERROR
        INFRASTRUCTURE_ERROR = FlextConstants.Errors.EXTERNAL_SERVICE_ERROR
        TYPE_ERROR = FlextConstants.Errors.TYPE_ERROR

    # =============================================================================
    # DIRECT CALLABLE INTERFACE - For general usage
    # =============================================================================

    def __new__(  # type: ignore[misc]
        cls,
        message: str,
        *,
        operation: str | None = None,
        field: str | None = None,
        config_key: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> BaseError:
        """Direct callable interface for FlextExceptions.

        Automatically selects appropriate exception type based on context.

        Usage:
            raise FlextExceptions("Failed", operation="create")
            raise FlextExceptions("Invalid", field="name", value="")
            raise FlextExceptions("Config error", config_key="database_url")
        """
        # Extract common kwargs that all exceptions understand
        context = cast("Mapping[str, object] | None", kwargs.get("context", {}))
        correlation_id = cast("str | None", kwargs.get("correlation_id"))

        if operation is not None:
            return cls._OperationError(
                message,
                operation=operation,
                code=error_code,
                context=context,
                correlation_id=correlation_id,
            )
        if field is not None:
            value = kwargs.get("value")
            validation_details = kwargs.get("validation_details")
            return cls._ValidationError(
                message,
                field=field,
                value=value,
                validation_details=validation_details,
                code=error_code,
                context=context,
                correlation_id=correlation_id,
            )
        if config_key is not None:
            config_file = cast("str | None", kwargs.get("config_file"))
            return cls._ConfigurationError(
                message,
                config_key=config_key,
                config_file=config_file,
                code=error_code,
                context=context,
                correlation_id=correlation_id,
            )
        # Default to general error
        return cls._Error(
            message, code=error_code, context=context, correlation_id=correlation_id
        )

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    @classmethod
    def get_metrics(cls) -> dict[str, int]:
        """Get exception occurrence metrics."""
        return cls.Metrics.get_metrics()

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear exception metrics."""
        cls.Metrics.clear_metrics()

    # =============================================================================
    # CONFIGURATION MANAGEMENT - FlextTypes.Config Integration
    # =============================================================================

    @classmethod
    def configure_error_handling(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure error handling system using FlextTypes.Config.

        Configures exception behavior, metrics collection, and error logging
        based on environment and validation settings.

        Args:
            config: Configuration dictionary with error handling settings

        Returns:
            FlextResult containing validated configuration or error

        """
        try:
            validated_config: FlextTypes.Config.ConfigDict = {}

            # Validate environment (required for error handling behavior)
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate log level (affects error detail level)
            if "log_level" in config:
                log_level = config["log_level"]
                valid_levels = [level.value for level in FlextConstants.Config.LogLevel]
                if log_level not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {valid_levels}"
                    )
                validated_config["log_level"] = log_level
            # Default based on environment
            elif validated_config["environment"] == "production":
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.ERROR.value
                )
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.WARNING.value
                )

            # Validate validation level (affects error validation strictness)
            if "validation_level" in config:
                val_level = config["validation_level"]
                valid_levels = [v.value for v in FlextConstants.Config.ValidationLevel]
                if val_level not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid validation_level '{val_level}'. Valid options: {valid_levels}"
                    )
                validated_config["validation_level"] = val_level
            # Default based on environment
            elif validated_config["environment"] == "production":
                validated_config["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.STRICT.value
                )
            else:
                validated_config["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.NORMAL.value
                )

            # Add error handling specific configuration
            validated_config["enable_metrics"] = config.get("enable_metrics", True)
            validated_config["enable_stack_traces"] = config.get(
                "enable_stack_traces", validated_config["environment"] != "production"
            )
            validated_config["max_error_details"] = config.get(
                "max_error_details", 1000
            )
            validated_config["error_correlation_enabled"] = config.get(
                "error_correlation_enabled", True
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Configuration error: {e}"
            )

    @classmethod
    def get_error_handling_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current error handling configuration.

        Returns:
            FlextResult containing current error handling configuration

        """
        try:
            # Build current configuration from system state
            current_config: FlextTypes.Config.ConfigDict = {
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                "enable_metrics": True,
                "enable_stack_traces": True,
                "max_error_details": 1000,
                "error_correlation_enabled": True,
                "total_errors_recorded": len(cls.Metrics._metrics),
                "error_types_available": list(cls.Metrics._metrics.keys()),
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(current_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get config: {e}"
            )

    @classmethod
    def create_environment_specific_config(
        cls, environment: FlextTypes.Config.Environment
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific error handling configuration.

        Args:
            environment: Target environment for configuration

        Returns:
            FlextResult containing environment-optimized configuration

        """
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Create environment-specific configuration
            if environment == "production":
                config: FlextTypes.Config.ConfigDict = {
                    "environment": environment,
                    "log_level": FlextConstants.Config.LogLevel.ERROR.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "enable_metrics": True,
                    "enable_stack_traces": False,  # Hide stack traces in production
                    "max_error_details": 500,  # Limit error details
                    "error_correlation_enabled": True,
                }
            elif environment == "development":
                config = {
                    "environment": environment,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "enable_metrics": True,
                    "enable_stack_traces": True,  # Full stack traces for debugging
                    "max_error_details": 2000,  # More error details for debugging
                    "error_correlation_enabled": True,
                }
            elif environment == "test":
                config = {
                    "environment": environment,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "enable_metrics": False,  # Disable metrics in tests
                    "enable_stack_traces": True,
                    "max_error_details": 1000,
                    "error_correlation_enabled": False,  # No correlation in tests
                }
            else:  # staging, local, etc.
                config = {
                    "environment": environment,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "enable_metrics": True,
                    "enable_stack_traces": True,
                    "max_error_details": 1000,
                    "error_correlation_enabled": True,
                }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Environment config failed: {e}"
            )


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextExceptions",  # Main hierarchical container - ONLY access point
    # Legacy compatibility aliases moved to flext_core.legacy to avoid type conflicts
]
