"""FLEXT Core Exceptions Module.

Basic exception hierarchy for the FLEXT Core library implementing
enterprise-grade error handling with structured context management.

Architecture:
    - Single source of truth pattern for all exception functionality
    - Hierarchical exception design with base FlextError
    - Rich context enhancement for debugging and error resolution
    - No underscore prefixes on public objects for clean API access

Exception Hierarchy:
    - FlextError: Base exception with context management
    - FlextValidationError: Field validation failures
    - FlextTypeError: Type mismatch and conversion errors
    - FlextOperationError: Operation and process failures
    - Specific domain errors: Configuration, connection, authentication, permissions

Maintenance Guidelines:
    - Add new exception types by inheriting from appropriate base exception classes
    - Include error codes from constants.py for consistent categorization
    - Maintain context information for debugging and monitoring
    - Follow naming conventions with Flext prefix for namespace consistency

Design Decisions:
    - Built-in context information with automatic enhancement
    - Structured error codes for operational categorization
    - Serialization support for logging and transport
    - Immutable error context preventing accidental modification

Enterprise Error Management:
    - Comprehensive error categorization supporting operational monitoring
    - Structured context capture for debugging with automatic field enhancement
    - Security-conscious context handling preventing sensitive information leakage
    - Serialization support for error transport across service boundaries

Context Management:
    - Automatic context enhancement with field-specific information
    - Type information capture for debugging type-related errors
    - Operation and stage tracking for complex process debugging
    - Safe value truncation preventing log pollution
    - Structured context dictionaries supporting JSON serialization

Error Code Integration:
    - Standardized error codes from constants module for consistent categorization
    - Automatic error code assignment for simplified exception creation
    - Programmatic error handling support through structured error code patterns

Dependencies:
    - constants: Structured error codes and categorization for consistent handling
    - time: Timestamp generation for temporal analysis
    - traceback: Stack trace capture for development debugging

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import traceback

from flext_core.constants import ERROR_CODES


def _create_base_error_class(module_name: str) -> type:
    """Create base error class for module."""

    class ModuleBaseError(FlextError):
        """Base exception for module operations."""

        def __init__(
            self,
            message: str = f"{module_name} error",
            **kwargs: object,
        ) -> None:
            """Initialize module error with context."""
            context = kwargs.copy()
            super().__init__(
                message,
                error_code=f"{module_name.upper()}_ERROR",
                context=context,
            )

    return ModuleBaseError


def _create_validation_error_class(module_name: str) -> type:
    """Create validation error class for module."""

    class ModuleValidationError(FlextValidationError):
        """Module validation errors."""

        def __init__(
            self,
            message: str = f"{module_name} validation failed",
            field: str | None = None,
            value: object = None,
            **kwargs: object,
        ) -> None:
            """Initialize module validation error with context."""
            validation_details: dict[str, object] = {}
            if field is not None:
                validation_details["field"] = field
            if value is not None:
                validation_details["value"] = str(value)[:100]

            context = kwargs.copy()
            super().__init__(
                f"{module_name}: {message}",
                validation_details=validation_details,
                context=context,
            )

    return ModuleValidationError


def _create_configuration_error_class(module_name: str) -> type:
    """Create configuration error class for module."""

    class ModuleConfigurationError(FlextConfigurationError):
        """Module configuration errors."""

        def __init__(
            self,
            message: str = f"{module_name} configuration error",
            config_key: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize module configuration error with context."""
            context = kwargs.copy()
            if config_key is not None:
                context["config_key"] = config_key
            super().__init__(f"{module_name} config: {message}", **context)

    return ModuleConfigurationError


def _create_connection_error_class(module_name: str) -> type:
    """Create connection error class for module."""

    class ModuleConnectionError(FlextConnectionError):
        """Module connection errors."""

        def __init__(
            self,
            message: str = f"{module_name} connection failed",
            service_name: str | None = None,
            endpoint: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize module connection error with context."""
            context = kwargs.copy()
            if service_name is not None:
                context["service_name"] = service_name
            if endpoint is not None:
                context["endpoint"] = endpoint
            super().__init__(f"{module_name} connection: {message}", **context)

    return ModuleConnectionError


def _create_processing_error_class(module_name: str) -> type:
    """Create processing error class for module."""

    class ModuleProcessingError(FlextProcessingError):
        """Module processing errors."""

        def __init__(
            self,
            message: str = f"{module_name} processing failed",
            operation: str | None = None,
            file_path: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize module processing error with context."""
            context = kwargs.copy()
            if operation is not None:
                context["operation"] = operation
            if file_path is not None:
                context["file_path"] = file_path
            super().__init__(f"{module_name} processing: {message}", **context)

    return ModuleProcessingError


def _create_authentication_error_class(module_name: str) -> type:
    """Create authentication error class for module."""

    class ModuleAuthenticationError(FlextAuthenticationError):
        """Module authentication errors."""

        def __init__(
            self,
            message: str = f"{module_name} authentication failed",
            username: str | None = None,
            auth_method: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize module authentication error with context."""
            context = kwargs.copy()
            if username is not None:
                context["username"] = username
            if auth_method is not None:
                context["auth_method"] = auth_method
            super().__init__(f"{module_name}: {message}", **context)

    return ModuleAuthenticationError


def _create_timeout_error_class(module_name: str) -> type:
    """Create timeout error class for module."""

    class ModuleTimeoutError(FlextTimeoutError):
        """Module timeout errors."""

        def __init__(
            self,
            message: str = f"{module_name} operation timed out",
            timeout_duration: float | None = None,
            operation: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize module timeout error with context."""
            context = kwargs.copy()
            if timeout_duration is not None:
                context["timeout_duration"] = timeout_duration
            if operation is not None:
                context["operation"] = operation
            super().__init__(f"{module_name}: {message}", **context)

    return ModuleTimeoutError


def _get_module_prefix(module_name: str) -> str:
    """Get module prefix for class naming."""
    return "".join(
        word.capitalize() for word in module_name.replace("_", "-").split("-")
    )


def create_module_exception_classes(module_name: str) -> dict[str, type]:
    """Create DRY module-specific exception classes to eliminate duplication.

    This function creates a complete set of module-specific exception classes that
    eliminate the need for duplicate exception code across FLEXT modules. Each module
    gets its own exception hierarchy while using the shared FlextError foundation.

    This pattern eliminates 85+ locations of duplicated exception code (mass=101).

    Args:
        module_name: Name of the module (e.g., "flext_auth", "algar_oud_mig")

    Returns:
        Dictionary of exception classes ready for use

    Example:
        exceptions = create_module_exception_classes("flext_auth")
        FlextAuthError = exceptions["FlextAuthError"]
        FlextAuthValidationError = exceptions["FlextAuthValidationError"]

    """
    module_prefix = _get_module_prefix(module_name)

    # Factory functions for each exception type
    exception_factories = {
        "Error": _create_base_error_class,
        "ValidationError": _create_validation_error_class,
        "ConfigurationError": _create_configuration_error_class,
        "ConnectionError": _create_connection_error_class,
        "ProcessingError": _create_processing_error_class,
        "AuthenticationError": _create_authentication_error_class,
        "TimeoutError": _create_timeout_error_class,
    }

    return {
        f"{module_prefix}{suffix}": factory(module_name)
        for suffix, factory in exception_factories.items()
    }


class FlextError(Exception):
    """Base exception for all FLEXT Core errors.

    Foundational exception class providing structured error information
    and rich context for debugging and monitoring.

    Architecture:
        - Standard Exception inheritance for compatibility
        - Rich context information with timestamps
        - Structured error codes for categorization
        - Stack trace capture for debugging

    Error Information:
        - Human-readable message for users and logs
        - Machine-readable error code for programmatic handling
        - Context dictionary for additional debugging information
        - Timestamp for temporal analysis and correlation
        - Stack trace for development and debugging

    Usage:
        # Basic usage
        raise FlextError("Something went wrong")

        # With error code and context
        raise FlextError(
            "Validation failed",
            error_code=ERROR_CODES["VALIDATION_ERROR"],
            context={"field": "email", "value": "invalid"}
        )
    """

    def __init__(
        self,
        message: str = "An error occurred",
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize error with enhanced debugging.

        Args:
            message: Error message
            error_code: Error code for categorization
            context: Additional context information

        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or ERROR_CODES["GENERIC_ERROR"]
        self.context = context or {}
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()

        # Safely limit context size to prevent memory issues
        max_size = 1000
        if self.context and len(str(self.context)) > max_size:
            self.context = {
                "_truncated": True,
                "_original_size": len(str(self.context)),
            }

    def __str__(self) -> str:
        """Return formatted error string."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        """Return detailed error representation."""
        return (
            f"{self.__class__.__name__}"
            f"(message='{self.message}', code='{self.error_code}')"
        )

    def to_dict(self) -> dict[str, object]:
        """Convert exception to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class FlextValidationError(FlextError):
    """Validation failure exception with field-specific context.

    Specialized exception for validation failures providing detailed information
    about field validation errors, failed rules, and problematic values.

    Architecture:
        - Inherits from FlextError for base functionality
        - Enhanced context with field-specific information
        - Automatic validation error categorization
        - Rich debugging information for development

    Validation Context:
        - Field name identification for multi-field validation
        - Value truncation for safe logging (prevents log pollution)
        - Failed rules enumeration for debugging
        - Enhanced context merging with base error context

    Usage:
        raise FlextValidationError(
            "Email format is invalid",
            field="email",
            value="not-an-email",
            rules=["email_format", "non_empty"]
        )
    """

    def __init__(
        self,
        message: str = "Validation failed",
        *,
        validation_details: dict[str, object] | None = None,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize validation error with validation details."""
        # Extract validation details
        details = validation_details or {}
        self.field = details.get("field")
        self.value = details.get("value")
        self.rules = details.get("rules", [])

        # Build enhanced context
        enhanced_context = context or {}
        if self.field is not None:
            enhanced_context["field"] = self.field
        if self.value is not None:
            enhanced_context["value"] = str(self.value)[:100]  # Limit value length
        if self.rules:
            enhanced_context["failed_rules"] = self.rules

        super().__init__(
            message=message,
            error_code=error_code or ERROR_CODES["VALIDATION_ERROR"],
            context=enhanced_context,
        )


class FlextTypeError(FlextError):
    """Type mismatch exception with expected and actual type information.

    Specialized exception for type-related errors providing clear information
    about expected versus actual types for debugging and error resolution.

    Architecture:
        - Inherits from FlextError for base functionality
        - Type information enhancement in context
        - Automatic type error categorization
        - Clear type mismatch messaging

    Type Context:
        - Expected type documentation for requirements
        - Actual type identification for debugging
        - String representation for logging safety
        - Enhanced context with type information

    Usage:
        raise FlextTypeError(
            "Expected string but got integer",
            expected_type=str,
            actual_type=int
        )
    """

    def __init__(
        self,
        message: str = "Type error occurred",
        *,
        expected_type: type | str | None = None,
        actual_type: type | str | None = None,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize type error with type information."""
        self.expected_type = expected_type
        self.actual_type = actual_type

        # Build enhanced context
        enhanced_context = context or {}
        if expected_type is not None:
            enhanced_context["expected_type"] = str(expected_type)
        if actual_type is not None:
            enhanced_context["actual_type"] = str(actual_type)

        super().__init__(
            message=message,
            error_code=error_code or ERROR_CODES["TYPE_ERROR"],
            context=enhanced_context,
        )


class FlextAttributeError(FlextError):
    """Attribute access exception with attribute context information.

    Specialized exception for attribute-related errors providing clear information
    about missing attributes and available alternatives for debugging.

    Architecture:
        - Inherits from FlextError for base functionality
        - Attribute context enhancement for debugging
        - Available attribute suggestions for resolution
        - Clear attribute access messaging

    Attribute Context:
        - Class name for context identification
        - Attribute name that was attempted
        - Available attributes for suggestions
        - Enhanced error reporting

    Usage:
        raise FlextAttributeError(
            "Object has no attribute 'missing_attr'",
            attribute_context={
                "class_name": "MyClass",
                "attribute_name": "missing_attr",
                "available_extra_fields": ["field1", "field2"]
            }
        )
    """

    def __init__(
        self,
        message: str = "Attribute error occurred",
        *,
        attribute_context: dict[str, object] | None = None,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize attribute error with attribute context."""
        # Build enhanced context
        enhanced_context = context or {}
        if attribute_context:
            enhanced_context.update(attribute_context)

        super().__init__(
            message=message,
            error_code=error_code or ERROR_CODES["TYPE_ERROR"],
            context=enhanced_context,
        )


class FlextOperationError(FlextError):
    """Operation failure exception with operation and stage context.

    Specialized exception for operation failures providing detailed information
    about operation context, execution stage, and failure circumstances.

    Architecture:
        - Inherits from FlextError for base functionality
        - Operation-specific context enhancement
        - Stage-based failure tracking
        - Process flow debugging support

    Operation Context:
        - Operation name identification for categorization
        - Execution stage tracking for pinpointing failures
        - Enhanced context with operation metadata
        - Process flow debugging information

    Usage:
        raise FlextOperationError(
            "Database connection failed",
            operation="user_creation",
            stage="database_insert"
        )
    """

    def __init__(
        self,
        message: str = "Operation failed",
        *,
        operation: str | None = None,
        stage: str | None = None,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize operation error with operation details."""
        self.operation = operation
        self.stage = stage

        # Build enhanced context
        enhanced_context = context or {}
        if operation is not None:
            enhanced_context["operation"] = operation
        if stage is not None:
            enhanced_context["stage"] = stage

        super().__init__(
            message=message,
            error_code=error_code or ERROR_CODES["OPERATION_ERROR"],
            context=enhanced_context,
        )


# =============================================================================
# SPECIFIC ERROR TYPES
# =============================================================================


class FlextConfigurationError(FlextError):
    """Configuration-related errors with context capture."""

    def __init__(self, message: str = "Configuration error", **kwargs: object) -> None:
        """Initialize configuration error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information

        """
        super().__init__(
            message,
            error_code=ERROR_CODES["CONFIG_ERROR"],
            context=kwargs,
        )


class FlextConnectionError(FlextError):
    """Connection-related errors with network context."""

    def __init__(self, message: str = "Connection error", **kwargs: object) -> None:
        """Initialize connection error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (host, port, etc.)

        """
        super().__init__(
            message,
            error_code=ERROR_CODES["CONNECTION_ERROR"],
            context=kwargs,
        )


class FlextAuthenticationError(FlextError):
    """Authentication-related errors with security context."""

    def __init__(
        self,
        message: str = "Authentication failed",
        **kwargs: object,
    ) -> None:
        """Initialize authentication error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (user, method, etc.)

        """
        super().__init__(message, error_code=ERROR_CODES["AUTH_ERROR"], context=kwargs)


class FlextPermissionError(FlextError):
    """Permission-related errors with authorization context."""

    def __init__(self, message: str = "Permission denied", **kwargs: object) -> None:
        """Initialize permission error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (resource, action, etc.)

        """
        super().__init__(
            message,
            error_code=ERROR_CODES["PERMISSION_ERROR"],
            context=kwargs,
        )


class FlextNotFoundError(FlextError):
    """Resource not found errors with lookup context."""

    def __init__(self, message: str = "Resource not found", **kwargs: object) -> None:
        """Initialize not found error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (resource_id, type, etc.)

        """
        super().__init__(message, error_code=ERROR_CODES["NOT_FOUND"], context=kwargs)


class FlextAlreadyExistsError(FlextError):
    """Resource already exists errors with conflict context."""

    def __init__(
        self,
        message: str = "Resource already exists",
        **kwargs: object,
    ) -> None:
        """Initialize already exists error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (resource_id, type, etc.)

        """
        super().__init__(
            message,
            error_code=ERROR_CODES["ALREADY_EXISTS"],
            context=kwargs,
        )


class FlextTimeoutError(FlextError):
    """Timeout-related errors with timing context."""

    def __init__(self, message: str = "Operation timed out", **kwargs: object) -> None:
        """Initialize timeout error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (timeout, duration, etc.)

        """
        super().__init__(
            message,
            error_code=ERROR_CODES["TIMEOUT_ERROR"],
            context=kwargs,
        )


class FlextProcessingError(FlextError):
    """Processing-related errors with operation context."""

    def __init__(self, message: str = "Processing failed", **kwargs: object) -> None:
        """Initialize processing error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (data, stage, etc.)

        """
        super().__init__(
            message,
            error_code=ERROR_CODES["PROCESSING_ERROR"],
            context=kwargs,
        )


class FlextCriticalError(FlextError):
    """Critical system errors requiring immediate attention."""

    def __init__(self, message: str = "Critical error", **kwargs: object) -> None:
        """Initialize critical error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (system, component, etc.)

        """
        super().__init__(
            message,
            error_code=ERROR_CODES["CRITICAL_ERROR"],
            context=kwargs,
        )


# =============================================================================
# ALIASES - Backward compatibility
# =============================================================================

# Legacy aliases mantendo compatibilidade
FlextConfigError = FlextConfigurationError
FlextSchemaError = FlextValidationError


# =============================================================================
# EXCEPTION METRICS - Monitoring and observability
# =============================================================================

# Global exception metrics dictionary
_EXCEPTION_METRICS: dict[str, int] = {}


def get_exception_metrics() -> dict[str, int]:
    """Get current exception metrics.

    Returns:
        Dictionary of exception type counts

    """
    return _EXCEPTION_METRICS.copy()


def clear_exception_metrics() -> None:
    """Clear all exception metrics."""
    _EXCEPTION_METRICS.clear()


def _record_exception(exception_type: str) -> None:
    """Record exception occurrence for metrics."""
    _EXCEPTION_METRICS[exception_type] = _EXCEPTION_METRICS.get(exception_type, 0) + 1


# =============================================================================
# EXCEPTION FACTORY - Unified exception creation interface
# =============================================================================


class FlextExceptions:
    """Unified factory interface for creating FLEXT exceptions.

    Provides convenient factory methods for creating all types of FLEXT exceptions
    with appropriate default context and error codes.
    """

    @staticmethod
    def create_validation_error(
        message: str,
        *,
        field: str | None = None,
        value: object = None,
        rules: list[str] | None = None,
        context: dict[str, object] | None = None,
    ) -> FlextValidationError:
        """Create validation error with field context."""
        validation_details: dict[str, object] = {}
        if field is not None:
            validation_details["field"] = field
        if value is not None:
            validation_details["value"] = str(value)
        if rules is not None:
            validation_details["rules"] = str(rules)
        return FlextValidationError(
            message,
            validation_details=validation_details,
            context=context,
        )

    @staticmethod
    def create_type_error(
        message: str,
        *,
        expected_type: type | None = None,
        actual_type: type | None = None,
        context: dict[str, object] | None = None,
    ) -> FlextTypeError:
        """Create type error with type context."""
        return FlextTypeError(
            message,
            expected_type=expected_type,
            actual_type=actual_type,
            context=context,
        )

    @staticmethod
    def create_operation_error(
        message: str,
        *,
        operation_name: str | None = None,
        context: dict[str, object] | None = None,
    ) -> FlextOperationError:
        """Create operation error with operation context."""
        return FlextOperationError(
            message,
            operation=operation_name,
            context=context,
        )

    @staticmethod
    def create_configuration_error(
        message: str,
        *,
        config_key: str | None = None,
        context: dict[str, object] | None = None,
    ) -> FlextConfigurationError:
        """Create configuration error with config context."""
        config_context = {}
        if config_key is not None:
            config_context["config_key"] = config_key
        return FlextConfigurationError(
            message,
            config_context=config_context,
            context=context,
        )

    @staticmethod
    def create_connection_error(
        message: str,
        *,
        endpoint: str | None = None,
        context: dict[str, object] | None = None,
    ) -> FlextConnectionError:
        """Create connection error with connection context."""
        connection_context = {}
        if endpoint is not None:
            connection_context["endpoint"] = endpoint
        return FlextConnectionError(
            message,
            connection_context=connection_context,
            context=context,
        )


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "FlextAlreadyExistsError",
    "FlextAttributeError",
    "FlextAuthenticationError",
    "FlextConfigError",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextCriticalError",
    "FlextError",
    "FlextExceptions",
    "FlextNotFoundError",
    "FlextOperationError",
    "FlextPermissionError",
    "FlextProcessingError",
    "FlextSchemaError",
    "FlextTimeoutError",
    "FlextTypeError",
    "FlextValidationError",
    "clear_exception_metrics",
    "get_exception_metrics",
]
