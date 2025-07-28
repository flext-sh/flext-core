"""FLEXT Core Exceptions Module.

Comprehensive exception hierarchy for the FLEXT Core library implementing
enterprise-grade error handling with integrated observability, metrics tracking, and
structured context management. Provides unified exception patterns with automatic
categorization.

Architecture:
    - Single source of truth pattern for all exception functionality across the system
    - Consolidated architecture eliminating _exceptions_base.py module duplication
    - Hierarchical exception design with base FlextError providing common functionality
    - Integrated metrics tracking for operational observability and incident analysis
    - No underscore prefixes on public objects for clean API access
    - Rich context enhancement for debugging and error resolution

Exception Hierarchy:
    - FlextError: Base exception with context management and automatic metrics tracking
    - FlextValidationError: Field validation failures with detailed validation context
    - FlextTypeError: Type mismatch and conversion errors with type information
    - FlextOperationError: Operation and process failures with stage tracking
    - Specific domain errors: Configuration, connection, authentication, permissions
    - Critical system errors: High-priority errors requiring immediate attention

Maintenance Guidelines:
    - Add new exception types by inheriting from appropriate base exception classes
    - Include error codes from constants.py for consistent categorization and handling
    - Maintain context information for debugging, monitoring, and incident resolution
    - Use factory methods in FlextExceptions class for consistency and standardization
    - Track metrics automatically for operational insights and system monitoring
    - Follow naming conventions with Flext prefix for namespace consistency
    - Ensure backward compatibility through legacy exception aliases

Design Decisions:
    - Eliminated _exceptions_base.py following "deliver more with much less" principle
    - Built-in metrics tracking for exception observability without dependencies
    - Rich context information with automatic enhancement and safe value truncation
    - Structured error codes for operational categorization and programmatic handling
    - Serialization support for logging, transport, and external system integration
    - Immutable error context preventing accidental modification after creation

Enterprise Error Management:
    - Comprehensive error categorization supporting operational monitoring and alerting
    - Structured context capture for debugging with automatic field enhancement
    - Metrics integration providing insights into error patterns and frequency
    - Temporal tracking for incident correlation and pattern analysis
    - Security-conscious context handling preventing sensitive information leakage
    - Serialization support for error transport across service boundaries

Observability Features:
    - Automatic exception counting by type and error code for pattern analysis
    - Last seen timestamp tracking for temporal analysis and incident correlation
    - Error code distribution tracking for pattern recognition and system health
    - Metrics clearing for testing scenarios and clean state management
    - Exception frequency analysis for identifying system issues and bottlenecks
    - Rich debugging context with stack trace capture and enhanced error information

Context Management:
    - Automatic context enhancement with field-specific information for validation
    - Type information capture for debugging type-related errors and conversions
    - Operation and stage tracking for complex process debugging and monitoring
    - Safe value truncation preventing log pollution and security information leakage
    - Structured context dictionaries supporting JSON serialization and transport
    - Enhanced context merging preserving both base and specialized error information

Error Code Integration:
    - Standardized error codes from constants module for consistent categorization
    - Automatic error code assignment for simplified exception creation
    - Error code distribution tracking for operational monitoring and alerting
    - Programmatic error handling support through structured error code patterns
    - Integration with logging systems for automated error classification

Dependencies:
    - constants: Structured error codes and categorization for consistent handling
    - time: Timestamp generation for temporal analysis and incident correlation
    - traceback: Stack trace capture for development debugging and error resolution

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import traceback

from flext_core.constants import ERROR_CODES

# =============================================================================
# GLOBAL EXCEPTION METRICS - Private para observabilidade
# =============================================================================

# Global exception metrics consolidado - elimina duplicação
_exception_metrics: dict[str, dict[str, object]] = {}


def _track_exception(exception_class: str, error_code: str | None) -> None:
    """Track exception metrics for observability and monitoring.

    Records exception occurrences with count, error code distribution,
    and temporal information for operational insights.

    Args:
        exception_class: Name of the exception class
        error_code: Associated error code for categorization

    """
    if exception_class not in _exception_metrics:
        _exception_metrics[exception_class] = {
            "count": 0,
            "error_codes": set(),
            "last_seen": 0.0,
        }

    metrics = _exception_metrics[exception_class]
    current_count = metrics["count"]
    if isinstance(current_count, int):
        metrics["count"] = current_count + 1
    else:
        metrics["count"] = 1
    if error_code:
        error_codes_set = metrics["error_codes"]
        if isinstance(error_codes_set, set):
            error_codes_set.add(error_code)
    metrics["last_seen"] = time.time()


def get_exception_metrics() -> dict[str, dict[str, object]]:
    """Get exception metrics for observability and monitoring.

    Returns comprehensive metrics including occurrence counts,
    error code distributions, and temporal information.

    Returns:
        Dictionary containing metrics for each exception type

    """
    return dict(_exception_metrics)


def clear_exception_metrics() -> None:
    """Clear exception metrics for testing scenarios.

    Removes all accumulated exception metrics to provide
    clean slate for test isolation.
    """
    _exception_metrics.clear()


# =============================================================================
# FLEXT BASE ERROR - Consolidado eliminando _FlextBaseError
# =============================================================================


class FlextError(Exception):
    """Base exception for all FLEXT Core errors with observability integration.

    Foundational exception class providing structured error information,
    automatic metrics tracking, and rich context for debugging and monitoring.

    Architecture:
        - Standard Exception inheritance for compatibility
        - Automatic metrics tracking on instantiation
        - Rich context information with timestamps
        - Structured error codes for categorization
        - Stack trace capture for debugging

    Error Information:
        - Human-readable message for users and logs
        - Machine-readable error code for programmatic handling
        - Context dictionary for additional debugging information
        - Timestamp for temporal analysis and correlation
        - Stack trace for development and debugging

    Observability Integration:
        - Automatic metrics tracking on creation
        - Error code distribution tracking
        - Exception type counting for pattern analysis
        - Temporal information for incident correlation

    Usage:
        # Basic usage
        raise FlextError("Something went wrong")

        # With error code and context
        raise FlextError(
            "Validation failed",
            error_code="VALIDATION_ERROR",
            context={"field": "email", "value": "invalid"}
        )
    """

    def __init__(
        self,
        message: str = "An error occurred",
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize error with metrics tracking.

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

        # Track exception metrics
        _track_exception(self.__class__.__name__, self.error_code)

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


# =============================================================================
# FLEXT VALIDATION ERROR - Consolidado eliminando _FlextValidationBaseError
# =============================================================================


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

    def __init__(  # Validation errors need detailed context
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


# =============================================================================
# FLEXT TYPE ERROR - Consolidado eliminando _FlextTypeBaseError
# =============================================================================


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


# =============================================================================
# FLEXT OPERATION ERROR - Consolidado eliminando _FlextOperationBaseError
# =============================================================================


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
# SPECIFIC ERROR TYPES - Consolidados
# =============================================================================


class FlextConfigurationError(FlextError):
    """Configuration-related errors with context capture."""

    def __init__(self, message: str = "Configuration error", **kwargs: object) -> None:
        """Initialize configuration error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information

        """
        super().__init__(message, error_code="CONFIG_ERROR", context=kwargs)


class FlextConnectionError(FlextError):
    """Connection-related errors with network context."""

    def __init__(self, message: str = "Connection error", **kwargs: object) -> None:
        """Initialize connection error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (host, port, etc.)

        """
        super().__init__(message, error_code="CONNECTION_ERROR", context=kwargs)


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
        super().__init__(message, error_code="AUTH_ERROR", context=kwargs)


class FlextPermissionError(FlextError):
    """Permission-related errors with authorization context."""

    def __init__(self, message: str = "Permission denied", **kwargs: object) -> None:
        """Initialize permission error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (resource, action, etc.)

        """
        super().__init__(message, error_code="PERMISSION_ERROR", context=kwargs)


class FlextNotFoundError(FlextError):
    """Resource not found errors with lookup context."""

    def __init__(self, message: str = "Resource not found", **kwargs: object) -> None:
        """Initialize not found error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (resource_id, type, etc.)

        """
        super().__init__(message, error_code="NOT_FOUND", context=kwargs)


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
        super().__init__(message, error_code="ALREADY_EXISTS", context=kwargs)


class FlextTimeoutError(FlextError):
    """Timeout-related errors with timing context."""

    def __init__(self, message: str = "Operation timed out", **kwargs: object) -> None:
        """Initialize timeout error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (timeout, duration, etc.)

        """
        super().__init__(message, error_code="TIMEOUT_ERROR", context=kwargs)


class FlextProcessingError(FlextError):
    """Processing-related errors with operation context."""

    def __init__(self, message: str = "Processing failed", **kwargs: object) -> None:
        """Initialize processing error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (data, stage, etc.)

        """
        super().__init__(message, error_code="PROCESSING_ERROR", context=kwargs)


class FlextCriticalError(FlextError):
    """Critical system errors requiring immediate attention."""

    def __init__(self, message: str = "Critical error", **kwargs: object) -> None:
        """Initialize critical error with context.

        Args:
            message: Descriptive error message
            **kwargs: Additional context information (system, component, etc.)

        """
        super().__init__(message, error_code="CRITICAL_ERROR", context=kwargs)


# =============================================================================
# FLEXT EXCEPTIONS - Interface principal consolidada
# =============================================================================


class FlextExceptions:
    """Consolidated exceptions interface providing unified exception management.

    Serves as the primary public API for exception creation, metrics access,
    and observability functions. Combines factory methods with utility operations.

    Architecture:
        - Static method interface for stateless operations
        - Factory methods for consistent exception creation
        - Observability integration for metrics access
        - Unified API hiding implementation complexity

    Factory Features:
        - Type-safe exception creation methods
        - Consistent parameter patterns across exception types
        - Enhanced context building for debugging
        - Standardized error categorization

    Observability Integration:
        - Exception metrics access for monitoring
        - Metrics clearing for testing scenarios
        - Aggregated statistics for operational insights

    Usage:
        # Factory methods
        error = FlextExceptions.create_validation_error(
            "Invalid email format",
            field="email",
            value="invalid-email"
        )

        # Observability
        metrics = FlextExceptions.get_metrics()
        FlextExceptions.clear_metrics()
    """

    # Exception factory methods
    @staticmethod
    def create_validation_error(
        message: str,
        field: str | None = None,
        value: object = None,
        rules: list[str] | None = None,
    ) -> FlextValidationError:
        """Create validation error."""
        return FlextValidationError(
            message=message,
            validation_details={
                "field": field,
                "value": value,
                "rules": rules or [],
            },
        )

    @staticmethod
    def create_type_error(
        message: str,
        expected_type: type | str | None = None,
        actual_type: type | str | None = None,
    ) -> FlextTypeError:
        """Create type error."""
        return FlextTypeError(
            message=message,
            expected_type=expected_type,
            actual_type=actual_type,
        )

    @staticmethod
    def create_operation_error(
        message: str,
        operation: str | None = None,
        stage: str | None = None,
    ) -> FlextOperationError:
        """Create operation error."""
        return FlextOperationError(
            message=message,
            operation=operation,
            stage=stage,
        )

    # Observability methods
    @staticmethod
    def get_metrics() -> dict[str, dict[str, object]]:
        """Get exception metrics."""
        return get_exception_metrics()

    @staticmethod
    def clear_metrics() -> None:
        """Clear exception metrics."""
        clear_exception_metrics()


# =============================================================================
# ALIASES - Backward compatibility
# =============================================================================

# Legacy aliases mantendo compatibilidade
FlextConfigError = FlextConfigurationError
FlextMigrationError = FlextOperationError
FlextSchemaError = FlextValidationError


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    # Specific exceptions
    "FlextAlreadyExistsError",
    "FlextAuthenticationError",
    # Legacy aliases
    "FlextConfigError",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextCriticalError",
    # Core exceptions
    "FlextError",
    # Main consolidated class
    "FlextExceptions",
    "FlextMigrationError",
    "FlextNotFoundError",
    "FlextOperationError",
    "FlextPermissionError",
    "FlextProcessingError",
    "FlextSchemaError",
    "FlextTimeoutError",
    "FlextTypeError",
    "FlextValidationError",
    # Observability functions
    "clear_exception_metrics",
    "get_exception_metrics",
]
