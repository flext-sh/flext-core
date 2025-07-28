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

import inspect
import threading
import traceback
from functools import wraps

from pydantic import BaseModel, ConfigDict

from flext_core._utilities_base import _BaseGenerators
from flext_core.constants import ERROR_CODES

# =============================================================================
# THREAD-SAFE DEBUGGING AND FALLBACK SYSTEM
# =============================================================================

# Thread-safe locks for concurrent access
_metrics_lock = threading.RLock()
_debug_lock = threading.RLock()

# Global exception metrics consolidado - elimina duplicação
_exception_metrics: dict[str, dict[str, object]] = {}

# Centralized debug configuration
_debug_config = {
    "enabled": True,
    "trace_level": "ERROR",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "max_context_size": 1000,
    "enable_stack_trace": True,
    "enable_frame_inspection": True,
    "fallback_enabled": True,
}

# Fallback registry for resilient error handling
_fallback_registry: dict[str, object] = {}


class FlextDebugConfig(BaseModel):
    """Configuration for FLEXT debug system."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    trace_level: str = "ERROR"
    max_context_size: int = 1000
    enable_stack_trace: bool = True
    enable_frame_inspection: bool = True
    fallback_enabled: bool = True


def configure_debug_system(
    config: FlextDebugConfig | None = None,
) -> None:
    """Configure the centralized debug system.

    Thread-safe configuration for debugging and error handling.

    Args:
        config: Debug configuration object

    """
    if config is None:
        config = FlextDebugConfig()

    with _debug_lock:
        _debug_config.update(
            {
                "enabled": config.enabled,
                "trace_level": config.trace_level,
                "max_context_size": config.max_context_size,
                "enable_stack_trace": config.enable_stack_trace,
                "enable_frame_inspection": config.enable_frame_inspection,
                "fallback_enabled": config.fallback_enabled,
            },
        )


def register_fallback(
    operation_name: str,
    fallback_func: object,
) -> None:
    """Register a fallback function for resilient error handling.

    Thread-safe registration of fallback functions.
    """
    with _debug_lock:
        _fallback_registry[operation_name] = fallback_func


def get_enhanced_traceback() -> dict[str, object]:
    """Get enhanced traceback with frame inspection.

    Returns detailed traceback information for debugging.
    """
    if not _debug_config["enabled"]:
        return {}

    try:
        frame_info: list[dict[str, object]] = []
        if _debug_config["enable_frame_inspection"]:
            frame_info.extend(
                {
                    "filename": frame_record.filename,
                    "lineno": frame_record.lineno,
                    "function": frame_record.function,
                    "code_context": frame_record.code_context[0].strip()
                    if frame_record.code_context
                    else "",
                }
                for frame_record in inspect.stack()[1:]  # Skip current frame
            )

        return {
            "thread_id": threading.current_thread().ident,
            "thread_name": threading.current_thread().name,
            "stack_trace": traceback.format_exc()
            if _debug_config["enable_stack_trace"]
            else "",
            "frame_info": frame_info,
            "timestamp": _BaseGenerators.generate_timestamp(),
        }
    except (AttributeError, TypeError, ValueError, OSError):
        # Fallback to minimal info if frame inspection fails
        return {
            "thread_id": threading.current_thread().ident,
            "timestamp": _BaseGenerators.generate_timestamp(),
            "fallback_mode": True,
        }


def safe_fallback(operation_name: str, *args: object, **kwargs: object) -> object:
    """Execute fallback function safely with error isolation.

    Thread-safe fallback execution with comprehensive error handling.
    """
    if not _debug_config["fallback_enabled"]:
        error_msg = f"Fallback disabled for operation: {operation_name}"
        raise FlextOperationError(error_msg)

    with _debug_lock:
        fallback_func = _fallback_registry.get(operation_name)

    if not fallback_func:
        error_msg = f"No fallback registered for operation: {operation_name}"
        raise FlextOperationError(error_msg)

    try:
        if callable(fallback_func):
            return fallback_func(*args, **kwargs)
    except (
        TypeError,
        ValueError,
        AttributeError,
        RuntimeError,
        OSError,
        KeyError,
    ) as e:
        # Even fallbacks can fail - provide ultimate fallback
        enhanced_trace = get_enhanced_traceback()
        error_msg = f"Fallback failed for operation '{operation_name}': {e}"
        raise FlextCriticalError(
            error_msg,
            error_code="FALLBACK_FAILURE",
            context={
                "original_args": args,
                "original_kwargs": kwargs,
                "enhanced_trace": enhanced_trace,
            },
        ) from e
    else:
        return fallback_func


def resilient_operation(
    operation_name: str,
) -> object:
    """Create decorator for resilient operations with automatic fallback.

    Provides thread-safe operation execution with automatic fallback on failure.
    """

    def decorator(func: object) -> object:
        if not callable(func):
            return func

        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            try:
                return func(*args, **kwargs)
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                OSError,
                KeyError,
            ) as e:
                # Enhanced error information
                enhanced_trace = get_enhanced_traceback()

                # Try fallback if available
                if (
                    _debug_config["fallback_enabled"]
                    and operation_name in _fallback_registry
                ):
                    try:
                        return safe_fallback(operation_name, *args, **kwargs)
                    except (
                        TypeError,
                        ValueError,
                        AttributeError,
                        RuntimeError,
                        OSError,
                        KeyError,
                    ) as fallback_error:
                        # Both primary and fallback failed
                        error_msg = (
                            f"Operation '{operation_name}' and its fallback both "
                            f"failed. Primary: {e}, Fallback: {fallback_error}"
                        )
                        raise FlextCriticalError(
                            error_msg,
                            error_code="OPERATION_AND_FALLBACK_FAILURE",
                            context={
                                "primary_error": str(e),
                                "fallback_error": str(fallback_error),
                                "enhanced_trace": enhanced_trace,
                                "operation_name": operation_name,
                            },
                        ) from fallback_error
                else:
                    # No fallback available
                    error_msg = f"Operation '{operation_name}' failed: {e}"
                    raise FlextOperationError(
                        error_msg,
                        error_code="OPERATION_FAILURE",
                        context={
                            "original_error": str(e),
                            "enhanced_trace": enhanced_trace,
                            "operation_name": operation_name,
                        },
                    ) from e

        return wrapper

    return decorator


def _track_exception(exception_class: str, error_code: str | None) -> None:
    """Track exception metrics for observability and monitoring.

    Records exception occurrences with count, error code distribution,
    and temporal information for operational insights.

    Thread-safe implementation with proper locking.

    Args:
        exception_class: Name of the exception class
        error_code: Associated error code for categorization

    """
    with _metrics_lock:
        if exception_class not in _exception_metrics:
            _exception_metrics[exception_class] = {
                "count": 0,
                "error_codes": set(),
                "last_seen": 0.0,
                "thread_info": {},
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

        # Track thread information and update timestamp
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        current_time = _BaseGenerators.generate_timestamp()
        thread_info = metrics.get("thread_info", {})
        if isinstance(thread_info, dict):
            thread_info[thread_id] = {
                "name": thread_name,
                "last_exception": current_time,
            }
            metrics["thread_info"] = thread_info
        metrics["last_seen"] = current_time


def get_exception_metrics() -> dict[str, dict[str, object]]:
    """Get exception metrics for observability and monitoring.

    Returns comprehensive metrics including occurrence counts,
    error code distributions, and temporal information.
    Thread-safe implementation.

    Returns:
        Dictionary containing metrics for each exception type

    """
    with _metrics_lock:
        return dict(_exception_metrics)


def clear_exception_metrics() -> None:
    """Clear exception metrics for testing scenarios.

    Removes all accumulated exception metrics to provide
    clean slate for test isolation.
    Thread-safe implementation.
    """
    with _metrics_lock:
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
        """Initialize error with enhanced debugging and metrics tracking.

        Args:
            message: Error message
            error_code: Error code for categorization
            context: Additional context information

        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or ERROR_CODES["GENERIC_ERROR"]
        self.context = context or {}
        self.timestamp = _BaseGenerators.generate_timestamp()
        self.stack_trace = traceback.format_stack()

        # Enhanced debugging information
        self.enhanced_trace = get_enhanced_traceback()

        # Safely limit context size to prevent memory issues
        max_size = _debug_config.get("max_context_size", 1000)
        if (
            self.context
            and isinstance(max_size, int)
            and len(str(self.context)) > max_size
        ):
            self.context = {
                "_truncated": True,
                "_original_size": len(str(self.context)),
            }

        # Track exception metrics with enhanced information
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
    # Core exceptions
    "FlextAlreadyExistsError",
    "FlextAuthenticationError",
    "FlextConfigError",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextCriticalError",
    "FlextError",
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
    # Debugging and observability functions
    "clear_exception_metrics",
    "configure_debug_system",
    "get_enhanced_traceback",
    "get_exception_metrics",
    "register_fallback",
    "resilient_operation",
    "safe_fallback",
]
