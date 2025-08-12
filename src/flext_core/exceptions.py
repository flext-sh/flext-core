"""FLEXT Core Exceptions - Enterprise-grade exception hierarchy for FLEXT ecosystem.

This module consolidates all exception patterns following PEP8 naming conventions and
provides structured error handling, context management, and cross-service serialization
for distributed data integration pipelines across the FLEXT ecosystem.

The exception hierarchy is designed to support enterprise-grade error handling with
proper categorization, context preservation, and metrics collection for monitoring
and debugging distributed systems.

Architecture:
    - Abstract Base Classes: Foundation exception patterns for inheritance
    - Concrete Implementations: Production-ready exception classes for common scenarios
    - Factory Methods: Module-specific exception creation with standardized patterns
    - Metrics System: Exception tracking and monitoring for observability

Exception Categories:
    - Validation Errors: Input validation and data integrity issues
    - Business Errors: Business rule violations and domain logic failures
    - Infrastructure Errors: Network, database, and service connectivity issues
    - Configuration Errors: Missing or invalid configuration values
    - Security Errors: Authentication and authorization failures

Example:
    Basic usage with structured error handling:

    >>> try:
    ...     # Some operation
    ...     raise FlextValidationError("Invalid data", error_code="INVALID_INPUT")
    ... except FlextValidationError as e:
    ...     logger.error(f"Validation failed: {e.to_dict()}")
    ...     print(e.error_code)
    'INVALID_INPUT'

    Creating module-specific exceptions:

    >>> DatabaseError = create_module_exception_classes("database")
    >>> raise DatabaseError.ConnectionError("Cannot connect to database")

"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Self, cast

if TYPE_CHECKING:
    from flext_core.typings import TAnyDict

# =============================================================================
# CONSTANTS - Error codes and categories
# =============================================================================

ERROR_CODES = {
    "VALIDATION_ERROR": "Invalid input data or parameters",
    "BUSINESS_ERROR": "Business rule violation",
    "INFRASTRUCTURE_ERROR": "Infrastructure service failure",
    "CONFIG_ERROR": "Configuration error or missing values",
    "CONNECTION_ERROR": "Network or service connection failure",
    "AUTHENTICATION_ERROR": "Authentication failure",
    "PERMISSION_ERROR": "Insufficient permissions",
    "NOT_FOUND_ERROR": "Resource not found",
    "ALREADY_EXISTS_ERROR": "Resource already exists",
    "TIMEOUT_ERROR": "Operation timeout",
    "PROCESSING_ERROR": "Data processing failure",
    "CRITICAL_ERROR": "Critical system error",
}

# =============================================================================
# EXCEPTION METRICS SYSTEM - Singleton pattern to avoid globals
# =============================================================================


class FlextExceptionMetrics:
    """Singleton class for tracking exception metrics."""

    _instance: FlextExceptionMetrics | None = None
    _metrics: ClassVar[dict[str, int]] = {}

    def __new__(cls) -> Self:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast("Self", cls._instance)

    def record_exception(self, exception_type: str) -> None:
        """Record exception occurrence."""
        self._metrics[exception_type] = self._metrics.get(exception_type, 0) + 1

    def get_metrics(self) -> dict[str, int]:
        """Get current metrics."""
        return dict(self._metrics)

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()


# Global instance for easy access
_exception_metrics = FlextExceptionMetrics()

# =============================================================================
# ABSTRACT BASE CLASSES - Foundation exception patterns
# =============================================================================


class FlextAbstractError(ABC, Exception):
    """Abstract base class for all FLEXT exceptions following SOLID principles.

    Provides foundation for implementing exceptions with proper separation
    of concerns and dependency inversion.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize abstract error."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._get_default_error_code()
        self.context = context or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        """Return a human-readable exception message."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    @property
    @abstractmethod
    def error_category(self) -> str:
        """Get error category - must be implemented by subclasses."""
        ...

    @abstractmethod
    def _get_default_error_code(self) -> str:
        """Get default error code - must be implemented by subclasses."""
        ...

    @abstractmethod
    def to_dict(self) -> TAnyDict:
        """Convert exception to dictionary - must be implemented by subclasses."""
        ...


class FlextAbstractValidationError(FlextAbstractError, ABC):
    """Abstract validation error for input validation."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        validation_details: dict[str, object] | None = None,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize abstract validation error."""
        super().__init__(message, error_code, context)
        self.field = field
        self.validation_details = validation_details or {}

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "VALIDATION"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "VALIDATION_ERROR"


class FlextAbstractBusinessError(FlextAbstractError, ABC):
    """Abstract business error for business rule violations."""

    def __init__(
        self,
        message: str,
        business_rule: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize abstract business error."""
        super().__init__(message, error_code, kwargs)
        self.business_rule = business_rule

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "BUSINESS"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "BUSINESS_ERROR"


class FlextAbstractInfrastructureError(FlextAbstractError, ABC):
    """Abstract infrastructure error for infrastructure issues."""

    def __init__(
        self,
        message: str,
        service: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize abstract infrastructure error."""
        super().__init__(message, error_code, kwargs)
        self.service = service

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "INFRASTRUCTURE"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "INFRASTRUCTURE_ERROR"


class FlextAbstractConfigurationError(FlextAbstractError, ABC):
    """Abstract configuration error for configuration issues."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize abstract configuration error."""
        super().__init__(message, error_code, kwargs)
        self.config_key = config_key

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "CONFIGURATION"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "CONFIG_ERROR"


class FlextAbstractErrorFactory(ABC):
    """Abstract factory for creating exceptions."""

    @abstractmethod
    def create_validation_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractValidationError:
        """Create validation error - must be implemented by subclasses."""
        ...

    @abstractmethod
    def create_business_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractBusinessError:
        """Create business error - must be implemented by subclasses."""
        ...

    @abstractmethod
    def create_infrastructure_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractInfrastructureError:
        """Create infrastructure error - must be implemented by subclasses."""
        ...

    @abstractmethod
    def create_configuration_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractConfigurationError:
        """Create configuration error - must be implemented by subclasses."""
        ...


# =============================================================================
# CONCRETE EXCEPTION IMPLEMENTATIONS - Production-ready exception classes
# =============================================================================


class FlextError(FlextAbstractError):
    """Base exception for all FLEXT operations.

    Provides structured error handling with context, serialization,
    and cross-service compatibility for distributed systems.
    """

    def __init__(
        self,
        message: str = "FLEXT operation failed",
        error_code: str | None = None,
        context: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize FLEXT error with rich context."""
        # Extract specific fields from kwargs
        correlation_id = kwargs.get("correlation_id", "unknown")
        timestamp = kwargs.get("timestamp", time.time())
        traceback_info = kwargs.get("traceback_info", traceback.format_exc())

        # Filter out specific fields from context
        filtered_context = {
            k: v
            for k, v in kwargs.items()
            if k not in {"correlation_id", "timestamp", "traceback_info"}
        }

        # Merge with provided context
        if context:
            filtered_context.update(context)

        super().__init__(message, error_code, filtered_context)

        # Set specific attributes
        self.correlation_id = correlation_id
        self.timestamp = timestamp
        self.traceback_info = traceback_info

        # Record metrics
        _exception_metrics.record_exception(self.__class__.__name__)

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "GENERAL"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "FLEXT_ERROR"

    def to_dict(self) -> TAnyDict:
        """Convert exception to serializable dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "error_category": self.error_category,
            "error_code": self.error_code,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "context": dict(self.context)
            if isinstance(self.context, Mapping)
            else self.context,
            "traceback": self.traceback_info,
        }

    def __repr__(self) -> str:
        """Return technical representation."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"correlation_id='{self.correlation_id}'"
            f")"
        )


class FlextValidationError(FlextAbstractValidationError, FlextError):
    """Validation error for input validation failures."""

    def __init__(
        self,
        message: str = "Validation failed",
        field: str | None = None,
        validation_details: dict[str, object] | None = None,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize validation error."""
        super().__init__(message, field, validation_details, error_code, context)

    def to_dict(self) -> TAnyDict:
        """Convert validation error to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "field": self.field,
                "validation_details": self.validation_details,
            },
        )
        return base_dict


class FlextTypeError(FlextError):
    """Type-related errors for type validation."""

    def __init__(
        self,
        message: str = "Type validation failed",
        expected_type: str | None = None,
        actual_type: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize type error."""
        context = dict(kwargs)
        if expected_type:
            context["expected_type"] = expected_type
        if actual_type:
            context["actual_type"] = actual_type
        super().__init__(message, error_code="TYPE_ERROR", context=context)


class FlextAttributeError(FlextError):
    """Attribute access errors."""

    def __init__(
        self,
        message: str = "Attribute error",
        attribute: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize attribute error."""
        context = dict(kwargs)
        if attribute:
            context["attribute"] = attribute
        super().__init__(message, error_code="ATTRIBUTE_ERROR", context=context)


class FlextOperationError(FlextError):
    """General operation errors."""

    def __init__(
        self,
        message: str = "Operation failed",
        operation: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize operation error."""
        context = dict(kwargs)
        if operation:
            context["operation"] = operation
        super().__init__(message, error_code="OPERATION_ERROR", context=context)


class FlextConfigurationError(FlextAbstractConfigurationError, FlextError):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str = "Configuration error",
        config_key: str | None = None,
        config_file: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize configuration error."""
        context = dict(kwargs)
        if config_file:
            context["config_file"] = config_file
        super().__init__(message, config_key, "CONFIG_ERROR", **context)

    def to_dict(self) -> TAnyDict:
        """Convert configuration error to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "config_key": self.config_key,
            },
        )
        return base_dict


class FlextConnectionError(FlextAbstractInfrastructureError, FlextError):
    """Connection-related errors."""

    def __init__(
        self,
        message: str = "Connection failed",
        service: str | None = None,
        endpoint: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize connection error."""
        context = dict(kwargs)
        if endpoint:
            context["endpoint"] = endpoint
        super().__init__(message, service, "CONNECTION_ERROR", **context)

    def to_dict(self) -> TAnyDict:
        """Convert connection error to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "service": self.service,
            },
        )
        return base_dict


class FlextAuthenticationError(FlextAbstractInfrastructureError, FlextError):
    """Authentication-related errors."""

    def __init__(
        self,
        message: str = "Authentication failed",
        service: str | None = None,
        user_id: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize authentication error."""
        context = dict(kwargs)
        if user_id:
            context["user_id"] = user_id
        super().__init__(message, service, "AUTHENTICATION_ERROR", **context)


class FlextPermissionError(FlextAbstractInfrastructureError, FlextError):
    """Permission-related errors."""

    def __init__(
        self,
        message: str = "Permission denied",
        service: str | None = None,
        required_permission: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize permission error."""
        context = dict(kwargs)
        if required_permission:
            context["required_permission"] = required_permission
        super().__init__(message, service, "PERMISSION_ERROR", **context)


class FlextNotFoundError(FlextError):
    """Resource not found errors."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize not found error."""
        context = dict(kwargs)
        if resource_type:
            context["resource_type"] = resource_type
        if resource_id:
            context["resource_id"] = resource_id
        super().__init__(message, error_code="NOT_FOUND_ERROR", context=context)


class FlextAlreadyExistsError(FlextError):
    """Resource already exists errors."""

    def __init__(
        self,
        message: str = "Resource already exists",
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize already exists error."""
        context = dict(kwargs)
        if resource_type:
            context["resource_type"] = resource_type
        if resource_id:
            context["resource_id"] = resource_id
        super().__init__(message, error_code="ALREADY_EXISTS_ERROR", context=context)


class FlextTimeoutError(FlextAbstractInfrastructureError, FlextError):
    """Timeout-related errors."""

    def __init__(
        self,
        message: str = "Operation timeout",
        service: str | None = None,
        timeout_seconds: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize timeout error."""
        context = dict(kwargs)
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        super().__init__(message, service, "TIMEOUT_ERROR", **context)


class FlextProcessingError(FlextAbstractBusinessError, FlextError):
    """Data processing errors."""

    def __init__(
        self,
        message: str = "Processing failed",
        business_rule: str | None = None,
        operation: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize processing error."""
        context = dict(kwargs)
        if operation:
            context["operation"] = operation
        super().__init__(message, business_rule, "PROCESSING_ERROR", **context)


class FlextCriticalError(FlextAbstractInfrastructureError, FlextError):
    """Critical system errors that require immediate attention."""

    def __init__(
        self,
        message: str = "Critical system error",
        service: str | None = None,
        severity: str = "CRITICAL",
        **kwargs: object,
    ) -> None:
        """Initialize critical error."""
        context = dict(kwargs)
        context["severity"] = severity
        super().__init__(message, service, "CRITICAL_ERROR", **context)


# =============================================================================
# MODULE-SPECIFIC EXCEPTION FACTORY METHODS
# =============================================================================


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
            context = dict(kwargs)
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

            context = dict(kwargs)
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
            config_file_value = kwargs.get("config_file")
            config_file_str = (
                config_file_value if isinstance(config_file_value, str) else None
            )
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != "config_file"}

            super().__init__(
                f"{module_name} config: {message}",
                config_key=config_key,
                config_file=config_file_str,
                **filtered_kwargs,
            )

    return ModuleConfigurationError


def _create_connection_error_class(module_name: str) -> type:
    """Create connection error class for module."""

    class ModuleConnectionError(FlextConnectionError):
        """Module connection errors."""

        def __init__(
            self,
            message: str = f"{module_name} connection failed",
            **kwargs: object,
        ) -> None:
            """Initialize module connection error with context."""
            super().__init__(
                f"{module_name} connection: {message}",
                service=f"{module_name}_connection",
                error_code=f"{module_name.upper()}_CONNECTION_ERROR",
                context=kwargs,
            )

    return ModuleConnectionError


def _create_processing_error_class(module_name: str) -> type:
    """Create processing error class for module."""

    class ModuleProcessingError(FlextProcessingError):
        """Module processing errors."""

        def __init__(
            self,
            message: str = f"{module_name} processing failed",
            **kwargs: object,
        ) -> None:
            """Initialize module processing error with context."""
            super().__init__(
                f"{module_name} processing: {message}",
                business_rule=f"{module_name}_processing",
                error_code=f"{module_name.upper()}_PROCESSING_ERROR",
                context=kwargs,
            )

    return ModuleProcessingError


def _create_authentication_error_class(module_name: str) -> type:
    """Create authentication error class for module."""

    class ModuleAuthenticationError(FlextAuthenticationError):
        """Module authentication errors."""

        def __init__(
            self,
            message: str = f"{module_name} authentication failed",
            **kwargs: object,
        ) -> None:
            """Initialize module authentication error with context."""
            super().__init__(
                f"{module_name} authentication: {message}",
                service=f"{module_name}_auth",
                error_code=f"{module_name.upper()}_AUTHENTICATION_ERROR",
                context=kwargs,
            )

    return ModuleAuthenticationError


def _create_timeout_error_class(module_name: str) -> type:
    """Create timeout error class for module."""

    class ModuleTimeoutError(FlextTimeoutError):
        """Module timeout errors."""

        def __init__(
            self,
            message: str = f"{module_name} operation timeout",
            timeout_seconds: int | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize module timeout error with context."""
            context = dict(kwargs)
            super().__init__(
                f"{module_name} timeout: {message}",
                service=f"{module_name}_service",
                timeout_seconds=timeout_seconds,
                **context,
            )

    return ModuleTimeoutError


def _get_module_prefix(module_name: str) -> str:
    """Get standardized module prefix."""
    return module_name.replace("-", "_").replace(".", "_").upper()


def create_context_exception_factory(module_name: str) -> type:
    """Create context exception factory for module."""

    class ContextExceptionFactory:
        """Factory for creating context-aware exceptions."""

        @staticmethod
        def create_error(message: str, **kwargs: object) -> FlextError:
            """Create base error with context."""
            error_class: type = _create_base_error_class(module_name)
            return cast("FlextError", error_class(message, **kwargs))

        @staticmethod
        def create_validation_error(
            message: str,
            **kwargs: object,
        ) -> FlextValidationError:
            """Create validation error with context."""
            validation_class = _create_validation_error_class(module_name)
            instance = validation_class(message, **kwargs)
            return cast("FlextValidationError", instance)

    return ContextExceptionFactory


def create_module_exception_classes(module_name: str) -> dict[str, type]:
    """Create comprehensive exception classes for a module."""
    prefix = _get_module_prefix(module_name)

    return {
        f"{prefix}Error": _create_base_error_class(module_name),
        f"{prefix}ValidationError": _create_validation_error_class(module_name),
        f"{prefix}ConfigurationError": _create_configuration_error_class(module_name),
        f"{prefix}ConnectionError": _create_connection_error_class(module_name),
        f"{prefix}ProcessingError": _create_processing_error_class(module_name),
        f"{prefix}AuthenticationError": _create_authentication_error_class(module_name),
        f"{prefix}TimeoutError": _create_timeout_error_class(module_name),
    }


# =============================================================================
# EXCEPTION FACTORY IMPLEMENTATION
# =============================================================================


class FlextExceptions(FlextAbstractErrorFactory):
    """Factory for creating FLEXT exceptions."""

    def create_validation_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextValidationError:
        """Create validation error."""
        # Extract validation-specific fields
        field = kwargs.get("field")
        validation_details = kwargs.get("validation_details")
        error_code = kwargs.get("error_code")

        # Filter out validation-specific kwargs
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"field", "validation_details", "error_code"}
        }

        return FlextValidationError(
            message,
            field=field if isinstance(field, str) else None,
            validation_details=validation_details
            if isinstance(validation_details, dict)
            else None,
            error_code=error_code if isinstance(error_code, str) else None,
            context=filtered_kwargs or None,
        )

    def create_business_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextProcessingError:
        """Create business error."""
        # Extract business-specific fields
        business_rule = kwargs.get("business_rule")
        operation = kwargs.get("operation")

        # Filter out business-specific kwargs
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in {"business_rule", "operation"}
        }

        return FlextProcessingError(
            message,
            business_rule=business_rule if isinstance(business_rule, str) else None,
            operation=operation if isinstance(operation, str) else None,
            **filtered_kwargs,
        )

    def create_infrastructure_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextConnectionError:
        """Create infrastructure error."""
        # Extract infrastructure-specific fields
        service = kwargs.get("service")
        endpoint = kwargs.get("endpoint")

        # Filter out infrastructure-specific kwargs
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in {"service", "endpoint"}
        }

        return FlextConnectionError(
            message,
            service=service if isinstance(service, str) else None,
            endpoint=endpoint if isinstance(endpoint, str) else None,
            **filtered_kwargs,
        )

    def create_configuration_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextConfigurationError:
        """Create configuration error."""
        # Extract configuration-specific fields
        config_key = kwargs.get("config_key")
        config_file = kwargs.get("config_file")

        # Filter out configuration-specific kwargs
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in {"config_key", "config_file"}
        }

        return FlextConfigurationError(
            message,
            config_key=config_key if isinstance(config_key, str) else None,
            config_file=config_file if isinstance(config_file, str) else None,
            **filtered_kwargs,
        )


# =============================================================================
# EXCEPTION METRICS AND MONITORING
# =============================================================================


def get_exception_metrics() -> dict[str, int]:
    """Get exception occurrence metrics."""
    return _exception_metrics.get_metrics()


def clear_exception_metrics() -> None:
    """Clear exception metrics."""
    _exception_metrics.clear_metrics()


def _record_exception(exception_type: str) -> None:
    """Record exception occurrence for metrics."""
    _exception_metrics.record_exception(exception_type)


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================


__all__: list[str] = [
    # Constants
    "ERROR_CODES",
    # Abstract Base Classes
    "FlextAbstractBusinessError",
    "FlextAbstractConfigurationError",
    "FlextAbstractError",
    "FlextAbstractErrorFactory",
    "FlextAbstractInfrastructureError",
    "FlextAbstractValidationError",
    "FlextAlreadyExistsError",
    "FlextAttributeError",
    "FlextAuthenticationError",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextCriticalError",
    # Concrete Exception Classes
    "FlextError",
    # Factory and Utility Classes
    "FlextExceptions",
    "FlextNotFoundError",
    "FlextOperationError",
    "FlextPermissionError",
    "FlextProcessingError",
    "FlextTimeoutError",
    "FlextTypeError",
    "FlextValidationError",
    "clear_exception_metrics",
    "create_context_exception_factory",
    # Factory Functions
    "create_module_exception_classes",
    # Metrics Functions
    "get_exception_metrics",
]
