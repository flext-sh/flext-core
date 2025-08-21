"""FLEXT exception hierarchy with DRY factory pattern.

Hierarchical exception system with dynamic class generation:
- FlextExceptions: Main container with Codes, Metrics, Base domains
- Dynamic generation: All exception classes created via factory pattern
- Clean Architecture: Domain separation with SOLID principles
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Self

from flext_core.constants import FlextConstants

if TYPE_CHECKING:
    from flext_core.protocols import FlextProtocols

    type ErrorHandlerProtocol = FlextProtocols.Foundation.ErrorHandler

# =============================================================================
# FlextExceptions - Hierarchical Exception Management System
# =============================================================================


class FlextExceptions:
    """Hierarchical exception system with DRY factory pattern.

    Domains: Codes (error enums), Metrics (tracking), Base (mixins + factory).
    All exception classes generated dynamically via Base.create_exception_type().
    """

    # =============================================================================
    # Metrics Domain: Exception metrics and monitoring functionality
    # =============================================================================

    class Metrics:
        """Exception metrics tracking (singleton pattern)."""

        class _FlextExceptionMetrics:
            """Internal singleton class for tracking exception metrics."""

            _instance: ClassVar[Self | None] = None
            _metrics: ClassVar[dict[str, int]] = {}

            def __new__(cls) -> Self:
                """Ensure singleton instance."""
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance

            def record_exception(self, exception_type: str) -> None:
                """Record exception occurrence."""
                self._metrics[exception_type] = self._metrics.get(exception_type, 0) + 1

            def get_metrics(self) -> dict[str, int]:
                """Get current metrics."""
                return dict(self._metrics)

            def clear_metrics(self) -> None:
                """Clear all metrics."""
                self._metrics.clear()

        # Private singleton instance
        _metrics_instance: ClassVar[_FlextExceptionMetrics] = _FlextExceptionMetrics()

        @classmethod
        def get_metrics(cls) -> dict[str, int]:
            """Get exception occurrence metrics."""
            return cls._metrics_instance.get_metrics()

        @classmethod
        def clear_metrics(cls) -> None:
            """Clear exception metrics."""
            cls._metrics_instance.clear_metrics()

        @classmethod
        def record_exception(cls, exception_type: str) -> None:
            """Record exception occurrence for metrics tracking."""
            cls._metrics_instance.record_exception(exception_type)

    # =============================================================================
    # Base Domain: Base exception classes and mixins
    # =============================================================================

    class Base:
        """Base exception classes, mixins, and DRY factory."""

        class FlextErrorMixin:
            """Common FLEXT error functionality with context tracking."""

            def __init__(
                self,
                message: str,
                *,
                code: object | None = None,
                error_code: object | None = None,
                context: Mapping[str, object] | None = None,
                correlation_id: str | None = None,
            ) -> None:
                self.message = message
                # Normalize error code
                resolved_code = error_code if error_code is not None else code
                if isinstance(resolved_code, Enum):
                    self.code = str(resolved_code.value)
                elif resolved_code is None:
                    self.code = FlextConstants.Errors.GENERIC_ERROR
                else:
                    self.code = str(resolved_code)
                self.context = dict(context or {})
                self.correlation_id = (
                    correlation_id or f"flext_{int(time.time() * 1000)}"
                )
                self.timestamp = time.time()
                FlextExceptions.Metrics.record_exception(self.__class__.__name__)

            def __str__(self) -> str:
                """Return formatted error message."""
                return f"[{self.code}] {self.message}"

            @property
            def error_code(self) -> str:
                return str(self.code)

        @classmethod
        def create_exception_type(
            cls,
            name: str,
            base_exception: type[Exception],
            default_code: str,
            doc: str = "",
            fields: list[str] | None = None,
        ) -> type[Exception]:
            """DRY factory to create exception classes dynamically."""
            if fields is None:
                fields = []

            # Create a proper class that inherits from both base_exception and FlextErrorMixin
            # MyPy limitation: Cannot handle dynamic inheritance properly
            class GeneratedExceptionClass(base_exception, cls.FlextErrorMixin):  # type: ignore[valid-type,misc,name-defined] # Dynamic inheritance
                """Dynamically generated exception class."""

                def __init__(self, message: str, **kwargs: object) -> None:
                    # Store field-specific parameters as attributes
                    for field_name in fields:  # type: ignore[union-attr] # fields is always a list after None check
                        if field_name in kwargs:
                            setattr(self, field_name, kwargs[field_name])

                    # Initialize base exception first
                    base_exception.__init__(self, message)

                    # Initialize mixin with normalized parameters
                    mixin_kwargs = {
                        "code": default_code,
                        "context": kwargs,
                    }
                    cls.FlextErrorMixin.__init__(self, message, **mixin_kwargs)  # type: ignore[arg-type] # Dynamic kwargs handling

            # Set class attributes
            GeneratedExceptionClass.__name__ = name
            GeneratedExceptionClass.__qualname__ = name
            GeneratedExceptionClass.__doc__ = (
                doc or f"{name} - {base_exception.__name__} with FLEXT context."
            )
            GeneratedExceptionClass.__module__ = __name__

            return GeneratedExceptionClass

        # Factory interface implementation will be added after class definition

    # =============================================================================
    # DRY EXCEPTION GENERATION (will be added after class definition)
    # =============================================================================

    # Exception specifications for dynamic generation - ALL EXCEPTIONS FOLLOW DRY PATTERN
    _EXCEPTION_SPECS: ClassVar[
        list[tuple[str, type[Exception], str, str, list[str]]]
    ] = [
        # Base errors
        (
            "FlextError",
            RuntimeError,
            FlextConstants.Errors.GENERIC_ERROR,
            "Base FLEXT error",
            [],
        ),
        (
            "FlextUserError",
            TypeError,
            FlextConstants.Errors.TYPE_ERROR,
            "User input/usage error",
            [],
        ),
        # Validation and configuration
        (
            "FlextValidationError",
            ValueError,
            FlextConstants.Errors.VALIDATION_ERROR,
            "Data validation failed",
            ["field", "value", "validation_details"],
        ),
        (
            "FlextConfigurationError",
            ValueError,
            FlextConstants.Errors.CONFIGURATION_ERROR,
            "Configuration error",
            ["config_key", "config_file"],
        ),
        # Network and connections
        (
            "FlextConnectionError",
            ConnectionError,
            FlextConstants.Errors.CONNECTION_ERROR,
            "Network/service connection failed",
            ["service", "endpoint"],
        ),
        # Operations and processing
        (
            "FlextOperationError",
            RuntimeError,
            FlextConstants.Errors.OPERATION_ERROR,
            "Operation failed",
            ["operation"],
        ),
        (
            "FlextProcessingError",
            RuntimeError,
            FlextConstants.Errors.PROCESSING_ERROR,
            "Processing failed",
            ["business_rule", "operation"],
        ),
        (
            "FlextTimeoutError",
            TimeoutError,
            FlextConstants.Errors.TIMEOUT_ERROR,
            "Operation timed out",
            ["timeout_seconds"],
        ),
        # Resources and access
        (
            "FlextNotFoundError",
            FileNotFoundError,
            FlextConstants.Errors.NOT_FOUND,
            "Resource not found",
            ["resource_id", "resource_type"],
        ),
        (
            "FlextAlreadyExistsError",
            FileExistsError,
            FlextConstants.Errors.ALREADY_EXISTS,
            "Resource already exists",
            ["resource_id"],
        ),
        (
            "FlextPermissionError",
            PermissionError,
            FlextConstants.Errors.PERMISSION_ERROR,
            "Insufficient permissions",
            ["required_permission"],
        ),
        (
            "FlextAuthenticationError",
            PermissionError,
            FlextConstants.Errors.AUTHENTICATION_ERROR,
            "Authentication failed",
            ["auth_method"],
        ),
        # Type and attributes
        (
            "FlextTypeError",
            TypeError,
            FlextConstants.Errors.TYPE_ERROR,
            "Type validation failed",
            ["expected_type", "actual_type"],
        ),
        (
            "FlextAttributeError",
            AttributeError,
            FlextConstants.Errors.OPERATION_ERROR,  # Using OPERATION_ERROR as there's no specific ATTRIBUTE_ERROR in constants
            "Attribute access failed",
            ["attribute_name"],
        ),
        # System critical
        (
            "FlextCriticalError",
            SystemError,
            FlextConstants.Errors.CRITICAL_ERROR,
            "Critical system error",
            ["severity"],
        ),
    ]

    # All exception classes are now generated dynamically via DRY factory pattern
    # Special cases with complex logic can be added here if needed

    # Simple module factory using dynamic pattern
    @staticmethod
    def create_context_exception_factory(module_name: str) -> type:
        """Create context exception factory for module."""

        class ContextExceptionFactory:
            @staticmethod
            def create_error(message: str, **kwargs: object) -> Exception:
                # Get the dynamically created exception class
                error_class = FlextExceptions.FlextError  # type: ignore[attr-defined] # Dynamic class creation
                return error_class(f"{module_name}: {message}", **kwargs)  # type: ignore[no-any-return] # Dynamic class instantiation

            @staticmethod
            def create_validation_error(message: str, **kwargs: object) -> Exception:
                # Get the dynamically created exception class
                validation_error_class = FlextExceptions.FlextValidationError  # type: ignore[attr-defined] # Dynamic class creation
                return validation_error_class(f"{module_name}: {message}", **kwargs)  # type: ignore[no-any-return] # Dynamic class instantiation

        return ContextExceptionFactory

    @staticmethod
    def create_module_exception_classes(module_name: str) -> dict[str, type]:
        """Create exception classes for a module using factory pattern."""
        prefix = module_name.replace("-", "_").replace(".", "_").upper()
        factory = FlextExceptions.Base.create_exception_type

        return {
            f"{prefix}Error": factory(
                f"{prefix}Error",
                RuntimeError,
                FlextConstants.Errors.GENERIC_ERROR,
                f"{module_name} error",
                [],
            ),
            f"{prefix}ValidationError": factory(
                f"{prefix}ValidationError",
                ValueError,
                FlextConstants.Errors.VALIDATION_ERROR,
                f"{module_name} validation error",
                ["field", "value"],
            ),
            f"{prefix}ConfigurationError": factory(
                f"{prefix}ConfigurationError",
                ValueError,
                FlextConstants.Errors.CONFIGURATION_ERROR,
                f"{module_name} config error",
                ["config_key", "config_file"],
            ),
        }

    # Abstract patterns removed - use dynamic factory instead


# =============================================================================
# DYNAMIC EXCEPTION GENERATION (DRY implementation)
# =============================================================================

# Generate dynamic exception classes using the factory pattern
for spec in FlextExceptions._EXCEPTION_SPECS:
    name, base_exception, default_code, doc, fields = spec
    # Create the exception class using the factory
    generated_class = FlextExceptions.Base.create_exception_type(
        name=name,
        base_exception=base_exception,
        default_code=default_code,
        doc=doc,
        fields=fields,
    )
    # Add to FlextExceptions namespace
    setattr(FlextExceptions, name, generated_class)


# =============================================================================
# EXCEPTION METRICS AND MONITORING
# =============================================================================


def get_exception_metrics() -> dict[str, int]:
    """Get exception occurrence metrics."""
    return FlextExceptions.Metrics.get_metrics()


def clear_exception_metrics() -> None:
    """Clear exception metrics."""
    FlextExceptions.Metrics.clear_metrics()


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================


# Error codes compatibility - using constants from flext_core.constants
class FlextErrorCodes:
    """Compatibility class for error codes using FlextConstants."""

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


# Base classes compatibility
FlextErrorMixin = FlextExceptions.Base.FlextErrorMixin

# Factory functions compatibility
create_context_exception_factory = FlextExceptions.create_context_exception_factory
create_module_exception_classes = FlextExceptions.create_module_exception_classes

# Global aliases for all dynamically generated exception classes
# (needed for external imports that expect global scope)
# MyPy limitation: Cannot detect dynamically created attributes
FlextError = FlextExceptions.FlextError  # type: ignore[attr-defined] # Dynamic class generation
FlextOperationError = FlextExceptions.FlextOperationError  # type: ignore[attr-defined] # Dynamic class generation
FlextProcessingError = FlextExceptions.FlextProcessingError  # type: ignore[attr-defined] # Dynamic class generation
FlextTimeoutError = FlextExceptions.FlextTimeoutError  # type: ignore[attr-defined] # Dynamic class generation
FlextNotFoundError = FlextExceptions.FlextNotFoundError  # type: ignore[attr-defined] # Dynamic class generation
FlextAlreadyExistsError = FlextExceptions.FlextAlreadyExistsError  # type: ignore[attr-defined] # Dynamic class generation
FlextPermissionError = FlextExceptions.FlextPermissionError  # type: ignore[attr-defined] # Dynamic class generation
FlextAuthenticationError = FlextExceptions.FlextAuthenticationError  # type: ignore[attr-defined] # Dynamic class generation
FlextTypeError = FlextExceptions.FlextTypeError  # type: ignore[attr-defined] # Dynamic class generation
FlextAttributeError = FlextExceptions.FlextAttributeError  # type: ignore[attr-defined] # Dynamic class generation
FlextCriticalError = FlextExceptions.FlextCriticalError  # type: ignore[attr-defined] # Dynamic class generation

# Global aliases for special exception classes (custom implementations)
FlextUserError = FlextExceptions.FlextUserError  # type: ignore[attr-defined] # Dynamic class generation
FlextValidationError = FlextExceptions.FlextValidationError  # type: ignore[attr-defined] # Dynamic class generation
FlextConfigurationError = FlextExceptions.FlextConfigurationError  # type: ignore[attr-defined] # Dynamic class generation
FlextConnectionError = FlextExceptions.FlextConnectionError  # type: ignore[attr-defined] # Dynamic class generation

# =============================================================================
# EXPORTS - Clean public API
# =============================================================================


__all__: list[str] = [
    # Dynamically generated exception classes
    "FlextAlreadyExistsError",
    "FlextAttributeError",
    "FlextAuthenticationError",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextCriticalError",
    "FlextError",
    "FlextErrorCodes",
    # Main hierarchical container
    "FlextExceptions",
    "FlextNotFoundError",
    "FlextOperationError",
    "FlextPermissionError",
    "FlextProcessingError",
    "FlextTimeoutError",
    "FlextTypeError",
    "FlextUserError",
    "FlextValidationError",
    # Factory functions
    "clear_exception_metrics",
    "create_context_exception_factory",
    "create_module_exception_classes",
    "get_exception_metrics",
]
