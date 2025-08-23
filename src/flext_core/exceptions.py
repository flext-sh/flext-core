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
from typing import TYPE_CHECKING, ClassVar, Self, cast, override

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

            @override
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

            # Build a dynamic class using type() to keep static checkers happy

            def _generated_init(self: object, message: str, **kwargs: object) -> None:
                # Store field-specific parameters as attributes
                for field_name in fields:
                    if field_name in kwargs:
                        setattr(self, field_name, kwargs[field_name])

                # Initialize base exception first
                base_exception.__init__(cast("Exception", self), message)

                # Initialize mixin with normalized parameters
                # Normalize error code inline
                resolved_code = (
                    kwargs.get("error_code") or kwargs.get("code") or default_code
                )
                if isinstance(resolved_code, Enum):
                    normalized_code = str(resolved_code.value)
                elif resolved_code is None:
                    normalized_code = FlextConstants.Errors.GENERIC_ERROR
                else:
                    normalized_code = str(resolved_code)

                cls.FlextErrorMixin.__init__(
                    cast("FlextExceptions.Base.FlextErrorMixin", self),
                    message,
                    code=normalized_code,
                    context=dict(kwargs),
                )

            def _generated_str(self: object) -> str:
                mixin_self = cast("FlextExceptions.Base.FlextErrorMixin", self)
                return cls.FlextErrorMixin.__str__(mixin_self)

            generated_class = type(
                name,
                (base_exception, cls.FlextErrorMixin),
                {
                    "__init__": _generated_init,
                    "__str__": _generated_str,
                    "error_code": property(
                        lambda self: str(
                            getattr(
                                self,
                                "code",
                                FlextConstants.Errors.GENERIC_ERROR,
                            )
                        )
                    ),
                },
            )

            # Set class attributes
            generated_class.__name__ = name
            generated_class.__qualname__ = name
            generated_class.__doc__ = (
                doc or f"{name} - {base_exception.__name__} with FLEXT context."
            )
            generated_class.__module__ = __name__
            # Class prepared

            return generated_class

        # Factory interface implementation will be added after class definition

    # =============================================================================
    # DRY EXCEPTION GENERATION (will be added after class definition)
    # =============================================================================

    # Exception specifications for dynamic generation - ALL EXCEPTIONS FOLLOW DRY PATTERN
    EXCEPTION_SPECS: ClassVar[
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
                # Use fallback for runtime access - classes may not exist at definition time
                try:
                    error_class = getattr(FlextExceptions, "FlextError", None)
                    if error_class is not None:
                        return cast(
                            "Exception",
                            error_class(f"{module_name}: {message}", **kwargs),
                        )
                except AttributeError:
                    pass
                # Fallback to standard exception if dynamic class not available
                return RuntimeError(f"{module_name}: {message}")

            @staticmethod
            def create_validation_error(message: str, **kwargs: object) -> Exception:
                # Use fallback for runtime access - classes may not exist at definition time
                try:
                    validation_error_class = getattr(
                        FlextExceptions, "FlextValidationError", None
                    )
                    if validation_error_class is not None:
                        return cast(
                            "Exception",
                            validation_error_class(
                                f"{module_name}: {message}", **kwargs
                            ),
                        )
                except AttributeError:
                    pass
                # Fallback to standard exception if dynamic class not available
                return ValueError(f"{module_name}: {message}")

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

    # =========================================================================
    # ERROR CODES - formerly FlextErrorCodes
    # =========================================================================

    class ErrorCodes:
        """Error code constants using FlextConstants for consistency."""

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

    # Abstract patterns removed - use dynamic factory instead


# =============================================================================
# DYNAMIC EXCEPTION GENERATION (DRY implementation)
# =============================================================================

# Generate dynamic exception classes using the factory pattern
for spec in FlextExceptions.EXCEPTION_SPECS:
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


# Error codes compatibility - facade for FlextExceptions.ErrorCodes
class FlextErrorCodes:
    """COMPATIBILITY FACADE: Use FlextExceptions.ErrorCodes instead.

    This class provides backward compatibility for existing code.
    All attributes delegate to FlextExceptions.ErrorCodes.

    DEPRECATED: Use FlextExceptions.ErrorCodes.[CODE] instead of FlextErrorCodes.[CODE]
    """

    GENERIC_ERROR = FlextExceptions.ErrorCodes.GENERIC_ERROR
    VALIDATION_ERROR = FlextExceptions.ErrorCodes.VALIDATION_ERROR
    CONFIGURATION_ERROR = FlextExceptions.ErrorCodes.CONFIGURATION_ERROR
    CONNECTION_ERROR = FlextExceptions.ErrorCodes.CONNECTION_ERROR
    AUTHENTICATION_ERROR = FlextExceptions.ErrorCodes.AUTHENTICATION_ERROR
    PERMISSION_ERROR = FlextExceptions.ErrorCodes.PERMISSION_ERROR
    NOT_FOUND = FlextExceptions.ErrorCodes.NOT_FOUND
    ALREADY_EXISTS = FlextExceptions.ErrorCodes.ALREADY_EXISTS
    TIMEOUT_ERROR = FlextExceptions.ErrorCodes.TIMEOUT_ERROR
    PROCESSING_ERROR = FlextExceptions.ErrorCodes.PROCESSING_ERROR
    CRITICAL_ERROR = FlextExceptions.ErrorCodes.CRITICAL_ERROR
    OPERATION_ERROR = FlextExceptions.ErrorCodes.OPERATION_ERROR
    UNWRAP_ERROR = FlextExceptions.ErrorCodes.UNWRAP_ERROR
    BUSINESS_ERROR = FlextExceptions.ErrorCodes.BUSINESS_ERROR
    INFRASTRUCTURE_ERROR = FlextExceptions.ErrorCodes.INFRASTRUCTURE_ERROR
    TYPE_ERROR = FlextExceptions.ErrorCodes.TYPE_ERROR


# Base classes compatibility
FlextErrorMixin = FlextExceptions.Base.FlextErrorMixin

# Factory functions compatibility
create_context_exception_factory = FlextExceptions.create_context_exception_factory
create_module_exception_classes = FlextExceptions.create_module_exception_classes

# Global aliases for all dynamically generated exception classes
# (needed for external imports that expect global scope)
# Type annotations for dynamic classes to help static analyzers
if TYPE_CHECKING:
    # Type annotations for static analyzers
    class FlextError(RuntimeError):
        """Base FLEXT error."""

        message: str
        code: str
        context: dict[str, object]
        correlation_id: str
        timestamp: float

        def __init__(self, message: str, **kwargs: object) -> None: ...

        @property
        def error_code(self) -> str: ...

    class FlextOperationError(RuntimeError):
        """Operation failed."""

        message: str
        code: str
        context: dict[str, object]
        correlation_id: str
        timestamp: float

        def __init__(self, message: str, **kwargs: object) -> None: ...

        @property
        def error_code(self) -> str: ...

    class FlextValidationError(ValueError):
        """Data validation failed."""

        message: str
        code: str
        context: dict[str, object]
        correlation_id: str
        timestamp: float

        def __init__(self, message: str, **kwargs: object) -> None: ...

        @property
        def error_code(self) -> str: ...

    class FlextConfigurationError(ValueError):
        """Configuration error."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextTypeError(TypeError):
        """Type error."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextAttributeError(AttributeError):
        """Attribute error."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextAlreadyExistsError(ValueError):
        """Resource already exists."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextAuthenticationError(RuntimeError):
        """Authentication failed."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextConnectionError(RuntimeError):
        """Connection failed."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextCriticalError(RuntimeError):
        """Critical system error."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextNotFoundError(ValueError):
        """Resource not found."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextPermissionError(PermissionError):
        """Permission denied."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextProcessingError(RuntimeError):
        """Processing failed."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextTimeoutError(TimeoutError):
        """Operation timed out."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

    class FlextUserError(ValueError):
        """User error."""

        def __init__(self, message: str, **kwargs: object) -> None: ...

else:
    # Runtime: Use dynamically created classes
    FlextError = FlextExceptions.FlextError
    FlextOperationError = FlextExceptions.FlextOperationError
    FlextValidationError = FlextExceptions.FlextValidationError
    FlextConfigurationError = FlextExceptions.FlextConfigurationError
    FlextTypeError = FlextExceptions.FlextTypeError
    FlextAttributeError = FlextExceptions.FlextAttributeError

    # Other dynamically generated classes
    FlextProcessingError = FlextExceptions.FlextProcessingError
    FlextTimeoutError = FlextExceptions.FlextTimeoutError
    FlextNotFoundError = FlextExceptions.FlextNotFoundError
    FlextAlreadyExistsError = FlextExceptions.FlextAlreadyExistsError
    FlextPermissionError = FlextExceptions.FlextPermissionError
    FlextAuthenticationError = FlextExceptions.FlextAuthenticationError
    FlextCriticalError = FlextExceptions.FlextCriticalError
    FlextUserError = FlextExceptions.FlextUserError
    FlextConnectionError = FlextExceptions.FlextConnectionError

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
