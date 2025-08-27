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
from typing import ClassVar, Self, cast, override

from flext_core.constants import FlextConstants

# =============================================================================
# FlextExceptions - Hierarchical Exception Management System
# =============================================================================


class FlextExceptions:
    """Hierarchical exception system with DRY factory pattern.

    Domains: Codes (error enums), Metrics (tracking), Base (mixins + factory).
    All exception classes generated dynamically via Base.create_exception_type().

    API Usage Patterns:
        Modern API (PRIMARY - use this):
            raise FlextExceptions.ValidationError("Invalid input")
            raise FlextExceptions.ConfigurationError("Missing key")

            except FlextExceptions.ValidationError as e:
                return FlextResult.fail(str(e))
            except FlextExceptions.ConfigurationError as e:
                return FlextResult.fail(str(e))

        Legacy API (backward compatibility only):
            except FlextExceptions.FlextValidationError as e:  # Still works
            from flext_core import FlextValidationError  # Root import for old code
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

                # Build context from field-specific parameters plus explicit context
                context_dict: dict[str, object] = {}

                # Add ALL kwargs as field parameters (except system parameters)
                system_params = ["code", "error_code", "context"]
                context_dict.update(
                    {
                        field_name: field_value
                        for field_name, field_value in kwargs.items()
                        if field_name not in system_params
                    }
                )

                # Handle explicit context parameter
                explicit_context = kwargs.get("context")
                if explicit_context is not None:
                    # Always nest explicit context under "context" key for consistency
                    # This matches the expectation in edge case tests
                    context_dict["context"] = explicit_context

                cls.FlextErrorMixin.__init__(
                    cast("FlextExceptions.Base.FlextErrorMixin", self),
                    message,
                    code=normalized_code,
                    context=context_dict,
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

    # =============================================================================
    # DYNAMIC CLASS GENERATION - Create exception classes from specifications
    # =============================================================================

    @classmethod
    def _generate_exception_classes(cls) -> None:
        """Generate all exception classes from EXCEPTION_SPECS."""
        for name, base_exception, default_code, doc, fields in cls.EXCEPTION_SPECS:
            # Create the dynamic class ONCE
            exception_class = cls.Base.create_exception_type(
                name, base_exception, default_code, doc, fields
            )

            # Set BOTH legacy and modern names to the SAME class object
            setattr(cls, name, exception_class)  # Legacy name (with Flext prefix)

            if name.startswith("Flext"):
                modern_name = name[5:]  # Remove "Flext" prefix
                setattr(
                    cls, modern_name, exception_class
                )  # Modern name (without prefix)

    # All exception classes are now generated dynamically via DRY factory pattern
    # Special cases with complex logic can be added here if needed

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

    # =============================================================================
    # TYPE DECLARATIONS FOR MYPY - Declare exception class attributes
    # =============================================================================

    # Modern API (without Flext prefix)
    Error: ClassVar[type[Exception]]
    UserError: ClassVar[type[Exception]]
    ValidationError: ClassVar[type[Exception]]
    ConfigurationError: ClassVar[type[Exception]]
    ConnectionError: ClassVar[type[Exception]]
    AuthenticationError: ClassVar[type[Exception]]
    PermissionError: ClassVar[type[Exception]]
    OperationError: ClassVar[type[Exception]]
    ProcessingError: ClassVar[type[Exception]]
    TimeoutError: ClassVar[type[Exception]]
    NotFoundError: ClassVar[type[Exception]]
    AlreadyExistsError: ClassVar[type[Exception]]
    CriticalError: ClassVar[type[Exception]]
    TypeError: ClassVar[type[Exception]]
    AttributeError: ClassVar[type[Exception]]

    # Legacy API (with Flext prefix)
    FlextError: ClassVar[type[Exception]]
    FlextUserError: ClassVar[type[Exception]]
    FlextValidationError: ClassVar[type[Exception]]
    FlextConfigurationError: ClassVar[type[Exception]]
    FlextConnectionError: ClassVar[type[Exception]]
    FlextAuthenticationError: ClassVar[type[Exception]]
    FlextPermissionError: ClassVar[type[Exception]]
    FlextOperationError: ClassVar[type[Exception]]
    FlextProcessingError: ClassVar[type[Exception]]
    FlextTimeoutError: ClassVar[type[Exception]]
    FlextNotFoundError: ClassVar[type[Exception]]
    FlextAlreadyExistsError: ClassVar[type[Exception]]
    FlextCriticalError: ClassVar[type[Exception]]
    FlextTypeError: ClassVar[type[Exception]]
    FlextAttributeError: ClassVar[type[Exception]]

    # =============================================================================
    # DYNAMIC CLASS INITIALIZATION
    # =============================================================================

    @classmethod
    def initialize(cls) -> None:
        """Initialize all dynamic exception classes."""
        cls._generate_exception_classes()

    @staticmethod
    def _get_exception_class(flext_class_name: str) -> type[Exception]:
        """Get a dynamically generated exception class from FlextExceptions."""
        return cast("type[Exception]", getattr(FlextExceptions, flext_class_name))

    # =============================================================================
    # MODERN API - Direct class aliases (no factory methods)
    # =============================================================================

    # Note: Factory methods have been removed in favor of direct class access.
    # Both FlextExceptions.ValidationError and FlextExceptions.FlextValidationError
    # now refer to the same exception CLASS, not a factory function.
    # This allows natural usage in both raise and except statements.
    #
    # The modern API aliases are created dynamically after initialization.
    # See the initialization code below for details.

    # ==========================================================================
    # METRICS MANAGEMENT METHODS
    # ==========================================================================

    # Class-level metrics storage
    _metrics: ClassVar[dict[str, object]] = {}

    @classmethod
    def get_metrics(cls) -> dict[str, object]:
        """Get current exception metrics.

        Returns:
            Dictionary containing exception metrics data

        """
        return cls._metrics.copy()

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics data."""
        cls._metrics.clear()

    @classmethod
    def record_metric(cls, key: str, value: object) -> None:
        """Record an exception metric.

        Args:
            key: Metric key identifier
            value: Metric value to store

        """
        cls._metrics[key] = value


# =============================================================================
# DYNAMIC CLASS INITIALIZATION
# =============================================================================

# Initialize all dynamic exception classes
# This also creates the modern API aliases
FlextExceptions.initialize()

# =============================================================================
# TYPE ANNOTATIONS FOR DYNAMIC EXCEPTIONS - Enable static type checking
# =============================================================================

# Add __annotations__ to help static type checkers understand dynamic attributes
# This approach works better with pyright than TYPE_CHECKING blocks
FlextExceptions.__annotations__ = {
    # Modern API (preferred - without Flext prefix)
    "Error": "type[Exception]",
    "UserError": "type[Exception]",
    "ValidationError": "type[Exception]",
    "ConfigurationError": "type[Exception]",
    "ConnectionError": "type[Exception]",
    "AuthenticationError": "type[Exception]",
    "PermissionError": "type[Exception]",
    "OperationError": "type[Exception]",
    "ProcessingError": "type[Exception]",
    "TimeoutError": "type[Exception]",
    "NotFoundError": "type[Exception]",
    "AlreadyExistsError": "type[Exception]",
    "CriticalError": "type[Exception]",
    "TypeError": "type[Exception]",
    "AttributeError": "type[Exception]",
    # Legacy API (backward compatibility - with Flext prefix)
    "FlextError": "type[Exception]",
    "FlextUserError": "type[Exception]",
    "FlextValidationError": "type[Exception]",
    "FlextConfigurationError": "type[Exception]",
    "FlextConnectionError": "type[Exception]",
    "FlextAuthenticationError": "type[Exception]",
    "FlextPermissionError": "type[Exception]",
    "FlextOperationError": "type[Exception]",
    "FlextProcessingError": "type[Exception]",
    "FlextTimeoutError": "type[Exception]",
    "FlextNotFoundError": "type[Exception]",
    "FlextAlreadyExistsError": "type[Exception]",
    "FlextCriticalError": "type[Exception]",
    "FlextTypeError": "type[Exception]",
    "FlextAttributeError": "type[Exception]",
}


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================


__all__: list[str] = [
    # Main hierarchical container - ONLY access point
    "FlextExceptions",
]
