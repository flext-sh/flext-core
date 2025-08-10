"""Enterprise-grade exception hierarchy for FLEXT.

Provides structured error handling, context management, and
cross-service serialization for distributed data integration
pipelines following SOLID principles.

Concrete implementations of abstract exception patterns from base_exceptions.py.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING

from flext_core.base_exceptions import (
    FlextAbstractBusinessError,
    FlextAbstractConfigurationError,
    FlextAbstractError,
    FlextAbstractErrorFactory,
    FlextAbstractInfrastructureError,
    FlextAbstractValidationError,
)
from flext_core.constants import ERROR_CODES

if TYPE_CHECKING:
    from flext_core.typings import TAnyDict


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
            context = dict(kwargs)
            if config_key is not None:
                context["config_key"] = config_key
            super().__init__(
                f"{module_name} config: {message}",
                config_key=config_key,
                **context,
            )

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
            context = dict(kwargs)
            if service_name is not None:
                context["service_name"] = service_name
            if endpoint is not None:
                context["endpoint"] = endpoint
            super().__init__(
                f"{module_name} connection: {message}",
                service=f"{module_name}_connection",
                **context,
            )

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
            context = dict(kwargs)
            if operation is not None:
                context["operation"] = operation
            if file_path is not None:
                context["file_path"] = file_path
            super().__init__(
                f"{module_name} processing: {message}",
                business_rule="processing",
                **context,
            )

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
            context = dict(kwargs)
            if username is not None:
                context["username"] = username
            if auth_method is not None:
                context["auth_method"] = auth_method
            super().__init__(
                f"{module_name}: {message}",
                service=f"{module_name}_auth",
                **context,
            )

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
            context = dict(kwargs)
            if timeout_duration is not None:
                context["timeout_duration"] = timeout_duration
            if operation is not None:
                context["operation"] = operation
            super().__init__(
                f"{module_name}: {message}",
                service=f"{module_name}_timeout",
                **context,
            )

    return ModuleTimeoutError


def _get_module_prefix(module_name: str) -> str:
    """Get module prefix for class naming."""
    return "".join(
        word.capitalize() for word in module_name.replace("_", "-").split("-")
    )


def create_context_exception_factory(
    error_prefix: str,
    default_message: str,
    **default_context: object,
) -> dict[str, object]:
    """Create DRY context factory for exception context building.

    SOLID Factory Pattern: Eliminates 18-line duplication in exception __init__ methods.
    This pattern eliminates the repetitive context building code found in 85+ locations.

    Args:
        error_prefix: Prefix for error message (e.g., "API request")
        default_message: Default message for the exception
        **default_context: Default context keys with their parameter names

    Returns:
        Dictionary with factory configuration.

    """
    return {
        "error_prefix": error_prefix,
        "default_message": default_message,
        "default_context": default_context,
    }


def create_module_exception_classes(module_name: str) -> dict[str, type]:
    """Create DRY module-specific exception classes to eliminate duplication.

    This function creates a complete set of module-specific exception classes that
    eliminate the need for duplicate exception code across FLEXT modules. Each module
    gets its own exception hierarchy while using the shared FlextError foundation.

    This pattern eliminates 85+ locations of duplicated exception code (mass=101).

    Args:
        module_name: Name of the module (e.g., "flext_auth", "client-a_oud_mig")

    Returns:
        Dictionary of exception classes ready for use.

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


class FlextError(FlextAbstractError):
    """Base exception for all FLEXT Core errors - implements FlextAbstractError.

    Provides structured error information with context, error codes,
    and cross-service serialization capabilities following SOLID principles.
    """

    def __init__(
        self,
        message: str = "An error occurred",
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize error with context and debugging info.

        Args:
            message: Error message
            error_code: Error code for categorization
            context: Additional context information

        """
        # Convert Mapping to dict for abstract base class
        context_dict = dict(context) if context else {}
        super().__init__(message, error_code, context_dict)

        # Additional FlextError-specific attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()

        # Safely limit context size to prevent memory issues
        max_size = 1000
        if self.context and len(str(self.context)) > max_size:
            self.context = {
                "_truncated": True,
                "_original_size": len(str(self.context)),
            }

    @property
    def error_category(self) -> str:
        """Get error category - implements abstract method."""
        return "GENERAL"

    def _get_default_error_code(self) -> str:
        """Get default error code - implements abstract method."""
        return ERROR_CODES["GENERIC_ERROR"]

    def __repr__(self) -> str:
        """Return detailed error representation."""
        return (
            f"{self.__class__.__name__}"
            f"(message='{self.message}', code='{self.error_code}')"
        )

    def to_dict(self) -> TAnyDict:
        """Convert exception to dictionary - implements abstract method.

        Returns:
            Dictionary suitable for JSON serialization

        """
        return {
            "error_type": "FlextError",
            "error_code": self.error_code,
            "error_category": self.error_category,
            "message": self.message,
            "context": self._sanitize_context(self.context),
            "timestamp": getattr(self, "timestamp", time.time()),
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "serialization_version": "1.0",
        }

    def _sanitize_context(self, context: dict[str, object]) -> dict[str, object]:
        """Sanitize context for safe serialization.

        Args:
            context: Original context dictionary

        Returns:
            Sanitized context safe for serialization

        """
        if not context:
            return {}

        # Constants for sanitization thresholds
        max_list_size = 10
        max_dict_size = 20

        sanitized: dict[str, object] = {}
        sensitive_keys = {
            "password",
            "token",
            "secret",
            "key",
            "auth",
            "credential",
            "api_key",
        }
        max_value_length = 500  # Prevent huge values in logs

        for key, value in context.items():
            # Skip sensitive keys
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
                continue

            # Truncate large values
            if isinstance(value, str) and len(value) > max_value_length:
                sanitized[key] = value[:max_value_length] + "... [TRUNCATED]"
            elif isinstance(value, (list, tuple)) and len(value) > max_list_size:
                sanitized[key] = (
                    f"[{type(value).__name__} with {len(value)} items - TRUNCATED]"
                )
            elif isinstance(value, dict) and len(value) > max_dict_size:
                sanitized[key] = f"[Dict with {len(value)} keys - TRUNCATED]"
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def from_dict(cls, error_dict: dict[str, object]) -> FlextError:
        """Reconstruct exception from dictionary.

        Args:
            error_dict: Serialized error dictionary from to_dict()

        Returns:
            Reconstructed FlextError instance

        Raises:
            FlextValidationError: If error_dict is malformed

        """
        # Validate required fields
        required_fields = {"type", "message", "error_code"}
        missing_fields = required_fields - set(error_dict.keys())
        if missing_fields:
            msg = f"Invalid error dictionary: missing fields {missing_fields}"
            raise FlextValidationError(
                msg,
                validation_details={"missing_fields": list(missing_fields)},
            )

        # Extract fields with defaults
        error_type = str(error_dict["type"])
        message = str(error_dict["message"])
        error_code = str(error_dict["error_code"])
        context_data = error_dict.get("context", {})
        context = dict(context_data) if isinstance(context_data, dict) else {}
        timestamp = error_dict.get("timestamp")

        # Attempt to resolve the correct exception class
        exception_class = cls._resolve_exception_class(error_type)

        # Create instance with appropriate constructor
        try:
            # Try to create with the specific class
            if exception_class != FlextError:
                return exception_class(
                    message=message,
                    error_code=error_code,
                    context=context,
                )
            # Use base FlextError as the final constructor
            instance = cls(message=message, error_code=error_code, context=context)
            # Preserve original timestamp if available
            if timestamp and isinstance(timestamp, (int, float)):
                instance.timestamp = float(timestamp)
            return instance

        except Exception as e:
            # REAL SOLUTION: Proper reconstruction error handling
            enhanced_context = {
                **context,
                "original_type": error_type,
                "reconstruction_error": str(e),
            }
            error_msg = f"Failed to reconstruct {error_type}: {e}"
            raise FlextValidationError(
                error_msg,
                validation_details={"original_type": error_type},
                context=enhanced_context,
            ) from e

    @staticmethod
    def _resolve_exception_class(error_type: str) -> type[FlextError]:
        """Resolve exception class from type name.

        Args:
            error_type: Exception class name

        Returns:
            Exception class with strict type resolution

        """
        # Map of known exception types (will be populated at module level)
        exception_mapping = {
            "FlextError": FlextError,
            "FlextValidationError": FlextValidationError,
            "FlextTypeError": FlextTypeError,
            "FlextOperationError": FlextOperationError,
            "FlextConfigurationError": FlextConfigurationError,
            "FlextConnectionError": FlextConnectionError,
            "FlextAuthenticationError": FlextAuthenticationError,
            "FlextPermissionError": FlextPermissionError,
            "FlextNotFoundError": FlextNotFoundError,
            "FlextAlreadyExistsError": FlextAlreadyExistsError,
            "FlextTimeoutError": FlextTimeoutError,
            "FlextProcessingError": FlextProcessingError,
            "FlextCriticalError": FlextCriticalError,
        }

        return exception_mapping.get(error_type, FlextError)  # type: ignore[return-value]


class FlextValidationError(FlextAbstractValidationError, FlextError):
    """Validation failure exception with field-specific context.

    Specialized exception for validation failures providing detailed information
    about field validation errors, failed rules, and problematic values.
    Implements FlextAbstractValidationError.
    """

    def __init__(
        self,
        message: str = "Validation failed",
        *,
        validation_details: Mapping[str, object] | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize validation error with validation details."""
        # Extract validation details
        details = validation_details or {}
        field = details.get("field")

        # Build enhanced context
        enhanced_context = dict(context) if context else {}
        if details:
            enhanced_context.update(details)

        # Initialize abstract validation error first
        FlextAbstractValidationError.__init__(
            self,
            message,
            field=str(field) if field else None,
            validation_details=dict(validation_details) if validation_details else None,
            error_code=error_code or ERROR_CODES["VALIDATION_ERROR"],
            context=enhanced_context,
        )

        # Initialize FlextError attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()

    @property
    def rules(self) -> list[str] | None:
        """Get validation rules from validation_details."""
        rules = (
            self.validation_details.get("rules") if self.validation_details else None
        )
        if isinstance(rules, list):
            return rules
        return None

    @property
    def value(self) -> object | None:
        """Get value from validation_details."""
        return self.validation_details.get("value") if self.validation_details else None


class FlextTypeError(FlextError):
    """Type mismatch exception with expected and actual type information.

    Specialized exception for type-related errors providing clear information
    about expected versus actual types for debugging and error resolution.
    """

    def __init__(
        self,
        message: str = "Type error occurred",
        *,
        expected_type: type | str | None = None,
        actual_type: type | str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize type error with type information."""
        self.expected_type = expected_type
        self.actual_type = actual_type

        # Build enhanced context
        enhanced_context = dict(context) if context else {}
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
    """

    def __init__(
        self,
        message: str = "Attribute error occurred",
        *,
        attribute_context: Mapping[str, object] | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize attribute error with attribute context."""
        # Build enhanced context
        enhanced_context = dict(context) if context else {}
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
    """

    def __init__(
        self,
        message: str = "Operation failed",
        *,
        operation: str | None = None,
        stage: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize operation error with operation details."""
        self.operation = operation
        self.stage = stage

        # Build enhanced context
        enhanced_context = dict(context) if context else {}
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


class FlextConfigurationError(FlextAbstractConfigurationError, FlextError):
    """Configuration-related errors with context capture.

    Implements FlextAbstractConfigurationError.
    """

    def __init__(
        self,
        message: str = "Configuration error",
        config_key: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize configuration error with context.

        Args:
            message: Descriptive error message
            config_key: Configuration key that failed
            **kwargs: Additional context information

        """
        FlextAbstractConfigurationError.__init__(
            self,
            message,
            config_key=config_key,
            error_code=ERROR_CODES["CONFIG_ERROR"],
            **kwargs,
        )

        # Initialize FlextError attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()


class FlextConnectionError(FlextAbstractInfrastructureError, FlextError):
    """Connection-related errors with network context.

    Implements FlextAbstractInfrastructureError for connection issues.
    """

    def __init__(
        self,
        message: str = "Connection error",
        service: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize connection error with context.

        Args:
            message: Descriptive error message
            service: Service name for connection
            **kwargs: Additional context information (host, port, etc.)

        """
        FlextAbstractInfrastructureError.__init__(
            self,
            message,
            service=service,
            error_code=ERROR_CODES["CONNECTION_ERROR"],
            **kwargs,
        )

        # Initialize FlextError attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()


class FlextAuthenticationError(FlextAbstractInfrastructureError, FlextError):
    """Authentication-related errors with security context.

    Implements FlextAbstractInfrastructureError for auth services.
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        service: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize authentication error with context.

        Args:
            message: Descriptive error message
            service: Authentication service name
            **kwargs: Additional context information (user, method, etc.)

        """
        FlextAbstractInfrastructureError.__init__(
            self,
            message,
            service=service or "authentication",
            error_code=ERROR_CODES["AUTH_ERROR"],
            **kwargs,
        )

        # Initialize FlextError attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()


class FlextPermissionError(FlextAbstractInfrastructureError, FlextError):
    """Permission-related errors with authorization context.

    Implements FlextAbstractInfrastructureError for authorization services.
    """

    def __init__(
        self,
        message: str = "Permission denied",
        service: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize permission error with context.

        Args:
            message: Descriptive error message
            service: Authorization service name
            **kwargs: Additional context information (resource, action, etc.)

        """
        FlextAbstractInfrastructureError.__init__(
            self,
            message,
            service=service or "authorization",
            error_code=ERROR_CODES["PERMISSION_ERROR"],
            **kwargs,
        )

        # Initialize FlextError attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()


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


class FlextTimeoutError(FlextAbstractInfrastructureError, FlextError):
    """Timeout-related errors with timing context.

    Implements FlextAbstractInfrastructureError for timeout services.
    """

    def __init__(
        self,
        message: str = "Operation timed out",
        service: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize timeout error with context.

        Args:
            message: Descriptive error message
            service: Service that timed out
            **kwargs: Additional context information (timeout, duration, etc.)

        """
        FlextAbstractInfrastructureError.__init__(
            self,
            message,
            service=service or "timeout_service",
            error_code=ERROR_CODES["TIMEOUT_ERROR"],
            **kwargs,
        )

        # Initialize FlextError attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()


class FlextProcessingError(FlextAbstractBusinessError, FlextError):
    """Processing-related errors with operation context.

    Implements FlextAbstractBusinessError for processing operations.
    """

    def __init__(
        self,
        message: str = "Processing failed",
        business_rule: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize processing error with context.

        Args:
            message: Descriptive error message
            business_rule: Business rule that failed
            **kwargs: Additional context information (data, stage, etc.)

        """
        FlextAbstractBusinessError.__init__(
            self,
            message,
            business_rule=business_rule or "data_processing",
            error_code=ERROR_CODES["PROCESSING_ERROR"],
            **kwargs,
        )

        # Initialize FlextError attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()


class FlextCriticalError(FlextAbstractInfrastructureError, FlextError):
    """Critical system errors requiring immediate attention.

    Implements FlextAbstractInfrastructureError for critical system issues.
    """

    def __init__(
        self,
        message: str = "Critical error",
        service: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize critical error with context.

        Args:
            message: Descriptive error message
            service: Critical service that failed
            **kwargs: Additional context information (system, component, etc.)

        """
        FlextAbstractInfrastructureError.__init__(
            self,
            message,
            service=service or "critical_system",
            error_code=ERROR_CODES["CRITICAL_ERROR"],
            **kwargs,
        )

        # Initialize FlextError attributes
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()


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


class FlextExceptions(FlextAbstractErrorFactory):
    """Unified factory interface for creating FLEXT exceptions.

    Provides convenient factory methods for creating all types of FLEXT exceptions
    with appropriate default context and error codes.
    Implements FlextAbstractErrorFactory.
    """

    @classmethod
    def create_validation_error(
        cls,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractValidationError:
        """Create validation error - implements abstract method."""
        field_obj = kwargs.pop("field", None)
        field = str(field_obj) if field_obj is not None else None
        value = kwargs.pop("value", None)
        rules_obj = kwargs.pop("rules", None)
        rules = list(rules_obj) if isinstance(rules_obj, (list, tuple)) else None
        context_obj = kwargs.pop("context", None)
        context = dict(context_obj) if isinstance(context_obj, Mapping) else None
        return cls.create_validation_error_with_field(
            message,
            field=field,
            value=value,
            rules=rules,
            context=context,
        )

    @staticmethod
    def create_business_error(
        message: str,
        **kwargs: object,
    ) -> FlextAbstractBusinessError:
        """Create business error - implements abstract method."""
        business_rule_obj = kwargs.pop("business_rule", None)
        business_rule = (
            str(business_rule_obj) if business_rule_obj is not None else None
        )
        return FlextProcessingError(message, business_rule=business_rule, **kwargs)

    @staticmethod
    def create_infrastructure_error(
        message: str,
        **kwargs: object,
    ) -> FlextAbstractInfrastructureError:
        """Create infrastructure error - implements abstract method."""
        service_obj = kwargs.pop("service", None)
        service = str(service_obj) if service_obj is not None else None
        return FlextConnectionError(message, service=service, **kwargs)

    @staticmethod
    def create_configuration_error(
        message: str,
        **kwargs: object,
    ) -> FlextAbstractConfigurationError:
        """Create configuration error - implements abstract method."""
        config_key_obj = kwargs.pop("config_key", None)
        config_key = str(config_key_obj) if config_key_obj is not None else None

        # Add config_key to context if provided
        if config_key is not None:
            raw_ctx = kwargs.get("context")
            context: dict[str, object]
            context = dict(raw_ctx) if isinstance(raw_ctx, dict) else {}
            context["config_key"] = config_key
            kwargs["context"] = context

        return FlextConfigurationError(message, config_key=config_key, **kwargs)

    @staticmethod
    def create_validation_error_with_field(
        message: str,
        *,
        field: str | None = None,
        value: object = None,
        rules: list[str] | None = None,
        context: Mapping[str, object] | None = None,
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
        context: Mapping[str, object] | None = None,
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
        context: Mapping[str, object] | None = None,
    ) -> FlextOperationError:
        """Create operation error with operation context."""
        return FlextOperationError(
            message,
            operation=operation_name,
            context=context,
        )

    @staticmethod
    def create_connection_error(
        message: str,
        *,
        endpoint: str | None = None,
        context: Mapping[str, object] | None = None,
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

__all__: list[str] = [
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
    "create_context_exception_factory",
    "create_module_exception_classes",
    "get_exception_metrics",
]
