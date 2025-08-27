"""FLEXT exception hierarchy with clean subclass approach.

Hierarchical exception system with real subclasses:
- FlextExceptions: Main container with Codes, Metrics, and real exception classes
- Clean subclasses: Each exception type has its own class with proper signatures
- Type Safety: Full mypy/pyright support without dynamic generation
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import ClassVar, cast

from flext_core.constants import FlextConstants

# =============================================================================
# FlextExceptions - Hierarchical Exception Management System
# =============================================================================


class FlextExceptions:
    """Hierarchical exception system with clean subclass approach.

    All exception classes are real subclasses with proper type signatures.

    Usage:
        raise FlextExceptions.AttributeError("Invalid field", attribute_name="test")
        raise FlextExceptions("Failed", operation="create")
        raise FlextExceptions("Invalid", field="name", value="")
    """

    # =============================================================================
    # Metrics Domain: Exception metrics and monitoring functionality
    # =============================================================================

    class Metrics:
        """Exception metrics tracking (singleton pattern)."""

        _metrics: ClassVar[dict[str, int]] = {}

        @classmethod
        def record_exception(cls, exception_type: str) -> None:
            """Record exception occurrence for metrics tracking."""
            cls._metrics[exception_type] = cls._metrics.get(exception_type, 0) + 1

        @classmethod
        def get_metrics(cls) -> dict[str, int]:
            """Get current exception metrics."""
            return dict(cls._metrics)

        @classmethod
        def clear_metrics(cls) -> None:
            """Clear all exception metrics."""
            cls._metrics.clear()

    # =============================================================================
    # BASE EXCEPTION CLASS - Clean hierarchical approach
    # =============================================================================

    class FlextExceptionBaseError(Exception):
        """Base class for all FLEXT exceptions with common functionality."""

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

    class _AttributeError(FlextExceptionBaseError, AttributeError):
        """Attribute access failed."""

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

    class _OperationError(FlextExceptionBaseError, RuntimeError):
        """Operation failed."""

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

    class _ValidationError(FlextExceptionBaseError, ValueError):
        """Data validation failed."""

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

    class _ConfigurationError(FlextExceptionBaseError, ValueError):
        """Configuration error."""

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

    class _ConnectionError(FlextExceptionBaseError, ConnectionError):
        """Network/service connection failed."""

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

    class _ProcessingError(FlextExceptionBaseError, RuntimeError):
        """Processing failed."""

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

    class _TimeoutError(FlextExceptionBaseError, TimeoutError):
        """Operation timed out."""

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

    class _NotFoundError(FlextExceptionBaseError, FileNotFoundError):
        """Resource not found."""

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

    class _AlreadyExistsError(FlextExceptionBaseError, FileExistsError):
        """Resource already exists."""

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

    class _PermissionError(FlextExceptionBaseError, PermissionError):
        """Insufficient permissions."""

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

    class _AuthenticationError(FlextExceptionBaseError, PermissionError):
        """Authentication failed."""

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

    class _TypeError(FlextExceptionBaseError, TypeError):
        """Type validation failed."""

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

    class _CriticalError(FlextExceptionBaseError, SystemError):
        """System critical error."""

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(
                message,
                code=FlextConstants.Errors.CRITICAL_ERROR,
                context=cast("Mapping[str, object] | None", kwargs.get("context")),
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _Error(FlextExceptionBaseError, RuntimeError):
        """Base FLEXT error."""

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(
                message,
                code=FlextConstants.Errors.GENERIC_ERROR,
                context=cast("Mapping[str, object] | None", kwargs.get("context")),
                correlation_id=cast("str | None", kwargs.get("correlation_id")),
            )

    class _UserError(FlextExceptionBaseError, TypeError):
        """User input/usage error."""

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
    # DIRECT CALLABLE INTERFACE - For general usage
    # =============================================================================

    def __new__(
        cls,
        message: str,
        *,
        operation: str | None = None,
        field: str | None = None,
        config_key: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> FlextExceptionBaseError:
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
                correlation_id=correlation_id
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
                correlation_id=correlation_id
            )
        if config_key is not None:
            config_file = cast("str | None", kwargs.get("config_file"))
            return cls._ConfigurationError(
                message,
                config_key=config_key,
                config_file=config_file,
                code=error_code,
                context=context,
                correlation_id=correlation_id
            )
        # Default to general error
        return cls._Error(message, code=error_code, context=context, correlation_id=correlation_id)

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
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    # Main hierarchical container - ONLY access point
    "FlextExceptions",
]
