"""Exception hierarchy for FLEXT ecosystem using modern Pydantic patterns."""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum, StrEnum
from typing import ClassVar, Self, cast

# Module-level logger (avoid using root logger)
logger = logging.getLogger(__name__)


class FlextErrorCodes(StrEnum):
    """Error codes for FLEXT exceptions."""

    GENERIC_ERROR = "FLEXT_GENERIC_ERROR"
    VALIDATION_ERROR = "FLEXT_VALIDATION_ERROR"
    TYPE_ERROR = "FLEXT_TYPE_ERROR"
    CONFIGURATION_ERROR = "FLEXT_CONFIG_ERROR"
    CONNECTION_ERROR = "FLEXT_CONNECTION_ERROR"
    AUTHENTICATION_ERROR = "FLEXT_AUTH_ERROR"
    PERMISSION_ERROR = "FLEXT_PERMISSION_ERROR"
    NOT_FOUND = "FLEXT_NOT_FOUND"
    ALREADY_EXISTS = "FLEXT_ALREADY_EXISTS"
    TIMEOUT_ERROR = "FLEXT_TIMEOUT_ERROR"
    PROCESSING_ERROR = "FLEXT_PROCESSING_ERROR"
    CRITICAL_ERROR = "FLEXT_CRITICAL_ERROR"
    OPERATION_ERROR = "FLEXT_OPERATION_ERROR"
    UNWRAP_ERROR = "FLEXT_UNWRAP_ERROR"
    BUSINESS_ERROR = "FLEXT_BUSINESS_ERROR"
    INFRASTRUCTURE_ERROR = "FLEXT_INFRASTRUCTURE_ERROR"


class FlextExceptionMetrics:
    """Singleton class for tracking exception metrics."""

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


# Global instance for easy access
_exception_metrics = FlextExceptionMetrics()


class FlextErrorMixin:
    """Common functionality shared by all FLEXT-specific errors.

    Args:
      message: A message describing the error
      code: Optional error code (enum or string). Normalized to string.
      error_code: Alias for ``code``; if provided, takes precedence.
      context: Additional context information
      correlation_id: Correlation ID for tracking

    Follows a modern, Pydantic-style structured error pattern.

    """

    def __init__(
      self,
      message: str,
      *,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
      correlation_id: str | None = None,
    ) -> None:
      """Initialize FLEXT error mixin.

      Args:
          message: A message describing the error
          code: Optional error code (enum or string)
          error_code: Optional alias for ``code`` (enum or string)
          context: Additional context information
          correlation_id: Correlation ID for tracking

      """
      self.message = message
      # Normalize error code from various enum/string types
      resolved_code = error_code if error_code is not None else code
      if isinstance(resolved_code, Enum):
          code_str = str(resolved_code.value)
      elif resolved_code is None:
          code_str = FlextErrorCodes.GENERIC_ERROR.value
      else:
          code_str = str(resolved_code)
      # Store normalized string code for universal compatibility
      self.code = code_str
      self.context = dict(context or {})
      self.correlation_id = correlation_id or f"flext_{int(time.time() * 1000)}"
      self.timestamp = time.time()
      self.stack_trace = traceback.format_stack()

      # Record metrics
      _exception_metrics.record_exception(self.__class__.__name__)

    def __str__(self) -> str:
      """Return human-readable error message."""
      return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, object]:
      """Convert error to dictionary for serialization."""
      return {
          "error_type": self.__class__.__name__,
          "code": self.code,
          "message": self.message,
          "context": self.context,
          "correlation_id": self.correlation_id,
          "timestamp": self.timestamp,
      }

    # Backward-compatibility accessor expected by some code
    @property
    def error_code(self) -> str:
      """Return normalized string error code."""
      return str(self.code)


class FlextUserError(FlextErrorMixin, TypeError):
    """An error raised due to incorrect use of FLEXT.

    Follows Pydantic's PydanticUserError pattern.
    """

    def __init__(
      self,
      message: str,
      *,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize user error."""
      FlextErrorMixin.__init__(
          self,
          message,
          code=code,
          error_code=error_code,
          context=context,
      )
      TypeError.__init__(self, message)


class FlextValidationError(FlextErrorMixin, ValueError):
    """A validation error raised when input validation fails.

    Follows Pydantic's validation error pattern.
    """

    def __init__(
      self,
      message: str,
      *,
      field: str | None = None,
      value: object = None,
      code: object | None = None,
      error_code: object | None = None,
      validation_details: Mapping[str, object] | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize validation error."""
      base_context: dict[str, object] = dict(context or {})
      if field is not None:
          base_context["field"] = field
      if value is not None:
          base_context["value"] = str(value)[:100]  # Truncate for safety
      if validation_details is not None:
          base_context.update(dict(validation_details))
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.VALIDATION_ERROR,
          error_code=error_code,
          context=base_context,
      )
      ValueError.__init__(self, message)

      self.field = field
      self.value = value


class FlextConfigurationError(FlextErrorMixin, ValueError):
    """Configuration-related error."""

    def __init__(
      self,
      message: str,
      *,
      config_key: str | None = None,
      config_file: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize configuration error."""
      base_context: dict[str, object] = dict(context or {})
      if config_key is not None:
          base_context["config_key"] = config_key
      if config_file is not None:
          base_context["config_file"] = config_file
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.CONFIGURATION_ERROR,
          error_code=error_code,
          context=base_context,
      )
      ValueError.__init__(self, message)

      self.config_key = config_key
      self.config_file = config_file


class FlextConnectionError(FlextErrorMixin, ConnectionError):
    """Connection-related error."""

    def __init__(
      self,
      message: str,
      *,
      service: str | None = None,
      endpoint: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize connection error."""
      base_context: dict[str, object] = dict(context or {})
      if service is not None:
          base_context["service"] = service
      if endpoint is not None:
          base_context["endpoint"] = endpoint
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.CONNECTION_ERROR,
          error_code=error_code,
          context=base_context,
      )
      ConnectionError.__init__(self, message)

      self.service = service
      self.endpoint = endpoint


class FlextNotFoundError(FlextErrorMixin, ValueError):
    """Resource not found error."""

    def __init__(
      self,
      message: str,
      *,
      resource_type: str | None = None,
      resource_id: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize not found error."""
      base_context: dict[str, object] = dict(context or {})
      if resource_type is not None:
          base_context["resource_type"] = resource_type
      if resource_id is not None:
          base_context["resource_id"] = resource_id
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.NOT_FOUND,
          error_code=error_code,
          context=base_context,
      )
      ValueError.__init__(self, message)

      self.resource_type = resource_type
      self.resource_id = resource_id


class FlextError(FlextErrorMixin, Exception):
    """Base exception for all FLEXT operations.

    Modern Pydantic-style error with structured error handling, context,
    and cross-service compatibility for distributed systems.
    """

    def __init__(
      self,
      message: str = "FLEXT operation failed",
      *,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize FLEXT error with rich context."""
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.GENERIC_ERROR,
          error_code=error_code,
          context=context,
      )
      Exception.__init__(self, message)

    def __repr__(self) -> str:
      """Return technical representation."""
      return (
          f"{self.__class__.__name__}("
          f"message='{self.message}', "
          f"code='{self.error_code}', "
          f"correlation_id='{self.correlation_id}'"
          f")"
      )


class FlextTypeError(FlextErrorMixin, TypeError):
    """Type-related errors for type validation."""

    def __init__(
      self,
      message: str = "Type validation failed",
      *,
      expected_type: object | None = None,
      actual_type: object | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize type error."""
      base_context: dict[str, object] = dict(context or {})
      if expected_type is not None:
          base_context["expected_type"] = str(expected_type)
      if actual_type is not None:
          base_context["actual_type"] = str(actual_type)
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.TYPE_ERROR,
          error_code=error_code,
          context=base_context,
      )
      TypeError.__init__(self, message)

      self.expected_type = expected_type
      self.actual_type = actual_type


class FlextOperationError(FlextErrorMixin, RuntimeError):
    """General operation errors."""

    def __init__(
      self,
      message: str = "Operation failed",
      *,
      operation: str | None = None,
      stage: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize operation error."""
      base_context: dict[str, object] = dict(context or {})
      if operation is not None:
          base_context["operation"] = operation
      if stage is not None:
          base_context["stage"] = stage
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.OPERATION_ERROR,
          error_code=error_code,
          context=base_context,
      )
      RuntimeError.__init__(self, message)

      self.operation = operation
      self.stage = stage


class FlextAttributeError(FlextErrorMixin, AttributeError):
    """Attribute access errors."""

    def __init__(
      self,
      message: str = "Attribute error",
      *,
      attribute: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      attribute_context: Mapping[str, object] | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize attribute error."""
      base_context: dict[str, object] = dict(context or {})
      if attribute is not None:
          base_context["attribute"] = attribute
      if attribute_context is not None:
          base_context.update(dict(attribute_context))
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.TYPE_ERROR,
          error_code=error_code,
          context=base_context,
      )
      AttributeError.__init__(self, message)

      self.attribute = attribute


class FlextAuthenticationError(FlextErrorMixin, PermissionError):
    """Authentication-related errors."""

    def __init__(
      self,
      message: str = "Authentication failed",
      *,
      service: str | None = None,
      user_id: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize authentication error."""
      base_context: dict[str, object] = dict(context or {})
      if service is not None:
          base_context["service"] = service
      if user_id is not None:
          base_context["user_id"] = user_id
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.AUTHENTICATION_ERROR,
          error_code=error_code,
          context=base_context,
      )
      PermissionError.__init__(self, message)

      self.service = service
      self.user_id = user_id


class FlextTimeoutError(FlextErrorMixin, TimeoutError):
    """Timeout-related errors."""

    def __init__(
      self,
      message: str = "Operation timeout",
      *,
      service: str | None = None,
      timeout_seconds: float | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize timeout error."""
      base_context: dict[str, object] = dict(context or {})
      if service is not None:
          base_context["service"] = service
      if timeout_seconds is not None:
          base_context["timeout_seconds"] = timeout_seconds
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.TIMEOUT_ERROR,
          error_code=error_code,
          context=base_context,
      )
      TimeoutError.__init__(self, message)

      self.service = service
      self.timeout_seconds = (
          int(timeout_seconds) if isinstance(timeout_seconds, (int, float)) else None
      )


class FlextProcessingError(FlextErrorMixin, RuntimeError):
    """Data processing errors."""

    def __init__(
      self,
      message: str = "Processing failed",
      *,
      business_rule: str | None = None,
      operation: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize processing error."""
      base_context: dict[str, object] = dict(context or {})
      if business_rule is not None:
          base_context["business_rule"] = business_rule
      if operation is not None:
          base_context["operation"] = operation
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.PROCESSING_ERROR,
          error_code=error_code,
          context=base_context,
      )
      RuntimeError.__init__(self, message)

      self.business_rule = business_rule
      self.operation = operation


class FlextPermissionError(FlextErrorMixin, PermissionError):
    """Permission-related errors."""

    def __init__(
      self,
      message: str = "Permission denied",
      *,
      service: str | None = None,
      required_permission: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize permission error."""
      base_context: dict[str, object] = dict(context or {})
      if service is not None:
          base_context["service"] = service
      if required_permission is not None:
          base_context["required_permission"] = required_permission
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.PERMISSION_ERROR,
          error_code=error_code,
          context=base_context,
      )
      PermissionError.__init__(self, message)

      self.service = service
      self.required_permission = required_permission


class FlextAlreadyExistsError(FlextErrorMixin, ValueError):
    """Resource already exists errors."""

    def __init__(
      self,
      message: str = "Resource already exists",
      *,
      resource_type: str | None = None,
      resource_id: str | None = None,
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize already exists error."""
      base_context: dict[str, object] = dict(context or {})
      if resource_type is not None:
          base_context["resource_type"] = resource_type
      if resource_id is not None:
          base_context["resource_id"] = resource_id
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.ALREADY_EXISTS,
          error_code=error_code,
          context=base_context,
      )
      ValueError.__init__(self, message)

      self.resource_type = resource_type
      self.resource_id = resource_id


class FlextCriticalError(FlextErrorMixin, SystemError):
    """Critical system errors that require immediate attention."""

    def __init__(
      self,
      message: str = "Critical system error",
      *,
      service: str | None = None,
      severity: str = "CRITICAL",
      code: object | None = None,
      error_code: object | None = None,
      context: Mapping[str, object] | None = None,
    ) -> None:
      """Initialize critical error."""
      base_context: dict[str, object] = dict(context or {})
      if service is not None:
          base_context["service"] = service
      base_context["severity"] = severity
      FlextErrorMixin.__init__(
          self,
          message,
          code=code or FlextErrorCodes.CRITICAL_ERROR,
          error_code=error_code,
          context=base_context,
      )
      SystemError.__init__(self, message)

      self.service = service
      self.severity = severity


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
              code=FlextErrorCodes.GENERIC_ERROR,
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
              field=field,
              value=value,
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
              context=filtered_kwargs,
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
              context=context,
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
# ABSTRACT BASE CLASSES FOR EXTENSION
# =============================================================================


class FlextAbstractError(FlextError, ABC):
    """Abstract base class for all FLEXT errors."""

    @abstractmethod
    def get_error_context(self) -> dict[str, object]:
      """Get error context information."""


class FlextAbstractConfigurationError(FlextConfigurationError, ABC):
    """Abstract base class for configuration errors."""

    @abstractmethod
    def get_config_context(self) -> dict[str, object]:
      """Get configuration context information."""


class FlextAbstractInfrastructureError(FlextConnectionError, ABC):
    """Abstract base class for infrastructure errors."""

    @abstractmethod
    def get_infrastructure_context(self) -> dict[str, object]:
      """Get infrastructure context information."""


class FlextAbstractBusinessError(FlextProcessingError, ABC):
    """Abstract base class for business logic errors."""

    @abstractmethod
    def get_business_context(self) -> dict[str, object]:
      """Get business rule context information."""


class FlextAbstractValidationError(FlextValidationError, ABC):
    """Abstract base class for validation errors."""

    @abstractmethod
    def get_validation_context(self) -> dict[str, object]:
      """Get validation context information."""


class FlextAbstractErrorFactory(ABC):
    """Abstract factory for creating FLEXT exceptions."""

    @abstractmethod
    def create_error(self, message: str, **kwargs: object) -> FlextError:
      """Create a generic error."""

    @abstractmethod
    def create_validation_error(
      self,
      message: str,
      **kwargs: object,
    ) -> FlextValidationError:
      """Create a validation error."""

    @abstractmethod
    def create_configuration_error(
      self,
      message: str,
      **kwargs: object,
    ) -> FlextConfigurationError:
      """Create a configuration error."""


# =============================================================================
# EXCEPTION FACTORY IMPLEMENTATION
# =============================================================================


class FlextExceptions(FlextAbstractErrorFactory):
    """Factory for creating FLEXT exceptions."""

    @staticmethod
    def create_validation_error(
      message: str,
      **kwargs: object,
    ) -> FlextValidationError:
      """Create validation error."""
      # Extract validation-specific fields
      field = kwargs.get("field")
      validation_details = kwargs.get("validation_details")

      # Filter out validation-specific kwargs
      filtered_kwargs = {
          k: v
          for k, v in kwargs.items()
          if k not in {"field", "validation_details", "error_code"}
      }

      # Normalize validation_details to a precise type for both argument and context
      validation_details_dict: dict[str, object] | None
      if isinstance(validation_details, dict):
          # Ensure keys are strings and values are objects
          validation_details_dict = {str(k): v for k, v in validation_details.items()}
      else:
          validation_details_dict = None

      merged_context: dict[str, object] | None
      if validation_details_dict is not None:
          merged_context = {**filtered_kwargs, **validation_details_dict}
      else:
          merged_context = filtered_kwargs or None

      # Extract value safely from validation_details_dict
      value = (
          validation_details_dict.get("value") if validation_details_dict else None
      )

      return FlextValidationError(
          message,
          field=field if isinstance(field, str) else None,
          value=value,
          context=merged_context,
      )

    @staticmethod
    def create_business_error(
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
          context=filtered_kwargs,
      )

    @staticmethod
    def create_infrastructure_error(
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
          context=filtered_kwargs,
      )

    @staticmethod
    def create_configuration_error(
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
          context=filtered_kwargs,
      )

    # Additional factories used by tests
    @staticmethod
    def create_connection_error(message: str, **kwargs: object) -> FlextConnectionError:
      """Create connection error for network/service connectivity issues."""
      service = kwargs.get("service")
      endpoint = kwargs.get("endpoint")
      filtered_kwargs = {
          k: v for k, v in kwargs.items() if k not in {"service", "endpoint"}
      }
      return FlextConnectionError(
          message,
          service=service if isinstance(service, str) else None,
          endpoint=endpoint if isinstance(endpoint, str) else None,
          context=filtered_kwargs,
      )

    @staticmethod
    def create_type_error(
      message: str,
      *,
      expected_type: object | None = None,
      actual_type: object | None = None,
      **kwargs: object,
    ) -> FlextTypeError:
      """Create type error for type validation failures."""
      expected_type_str: str | None
      if isinstance(expected_type, str):
          expected_type_str = expected_type
      elif isinstance(expected_type, type):
          expected_type_str = expected_type.__name__
      else:
          expected_type_str = None

      actual_type_str: str | None
      if isinstance(actual_type, str):
          actual_type_str = actual_type
      elif isinstance(actual_type, type):
          actual_type_str = actual_type.__name__
      else:
          actual_type_str = None

      return FlextTypeError(
          message,
          expected_type=expected_type_str,
          actual_type=actual_type_str,
          context=kwargs,
      )

    @staticmethod
    def create_operation_error(message: str, **kwargs: object) -> FlextOperationError:
      """Create operation error for failed operations or processes."""
      operation_name_obj = kwargs.get("operation_name") or kwargs.get("operation")
      operation_name = (
          operation_name_obj if isinstance(operation_name_obj, str) else None
      )
      filtered_kwargs = {
          k: v for k, v in kwargs.items() if k not in {"operation_name", "operation"}
      }
      # Build context dict explicitly to satisfy type expectations
      extra_context: dict[str, object] = dict(filtered_kwargs)
      # Pass context via explicit keyword to avoid signature conflicts
      return FlextOperationError(
          message,
          operation=operation_name,
          context=extra_context,
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
    # Concrete Exception Classes
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
