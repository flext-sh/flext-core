from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Self

from _typeshed import Incomplete

__all__ = [
    "FlextAlreadyExistsError",
    "FlextAttributeError",
    "FlextAuthenticationError",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextCriticalError",
    "FlextError",
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
    "create_module_exception_classes",
    "get_exception_metrics",
]

class FlextErrorCodes(StrEnum):
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
    def __new__(cls) -> Self: ...
    def record_exception(self, exception_type: str) -> None: ...
    def get_metrics(self) -> dict[str, int]: ...
    def clear_metrics(self) -> None: ...

class FlextErrorMixin:
    message: Incomplete
    code: Incomplete
    context: Incomplete
    correlation_id: Incomplete
    timestamp: Incomplete
    stack_trace: Incomplete
    def __init__(
        self,
        message: str,
        *,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
        correlation_id: str | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, object]: ...
    @property
    def error_code(self) -> str: ...

class FlextUserError(FlextErrorMixin, TypeError):
    def __init__(
        self,
        message: str,
        *,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextValidationError(FlextErrorMixin, ValueError):
    field: Incomplete
    value: Incomplete
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
    ) -> None: ...

class FlextConfigurationError(FlextErrorMixin, ValueError):
    config_key: Incomplete
    config_file: Incomplete
    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        config_file: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextConnectionError(FlextErrorMixin, ConnectionError):
    service: Incomplete
    endpoint: Incomplete
    def __init__(
        self,
        message: str,
        *,
        service: str | None = None,
        endpoint: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextNotFoundError(FlextErrorMixin, ValueError):
    resource_type: Incomplete
    resource_id: Incomplete
    def __init__(
        self,
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextError(FlextErrorMixin, Exception):
    def __init__(
        self,
        message: str = "FLEXT operation failed",
        *,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextTypeError(FlextErrorMixin, TypeError):
    expected_type: Incomplete
    actual_type: Incomplete
    def __init__(
        self,
        message: str = "Type validation failed",
        *,
        expected_type: object | None = None,
        actual_type: object | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextOperationError(FlextErrorMixin, RuntimeError):
    operation: Incomplete
    stage: Incomplete
    def __init__(
        self,
        message: str = "Operation failed",
        *,
        operation: str | None = None,
        stage: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextAttributeError(FlextErrorMixin, AttributeError):
    attribute: Incomplete
    def __init__(
        self,
        message: str = "Attribute error",
        *,
        attribute: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        attribute_context: Mapping[str, object] | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextAuthenticationError(FlextErrorMixin, PermissionError):
    service: Incomplete
    user_id: Incomplete
    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        service: str | None = None,
        user_id: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextTimeoutError(FlextErrorMixin, TimeoutError):
    service: Incomplete
    timeout_seconds: Incomplete
    def __init__(
        self,
        message: str = "Operation timeout",
        *,
        service: str | None = None,
        timeout_seconds: float | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextProcessingError(FlextErrorMixin, RuntimeError):
    business_rule: Incomplete
    operation: Incomplete
    def __init__(
        self,
        message: str = "Processing failed",
        *,
        business_rule: str | None = None,
        operation: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextPermissionError(FlextErrorMixin, PermissionError):
    service: Incomplete
    required_permission: Incomplete
    def __init__(
        self,
        message: str = "Permission denied",
        *,
        service: str | None = None,
        required_permission: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextAlreadyExistsError(FlextErrorMixin, ValueError):
    resource_type: Incomplete
    resource_id: Incomplete
    def __init__(
        self,
        message: str = "Resource already exists",
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextCriticalError(FlextErrorMixin, SystemError):
    service: Incomplete
    severity: Incomplete
    def __init__(
        self,
        message: str = "Critical system error",
        *,
        service: str | None = None,
        severity: str = "CRITICAL",
        code: object | None = None,
        error_code: object | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

def create_context_exception_factory(module_name: str) -> type: ...
def create_module_exception_classes(module_name: str) -> dict[str, type]: ...

class FlextAbstractError(FlextError, ABC):
    @abstractmethod
    def get_error_context(self) -> dict[str, object]: ...

class FlextAbstractConfigurationError(FlextConfigurationError, ABC):
    @abstractmethod
    def get_config_context(self) -> dict[str, object]: ...

class FlextAbstractInfrastructureError(FlextConnectionError, ABC):
    @abstractmethod
    def get_infrastructure_context(self) -> dict[str, object]: ...

class FlextAbstractBusinessError(FlextProcessingError, ABC):
    @abstractmethod
    def get_business_context(self) -> dict[str, object]: ...

class FlextAbstractValidationError(FlextValidationError, ABC):
    @abstractmethod
    def get_validation_context(self) -> dict[str, object]: ...

class FlextAbstractErrorFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_error(message: str, **kwargs: object) -> FlextError: ...
    @staticmethod
    @abstractmethod
    def create_validation_error(
        message: str, **kwargs: object
    ) -> FlextValidationError: ...
    @staticmethod
    @abstractmethod
    def create_configuration_error(
        message: str, **kwargs: object
    ) -> FlextConfigurationError: ...

class FlextExceptions(FlextAbstractErrorFactory, ABC):
    @staticmethod
    def create_validation_error(
        message: str, **kwargs: object
    ) -> FlextValidationError: ...
    @staticmethod
    def create_business_error(
        message: str, **kwargs: object
    ) -> FlextProcessingError: ...
    @staticmethod
    def create_infrastructure_error(
        message: str, **kwargs: object
    ) -> FlextConnectionError: ...
    @staticmethod
    def create_configuration_error(
        message: str, **kwargs: object
    ) -> FlextConfigurationError: ...
    @staticmethod
    def create_connection_error(
        message: str, **kwargs: object
    ) -> FlextConnectionError: ...
    @staticmethod
    def create_type_error(
        message: str,
        *,
        expected_type: object | None = None,
        actual_type: object | None = None,
        **kwargs: object,
    ) -> FlextTypeError: ...
    @staticmethod
    def create_operation_error(
        message: str, **kwargs: object
    ) -> FlextOperationError: ...

def get_exception_metrics() -> dict[str, int]: ...
def clear_exception_metrics() -> None: ...
