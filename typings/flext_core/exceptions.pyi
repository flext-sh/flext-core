from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Self

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
    message: str
    code: str | None
    context: dict[str, object] | None
    correlation_id: str | None
    timestamp: float | None
    stack_trace: str | None
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        error_code: str | None = None,
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
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextValidationError(FlextErrorMixin, ValueError):
    field: str | None
    value: object
    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: object = None,
        code: str | None = None,
        error_code: str | None = None,
        validation_details: Mapping[str, object] | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextConfigurationError(FlextErrorMixin, ValueError):
    config_key: str | None
    config_file: str | None
    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        config_file: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextConnectionError(FlextErrorMixin, ConnectionError):
    service: str | None
    endpoint: str | None
    host: str | None
    port: int | None
    def __init__(
        self,
        message: str,
        *,
        service: str | None = None,
        endpoint: str | None = None,
        host: str | None = None,
        port: int | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextNotFoundError(FlextErrorMixin, ValueError):
    resource_type: str | None
    resource_id: str | None
    def __init__(
        self,
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextError(FlextErrorMixin, Exception):
    def __init__(
        self,
        message: str = "FLEXT operation failed",
        *,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextTypeError(FlextErrorMixin, TypeError):
    expected_type: str | None
    actual_type: str | None
    def __init__(
        self,
        message: str = "Type validation failed",
        *,
        expected_type: str | None = None,
        actual_type: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextOperationError(FlextErrorMixin, RuntimeError):
    operation: str | None
    stage: str | None
    def __init__(
        self,
        message: str = "Operation failed",
        *,
        operation: str | None = None,
        stage: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextAttributeError(FlextErrorMixin, AttributeError):
    attribute: str | None
    def __init__(
        self,
        message: str = "Attribute error",
        *,
        attribute: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        attribute_context: Mapping[str, object] | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextAuthenticationError(FlextErrorMixin, PermissionError):
    service: str | None
    user_id: str | None
    api_url: str | None
    authentication_method: str | None
    token_type: str | None
    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        service: str | None = None,
        user_id: str | None = None,
        api_url: str | None = None,
        authentication_method: str | None = None,
        token_type: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextTimeoutError(FlextErrorMixin, TimeoutError):
    service: str | None
    timeout_seconds: float | None
    operation: str | None
    def __init__(
        self,
        message: str = "Operation timeout",
        *,
        service: str | None = None,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextProcessingError(FlextErrorMixin, RuntimeError):
    business_rule: str | None
    operation: str | None
    stage: str | None
    api_url: str | None
    def __init__(
        self,
        message: str = "Processing failed",
        *,
        business_rule: str | None = None,
        operation: str | None = None,
        stage: str | None = None,
        api_url: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextPermissionError(FlextErrorMixin, PermissionError):
    service: str | None
    required_permission: str | None
    resource: str | None
    action: str | None
    def __init__(
        self,
        message: str = "Permission denied",
        *,
        service: str | None = None,
        required_permission: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextAlreadyExistsError(FlextErrorMixin, ValueError):
    resource_type: str | None
    resource_id: str | None
    def __init__(
        self,
        message: str = "Resource already exists",
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None: ...

class FlextCriticalError(FlextErrorMixin, SystemError):
    service: str | None
    severity: str
    operation: str | None
    component: str | None
    def __init__(
        self,
        message: str = "Critical system error",
        *,
        service: str | None = None,
        severity: str = "CRITICAL",
        operation: str | None = None,
        component: str | None = None,
        code: str | None = None,
        error_code: str | None = None,
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
    def create_validation_error(
        message: str,
        *,
        field: str | None = None,
        value: object = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextValidationError: ...
    @staticmethod
    def create_configuration_error(
        message: str,
        *,
        config_key: str | None = None,
        config_file: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextConfigurationError: ...
    @staticmethod
    def create_connection_error(
        message: str,
        *,
        service: str | None = None,
        endpoint: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextConnectionError: ...
    @staticmethod
    def create_authentication_error(
        message: str,
        *,
        service: str | None = None,
        user_id: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextAuthenticationError: ...
    @staticmethod
    def create_permission_error(
        message: str,
        *,
        service: str | None = None,
        required_permission: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextPermissionError: ...
    @staticmethod
    def create_not_found_error(
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextNotFoundError: ...
    @staticmethod
    def create_already_exists_error(
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextAlreadyExistsError: ...
    @staticmethod
    def create_timeout_error(
        message: str,
        *,
        service: str | None = None,
        timeout_seconds: float | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextTimeoutError: ...
    @staticmethod
    def create_processing_error(
        message: str,
        *,
        business_rule: str | None = None,
        operation: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextProcessingError: ...
    @staticmethod
    def create_critical_error(
        message: str,
        *,
        service: str | None = None,
        severity: str = "CRITICAL",
        context: Mapping[str, object] | None = None,
    ) -> FlextCriticalError: ...

class FlextExceptions(FlextAbstractErrorFactory):
    @staticmethod
    def create_validation_error(
        message: str,
        *,
        field: str | None = None,
        value: object = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextValidationError: ...
    @staticmethod
    def create_configuration_error(
        message: str,
        *,
        config_key: str | None = None,
        config_file: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextConfigurationError: ...
    @staticmethod
    def create_connection_error(
        message: str,
        *,
        service: str | None = None,
        endpoint: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextConnectionError: ...
    @staticmethod
    def create_authentication_error(
        message: str,
        *,
        service: str | None = None,
        user_id: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextAuthenticationError: ...
    @staticmethod
    def create_permission_error(
        message: str,
        *,
        service: str | None = None,
        required_permission: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextPermissionError: ...
    @staticmethod
    def create_not_found_error(
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextNotFoundError: ...
    @staticmethod
    def create_already_exists_error(
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextAlreadyExistsError: ...
    @staticmethod
    def create_timeout_error(
        message: str,
        *,
        service: str | None = None,
        timeout_seconds: float | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextTimeoutError: ...
    @staticmethod
    def create_processing_error(
        message: str,
        *,
        business_rule: str | None = None,
        operation: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> FlextProcessingError: ...
    @staticmethod
    def create_critical_error(
        message: str,
        *,
        service: str | None = None,
        severity: str = "CRITICAL",
        context: Mapping[str, object] | None = None,
    ) -> FlextCriticalError: ...

def get_exception_metrics() -> dict[str, int]: ...
def clear_exception_metrics() -> None: ...
