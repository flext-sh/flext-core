from contextlib import AbstractContextManager
from typing import Protocol

from _typeshed import Incomplete

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult

__all__ = [
    "FlextBusinessError",
    "FlextError",
    "FlextSemantic",
    "FlextValidationError",
    "configure_observability",
    "create_business_error",
    "create_validation_error",
    "get_observability",
]

class FlextSemanticModel:
    class Foundation:
        @staticmethod
        def get_base_config() -> dict[str, object]: ...

    class Namespace:
        class FlextData: ...
        class FlextAuth: ...
        class FlextService: ...
        class FlextInfrastructure: ...

    class Factory:
        @staticmethod
        def create_model_with_defaults(**kwargs: object) -> dict[str, object]: ...
        @staticmethod
        def validate_with_business_rules(instance: object) -> FlextResult[None]: ...

class FlextSemanticObservability:
    class Protocol:
        class Logger(Protocol):
            def trace(self, message: str, **context: object) -> None: ...
            def debug(self, message: str, **context: object) -> None: ...
            def info(self, message: str, **context: object) -> None: ...
            def warn(self, message: str, **context: object) -> None: ...
            def error(
                self,
                message: str,
                *,
                error_code: str | None = None,
                exception: Exception | None = None,
                **context: object,
            ) -> None: ...
            def fatal(self, message: str, **context: object) -> None: ...
            def audit(
                self,
                message: str,
                *,
                user_id: str | None = None,
                action: str | None = None,
                resource: str | None = None,
                outcome: str | None = None,
                **context: object,
            ) -> None: ...

        class SpanProtocol(Protocol):
            def add_context(self, key: str, value: object) -> None: ...
            def add_error(self, error: Exception) -> None: ...

        class Tracer(Protocol):
            def business_span(
                self, operation_name: str, **context: object
            ) -> AbstractContextManager[object]: ...
            def technical_span(
                self,
                operation_name: str,
                *,
                component: str | None = None,
                resource: str | None = None,
                **context: object,
            ) -> AbstractContextManager[object]: ...

        class Metrics(Protocol):
            def increment(
                self,
                metric_name: str,
                value: int = 1,
                tags: dict[str, str] | None = None,
            ) -> None: ...
            def histogram(
                self, metric_name: str, value: float, tags: dict[str, str] | None = None
            ) -> None: ...
            def gauge(
                self, metric_name: str, value: float, tags: dict[str, str] | None = None
            ) -> None: ...

        class Observability(Protocol):
            @property
            def log(self) -> FlextSemanticObservability.Protocol.Logger: ...
            @property
            def trace(self) -> FlextSemanticObservability.Protocol.Tracer: ...
            @property
            def metrics(self) -> FlextSemanticObservability.Protocol.Metrics: ...

    class Factory:
        @staticmethod
        def get_minimal_observability() -> object: ...
        @staticmethod
        def configure_observability(
            service_name: str, *, log_level: str = ...
        ) -> object: ...

class FlextSemanticError:
    class Hierarchy:
        class FlextError(Exception):
            message: Incomplete
            error_code: Incomplete
            context: Incomplete
            cause: Incomplete
            def __init__(
                self,
                message: str,
                *,
                error_code: str = ...,
                context: dict[str, object] | None = None,
                cause: Exception | None = None,
            ) -> None: ...

        class FlextBusinessError(FlextError):
            def __init__(self, message: str, **kwargs: object) -> None: ...

        class FlextTechnicalError(FlextError):
            def __init__(self, message: str, **kwargs: object) -> None: ...

        class FlextValidationError(FlextError):
            def __init__(self, message: str, **kwargs: object) -> None: ...

        class FlextSecurityError(FlextError):
            def __init__(self, message: str, **kwargs: object) -> None: ...

    class Namespace: ...

    class Factory:
        @staticmethod
        def create_business_error(
            message: str,
            *,
            context: dict[str, object] | None = None,
            cause: Exception | None = None,
        ) -> FlextSemanticError.Hierarchy.FlextBusinessError: ...
        @staticmethod
        def create_validation_error(
            message: str,
            *,
            field_name: str | None = None,
            field_value: object = None,
            context: dict[str, object] | None = None,
        ) -> FlextSemanticError.Hierarchy.FlextValidationError: ...
        @staticmethod
        def from_exception(
            exception: Exception,
            *,
            message: str | None = None,
            context: dict[str, object] | None = None,
        ) -> FlextSemanticError.Hierarchy.FlextError: ...

class FlextSemantic:
    Constants = FlextConstants
    Models = FlextSemanticModel
    Observability = FlextSemanticObservability
    Errors = FlextSemanticError

FlextError = FlextSemanticError.Hierarchy.FlextError
FlextBusinessError = FlextSemanticError.Hierarchy.FlextBusinessError
FlextValidationError = FlextSemanticError.Hierarchy.FlextValidationError
get_observability: Incomplete
configure_observability: Incomplete
create_business_error: Incomplete
create_validation_error: Incomplete
