from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ParamSpec, Protocol, TypeVar

from _typeshed import Incomplete

from flext_core.protocols import (
    FlextDecoratedFunction as FlextDecoratedFunction,
    FlextLoggerProtocol,
)
from flext_core.result import FlextResult
from flext_core.typings import TErrorHandler

__all__ = [
    "FlextDecoratedFunction",
    "FlextDecoratorFactory",
    "FlextDecoratorUtils",
    "FlextDecorators",
    "FlextErrorHandlingDecorators",
    "FlextFunctionalDecorators",
    "FlextImmutabilityDecorators",
    "FlextLoggingDecorators",
    "FlextPerformanceDecorators",
    "FlextValidationDecorators",
    "_decorators_base",
    "_flext_cache_decorator",
    "_flext_safe_call_decorator",
    "_flext_timing_decorator",
    "_flext_validate_input_decorator",
    "_safe_call_decorator",
    "_validate_input_decorator",
]

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., object])

class FlextAbstractDecorator(ABC):
    name: Incomplete
    def __init__(self, name: str | None = None) -> None: ...
    @abstractmethod
    def __call__(self, func: F) -> F: ...

class FlextAbstractDecoratorFactory(ABC):
    @abstractmethod
    def create_validation_decorator(self, **kwargs: object) -> object: ...

class FlextAbstractValidationDecorator(FlextAbstractDecorator, ABC): ...
class FlextAbstractErrorHandlingDecorator(FlextAbstractDecorator, ABC): ...
class FlextAbstractPerformanceDecorator(FlextAbstractDecorator, ABC): ...
class FlextAbstractLoggingDecorator(FlextAbstractDecorator, ABC): ...

class FlextDecoratorUtils:
    @staticmethod
    def preserve_metadata(
        original: Callable[P, R], wrapper: Callable[P, R]
    ) -> Callable[P, R]: ...

class FlextValidationDecorators(FlextAbstractValidationDecorator):
    def __init__(self, name: str | None = None) -> None: ...
    def __call__(self, func: F) -> F: ...
    def validate_input(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> FlextResult[None]: ...
    def validate_output(self, result: object) -> FlextResult[object]: ...
    def apply_decoration(self, func: Callable[P, object]) -> Callable[P, object]: ...
    @staticmethod
    def create_validation_decorator(
        validator: Callable[[object], bool],
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def validate_arguments(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...
    @staticmethod
    def create_input_validator(
        validator: Callable[[object], bool],
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...

class FlextErrorHandlingDecorators(FlextAbstractErrorHandlingDecorator):
    handled_exceptions: Incomplete
    def __init__(
        self,
        name: str | None = None,
        handled_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None: ...
    def __call__(self, func: F) -> F: ...
    def handle_error(self, func_name: str, error: Exception) -> object: ...
    def should_handle_error(self, error: Exception) -> bool: ...
    def create_error_result(self, func_name: str, error: Exception) -> object: ...
    def apply_decoration(self, func: Callable[P, object]) -> Callable[P, object]: ...
    @staticmethod
    def create_safe_decorator(
        error_handler: TErrorHandler | None = None,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def get_safe_decorator() -> Callable[
        [FlextDecoratedFunction], FlextDecoratedFunction
    ]: ...
    @staticmethod
    def retry_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...
    @staticmethod
    def safe_call(
        error_handler: TErrorHandler | None = None,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...

class FlextPerformanceDecorators(FlextAbstractPerformanceDecorator):
    threshold_seconds: Incomplete
    metrics: dict[str, dict[str, object]]
    def __init__(
        self, name: str | None = None, threshold_seconds: float = 1.0
    ) -> None: ...
    def __call__(self, func: F) -> F: ...
    def start_timing(self) -> float: ...
    def stop_timing(self, start_time: float) -> float: ...
    def record_metrics(
        self, func_name: str, duration: float, args: tuple[object, ...]
    ) -> None: ...
    def apply_decoration(self, func: Callable[P, object]) -> Callable[P, object]: ...
    @staticmethod
    def create_cache_decorator(
        max_size: int = 128,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def get_timing_decorator(
        func: FlextDecoratedFunction | None = None,
    ) -> (
        FlextDecoratedFunction
        | Callable[[FlextDecoratedFunction], FlextDecoratedFunction]
    ): ...
    @staticmethod
    def memoize_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...
    @staticmethod
    def cache_results(
        max_size: int = 128,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def time_execution(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...

class FlextLoggingDecorators(FlextAbstractLoggingDecorator):
    log_level: Incomplete
    def __init__(self, name: str | None = None, log_level: str = "INFO") -> None: ...
    def __call__(self, func: F) -> F: ...
    @property
    def logger(self) -> FlextLoggerProtocol: ...
    def log_entry(
        self, func_name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None: ...
    def log_exit(self, func_name: str, _result: object, duration: float) -> None: ...
    def log_error(self, func_name: str, error: Exception) -> None: ...
    def apply_decoration(
        self, func: Callable[..., object]
    ) -> Callable[..., object]: ...
    @staticmethod
    def log_calls_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...
    @staticmethod
    def log_exceptions_decorator(
        func: FlextDecoratedFunction,
    ) -> FlextDecoratedFunction: ...
    @staticmethod
    def log_function_calls(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...

class FlextImmutabilityDecorators(FlextAbstractDecorator):
    def __init__(self, name: str | None = None) -> None: ...
    def __call__(self, func: F) -> F: ...
    def apply_decoration(
        self, func: Callable[..., object]
    ) -> Callable[..., object]: ...
    def validate_function(self, func: Callable[..., object]) -> bool: ...
    @staticmethod
    def immutable_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...
    @staticmethod
    def freeze_args_decorator(
        func: FlextDecoratedFunction,
    ) -> FlextDecoratedFunction: ...
    @staticmethod
    def readonly_result(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...

class FlextFunctionalDecorators(FlextAbstractDecorator):
    def __init__(self, name: str | None = None) -> None: ...
    def __call__(self, func: F) -> F: ...
    def apply_decoration(
        self, func: Callable[..., object]
    ) -> Callable[..., object]: ...
    def validate_function(self, func: Callable[..., object]) -> bool: ...
    @staticmethod
    def curry_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...
    @staticmethod
    def compose_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...
    @staticmethod
    def pipeline_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...

class FlextDecoratorFactory(FlextAbstractDecoratorFactory):
    def create_validation_decorator(
        self, **kwargs: object
    ) -> FlextAbstractValidationDecorator: ...
    def create_performance_decorator(
        self, **kwargs: object
    ) -> FlextAbstractPerformanceDecorator: ...
    def create_logging_decorator(
        self, **kwargs: object
    ) -> FlextAbstractLoggingDecorator: ...
    def create_error_handling_decorator(
        self, **kwargs: object
    ) -> FlextAbstractErrorHandlingDecorator: ...
    @staticmethod
    def create_cache_decorator(
        max_size: int = 128,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def create_timing_decorator() -> Callable[
        [FlextDecoratedFunction], FlextDecoratedFunction
    ]: ...
    @staticmethod
    def create_safe_decorator(
        error_handler: TErrorHandler | None = None,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def create_static_validation_decorator(
        validator: Callable[[object], bool],
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...

def _flext_safe_call_decorator(
    error_handler: TErrorHandler | None = None,
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
def _flext_timing_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction: ...
def _flext_validate_input_decorator(
    validator: Callable[[object], bool],
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
def _flext_cache_decorator(
    max_size: int = 128,
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...

class FlextDecorators:
    @staticmethod
    def validated_with_result(
        model_class: object | None = None,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def safe_result(func: Callable[P, object]) -> Callable[P, object]: ...
    @staticmethod
    def cached_with_timing(
        max_size: int = 128,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def safe_cached(
        max_size: int = 128,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def validated_cached(
        model_class: object, max_size: int = 128
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def complete_decorator(
        model_class: object | None = None,
        *,
        cache_size: int = 128,
        with_timing: bool = False,
        with_logging: bool = False,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    Validation = FlextValidationDecorators
    ErrorHandling = FlextErrorHandlingDecorators
    Performance = FlextPerformanceDecorators
    Functional = FlextFunctionalDecorators
    Immutability = FlextImmutabilityDecorators
    Logging = FlextLoggingDecorators

class _DecoratedFunction(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...

class _BaseDecoratorUtils: ...

_decorators_base: Incomplete
_validate_input_decorator = _flext_validate_input_decorator
_safe_call_decorator = _flext_safe_call_decorator

class _BaseImmutabilityDecorators(FlextImmutabilityDecorators):
    @staticmethod
    def freeze_args_decorator(
        func: FlextDecoratedFunction | None = None,
    ) -> (
        Callable[[FlextDecoratedFunction], FlextDecoratedFunction]
        | FlextDecoratedFunction
    ): ...

class _BaseDecoratorFactory(FlextDecoratorFactory):
    @staticmethod
    def create_cache_decorator(
        size: int = 128,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
    @staticmethod
    def create_safe_decorator(
        error_handler: Callable[[Exception], str] | None = None,
    ) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]: ...
