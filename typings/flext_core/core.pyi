from collections.abc import Callable
from typing import ParamSpec, TypeVar

from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")

__all__ = ["FlextCore", "flext_core"]

class FlextCore:
    def __init__(self) -> None: ...
    @classmethod
    def get_instance(cls) -> FlextCore: ...
    @property
    def container(self) -> FlextContainer: ...
    def register_service(self, key: str, service: object) -> FlextResult[None]: ...
    def get_service(self, key: str) -> FlextResult[object]: ...
    @staticmethod
    def get_logger(name: str) -> FlextLogger: ...
    @staticmethod
    def configure_logging(
        *, log_level: str = "INFO", _json_output: bool | None = None
    ) -> None: ...
    @staticmethod
    def ok(value: object) -> FlextResult[object]: ...
    @staticmethod
    def fail(error: str) -> FlextResult[object]: ...
    @staticmethod
    def pipe(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]: ...
    @staticmethod
    def compose(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]: ...
    @staticmethod
    def when(
        predicate: Callable[[object], bool],
        then_func: Callable[[object], FlextResult[object]],
        else_func: Callable[[object], FlextResult[object]] | None = None,
    ) -> Callable[[object], FlextResult[object]]: ...
    @staticmethod
    def tap(
        side_effect: Callable[[object], None],
    ) -> Callable[[object], FlextResult[object]]: ...
    def get_settings(self, settings_class: type[object]) -> object: ...
    @property
    def constants(self) -> type[FlextConstants]: ...
    @staticmethod
    def validate_type(
        obj: object, expected_type: type[object]
    ) -> FlextResult[object]: ...
    @staticmethod
    def validate_dict_structure(
        obj: object, value_type: type[object]
    ) -> FlextResult[dict[str, object]]: ...
    @staticmethod
    def create_validated_model(
        model_class: type[object], **data: object
    ) -> FlextResult[object]: ...
    @staticmethod
    def make_immutable(target_class: type[_T]) -> type[_T]: ...
    @staticmethod
    def make_pure(func: Callable[_P, _R]) -> Callable[_P, _R]: ...

def flext_core() -> FlextCore: ...
