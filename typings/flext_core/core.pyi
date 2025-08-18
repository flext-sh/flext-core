from collections.abc import Callable
from typing import ParamSpec, TypeVar

from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer, FlextServiceKey
from flext_core.guards import ValidatedModel
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult

__all__ = ["FlextCore", "flext_core"]

_P = ParamSpec("_P")
_R = TypeVar("_R")

class FlextCore:
    def __init__(self) -> None: ...
    @classmethod
    def get_instance(cls) -> FlextCore: ...
    @property
    def container(self) -> FlextContainer: ...
    def register_service[S](
        self, key: FlextServiceKey[S], service: S
    ) -> FlextResult[None]: ...
    def get_service[S](self, key: FlextServiceKey[S]) -> FlextResult[S]: ...
    @staticmethod
    def get_logger(name: str) -> FlextLogger: ...
    @staticmethod
    def configure_logging(
        *, log_level: str = "INFO", _json_output: bool | None = None
    ) -> None: ...
    @staticmethod
    def ok[V](value: V) -> FlextResult[V]: ...
    @staticmethod
    def fail[V](error: str) -> FlextResult[V]: ...
    @staticmethod
    def pipe(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]: ...
    @staticmethod
    def compose(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]: ...
    @staticmethod
    def when[V](
        predicate: Callable[[V], bool],
        then_func: Callable[[V], FlextResult[V]],
        else_func: Callable[[V], FlextResult[V]] | None = None,
    ) -> Callable[[V], FlextResult[V]]: ...
    @staticmethod
    def tap[V](side_effect: Callable[[V], None]) -> Callable[[V], FlextResult[V]]: ...
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
    def create_validated_model[T: ValidatedModel](
        model_class: type[T], **data: object
    ) -> FlextResult[T]: ...
    @staticmethod
    def make_immutable[T](target_class: type[T]) -> type[T]: ...
    @staticmethod
    def make_pure(func: Callable[_P, _R]) -> Callable[_P, _R]: ...

def flext_core() -> FlextCore: ...
