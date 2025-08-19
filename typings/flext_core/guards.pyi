from collections.abc import Callable
from typing import ParamSpec, TypeVar

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.mixins import FlextSerializableMixin
from flext_core.result import FlextResult
from flext_core.typings import T

P = ParamSpec("P")  # noqa: PYI001
R = TypeVar("R")  # noqa: PYI001

__all__ = [
    "FlextGuards",
    "FlextValidatedModel",
    "FlextValidationUtils",
    "ValidatedModel",
    "immutable",
    "is_dict_of",
    "is_instance_of",
    "is_list_of",
    "is_not_none",
    "make_builder",
    "make_factory",
    "pure",
    "require_in_range",
    "require_non_empty",
    "require_not_none",
    "require_positive",
    "safe",
    "validated",
]

Platform = FlextConstants.Platform

class FlextGuards:
    @staticmethod
    def is_dict_of(obj: object, value_type: type[object]) -> bool: ...
    @staticmethod
    def is_not_none(obj: object) -> bool: ...
    @staticmethod
    def is_list_of(obj: object, value_type: type[object]) -> bool: ...
    @staticmethod
    def is_instance_of(obj: object, value_type: type[object]) -> bool: ...
    @staticmethod
    def immutable(target_class: type[T]) -> type[T]: ...
    @staticmethod
    def pure(func: Callable[P, R]) -> Callable[P, R]: ...
    @staticmethod
    def make_factory(target_class: type[object]) -> Callable[[], object]: ...
    @staticmethod
    def make_builder(target_class: type[object]) -> Callable[[], object]: ...

class FlextValidatedModel(BaseModel, FlextSerializableMixin):
    def __init__(self, **data: object) -> None: ...
    def validate_flext(self) -> FlextResult[None]: ...
    @property
    def is_valid(self) -> bool: ...
    @classmethod
    def create(cls, **data: object) -> FlextResult[FlextValidatedModel]: ...

class FlextValidationUtils:
    @staticmethod
    def require_not_none(
        value: object, message: str = "Value cannot be None"
    ) -> object: ...
    @staticmethod
    def require_positive(
        value: object, message: str = "Value must be positive"
    ) -> object: ...
    @staticmethod
    def require_in_range(
        value: object, min_val: int, max_val: int, message: str | None = None
    ) -> object: ...
    @staticmethod
    def require_non_empty(
        value: object, message: str = "Value cannot be empty"
    ) -> object: ...

# Type aliases for backward compatibility
is_not_none = FlextGuards.is_not_none  # Direct reference
is_list_of = FlextGuards.is_list_of  # Direct reference
is_instance_of = FlextGuards.is_instance_of  # Direct reference
is_dict_of = FlextGuards.is_dict_of  # Direct reference

# Type aliases for backward compatibility
type validated = Callable[[object], object]  # noqa: PYI001, PYI042
type safe = Callable[[object], object]  # noqa: PYI001, PYI042
immutable = FlextGuards.immutable  # Direct reference instead of type alias
pure = FlextGuards.pure  # Direct reference instead of type alias
type make_factory = Callable[[type[object]], Callable[[], object]]  # noqa: PYI001, PYI042
type make_builder = Callable[[type[object]], Callable[[], object]]  # noqa: PYI001, PYI042
type require_not_none = Callable[[object, str], object]  # noqa: PYI001, PYI042
type require_positive = Callable[[object, str], object]  # noqa: PYI001, PYI042
type require_in_range = Callable[[object, int, int, str | None], object]  # noqa: PYI001, PYI042
type require_non_empty = Callable[[object, str], object]  # noqa: PYI001, PYI042

ValidatedModel = FlextValidatedModel
