from collections.abc import Callable
from typing import ParamSpec, Self, TypeVar

from _typeshed import Incomplete
from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.mixins import FlextSerializableMixin, FlextValidatableMixin
from flext_core.result import FlextResult

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
is_not_none: Incomplete
is_list_of: Incomplete
is_instance_of: Incomplete
T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

class FlextGuards:
    @staticmethod
    def is_dict_of(obj: object, value_type: type) -> bool: ...
    @staticmethod
    def immutable(target_class: type[T]) -> type[T]: ...
    @staticmethod
    def pure(func: Callable[P, R]) -> Callable[P, R]: ...
    @staticmethod
    def make_factory(target_class: type) -> Callable[[], object]: ...
    @staticmethod
    def make_builder(target_class: type) -> Callable[[], object]: ...

class FlextValidatedModel(BaseModel, FlextSerializableMixin, FlextValidatableMixin):
    def __init__(self, **data: object) -> None: ...
    @classmethod
    def create(cls, **data: object) -> FlextResult[Self]: ...

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

validated: Incomplete
safe: Incomplete
is_dict_of: Incomplete
immutable: Incomplete
pure: Incomplete
make_factory: Incomplete
make_builder: Incomplete
require_not_none: Incomplete
require_positive: Incomplete
require_in_range: Incomplete
require_non_empty: Incomplete
ValidatedModel = FlextValidatedModel
