from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from _typeshed import Incomplete

from flext_core.mixins import FlextSerializableMixin
from flext_core.models import FlextModel
from flext_core.result import FlextResult

__all__ = ["FlextDomainService"]

type OperationType = (
    Callable[[], object]
    | Callable[[object], object]
    | Callable[[object, object], object]
    | Callable[[object, object, object], object]
)
TDomainResult = TypeVar("TDomainResult")

class FlextDomainService[TDomainResult](FlextModel, FlextSerializableMixin, ABC):
    model_config: Incomplete
    def is_valid(self) -> bool: ...
    def validate_business_rules(self) -> FlextResult[None]: ...
    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]: ...
    def validate_config(self) -> FlextResult[None]: ...
    def execute_operation(
        self, operation_name: str, operation: object, *args: object, **kwargs: object
    ) -> FlextResult[object]: ...
    def get_service_info(self) -> dict[str, object]: ...
