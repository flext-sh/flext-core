from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import ClassVar, Generic, Self, TypeVar

from _typeshed import Incomplete
from pydantic import BaseModel

from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimingMixin,
)
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.typings import TAnyDict, TServiceName

__all__ = ["FlextCommands"]

_TCommand = TypeVar("_TCommand")
_TResult = TypeVar("_TResult")
_TQuery = TypeVar("_TQuery")
_TQueryResult = TypeVar("_TQueryResult")

# Public aliases for external use
TCommand = _TCommand
TResult = _TResult
TQuery = _TQuery
TQueryResult = _TQueryResult

class FlextAbstractCommand(ABC):
    @abstractmethod
    def validate_command(self) -> FlextResult[None]: ...

class FlextAbstractCommandHandler[TCommand, TResult](ABC):
    @property
    @abstractmethod
    def handler_name(self) -> str: ...
    @abstractmethod
    def handle(self, command: TCommand) -> FlextResult[TResult]: ...
    @abstractmethod
    def can_handle(self, command: object) -> bool: ...

class FlextAbstractCommandBus(ABC):
    @abstractmethod
    def send_command(self, command: FlextAbstractCommand) -> FlextResult[object]: ...
    @abstractmethod
    def unregister_handler(self, command_type: str) -> bool: ...
    @abstractmethod
    def get_registered_handlers(self) -> dict[str, object]: ...

class FlextAbstractQueryHandler[TQuery, TQueryResult](ABC):
    @property
    @abstractmethod
    def handler_name(self) -> str: ...
    @abstractmethod
    def handle(self, query: TQuery) -> FlextResult[TQueryResult]: ...

class FlextCommands:
    class Command(
        BaseModel, FlextAbstractCommand, FlextSerializableMixin, FlextLoggableMixin
    ):
        model_config: ClassVar[Incomplete]
        command_id: str
        command_type: str
        timestamp: datetime
        user_id: str | None
        correlation_id: str
        legacy_mixin_setup: object | None
        def model_post_init(self, __context: object, /) -> None: ...
        def to_payload(self) -> FlextPayload[TAnyDict]: ...
        @classmethod
        def from_payload(cls, payload: FlextPayload[TAnyDict]) -> FlextResult[Self]: ...
        def validate_command(self) -> FlextResult[None]: ...
        @staticmethod
        def require_field(
            field_name: str, value: object, error_msg: str = ""
        ) -> FlextResult[None]: ...
        @staticmethod
        def require_email(
            email: str, field_name: str = "email"
        ) -> FlextResult[None]: ...
        @staticmethod
        def require_min_length(
            value: str, min_len: int, field_name: str
        ) -> FlextResult[None]: ...
        def get_metadata(self) -> TAnyDict: ...

    class Result(FlextResult[_TResult], Generic[_TResult]):
        metadata: Incomplete
        def __init__(
            self,
            data: _TResult | None = None,
            error: str | None = None,
            metadata: TAnyDict | None = None,
        ) -> None: ...
        @classmethod
        def ok(
            cls, data: _TResult, /, *, metadata: TAnyDict | None = None
        ) -> FlextCommands.Result[_TResult]: ...
        @classmethod
        def fail(
            cls,
            error: str,
            /,
            *,
            error_code: str | None = None,
            error_data: dict[str, object] | None = None,
        ) -> FlextCommands.Result[_TResult]: ...

    class Handler(
        FlextAbstractCommandHandler[_TCommand, _TResult],
        FlextLoggableMixin,
        FlextTimingMixin,
        ABC,
        Generic[_TCommand, _TResult],
    ):
        handler_id: Incomplete
        def __init__(
            self,
            handler_name: TServiceName | None = None,
            handler_id: TServiceName | None = None,
        ) -> None: ...
        @property
        def handler_name(self) -> str: ...
        def validate_command(self, command: _TCommand) -> FlextResult[None]: ...
        @abstractmethod
        def handle(self, command: _TCommand) -> FlextResult[_TResult]: ...
        def process_command(self, command: _TCommand) -> FlextResult[_TResult]: ...
        def can_handle(self, command: object) -> bool: ...
        def handle_command(self, command: _TCommand) -> FlextResult[_TResult]: ...
        def get_command_type(self) -> str: ...
        def execute(self, command: _TCommand) -> FlextResult[_TResult]: ...

    class Bus(FlextAbstractCommandBus, FlextLoggableMixin):
        def __init__(self) -> None: ...
        def register_handler(self, *args: object) -> None: ...
        def register_handler_flexible(
            self,
            handler_or_command_type: object | type[_TCommand],
            handler: FlextCommands.Handler[_TCommand, _TResult] | None = None,
        ) -> FlextResult[None]: ...
        def execute(self, command: FlextAbstractCommand) -> FlextResult[object]: ...
        def add_middleware(self, middleware: object) -> None: ...
        def get_all_handlers(self) -> list[object]: ...
        def find_handler(self, command: object) -> object | None: ...
        def unregister_handler(self, command_type: str) -> bool: ...
        def send_command(
            self, command: FlextAbstractCommand
        ) -> FlextResult[object]: ...
        def get_registered_handlers(self) -> dict[str, object]: ...

    class Decorators:
        @staticmethod
        def command_handler(
            command_type: type[object],
        ) -> Callable[[Callable[[object], object]], Callable[[object], object]]: ...

    class Query(BaseModel, FlextSerializableMixin):
        model_config: ClassVar[Incomplete]
        query_id: str | None
        query_type: str | None
        page_size: int
        page_number: int
        sort_by: str | None
        sort_order: str
        def validate_query(self) -> FlextResult[None]: ...

    class QueryHandler(
        FlextAbstractQueryHandler[_TQuery, _TQueryResult],
        ABC,
        Generic[_TQuery, _TQueryResult],
    ):
        def __init__(self, handler_name: str | None = None) -> None: ...
        @property
        def handler_name(self) -> str: ...
        def can_handle(self, query: _TQuery) -> bool: ...
        def validate_query(self, query: _TQuery) -> FlextResult[None]: ...
        @abstractmethod
        def handle(self, query: _TQuery) -> FlextResult[_TQueryResult]: ...

    @staticmethod
    def create_command_bus() -> FlextCommands.Bus: ...
    @staticmethod
    def create_simple_handler(
        handler_func: Callable[[object], object],
    ) -> FlextCommands.Handler[object, object]: ...
