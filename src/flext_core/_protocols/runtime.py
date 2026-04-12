"""Runtime protocol surface used by service type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, Self, runtime_checkable

from structlog.typing import BindableLogger

from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers


class FlextProtocolsRuntime:
    """Structural runtime contracts for service alias evaluation."""

    @runtime_checkable
    class Model(Protocol):
        """Structural model contract for runtime-evaluated aliases."""

        model_fields: Mapping[str, object]

        def model_dump(
            self,
            *,
            mode: str = "python",
            include: object | None = None,
            exclude: object | None = None,
            context: object | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | str = True,
            fallback: Callable[[object], object] | None = None,
            serialize_as_any: bool = False,
        ) -> Mapping[str, object]: ...

        @classmethod
        def model_validate(
            cls,
            obj: object,
            *,
            strict: bool | None = None,
            extra: object | None = None,
            from_attributes: bool | None = None,
            context: object | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> Self: ...

        def model_copy(
            self,
            *,
            update: Mapping[str, object] | None = None,
            deep: bool = False,
        ) -> Self: ...

    @runtime_checkable
    class Result[T_co](Protocol):
        """Observable result contract for runtime alias evaluation."""

        @property
        def error(self) -> str | None: ...

        @property
        def error_code(self) -> str | None: ...

        @property
        def error_data(self) -> FlextTypingContainers.ConfigMap | None: ...

        @property
        def exception(self) -> BaseException | None: ...

        @property
        def success(self) -> bool: ...

        @property
        def failure(self) -> bool: ...

        @property
        def value(self) -> T_co: ...

        def unwrap(self) -> T_co: ...

    @runtime_checkable
    class Routable(Protocol):
        """Message objects exposing canonical route fields."""

        @property
        def command_type(self) -> str | None: ...

        @property
        def event_type(self) -> str | None: ...

        @property
        def query_type(self) -> str | None: ...

    @runtime_checkable
    class Flushable(Protocol):
        """Values exposing a flush() operation."""

        def flush(self) -> None: ...

    @runtime_checkable
    class Settings(Protocol):
        """Minimal settings contract used by service aliases."""

        app_name: str
        version: str
        enable_caching: bool
        timeout_seconds: float
        dispatcher_auto_context: bool
        dispatcher_enable_logging: bool

        @classmethod
        def fetch_global(
            cls,
            *,
            overrides: FlextTypingBase.ScalarMapping | None = None,
        ) -> Self: ...

        def model_copy(
            self,
            *,
            update: Mapping[str, FlextTypingBase.Container] | None = None,
            deep: bool = False,
        ) -> Self: ...

        def model_dump(self) -> FlextTypingBase.ScalarMapping: ...

    @runtime_checkable
    class Context(Protocol):
        """Minimal context contract used by service aliases."""

        def get(
            self,
            key: str,
            scope: str = ...,
        ) -> FlextProtocolsRuntime.Result[
            FlextTypingBase.Container | FlextProtocolsRuntime.Model
        ]: ...

        def set(
            self,
            key_or_data: str | FlextTypingContainers.ConfigMap,
            value: FlextTypingBase.Container
            | FlextProtocolsRuntime.Model
            | None = None,
            *,
            scope: str = ...,
        ) -> FlextProtocolsRuntime.Result[bool]: ...

        def has(self, key: str, scope: str = ...) -> bool: ...

        def keys(self) -> Sequence[str]: ...

        def values(self) -> Sequence[FlextTypingBase.Container]: ...

        def items(self) -> Sequence[tuple[str, FlextTypingBase.RecursiveContainer]]: ...

        def remove(self, key: str, scope: str = ...) -> None: ...

        def clear(self) -> None: ...

        def clone(self) -> Self: ...

        def merge(
            self,
            other: Self
            | FlextTypingContainers.ConfigMap
            | Mapping[str, FlextTypingBase.Container],
        ) -> Self: ...

        def validate_context(self) -> FlextProtocolsRuntime.Result[bool]: ...

        def export(
            self,
            *,
            include_statistics: bool = ...,
            include_metadata: bool = ...,
            as_dict: bool = ...,
        ) -> FlextProtocolsRuntime.Model | Mapping[str, FlextTypingBase.Container]: ...

        def resolve_metadata(
            self,
            key: str,
        ) -> FlextProtocolsRuntime.Result[
            FlextTypingBase.Container | FlextProtocolsRuntime.Model
        ]: ...

        def apply_metadata(
            self,
            key: str,
            value: (
                FlextTypingBase.Scalar
                | Mapping[
                    str, FlextTypingBase.Scalar | Sequence[FlextTypingBase.Scalar]
                ]
                | Sequence[FlextTypingBase.Scalar]
            ),
        ) -> None: ...

    @runtime_checkable
    class DispatchMessage(Protocol):
        """Values routing messages through dispatch_message()."""

        def dispatch_message(
            self,
            message: FlextProtocolsRuntime.Routable,
            operation: str = ...,
        ) -> (
            FlextProtocolsRuntime.Result[
                FlextTypingBase.Container | FlextProtocolsRuntime.Model
            ]
            | FlextTypingBase.Container
            | FlextProtocolsRuntime.Model
            | None
        ): ...

    @runtime_checkable
    class Handle(Protocol):
        """Handler values exposing handle()."""

        def handle(
            self,
            message: FlextProtocolsRuntime.Routable,
        ) -> (
            FlextProtocolsRuntime.Result[
                FlextTypingBase.Container | FlextProtocolsRuntime.Model
            ]
            | FlextTypingBase.Container
            | FlextProtocolsRuntime.Model
            | None
        ): ...

    @runtime_checkable
    class Execute(Protocol):
        """Executable handler values."""

        def execute(
            self,
            message: FlextProtocolsRuntime.Routable,
        ) -> (
            FlextProtocolsRuntime.Result[
                FlextTypingBase.Container | FlextProtocolsRuntime.Model
            ]
            | FlextTypingBase.Container
            | FlextProtocolsRuntime.Model
            | None
        ): ...

    @runtime_checkable
    class AutoDiscoverableHandler(Protocol):
        """Handlers that inspect message types dynamically."""

        def can_handle(self, message_type: type) -> bool: ...

    @runtime_checkable
    class Dispatcher(Protocol):
        """Minimal dispatcher surface needed by service aliases."""

        def dispatch(
            self,
            message: FlextProtocolsRuntime.Routable,
        ) -> FlextProtocolsRuntime.Result[
            FlextTypingBase.Container | FlextProtocolsRuntime.Model
        ]: ...

        def publish(
            self,
            event: FlextProtocolsRuntime.Routable
            | Sequence[FlextProtocolsRuntime.Routable],
        ) -> FlextProtocolsRuntime.Result[bool]: ...

        def register_handler(
            self,
            handler: Callable[
                ...,
                FlextTypingBase.Container | FlextProtocolsRuntime.Model | None,
            ],
            *,
            is_event: bool = False,
        ) -> FlextProtocolsRuntime.Result[bool]: ...

    @runtime_checkable
    class ProviderLike[T_co](Protocol):
        """Dependency-provider abstraction for service aliases."""

        def __call__(self) -> T_co: ...

    @runtime_checkable
    class DispatchableService(Protocol):
        """Minimal dispatch-capable service contract."""

        def dispatch(
            self,
            message: FlextProtocolsRuntime.Model,
            /,
        ) -> FlextProtocolsRuntime.Model: ...

    @runtime_checkable
    class SuccessCheckable(Protocol):
        """Values exposing success/failure outcome semantics."""

        @property
        def success(self) -> bool: ...

        @property
        def failure(self) -> bool: ...

    @runtime_checkable
    class StructuredError(Protocol):
        """Error carriers with domain, code, and message metadata."""

        @property
        def error_domain(self) -> str | None: ...

        @property
        def error_code(self) -> str | None: ...

        @property
        def error_message(self) -> str | None: ...

    @runtime_checkable
    class ErrorDomainProtocol(Protocol):
        """Enumeration-like error domain contract."""

        value: str
        name: str

    @runtime_checkable
    class Configurable(Protocol):
        """Configurable component contract."""

        def configure(
            self,
            settings: FlextTypingBase.ContainerMapping | None = None,
        ) -> Self: ...

    @runtime_checkable
    class Logger(BindableLogger, Protocol):
        """Logger protocol with explicit methods used by service aliases."""

        def critical(
            self,
            msg: str,
            *args: FlextTypingBase.Container | FlextProtocolsRuntime.Model,
            **kw: FlextTypingBase.Container | FlextProtocolsRuntime.Model | Exception,
        ) -> FlextProtocolsRuntime.Result[bool] | None: ...

        def debug(
            self,
            msg: str,
            *args: FlextTypingBase.Container | FlextProtocolsRuntime.Model,
            **kw: FlextTypingBase.Container | FlextProtocolsRuntime.Model | Exception,
        ) -> FlextProtocolsRuntime.Result[bool] | None: ...

        def error(
            self,
            msg: str,
            *args: FlextTypingBase.Container | FlextProtocolsRuntime.Model,
            **kw: FlextTypingBase.Container | FlextProtocolsRuntime.Model | Exception,
        ) -> FlextProtocolsRuntime.Result[bool] | None: ...

        def exception(
            self,
            msg: str,
            *args: FlextTypingBase.Container | FlextProtocolsRuntime.Model,
            **kw: FlextTypingBase.Container | FlextProtocolsRuntime.Model | Exception,
        ) -> FlextProtocolsRuntime.Result[bool] | None: ...

        def info(
            self,
            msg: str,
            *args: FlextTypingBase.Container | FlextProtocolsRuntime.Model,
            **kw: FlextTypingBase.Container | FlextProtocolsRuntime.Model | Exception,
        ) -> FlextProtocolsRuntime.Result[bool] | None: ...

        def warning(
            self,
            msg: str,
            *args: FlextTypingBase.Container | FlextProtocolsRuntime.Model,
            **kw: FlextTypingBase.Container | FlextProtocolsRuntime.Model | Exception,
        ) -> FlextProtocolsRuntime.Result[bool] | None: ...

    @runtime_checkable
    class OutputLogger(Protocol):
        """Wrapped output logger contract returned by logger factories."""

        def critical(self, message: str) -> None: ...

        def debug(self, message: str) -> None: ...

        def error(self, message: str) -> None: ...

        def exception(self, message: str) -> None: ...

        def info(self, message: str) -> None: ...

        def msg(self, message: str) -> None: ...

        def warn(self, message: str) -> None: ...

        def warning(self, message: str) -> None: ...


__all__: list[str] = ["FlextProtocolsRuntime"]
