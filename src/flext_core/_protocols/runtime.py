"""Runtime-safe protocol support for private service type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# from collections.abc import Callable, Mapping, Sequence
# from typing import Protocol, Self, runtime_checkable
#
# from structlog.typing import BindableLogger
#
# from flext_core._protocols.base import FlextProtocolsBase
# from flext_core._protocols.runtime import FlextProtocolsRuntime
# from flext_core._protocols.pydantic import FlextProtocolsPydantic
#
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from flext_core import t, m


class FlextProtocolsRuntime:
    """Minimal structural protocols used by runtime-evaluated service aliases."""


#     @runtime_checkable
#     class Settings(Protocol):
#         """Minimal settings contract used by type aliases and guard inputs."""
#
#         app_name: str
#         version: str
#         enable_caching: bool
#         timeout_seconds: float
#         dispatcher_auto_context: bool
#         dispatcher_enable_logging: bool
#
#         @classmethod
#         def fetch_global(
#             cls,
#             *,
#             overrides: t.ScalarMapping | None = None,
#         ) -> Self: ...
#
#         def model_copy(
#             self,
#             *,
#             update: Mapping[str, t.Container] | None = None,
#             deep: bool = False,
#         ) -> Self: ...
#
#         def model_dump(self) -> t.ScalarMapping: ...
#
#     @runtime_checkable
#     class Context(Protocol):
#         """Minimal context contract for runtime-safe alias evaluation."""
#
#         def get(
#             self,
#             key: str,
#             scope: str = ...,
#         ) -> FlextProtocolsRuntime.ResultLike[
#             t.Container | m.Model
#         ]: ...
#
#         def set(
#             self,
#             key_or_data: str | t.ConfigMap,
#             value: t.Container | FlextProtocolsRuntime.Model | None = None,
#             *,
#             scope: str = ...,
#         ) -> FlextProtocolsRuntime.ResultLike[bool]: ...
#
#         def has(self, key: str, scope: str = ...) -> bool: ...
#
#         def keys(self) -> Sequence[str]: ...
#
#         def values(self) -> Sequence[t.Container]: ...
#
#         def items(self) -> Sequence[tuple[str, t.RecursiveContainer]]: ...
#
#         def remove(self, key: str, scope: str = ...) -> None: ...
#
#         def clear(self) -> None: ...
#
#         def clone(self) -> Self: ...
#
#         def merge(
#             self,
#             other: (
#                 Self
#                 | t.ConfigMap
#                 | Mapping[str, t.Container]
#             ),
#         ) -> Self: ...
#
#         def validate_context(self) -> FlextProtocolsRuntime.ResultLike[bool]: ...
#
#         def export(
#             self,
#             *,
#             include_statistics: bool = ...,
#             include_metadata: bool = ...,
#             as_dict: bool = ...,
#         ) -> FlextProtocolsRuntime.Model | Mapping[str, t.Container]: ...
#
#         def resolve_metadata(
#             self,
#             key: str,
#         ) -> FlextProtocolsRuntime.ResultLike[
#             t.Container | FlextProtocolsRuntime.Model
#         ]: ...
#
#         def apply_metadata(
#             self,
#             key: str,
#             value: (
#                 t.Scalar
#                 | Mapping[
#                     str,
#                     t.Scalar
#                     | Sequence[t.Scalar],
#                 ]
#                 | Sequence[t.Scalar]
#             ),
#         ) -> None: ...
#
#     @runtime_checkable
#     class DispatchMessage(Protocol):
#         """Protocol for values that route messages through dispatch_message()."""
#
#         def dispatch_message(
#             self,
#             message: FlextProtocolsRuntime.Routable,
#             operation: str = ...,
#         ) -> (
#             FlextProtocolsRuntime.ResultLike[
#                 t.Container | FlextProtocolsRuntime.Model
#             ]
#             | t.Container
#             | FlextProtocolsRuntime.Model
#             | None
#         ): ...
#
#     @runtime_checkable
#     class Handler(Protocol):
#         """Protocol for handler objects with route discovery and handle()."""
#
#         def can_handle(self, message_type: type) -> bool: ...
#
#         def handle(
#             self,
#             message: FlextProtocolsRuntime.Routable,
#         ) -> (
#             FlextProtocolsRuntime.ResultLike[
#                 t.Container | FlextProtocolsRuntime.Model
#             ]
#             | t.Container
#             | FlextProtocolsRuntime.Model
#             | None
#         ): ...
#
#     @runtime_checkable
#     class Handle(Protocol):
#         """Protocol for handler values exposing a handle() method."""
#
#         def handle(
#             self,
#             message: FlextProtocolsRuntime.Routable,
#         ) -> (
#             FlextProtocolsRuntime.ResultLike[
#                 t.Container | FlextProtocolsRuntime.Model
#             ]
#             | t.Container
#             | FlextProtocolsRuntime.Model
#             | None
#         ): ...
#
#     @runtime_checkable
#     class Execute(Protocol):
#         """Protocol for executable handler values."""
#
#         def execute(
#             self,
#             message: FlextProtocolsRuntime.Routable,
#         ) -> (
#             FlextProtocolsRuntime.ResultLike[
#                 t.Container | FlextProtocolsRuntime.Model
#             ]
#             | t.Container
#             | FlextProtocolsRuntime.Model
#             | None
#         ): ...
#
#     @runtime_checkable
#     class AutoDiscoverableHandler(Protocol):
#         """Protocol for handlers that inspect message types dynamically."""
#
#         def can_handle(self, message_type: type) -> bool: ...
#
#     @runtime_checkable
#     class Dispatcher(Protocol):
#         """Minimal dispatcher surface needed by runtime aliases."""
#
#         def dispatch(
#             self,
#             message: FlextProtocolsRuntime.Routable,
#         ) -> FlextProtocolsRuntime.ResultLike[
#             t.Container | FlextProtocolsRuntime.Model
#         ]: ...
#
#         def publish(
#             self,
#             event: (
#                 FlextProtocolsRuntime.Routable
#                 | Sequence[FlextProtocolsRuntime.Routable]
#             ),
#         ) -> FlextProtocolsRuntime.ResultLike[bool]: ...
#
#         def register_handler(
#             self,
#             handler: Callable[
#                 ...,
#                 t.Container | FlextProtocolsRuntime.Model | None,
#             ],
#             *,
#             is_event: bool = False,
#         ) -> FlextProtocolsRuntime.ResultLike[bool]: ...
#
#     @runtime_checkable
#     class ProviderLike[T_co](Protocol):
#         """DI-free provider abstraction for runtime-safe typing."""
#
#         def __call__(self) -> T_co: ...
#
#     @runtime_checkable
#     class DispatchableService(Protocol):
#         """Minimal dispatch-capable service contract."""
#
#         def dispatch(
#             self,
#             message: FlextProtocolsRuntime.Model,
#             /,
#         ) -> FlextProtocolsRuntime.Model: ...
#
#     @runtime_checkable
#     class SuccessCheckable(Protocol):
#         """Models or results that expose success and failure outcomes."""
#
#         @property
#         def success(self) -> bool: ...
#
#         @property
#         def failure(self) -> bool: ...
#
#     @runtime_checkable
#     class StructuredError(Protocol):
#         """Error carriers with domain, code, and message metadata."""
#
#         @property
#         def error_domain(self) -> str | None: ...
#
#         @property
#         def error_code(self) -> str | None: ...
#
#         @property
#         def error_message(self) -> str | None: ...
#
#         def matches_error_domain(self, domain: str) -> bool: ...
#
#     @runtime_checkable
#     class ErrorDomainProtocol(Protocol):
#         """Error domain enumeration contract."""
#
#         value: str
#         name: str
#
#     @runtime_checkable
#     class Configurable(Protocol):
#         """Values that can be configured from a mapping payload."""
#
#         def configure(
#             self,
#             settings: Mapping[str, t.Container] | None = None,
#         ) -> Self: ...
#
#     @runtime_checkable
#     class Flushable(Protocol):
#         """Values that expose a flush() method."""
#
#         def flush(self) -> None: ...
#
#     @runtime_checkable
#     class Logger(BindableLogger, Protocol):
#         """Bindable logger surface required by registration/runtime aliases."""
#
#         def critical(
#             self,
#             msg: str,
#             *args: t.Container | FlextProtocolsRuntime.Model,
#             **kw: t.Container | FlextProtocolsRuntime.Model | Exception,
#         ) -> FlextProtocolsRuntime.ResultLike[bool] | None: ...
#
#         def debug(
#             self,
#             msg: str,
#             *args: t.Container | FlextProtocolsRuntime.Model,
#             **kw: t.Container | FlextProtocolsRuntime.Model | Exception,
#         ) -> FlextProtocolsRuntime.ResultLike[bool] | None: ...
#
#         def error(
#             self,
#             msg: str,
#             *args: t.Container | FlextProtocolsRuntime.Model,
#             **kw: t.Container | FlextProtocolsRuntime.Model | Exception,
#         ) -> FlextProtocolsRuntime.ResultLike[bool] | None: ...
#
#         def exception(
#             self,
#             msg: str,
#             *args: t.Container | FlextProtocolsRuntime.Model,
#             **kw: t.Container | FlextProtocolsRuntime.Model | Exception,
#         ) -> FlextProtocolsRuntime.ResultLike[bool] | None: ...
#
#         def info(
#             self,
#             msg: str,
#             *args: t.Container | FlextProtocolsRuntime.Model,
#             **kw: t.Container | FlextProtocolsRuntime.Model | Exception,
#         ) -> FlextProtocolsRuntime.ResultLike[bool] | None: ...
#
#         def warning(
#             self,
#             msg: str,
#             *args: t.Container | FlextProtocolsRuntime.Model,
#             **kw: t.Container | FlextProtocolsRuntime.Model | Exception,
#         ) -> FlextProtocolsRuntime.ResultLike[bool] | None: ...
#
#     @runtime_checkable
#     class OutputLogger(Protocol):
#         """Raw wrapped logger surface returned by runtime logger factories."""
#
#         def critical(self, message: str) -> None: ...
#
#         def debug(self, message: str) -> None: ...
#
#         def error(self, message: str) -> None: ...
#
#         def exception(self, message: str) -> None: ...
#
#         def info(self, message: str) -> None: ...
#
#         def msg(self, message: str) -> None: ...
#
#         def warn(self, message: str) -> None: ...
#
#         def warning(self, message: str) -> None: ...
#

__all__: list[str] = ["FlextProtocolsRuntime"]
