"""FlextProtocolsHandler - handler, bus, registry, middleware protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from flext_core._protocols.base import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core import FlextProtocolsResult, r, t


class FlextProtocolsHandler:
    """Protocols for CQRS handlers and message routing."""

    @runtime_checkable
    class Handler[MessageT: FlextProtocolsBase.Model, ResultT](
        FlextProtocolsBase.Base,
        Protocol,
    ):
        def can_handle(self, message_type: type) -> bool: ...

        def handle(self, message: MessageT) -> r[ResultT]: ...

    @runtime_checkable
    class DispatchMessage(Protocol):
        def dispatch_message(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> (
            FlextProtocolsResult.ResultLike[t.RuntimeAtomic]
            | t.Container
            | BaseModel
            | None
        ): ...

    @runtime_checkable
    class Handle(Protocol):
        def handle(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> (
            FlextProtocolsResult.ResultLike[t.RuntimeAtomic]
            | t.Container
            | BaseModel
            | None
        ): ...

    @runtime_checkable
    class Execute(Protocol):
        def execute(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> (
            FlextProtocolsResult.ResultLike[t.RuntimeAtomic]
            | t.Container
            | BaseModel
            | None
        ): ...

    @runtime_checkable
    class Dispatcher(FlextProtocolsBase.Base, Protocol):
        def dispatch(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.Result[FlextProtocolsBase.Model]: ...

        def publish(
            self,
            event: FlextProtocolsBase.Routable | Sequence[FlextProtocolsBase.Routable],
        ) -> FlextProtocolsResult.Result[bool]: ...

        def register_handler(
            self,
            handler: t.HandlerLike,
            *,
            is_event: bool = False,
        ) -> FlextProtocolsResult.Result[bool]: ...

    @runtime_checkable
    class Middleware(FlextProtocolsBase.Base, Protocol):
        def process[TResult](
            self,
            command: FlextProtocolsBase.Model,
            next_handler: Callable[
                [FlextProtocolsBase.Model],
                FlextProtocolsResult.Result[TResult],
            ],
        ) -> FlextProtocolsResult.Result[TResult]: ...


__all__ = ["FlextProtocolsHandler"]
