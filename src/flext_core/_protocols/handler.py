"""FlextProtocolsHandler - handler, bus, registry, middleware protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from flext_core import FlextProtocolsBase, FlextProtocolsResult, t


class FlextProtocolsHandler:
    """Protocols for CQRS handlers and message routing."""

    @runtime_checkable
    class Handler[MessageT: FlextProtocolsBase.Model, ResultT](
        FlextProtocolsBase.Base,
        Protocol,
    ):
        """Protocol that defines a typed message handler contract."""

        def can_handle(self, message_type: type) -> bool: ...

        def handle(
            self,
            message: MessageT,
        ) -> FlextProtocolsResult.Result[ResultT]: ...

    @runtime_checkable
    class DispatchMessage(Protocol):
        """Protocol for routing a message through a dispatch path."""

        def dispatch_message(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> (
            FlextProtocolsResult.Result[t.RuntimeAtomic]
            | t.Container
            | BaseModel
            | None
        ): ...

    @runtime_checkable
    class Handle(Protocol):
        """Protocol for handle behaviors in CQRS message workflows."""

        def handle(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> (
            FlextProtocolsResult.Result[t.RuntimeAtomic]
            | t.Container
            | BaseModel
            | None
        ): ...

    @runtime_checkable
    class Execute(Protocol):
        """Protocol to execute routed messages and return transformed results."""

        def execute(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> (
            FlextProtocolsResult.Result[t.RuntimeAtomic]
            | t.Container
            | BaseModel
            | None
        ): ...

    @runtime_checkable
    class Dispatcher(FlextProtocolsBase.Base, Protocol):
        """Protocol for dispatching and publishing messages in CQRS systems."""

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
    class AutoDiscoverableHandler(Protocol):
        """Protocol for handlers that can inspect message types at runtime."""

        def can_handle(self, message_type: type) -> bool: ...

    @runtime_checkable
    class CommandBus(Protocol):
        """Protocol for command bus implementations with dispatch/publish/register."""

        def dispatch(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.Result[t.RuntimeAtomic]: ...

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
        """Protocol for middleware layers in handler execution chains."""

        def process[TResult](
            self,
            command: FlextProtocolsBase.Model,
            next_handler: Callable[
                [FlextProtocolsBase.Model],
                FlextProtocolsResult.Result[TResult],
            ],
        ) -> FlextProtocolsResult.Result[TResult]: ...


__all__ = ["FlextProtocolsHandler"]
