"""FlextProtocolsHandler - handler, bus, registry, middleware protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.result import FlextProtocolsResult

if TYPE_CHECKING:
    from flext_core._typings.services import FlextTypesServices


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
            operation: str = ...,
        ) -> (
            FlextProtocolsResult.Result[FlextTypesServices.RuntimeAtomic]
            | FlextTypesServices.RuntimeAtomic
            | None
        ): ...

    @runtime_checkable
    class Handle(Protocol):
        """Protocol for handle behaviors in CQRS message workflows."""

        def handle(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> (
            FlextProtocolsResult.Result[FlextTypesServices.RuntimeAtomic]
            | FlextTypesServices.RuntimeAtomic
            | None
        ): ...

    @runtime_checkable
    class Execute(Protocol):
        """Protocol to execute routed messages and return transformed results."""

        def execute(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> (
            FlextProtocolsResult.Result[FlextTypesServices.RuntimeAtomic]
            | FlextTypesServices.RuntimeAtomic
            | None
        ): ...

    @runtime_checkable
    class _MessageBusBase(Protocol):
        """Shared protocol for publish/register_handler on bus-like types."""

        def publish(
            self,
            event: FlextProtocolsBase.Routable | Sequence[FlextProtocolsBase.Routable],
        ) -> FlextProtocolsResult.Result[bool]: ...

        def register_handler(
            self,
            handler: FlextTypesServices.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> FlextProtocolsResult.Result[bool]: ...

    @runtime_checkable
    class Dispatcher(_MessageBusBase, FlextProtocolsBase.Base, Protocol):
        """Protocol for dispatching and publishing messages in CQRS systems."""

        def dispatch(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.Result[FlextTypesServices.RuntimeAtomic]: ...

    @runtime_checkable
    class AutoDiscoverableHandler(Protocol):
        """Protocol for handlers that can inspect message types at runtime."""

        def can_handle(self, message_type: type) -> bool: ...

    @runtime_checkable
    class CommandBus(_MessageBusBase, Protocol):
        """Protocol for command bus implementations with dispatch/publish/register."""

        def dispatch(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.Result[FlextTypesServices.RuntimeAtomic]: ...

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


__all__: list[str] = ["FlextProtocolsHandler"]
