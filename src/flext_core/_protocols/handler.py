"""FlextProtocolsHandler - handler, bus, registry, middleware protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.result import FlextProtocolsResult

if TYPE_CHECKING:
    from flext_core import r


class FlextProtocolsHandler:
    """Protocols for CQRS handlers and message routing."""

    @runtime_checkable
    class Handler[MessageT: FlextProtocolsBase.Model, ResultT](
        FlextProtocolsBase.Base, Protocol
    ):
        """Command/Query handler interface (generic).

        Reflects real implementations like FlextHandlers which provide
        comprehensive validation and execution pipelines for CQRS handlers.

        Type Parameters:
        - MessageT: Type of message handled (command, query, or event)
        - ResultT: Type of result returned by handler
        """

        def can_handle(self, message_type: type) -> bool:
            """Check if handler can handle the specified message type.

            Reflects real implementations like FlextHandlers.can_handle() which
            checks message type compatibility using duck typing and class hierarchy.
            """
            ...

        def handle(self, message: MessageT) -> r[ResultT]:
            """Handle message - core business logic method.

            Reflects real implementations like FlextHandlers.handle() which
            executes handler business logic for commands, queries, or events.
            """
            ...

    @runtime_checkable
    class CommandBus(FlextProtocolsBase.Base, Protocol):
        """Command routing and execution protocol.

        Matches FlextDispatcher: strict handler registration and message dispatch.
        """

        def dispatch(
            self, message: FlextProtocolsBase.Routable
        ) -> FlextProtocolsResult.Result[FlextProtocolsBase.Model]:
            """Dispatch a CQRS message to its registered handler."""
            ...

        def publish(
            self,
            event: FlextProtocolsBase.Routable | Sequence[FlextProtocolsBase.Routable],
        ) -> FlextProtocolsResult.Result[bool]:
            """Publish events to registered subscribers."""
            ...

        def register_handler(
            self, handler: t.HandlerLike, *, is_event: bool = False
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a handler with route auto-discovery.

            Handler must expose message_type, event_type, or can_handle
            for route resolution.
            """
            ...

    @runtime_checkable
    class Middleware(FlextProtocolsBase.Base, Protocol):
        """Processing pipeline middleware."""

        def process[TResult](
            self,
            command: FlextProtocolsBase.Model,
            next_handler: Callable[
                [FlextProtocolsBase.Model], FlextProtocolsResult.Result[TResult]
            ],
        ) -> FlextProtocolsResult.Result[TResult]:
            """Process command."""
            ...


__all__ = ["FlextProtocolsHandler"]
