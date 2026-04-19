"""FlextProtocolsHandler - handler, bus, registry, middleware protocols.

Mirrors the public surface of ``FlextHandlers``, ``FlextDispatcher``, and
related concrete classes so that ``p.*`` protocols can be used in type
annotations everywhere instead of concrete types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocolsBase, FlextProtocolsResult

if TYPE_CHECKING:
    from flext_core import (
        c,
        t,
    )


class FlextProtocolsHandler:
    """Protocols for CQRS handlers and message routing."""

    # ------------------------------------------------------------------
    # Handler — mirrors FlextHandlers public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class Handler[MessageT: FlextProtocolsBase.Model, ResultT](
        FlextProtocolsBase.Base,
        Protocol,
    ):
        """Typed message handler contract.

        Mirrors the public instance API of ``FlextHandlers[MessageT, ResultT]``
        so consumers can depend on ``p.Handler`` for typing instead of the
        concrete class.
        """

        # --- identity ---

        @property
        def handler_name(self) -> str:
            """Handler name from configuration."""
            ...

        @property
        def mode(self) -> c.HandlerType:
            """Handler mode (command, query, event, operation, saga)."""
            ...

        # --- core contract ---

        def can_handle(self, message_type: type) -> bool:
            """Check if handler can process the given message type."""
            ...

        def handle(
            self,
            message: MessageT,
        ) -> FlextProtocolsResult.Result[ResultT]:
            """Core business logic — must be implemented by concrete handlers."""
            ...

        # --- pipeline ---

        def execute(
            self,
            message: MessageT,
        ) -> FlextProtocolsResult.Result[ResultT]:
            """Execute handler with validation and error handling pipeline."""
            ...

        def dispatch_message(
            self,
            message: MessageT,
            operation: str = ...,
        ) -> FlextProtocolsResult.Result[ResultT]:
            """Dispatch message through the full handler pipeline."""
            ...

        def validate_message(
            self,
            data: MessageT,
        ) -> FlextProtocolsResult.Result[bool]:
            """Validate input data before execution."""
            ...

        # --- callable ---

        def __call__(
            self,
            message: MessageT,
        ) -> FlextProtocolsResult.Result[ResultT]:
            """Callable interface for dispatcher integration."""
            ...

        # --- context & metrics ---

        def push_context(
            self,
            ctx: Mapping[str, t.Container],
        ) -> FlextProtocolsResult.Result[bool]:
            """Push execution context onto the local handler stack."""
            ...

        def pop_context(
            self,
        ) -> FlextProtocolsResult.Result[t.FlatContainerMapping]:
            """Pop execution context from the local handler stack."""
            ...

        def record_metric(
            self,
            name: str,
            value: t.MetadataAttributeValue,
        ) -> FlextProtocolsResult.Result[bool]:
            """Record a metric value in the current handler state."""
            ...

    # ------------------------------------------------------------------
    # Dispatch-style structural protocols (used by dispatcher routing)
    # ------------------------------------------------------------------

    @runtime_checkable
    class DispatchMessage(Protocol):
        """Protocol for routing a message through a dispatch path."""

        def dispatch_message(
            self,
            message: FlextProtocolsBase.Routable,
            operation: str = ...,
        ) -> FlextProtocolsResult.ResultLike[t.RuntimeData] | t.RuntimeData | None: ...

    @runtime_checkable
    class Handle(Protocol):
        """Protocol for handle behaviors in CQRS message workflows."""

        def handle(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.ResultLike[t.RuntimeData] | t.RuntimeData | None: ...

    @runtime_checkable
    class Execute(Protocol):
        """Protocol to execute routed messages and return transformed results."""

        def execute(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.ResultLike[t.RuntimeData] | t.RuntimeData | None: ...

    @runtime_checkable
    class AutoDiscoverableHandler(Protocol):
        """Protocol for handlers that can inspect message types at runtime."""

        def can_handle(self, message_type: type) -> bool: ...

    # ------------------------------------------------------------------
    # Dispatcher — inlined from _MessageBusBase, mirrors FlextDispatcher
    # ------------------------------------------------------------------

    @runtime_checkable
    class Dispatcher(FlextProtocolsBase.Base, Protocol):
        """Protocol for dispatching and publishing messages in CQRS systems.

        Mirrors the public surface of ``FlextDispatcher``.
        """

        def dispatch(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.Result[t.RuntimeData]:
            """Route a CQRS message to a registered handler."""
            ...

        def publish(
            self,
            event: FlextProtocolsBase.Routable | Sequence[FlextProtocolsBase.Routable],
        ) -> FlextProtocolsResult.Result[bool]:
            """Publish event(s) to all registered subscribers."""
            ...

        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a handler for message routing."""
            ...

    # ------------------------------------------------------------------
    # CommandBus — dispatch + register only (no publish — SRP)
    # ------------------------------------------------------------------

    @runtime_checkable
    class CommandBus(Protocol):
        """Protocol for command bus implementations with dispatch and registration.

        Unlike ``Dispatcher``, a command bus does NOT publish events.
        """

        def dispatch(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.Result[t.RuntimeData]:
            """Dispatch a command to a registered handler."""
            ...

        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a handler for command routing."""
            ...

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

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
