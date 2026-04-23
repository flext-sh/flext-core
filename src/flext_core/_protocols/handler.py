"""FlextProtocolsHandler - handler, bus, registry, middleware protocols.

Mirrors the public surface of ``FlextHandlers``, ``FlextDispatcher``, and
related concrete classes so that ``p.*`` protocols can be used in type
annotations everywhere instead of concrete types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence,
)
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase as p
from flext_core._protocols.result import FlextProtocolsResult as pr

if TYPE_CHECKING:
    from flext_core.constants import c
    from flext_core.typings import t


class FlextProtocolsHandler:
    """Protocols for CQRS handlers and message routing."""

    # ------------------------------------------------------------------
    # Handler — mirrors FlextHandlers public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class Handler[MessageT: p.Model, ResultT](
        p.Base,
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
        ) -> pr.Result[ResultT]:
            """Core business logic — must be implemented by concrete handlers."""
            ...

        # --- pipeline ---

        def execute(
            self,
            message: MessageT,
        ) -> pr.Result[ResultT]:
            """Execute handler with validation and error handling pipeline."""
            ...

        def dispatch_message(
            self,
            message: MessageT,
            operation: str = ...,
        ) -> pr.Result[ResultT]:
            """Dispatch message through the full handler pipeline."""
            ...

        def validate_message(
            self,
            data: MessageT,
        ) -> pr.Result[bool]:
            """Validate input data before execution."""
            ...

        # --- callable ---

        def __call__(
            self,
            message: MessageT,
        ) -> pr.Result[ResultT]:
            """Callable interface for dispatcher integration."""
            ...

        # --- context & metrics ---

        def push_context(
            self,
            ctx: t.JsonMapping,
        ) -> pr.Result[bool]:
            """Push execution context onto the local handler stack."""
            ...

        def pop_context(
            self,
        ) -> pr.Result[t.JsonMapping]:
            """Pop execution context from the local handler stack."""
            ...

        def record_metric(
            self,
            name: str,
            value: t.MetadataData,
        ) -> pr.Result[bool]:
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
            message: p.Routable,
            operation: str = ...,
        ) -> pr.ResultLike[t.RuntimeData] | t.RuntimeData | None: ...

    @runtime_checkable
    class Handle(Protocol):
        """Protocol for handle behaviors in CQRS message workflows."""

        def handle(
            self,
            message: p.Routable,
        ) -> pr.ResultLike[t.RuntimeData] | t.RuntimeData | None: ...

    @runtime_checkable
    class Execute(Protocol):
        """Protocol to execute routed messages and return transformed results."""

        def execute(
            self,
            message: p.Routable,
        ) -> pr.ResultLike[t.RuntimeData] | t.RuntimeData | None: ...

    @runtime_checkable
    class AutoDiscoverableHandler(Protocol):
        """Protocol for handlers that can inspect message types at runtime."""

        def can_handle(self, message_type: type) -> bool: ...

    # ------------------------------------------------------------------
    # Dispatcher — inlined from _MessageBusBase, mirrors FlextDispatcher
    # ------------------------------------------------------------------

    @runtime_checkable
    class Dispatcher(p.Base, Protocol):
        """Protocol for dispatching and publishing messages in CQRS systems.

        Mirrors the public surface of ``FlextDispatcher``.
        """

        def dispatch(
            self,
            message: p.Routable,
        ) -> pr.Result[t.RuntimeData]:
            """Route a CQRS message to a registered handler."""
            ...

        def publish(
            self,
            event: p.Routable | Sequence[p.Routable],
        ) -> pr.Result[bool]:
            """Publish event(s) to all registered subscribers."""
            ...

        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> pr.Result[bool]:
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
            message: p.Routable,
        ) -> pr.Result[t.RuntimeData]:
            """Dispatch a command to a registered handler."""
            ...

        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> pr.Result[bool]:
            """Register a handler for command routing."""
            ...

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    @runtime_checkable
    class Middleware(p.Base, Protocol):
        """Protocol for middleware layers in handler execution chains."""

        def process[TResult](
            self,
            command: p.Model,
            next_handler: Callable[
                [p.Model],
                pr.Result[TResult],
            ],
        ) -> pr.Result[TResult]: ...


__all__: list[str] = ["FlextProtocolsHandler"]
