"""FlextProtocolsHandler - handler, bus, registry, middleware protocols.

Mirrors the public surface of ``FlextHandlers``, ``FlextDispatcher``, and
related concrete classes so that ``p.*`` protocols can be used in type
annotations everywhere instead of concrete types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from .base import FlextProtocolsBase as p

from collections.abc import Callable

from flext_core import FlextConstants as c

if TYPE_CHECKING:
    # NOTE (multi-agent, mro-wkii.17.26): FlextTypesServices imports this
    # protocol while t is still being composed; t is annotation-only here.
    from flext_core import FlextTypes as t

from .container import FlextProtocolsContainer as pc
from .result import FlextProtocolsResult as pr


class FlextProtocolsHandler:
    """Protocols for CQRS handlers and message routing."""

    # FLEXT: annotations remain structural; concrete models are construction-only.
    @runtime_checkable
    class HandlerConfig(p.BaseModel, Protocol):
        """Validated handler configuration consumed by the runtime pipeline."""

        @property
        def handler_id(self) -> str: ...

        @property
        def handler_name(self) -> str: ...

        @property
        def handler_mode(self) -> c.HandlerType: ...

    @runtime_checkable
    class ExecutionContext(p.BaseModel, Protocol):
        """Field-level contract for active handler execution state."""

        @property
        def handler_name(self) -> str: ...

        @property
        def handler_mode(self) -> c.HandlerType: ...

        @property
        def started_at(self) -> float | None: ...

        @property
        def metrics_state_data(self) -> pc.MutableRootDict[t.JsonPayload]: ...

        @property
        def execution_time_ms(self) -> float: ...

        @override
        def model_copy(
            self,
            *,
            update: t.MappingKV[
                str, t.JsonPayload | p.BaseModel | t.SequenceOf[p.BaseModel]
            ]
            | None = None,
            deep: bool = False,
        ) -> FlextProtocolsHandler.ExecutionContext: ...

    @runtime_checkable
    class HandlerRuntimeState(p.BaseModel, Protocol):
        """Field-level contract for copy-on-write handler pipeline state."""

        @property
        def execution_context(self) -> FlextProtocolsHandler.ExecutionContext: ...

        @property
        def context_stack(
            self,
        ) -> t.SequenceOf[FlextProtocolsHandler.ExecutionContext]: ...

        @override
        def model_copy(
            self,
            *,
            update: t.MappingKV[
                str, t.JsonPayload | p.BaseModel | t.SequenceOf[p.BaseModel]
            ]
            | None = None,
            deep: bool = False,
        ) -> FlextProtocolsHandler.HandlerRuntimeState: ...

    @runtime_checkable
    class DecoratorConfig(p.BaseModel, Protocol):
        """Handler decorator metadata consumed by discovery."""

        @property
        def command(self) -> type: ...

        @property
        def priority(self) -> int: ...

    # ------------------------------------------------------------------
    # Handler — mirrors FlextHandlers public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class Handler[MessageT, ResultT](p.Base, Protocol):
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

        def handle(self, message: MessageT) -> pr.Result[ResultT]:
            """Core business logic — must be implemented by concrete handlers."""
            ...

        # --- pipeline ---

        def execute(self, message: MessageT) -> pr.Result[ResultT]:
            """Execute handler with validation and error handling pipeline."""
            ...

        def dispatch_message(
            self, message: MessageT, operation: str = ...
        ) -> pr.Result[ResultT]:
            """Dispatch message through the full handler pipeline."""
            ...

        def validate_message(self, data: MessageT) -> pr.Result[bool]:
            """Validate input data before execution."""
            ...

        # --- callable ---

        def __call__(self, message: MessageT) -> pr.Result[ResultT]:
            """Callable interface for dispatcher integration."""
            ...

        # --- context & metrics ---

        def push_context(
            self, ctx: t.JsonMapping | FlextProtocolsHandler.ExecutionContext
        ) -> pr.Result[bool]:
            """Push execution context onto the local handler stack."""
            ...

        def pop_context(self) -> pr.Result[pc.RootDict[t.JsonPayload]]:
            """Pop execution context from the local handler stack."""
            ...

        def record_metric(self, name: str, value: t.JsonPayload) -> pr.Result[bool]:
            """Record a metric value in the current handler state."""
            ...

    # ------------------------------------------------------------------
    # Dispatch-style structural protocols (used by dispatcher routing)
    # ------------------------------------------------------------------

    @runtime_checkable
    class DispatchMessage(Protocol):
        """Protocol for routing a message through a dispatch path."""

        def dispatch_message(
            self, message: p.Routable, operation: str = ...
        ) -> pr.ResultLike[t.JsonPayload] | t.JsonPayload | None: ...

    @runtime_checkable
    class Handle(Protocol):
        """Protocol for handle behaviors in CQRS message workflows."""

        def handle(
            self, message: p.Routable
        ) -> pr.ResultLike[t.JsonPayload] | t.JsonPayload | None: ...

    @runtime_checkable
    class Execute(Protocol):
        """Protocol to execute routed messages and return transformed results."""

        def execute(
            self, message: p.Routable
        ) -> pr.ResultLike[t.JsonPayload] | t.JsonPayload | None: ...

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

        def dispatch(self, message: p.Routable) -> pr.Result[t.JsonPayload]:
            """Route a CQRS message to a registered handler."""
            ...

        def publish(
            self, event: p.Routable | t.SequenceOf[p.Routable]
        ) -> pr.Result[bool]:
            """Publish event(s) to all registered subscribers."""
            ...

        def register_handler(
            self, handler: t.DispatchableHandler, *, is_event: bool = False
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

        def dispatch(self, message: p.Routable) -> pr.Result[t.JsonPayload]:
            """Dispatch a command to a registered handler."""
            ...

        def register_handler(
            self, handler: t.DispatchableHandler, *, is_event: bool = False
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
            command: p.BaseModel,
            next_handler: Callable[[p.BaseModel], pr.Result[TResult]],
        ) -> pr.Result[TResult]: ...


__all__: list[str] = ["FlextProtocolsHandler"]
