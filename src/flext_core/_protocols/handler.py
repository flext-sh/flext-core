"""FlextProtocolsHandler - handler, bus, registry, middleware protocols.

Mirrors the public surface of ``FlextHandlers``, ``FlextDispatcher``, and
related concrete classes so that ``p.*`` protocols can be used in type
annotations everywhere instead of concrete types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols import FlextProtocolsBase, FlextProtocolsResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core import c
    from flext_core._constants import FlextConstantsCqrs, FlextConstantsStatus
    from flext_core._protocols import FlextProtocolsContainer, FlextProtocolsContainers
    from flext_core._typings import (
        FlextTypesAnnotateds,
        FlextTypesCore,
        FlextTypesPydantic,
    )


class FlextProtocolsHandler:
    """Protocols for CQRS handlers and message routing."""

    type HandlerCallable = t.HandlerCallable[
        FlextProtocolsResult.ResultView[t.ScalarOrModel]
    ]
    type DispatchableHandler = t.DispatchableHandler[
        FlextProtocolsHandler.DispatchMessage
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Execute
        | FlextProtocolsHandler.AutoDiscoverableHandler,
        FlextProtocolsBase.Routable,
        FlextProtocolsResult.ResultView[t.JsonPayload],
    ]
    type ResolvedHandlerCallable = t.ResolvedHandlerCallable[
        FlextProtocolsResult.ResultView[t.JsonPayload]
    ]
    type RoutedHandlerCallable = t.RoutedHandlerCallable[
        FlextProtocolsBase.Routable,
        FlextProtocolsResult.ResultView[t.JsonPayload],
    ]
    type DispatcherRouteEntry = t.DispatcherRouteEntry[
        FlextProtocolsHandler.DispatchableHandler,
        FlextProtocolsHandler.RoutedHandlerCallable,
    ]

    # FLEXT: annotations remain structural; concrete models are construction-only.
    @runtime_checkable
    class ExecutionContext(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def handler_name(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Name of the handler being executed."""
            ...

        @property
        def handler_mode(self) -> FlextConstantsCqrs.HandlerType:
            """Mode of handler execution."""
            ...

        @property
        def started_at(self) -> float | None:
            """Monotonic start timestamp used to compute execution time."""
            ...

        @property
        def metrics_state_data(self) -> FlextProtocolsContainers.Dict:
            """Mutable metrics payload for the active handler execution."""
            ...

        @property
        def execution_time_ms(self) -> float:
            """Elapsed execution time in milliseconds (0 until started)."""
            ...

    @runtime_checkable
    class HandlerRuntimeState(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def execution_context(self) -> FlextProtocolsHandler.ExecutionContext:
            """Execution context for the active handler."""
            ...

        @property
        def context_stack(self) -> Sequence[FlextProtocolsHandler.ExecutionContext]:
            """Stack of nested execution contexts."""
            ...

        @property
        def handler_name(self) -> str:
            """Active handler name taken from the execution context."""
            ...

        @property
        def handler_mode(self) -> FlextConstantsCqrs.HandlerType:
            """Active handler mode taken from the execution context."""
            ...

    @runtime_checkable
    class DecoratorConfig(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def command(self) -> type:
            """Command type this handler processes."""
            ...

        @property
        def priority(self) -> FlextTypesPydantic.NonNegativeInt:
            """Handler priority (higher = processed first)."""
            ...

        @property
        def timeout(self) -> float | None:
            """Handler execution timeout in seconds."""
            ...

        @property
        def middleware(self) -> Sequence[type[FlextProtocolsHandler.Middleware]]:
            """Middleware types to apply to this handler."""
            ...

    # ------------------------------------------------------------------
    # Handler — mirrors FlextHandlers public instance surface
    # ------------------------------------------------------------------

    @runtime_checkable
    class Handler[MessageT, ResultT](FlextProtocolsBase.Base, Protocol):
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

        def handle(self, message: MessageT) -> FlextProtocolsResult.Result[ResultT]:
            """Core business logic — must be implemented by concrete handlers."""
            ...

        # --- pipeline ---

        def execute(self, message: MessageT) -> FlextProtocolsResult.Result[ResultT]:
            """Execute handler with validation and error handling pipeline."""
            ...

        def dispatch_message(
            self,
            message: MessageT,
            operation: str = ...,
        ) -> FlextProtocolsResult.Result[ResultT]:
            """Dispatch message through the full handler pipeline."""
            ...

        def validate_message(self, data: MessageT) -> FlextProtocolsResult.Result[bool]:
            """Validate input data before execution."""
            ...

        # --- callable ---

        def __call__(self, message: MessageT) -> FlextProtocolsResult.Result[ResultT]:
            """Callable interface for dispatcher integration."""
            ...

        # --- context & metrics ---

        def push_context(
            self,
            ctx: t.JsonMapping | FlextProtocolsHandler.ExecutionContext,
        ) -> FlextProtocolsResult.Result[bool]:
            """Push execution context onto the local handler stack."""
            ...

        def pop_context(
            self,
        ) -> FlextProtocolsResult.Result[
            FlextProtocolsContainer.RootDict[t.JsonPayload]
        ]:
            """Pop execution context from the local handler stack."""
            ...

        def record_metric(
            self,
            name: str,
            value: t.JsonPayload,
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
        ) -> FlextProtocolsResult.ResultView[t.JsonPayload] | t.JsonPayload | None: ...

    @runtime_checkable
    class Handle(Protocol):
        """Protocol for handle behaviors in CQRS message workflows."""

        def handle(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.ResultView[t.JsonPayload] | t.JsonPayload | None: ...

    @runtime_checkable
    class Execute(Protocol):
        """Protocol to execute routed messages and return transformed results."""

        def execute(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.ResultView[t.JsonPayload] | t.JsonPayload | None: ...

    @runtime_checkable
    class AutoDiscoverableHandler(Protocol):
        """Protocol for handlers that can inspect message types at runtime."""

        def can_handle(self, message_type: type) -> bool: ...

    # ------------------------------------------------------------------
    # Dispatcher — inlined from _MessageBusBase, mirrors FlextDispatcher
    # ------------------------------------------------------------------

    @runtime_checkable
    class HandlerRegistration(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def handler(self) -> FlextProtocolsHandler.DispatchableHandler:
            """Original handler whose identity owns the binding."""
            ...

        @property
        def message_type(self) -> FlextTypesCore.RegistryBindingKey | None:
            """Explicit message type or route name."""
            ...

        @property
        def is_event(self) -> bool:
            """Whether this binding subscribes to events."""
            ...

    @runtime_checkable
    class HandlerRegistrationReport(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def registered(self) -> tuple[FlextProtocolsHandler.HandlerRegistration, ...]:
            """Bindings newly published by the batch."""
            ...

        @property
        def skipped(self) -> tuple[FlextProtocolsHandler.HandlerRegistration, ...]:
            """Bindings already owned by the same handler."""
            ...

    @runtime_checkable
    class Dispatcher(FlextProtocolsBase.Base, Protocol):
        """Protocol for dispatching and publishing messages in CQRS systems.

        Mirrors the public surface of ``FlextDispatcher``.
        """

        def dispatch(
            self,
            message: FlextProtocolsBase.Routable,
        ) -> FlextProtocolsResult.Result[t.JsonPayload]:
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
            handler: FlextProtocolsHandler.DispatchableHandler,
            *,
            message_type: t.RegistryBindingKey | None = None,
            is_event: bool = False,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a handler for message routing."""
            ...

        def register_handlers(
            self, bindings: Sequence[FlextProtocolsHandler.HandlerRegistration]
        ) -> FlextProtocolsResult.Result[
            FlextProtocolsHandler.HandlerRegistrationReport
        ]:
            """Atomically bind handlers and report new and idempotent bindings."""
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
        ) -> FlextProtocolsResult.Result[t.JsonPayload]:
            """Dispatch a command to a registered handler."""
            ...

        def register_handler(
            self,
            handler: FlextProtocolsHandler.DispatchableHandler,
            *,
            message_type: t.RegistryBindingKey | None = None,
            is_event: bool = False,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a handler for command routing."""
            ...

        def register_handlers(
            self, bindings: Sequence[FlextProtocolsHandler.HandlerRegistration]
        ) -> FlextProtocolsResult.Result[
            FlextProtocolsHandler.HandlerRegistrationReport
        ]:
            """Atomically bind handlers and report new and idempotent bindings."""
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

    @runtime_checkable
    class RegistrationDetails(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def registration_id(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Unique registration identifier."""
            ...

        @property
        def handler_mode(self) -> FlextConstantsCqrs.HandlerType:
            """Handler mode (command, query, or event)."""
            ...

        @property
        def timestamp(self) -> str:
            """ISO 8601 timestamp recording when the registration entry was created."""
            ...

        @property
        def status(self) -> FlextConstantsStatus.Status:
            """Current registration status."""
            ...

    @runtime_checkable
    class CombinedRailwayOptions(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def enabled(self) -> bool:
            """Whether combined() applies railway wrapping."""
            ...

        @property
        def error_code(self) -> str | None:
            """Error code passed to railway() when railway wrapping is enabled."""
            ...


__all__: list[str] = ["FlextProtocolsHandler"]
