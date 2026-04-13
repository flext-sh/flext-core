"""Handler management patterns extracted from FlextModels.

This module contains the FlextModelsHandler class with all handler-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Handler instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import MutableSequence, Sequence
from typing import Annotated, ClassVar, Self

from flext_core import (
    FlextModelsBase as m,
    FlextUtilitiesPydantic,
    c,
    p,
    r,
    t,
)


class FlextModelsHandler:
    """Handler management pattern container class.

    This class acts as a namespace container for handler management patterns.
    All nested classes are accessed via FlextModels.Handler.* in the main models.py.
    """

    class RegistrationResult(m.ArbitraryTypesModel):
        """Result of a handler registration operation.

        Provides structured feedback on the outcome of a handler registration,
        including status, mode, and identification.
        """

        handler_name: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(description="Name of the handler"),
        ]
        status: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(
                description="Registration status (registered, skipped, failed)",
            ),
        ]
        mode: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(
                description="Registration mode (auto_discovery, explicit)"
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType | None,
            FlextUtilitiesPydantic.Field(
                None, description="Handler mode (command/query/event)"
            ),
        ] = None
        message_type: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                None,
                description="Message type bound (for explicit mode)",
            ),
        ] = None
        _GETITEM_FIELDS: ClassVar[frozenset[str]] = frozenset({
            "handler_name",
            c.FIELD_STATUS,
            "mode",
            c.FIELD_HANDLER_MODE,
            "message_type",
        })

        def __getitem__(self, key: str) -> t.ValueOrModel:
            match key:
                case "handler_name":
                    return self.handler_name
                case c.FIELD_STATUS:
                    return self.status
                case "mode":
                    return self.mode
                case c.FIELD_HANDLER_MODE:
                    return self.handler_mode
                case "message_type":
                    return self.message_type
                case _:
                    raise KeyError(key)

    class RegistrationDetails(m.ArbitraryTypesModel):
        """Registration details for handler registration tracking.

        Tracks metadata about handler registrations in the CQRS system,
        including unique identification, timing, and status information.

        This model is used by FlextRegistry to track which handlers have been
        registered with the dispatcher and monitor their lifecycle.

        Attributes:
            registration_id: Unique identifier for this registration
            handler_mode: Mode of handler (command, query, or event)
            timestamp: ISO 8601 timestamp when registration occurred
            status: Current status of the registration

        Examples:
            >>> details = FlextModelsHandler.RegistrationDetails(
            ...     registration_id="reg-123",
            ...     handler_mode="command",
            ...     timestamp="2025-01-01T00:00:00Z",
            ...     status="running",
            ... )
            >>> details.registration_id
            'reg-123'

        """

        model_config: ClassVar[c.ConfigDict] = c.ConfigDict(
            json_schema_extra={
                "title": "RegistrationDetails",
                "description": "Handler registration tracking details",
            },
        )
        registration_id: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            FlextUtilitiesPydantic.Field(
                default=c.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = c.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            FlextUtilitiesPydantic.Field(
                description="ISO 8601 timestamp recording when the registration entry was created.",
                title="Registration Timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=c.PATTERN_ISO8601_TIMESTAMP,
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: c.DEFAULT_EMPTY_STRING)
        status: Annotated[
            c.CommonStatus,
            FlextUtilitiesPydantic.Field(
                default=c.CommonStatus.RUNNING,
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = c.CommonStatus.RUNNING

    class ExecutionContext(m.ArbitraryTypesModel):
        """Handler execution context for tracking handler performance and state.

        Attributes:
            handler_name: Name of the handler being executed
            handler_mode: Mode of execution (command, query, or event)

        """

        _start_time: float | None = FlextUtilitiesPydantic.PrivateAttr(
            default_factory=lambda: None,
        )
        _metrics_state: t.Dict | None = FlextUtilitiesPydantic.PrivateAttr(
            default_factory=lambda: None,
        )

        model_config: ClassVar[c.ConfigDict] = c.ConfigDict(
            json_schema_extra={
                "title": "HandlerExecutionContext",
                "description": "Handler execution context for tracking performance and state",
            },
        )
        handler_name: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            FlextUtilitiesPydantic.Field(
                description="Mode of handler execution",
                examples=["command", "query", "event"],
            ),
        ]

        @FlextUtilitiesPydantic.computed_field()
        def execution_time_ms(self) -> float:
            """Get execution time in milliseconds."""
            if self._start_time is None:
                return 0.0
            start_time: float = self._start_time
            elapsed: float = time.time() - start_time
            return round(elapsed * c.DEFAULT_SIZE, 2)

        @FlextUtilitiesPydantic.computed_field()
        def has_metrics(self) -> bool:
            """Check if metrics have been recorded."""
            return self._metrics_state is not None and bool(self._metrics_state)

        @FlextUtilitiesPydantic.computed_field()
        def running(self) -> bool:
            """Check if execution is currently running."""
            return self._start_time is not None

        @FlextUtilitiesPydantic.computed_field()
        def metrics_state(self) -> t.Dict:
            """Get current metrics state.

            Returns:
                Dictionary containing metrics state (empty ConfigurationDict if not set)

            """
            if self._metrics_state is None:
                self._metrics_state = t.Dict(root={})
            metrics_state_val: t.Dict = self._metrics_state
            return metrics_state_val

        @classmethod
        def create_for_handler(
            cls,
            handler_name: str,
            handler_mode: c.HandlerType,
        ) -> Self:
            """Create execution context for a handler.

            Args:
                handler_name: Name of the handler
                handler_mode: Mode of the handler (command/query/event)

            Returns:
                New HandlerExecutionContext instance

            """
            return cls(handler_name=handler_name, handler_mode=handler_mode)

        def reset(self) -> None:
            """Reset execution context."""
            self._start_time = None
            self._metrics_state = None

        def apply_metrics_state(self, state: t.Dict) -> None:
            """Apply metrics state to the execution context."""
            self._metrics_state = state

        def start_execution(self) -> None:
            """Start execution timing.

            Records the current time as the start time for execution metrics.
            Should be called at the beginning of handler execution.

            Examples:
                >>> context = FlextModelsHandler.ExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.start_execution()

            """
            self._start_time = time.time()

    class MetricsTracker(m.ArbitraryTypesModel):
        """Tracks handler execution metrics via ExecutionContext.

        Delegates metric storage to an internal ExecutionContext instance,
        providing a simplified record/get interface for CQRS handler pipelines.
        """

        _context: FlextModelsHandler.ExecutionContext = (
            FlextUtilitiesPydantic.PrivateAttr(
                default_factory=lambda: (
                    FlextModelsHandler.ExecutionContext.create_for_handler(
                        handler_name="metrics",
                        handler_mode=c.HandlerType.OPERATION,
                    )
                ),
            )
        )

        @FlextUtilitiesPydantic.computed_field()
        @property
        def metrics(self) -> t.ConfigMap:
            """Return all recorded metrics as a ConfigMap."""
            raw_state = self._context.metrics_state
            state: t.Dict = (
                raw_state if isinstance(raw_state, t.Dict) else t.Dict(root={})
            )
            return t.ConfigMap(root=dict(state.root))

        @classmethod
        def create_for_context(
            cls,
            context: FlextModelsHandler.ExecutionContext,
        ) -> Self:
            """Create a tracker bound to an existing execution context."""
            tracker = cls()
            tracker._context = context
            return tracker

        def record_metric(
            self,
            name: str,
            value: t.MetadataAttributeValue,
        ) -> p.Result[bool]:
            """Record a named metric value in the tracker."""
            raw_state = self._context.metrics_state
            state: t.Dict = (
                raw_state if isinstance(raw_state, t.Dict) else t.Dict(root={})
            )
            current = dict(state.root)
            current[name] = value
            self._context.apply_metrics_state(t.Dict(root=current))
            return r[bool].ok(True)

    class ContextStack(m.ArbitraryTypesModel):
        """Manages a stack of ExecutionContext instances for CQRS handler pipelines."""

        _stack: MutableSequence[FlextModelsHandler.ExecutionContext] = (
            FlextUtilitiesPydantic.PrivateAttr(
                default_factory=lambda: list[FlextModelsHandler.ExecutionContext](),
            )
        )

        def current_context(self) -> FlextModelsHandler.ExecutionContext | None:
            """Return the current top-of-stack execution context, or None."""
            if self._stack:
                return self._stack[-1]
            return None

        def pop_context(self) -> p.Result[t.ScalarMapping]:
            """Pop and return the top context from the stack as a scalar dict."""
            if self._stack:
                popped = self._stack.pop()
                return r[t.ScalarMapping].ok({
                    "handler_name": popped.handler_name,
                    c.FIELD_HANDLER_MODE: popped.handler_mode,
                })
            return r[t.ScalarMapping].ok({})

        def push_context(
            self,
            ctx: FlextModelsHandler.ExecutionContext | t.RecursiveContainerMapping,
        ) -> p.Result[bool]:
            """Push an execution context or mapping onto the context stack."""
            if isinstance(ctx, FlextModelsHandler.ExecutionContext):
                self._stack.append(ctx)
                return r[bool].ok(True)
            ctx_mapping: t.RecursiveContainerMapping = {
                str(k): v for k, v in ctx.items()
            }
            handler_name: str = str(
                ctx_mapping.get("handler_name", c.IDENTIFIER_UNKNOWN),
            )
            handler_mode_str: str = str(
                ctx_mapping.get(c.FIELD_HANDLER_MODE, c.HandlerType.OPERATION),
            )
            handler_mode_literal: c.HandlerType = (
                c.HandlerType.COMMAND
                if handler_mode_str == c.HandlerType.COMMAND
                else c.HandlerType.QUERY
                if handler_mode_str == c.HandlerType.QUERY
                else c.HandlerType.EVENT
                if handler_mode_str == c.HandlerType.EVENT
                else c.HandlerType.SAGA
                if handler_mode_str == "saga"
                else c.HandlerType.OPERATION
            )
            execution_ctx = FlextModelsHandler.ExecutionContext.create_for_handler(
                handler_name=handler_name,
                handler_mode=handler_mode_literal,
            )
            self._stack.append(execution_ctx)
            return r[bool].ok(True)

    class HandlerRuntimeState(m.ArbitraryTypesModel):
        """Centralized mutable runtime state for a handler pipeline."""

        execution_context: Annotated[
            FlextModelsHandler.ExecutionContext,
            FlextUtilitiesPydantic.Field(
                description="Execution context for the active handler"
            ),
        ]
        metrics_tracker: Annotated[
            FlextModelsHandler.MetricsTracker,
            FlextUtilitiesPydantic.Field(
                default_factory=lambda: FlextModelsHandler.MetricsTracker(),
                description="Metrics tracker bound to the handler execution context",
            ),
        ]
        context_stack: Annotated[
            FlextModelsHandler.ContextStack,
            FlextUtilitiesPydantic.Field(
                default_factory=lambda: FlextModelsHandler.ContextStack(),
                description="Context stack used by nested handler execution",
            ),
        ]
        accepted_message_types: Annotated[
            tuple[t.TypeHintSpecifier, ...],
            FlextUtilitiesPydantic.Field(
                default_factory=tuple,
                description="Accepted message types computed for dispatch routing",
            ),
        ]
        revalidate_pydantic_messages: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                default=False,
                description="Whether Pydantic messages must be revalidated on dispatch",
            ),
        ] = False
        type_warning_emitted: Annotated[
            bool,
            FlextUtilitiesPydantic.Field(
                default=False,
                description="Whether the handler already emitted a type warning",
            ),
        ] = False

        @FlextUtilitiesPydantic.computed_field()
        @property
        def handler_name(self) -> str:
            """Expose the active handler name from the execution context."""
            return self.execution_context.handler_name

        @FlextUtilitiesPydantic.computed_field()
        @property
        def handler_mode(self) -> c.HandlerType:
            """Expose the active handler mode from the execution context."""
            return self.execution_context.handler_mode

        @FlextUtilitiesPydantic.computed_field()
        @property
        def metrics(self) -> t.ConfigMap:
            """Expose normalized metrics collected for the handler."""
            return self.metrics_tracker.metrics

        @classmethod
        def create_for_handler(
            cls,
            handler_name: str,
            handler_mode: c.HandlerType,
        ) -> Self:
            """Create runtime state with a shared execution context and metrics."""
            execution_context = FlextModelsHandler.ExecutionContext.create_for_handler(
                handler_name=handler_name,
                handler_mode=handler_mode,
            )
            return cls(
                execution_context=execution_context,
                metrics_tracker=FlextModelsHandler.MetricsTracker.create_for_context(
                    execution_context,
                ),
            )

        def pop_context(self) -> p.Result[t.ScalarMapping]:
            """Pop handler context from the runtime stack."""
            return self.context_stack.pop_context()

        def push_context(
            self,
            ctx: FlextModelsHandler.ExecutionContext | t.RecursiveContainerMapping,
        ) -> p.Result[bool]:
            """Push handler context onto the runtime stack."""
            return self.context_stack.push_context(ctx)

        def record_metric(
            self,
            name: str,
            value: t.MetadataAttributeValue,
        ) -> p.Result[bool]:
            """Record a metric via the bound metrics tracker."""
            return self.metrics_tracker.record_metric(name, value)

        def start_execution(self) -> None:
            """Start timing for the active execution context."""
            self.execution_context.start_execution()

    class DecoratorConfig(m.ArbitraryTypesModel):
        """Configuration extracted from @FlextHandlers.handler() decorator."""

        model_config: ClassVar[c.ConfigDict] = c.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        command: Annotated[
            type,
            FlextUtilitiesPydantic.Field(
                description="Command type this handler processes"
            ),
        ]
        priority: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Handler priority (higher = processed first)",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        timeout: Annotated[
            float | None,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        middleware: Annotated[
            Sequence[type[p.Middleware]],
            FlextUtilitiesPydantic.Field(
                description="Middleware types to apply to this handler",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=list[type[p.Middleware]])


__all__: list[str] = ["FlextModelsHandler"]
