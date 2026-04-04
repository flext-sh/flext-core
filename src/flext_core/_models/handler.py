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

from pydantic import ConfigDict, Field, PrivateAttr, computed_field

from flext_core import FlextModelsBase, c, p, r, t


class FlextModelsHandler:
    """Handler management pattern container class.

    This class acts as a namespace container for handler management patterns.
    All nested classes are accessed via FlextModels.Handler.* in the main models.py.
    """

    class RegistrationResult(FlextModelsBase.ArbitraryTypesModel):
        """Result of a handler registration operation.

        Provides structured feedback on the outcome of a handler registration,
        including status, mode, and identification.
        """

        handler_name: Annotated[t.NonEmptyStr, Field(description="Name of the handler")]
        status: Annotated[
            t.NonEmptyStr,
            Field(
                description="Registration status (registered, skipped, failed)",
            ),
        ]
        mode: Annotated[
            t.NonEmptyStr,
            Field(description="Registration mode (auto_discovery, explicit)"),
        ]
        handler_mode: Annotated[
            c.HandlerType | None,
            Field(default=None, description="Handler mode (command/query/event)"),
        ] = None
        message_type: Annotated[
            str | None,
            Field(
                default=None,
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

    class RegistrationDetails(FlextModelsBase.ArbitraryTypesModel):
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

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "RegistrationDetails",
                "description": "Handler registration tracking details",
            },
        )
        registration_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            Field(
                default=c.HandlerType.COMMAND,
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = c.HandlerType.COMMAND
        timestamp: Annotated[
            str,
            Field(
                description="ISO 8601 timestamp recording when the registration entry was created.",
                title="Registration Timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
                pattern=c.PATTERN_ISO8601_TIMESTAMP,
            ),
        ] = Field(default_factory=lambda: c.DEFAULT_EMPTY_STRING)
        status: Annotated[
            c.CommonStatus,
            Field(
                default=c.CommonStatus.RUNNING,
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = c.CommonStatus.RUNNING

    class ExecutionContext(FlextModelsBase.ArbitraryTypesModel):
        """Handler execution context for tracking handler performance and state.

        Attributes:
            handler_name: Name of the handler being executed
            handler_mode: Mode of execution (command, query, or event)

        """

        _start_time: float | None = PrivateAttr(default=None)
        _metrics_state: t.Dict | None = PrivateAttr(default=None)

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "HandlerExecutionContext",
                "description": "Handler execution context for tracking performance and state",
            },
        )
        handler_name: Annotated[
            t.NonEmptyStr,
            Field(
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            c.HandlerType,
            Field(
                description="Mode of handler execution",
                examples=["command", "query", "event"],
            ),
        ]

        @computed_field
        def execution_time_ms(self) -> float:
            """Get execution time in milliseconds."""
            if self._start_time is None:
                return 0.0
            start_time: float = self._start_time
            elapsed: float = time.time() - start_time
            return round(elapsed * c.DEFAULT_SIZE, 2)

        @computed_field
        def has_metrics(self) -> bool:
            """Check if metrics have been recorded."""
            return self._metrics_state is not None and bool(self._metrics_state)

        @computed_field
        def is_running(self) -> bool:
            """Check if execution is currently running."""
            return self._start_time is not None

        @computed_field
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

        def set_metrics_state(self, state: t.Dict) -> None:
            """Set metrics state."""
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

    class MetricsTracker(FlextModelsBase.ArbitraryTypesModel):
        """Tracks handler execution metrics via ExecutionContext.

        Delegates metric storage to an internal ExecutionContext instance,
        providing a simplified record/get interface for CQRS handler pipelines.
        """

        _context: FlextModelsHandler.ExecutionContext = PrivateAttr(
            default_factory=lambda: (
                FlextModelsHandler.ExecutionContext.create_for_handler(
                    handler_name="metrics",
                    handler_mode=c.HandlerType.OPERATION,
                )
            ),
        )

        def get_metrics(self) -> r[t.ConfigMap]:
            """Return all recorded metrics as a ConfigMap result."""
            raw_state = self._context.metrics_state
            state: t.Dict = (
                raw_state if isinstance(raw_state, t.Dict) else t.Dict(root={})
            )
            return r[t.ConfigMap].ok(t.ConfigMap(root=dict(state.root)))

        def record_metric(self, name: str, value: t.Scalar) -> r[bool]:
            """Record a named metric value in the tracker."""
            raw_state = self._context.metrics_state
            state: t.Dict = (
                raw_state if isinstance(raw_state, t.Dict) else t.Dict(root={})
            )
            current = dict(state.root)
            current[name] = value
            self._context.set_metrics_state(t.Dict(root=current))
            return r[bool].ok(True)

    class ContextStack(FlextModelsBase.ArbitraryTypesModel):
        """Manages a stack of ExecutionContext instances for CQRS handler pipelines."""

        _stack: MutableSequence[FlextModelsHandler.ExecutionContext] = PrivateAttr(
            default_factory=lambda: list[FlextModelsHandler.ExecutionContext](),
        )

        def current_context(self) -> FlextModelsHandler.ExecutionContext | None:
            """Return the current top-of-stack execution context, or None."""
            if self._stack:
                return self._stack[-1]
            return None

        def pop_context(self) -> r[t.ScalarMapping]:
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
            ctx: FlextModelsHandler.ExecutionContext | t.ScalarMapping,
        ) -> r[bool]:
            """Push an execution context or mapping onto the context stack."""
            if isinstance(ctx, FlextModelsHandler.ExecutionContext):
                self._stack.append(ctx)
                return r[bool].ok(True)
            ctx_mapping: t.ScalarMapping = {str(k): v for k, v in ctx.items()}
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

    class DecoratorConfig(FlextModelsBase.ArbitraryTypesModel):
        """Configuration extracted from @FlextHandlers.handler() decorator."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        command: Annotated[
            type,
            Field(description="Command type this handler processes"),
        ]
        priority: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Handler priority (higher = processed first)",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        timeout: Annotated[
            float | None,
            Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                description="Handler execution timeout in seconds",
                gt=0.0,
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        middleware: Annotated[
            Sequence[type[p.Middleware]],
            Field(
                description="Middleware types to apply to this handler",
            ),
        ] = Field(default_factory=list[type[p.Middleware]])


__all__ = ["FlextModelsHandler"]
