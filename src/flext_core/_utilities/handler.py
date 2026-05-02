"""Orchestration helpers for handler runtime state.

Pure Pydantic-v2 operations over ``FlextModelsHandler`` state objects.
Only the operations proven to be consumed by ``src/`` (the ``FlextHandlers``
service, registry, dispatcher) are exposed. Examples / tests are rewired
to the same surface - no extra wrappers for their convenience.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from flext_core import (
    FlextConstants as c,
    FlextModelsHandler,
    FlextProtocols as p,
    FlextResult as r,
    FlextRuntime,
    FlextTypes as t,
)


class FlextUtilitiesHandler:
    """Stateless orchestration helpers for handler pipelines."""

    @staticmethod
    def create_runtime_state(
        handler_name: str,
        handler_mode: c.HandlerType,
    ) -> FlextModelsHandler.HandlerRuntimeState:
        """Build runtime state with a fresh execution context."""
        return FlextModelsHandler.HandlerRuntimeState(
            execution_context=FlextModelsHandler.ExecutionContext(
                handler_name=handler_name,
                handler_mode=handler_mode,
            ),
        )

    @staticmethod
    def start_execution(
        state: FlextModelsHandler.HandlerRuntimeState,
    ) -> FlextModelsHandler.HandlerRuntimeState:
        """Stamp the current monotonic time on the active execution context."""
        return state.model_copy(
            update={
                "execution_context": state.execution_context.model_copy(
                    update={"started_at": time.time()},
                ),
            },
        )

    @staticmethod
    def record_metric(
        ctx: FlextModelsHandler.ExecutionContext,
        name: str,
        value: t.JsonPayload,
    ) -> p.Result[bool]:
        """Record a metric value onto an execution context's payload."""
        normalized = FlextRuntime.normalize_to_container(value)
        ctx.metrics_state_data.root[name] = (
            normalized
            if isinstance(normalized, (str, int, float, bool, datetime, Path))
            else str(normalized)
        )
        return r[bool].ok(True)

    @staticmethod
    def push_context(
        state: FlextModelsHandler.HandlerRuntimeState,
        ctx: t.JsonMapping | FlextModelsHandler.ExecutionContext,
    ) -> p.Result[bool]:
        """Coerce a flat mapping into an ExecutionContext and push it."""
        if isinstance(ctx, FlextModelsHandler.ExecutionContext):
            state.context_stack.append(ctx.model_copy())
            return r[bool].ok(True)
        handler_name = str(ctx.get("handler_name", c.IDENTIFIER_UNKNOWN))
        handler_mode_str = str(ctx.get(c.FIELD_HANDLER_MODE, c.HandlerType.OPERATION))
        handler_mode = (
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
        state.context_stack.append(
            FlextModelsHandler.ExecutionContext(
                handler_name=handler_name,
                handler_mode=handler_mode,
            ),
        )
        return r[bool].ok(True)

    @staticmethod
    def pop_context(
        state: FlextModelsHandler.HandlerRuntimeState,
    ) -> p.Result[t.ScalarMapping]:
        """Pop the top context and return its identity as a scalar mapping."""
        if not state.context_stack:
            return r[t.ScalarMapping].ok({})
        popped = state.context_stack.pop()
        return r[t.ScalarMapping].ok({
            "handler_name": popped.handler_name,
            c.FIELD_HANDLER_MODE: popped.handler_mode,
        })


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesHandler"]
