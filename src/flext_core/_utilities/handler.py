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
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from flext_core import c, r
from flext_core.runtime import FlextRuntime

from flext_core import p, t


class FlextUtilitiesHandler:
    """Stateless orchestration helpers for handler pipelines."""

    @staticmethod
    def create_runtime_state(
        handler_name: str, handler_mode: c.HandlerType
    ) -> p.HandlerRuntimeState:
        """Build runtime state with a fresh execution context."""
        from flext_core import m

        return m.HandlerRuntimeState(
            execution_context=m.ExecutionContext(
                handler_name=handler_name, handler_mode=handler_mode
            )
        )

    @staticmethod
    def start_execution(state: p.HandlerRuntimeState) -> p.HandlerRuntimeState:
        """Stamp the current monotonic time on the active execution context."""
        execution_context = state.execution_context.model_copy(
            update={"started_at": time.time()}
        )
        return state.model_copy(update={"execution_context": execution_context})

    @staticmethod
    def record_metric(
        ctx: p.ExecutionContext, name: str, value: t.JsonPayload
    ) -> p.Result[bool]:
        """Record a metric value onto an execution context's payload."""
        normalized = FlextRuntime.normalize_to_container(value)
        ctx.metrics_state_data.root[name] = (
            normalized
            if isinstance(normalized, (str, int, float, bool, datetime, Path))
            else str(normalized)
        )
        return r.ok(True)

    @staticmethod
    def push_context(
        state: p.HandlerRuntimeState, ctx: t.JsonMapping | p.ExecutionContext
    ) -> p.Result[p.HandlerRuntimeState]:
        """Validate a context and return state with an extended stack."""
        if not isinstance(ctx, Mapping):
            execution_context = ctx.model_copy()
        else:
            from flext_core import m

            validated = r.from_validation(ctx, m.ExecutionContext)
            if validated.failure:
                return r.fail_op("push handler context", validated.error)
            execution_context: p.ExecutionContext = validated.unwrap()
        return r.ok(
            state.model_copy(
                update={"context_stack": (*state.context_stack, execution_context)}
            )
        )

    @staticmethod
    def pop_context(
        state: p.HandlerRuntimeState,
    ) -> p.Result[t.Pair[p.HandlerRuntimeState, p.RootDict[t.JsonPayload]]]:
        """Return state without the top context plus its validated identity."""
        from flext_core import m

        if not state.context_stack:
            empty_context: p.RootDict[t.JsonPayload] = m.ConfigMap(root={})
            return r.ok((state, empty_context))
        popped = state.context_stack[-1]
        next_state = state.model_copy(
            update={"context_stack": state.context_stack[:-1]}
        )
        popped_context: p.RootDict[t.JsonPayload] = m.ConfigMap(
            root={
                "handler_name": popped.handler_name,
                c.FIELD_HANDLER_MODE: popped.handler_mode,
            }
        )
        return r.ok((next_state, popped_context))


__all__: list[str] = ["FlextUtilitiesHandler"]
