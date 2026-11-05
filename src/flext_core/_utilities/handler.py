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
from collections.abc import Mapping, MutableSequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._utilities.pydantic import FlextUtilitiesPydantic


class FlextUtilitiesHandler:
    """Stateless orchestration helpers for handler pipelines."""

    @staticmethod
    def create_runtime_state(
        handler_name: str,
        handler_mode: flext_core.c.HandlerType,
    ) -> flext_core.p.HandlerRuntimeState:
        """Build runtime state with a fresh execution context.

        Returns:
            The resulting ``p.HandlerRuntimeState``.

        """
        return flext_core.m.HandlerRuntimeState(
            execution_context=flext_core.m.ExecutionContext(
                handler_name=handler_name,
                handler_mode=handler_mode,
            ),
        )

    @staticmethod
    def start_execution(
        state: flext_core.p.HandlerRuntimeState,
    ) -> flext_core.p.HandlerRuntimeState:
        """Stamp the current monotonic time on the active execution context.

        Returns:
            The resulting ``p.HandlerRuntimeState``.

        """
        execution_context = state.execution_context.model_copy(
            update={"started_at": time.time()},
        )
        return state.model_copy(update={"execution_context": execution_context})

    @staticmethod
    def record_metric(
        ctx: flext_core.p.ExecutionContext,
        name: str,
        value: flext_core.t.JsonPayload,
    ) -> flext_core.p.Result[bool]:
        """Record a metric value onto an execution context's payload.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        normalized = FlextUtilitiesPydantic.normalize_to_container(value)
        ctx.metrics_state_data.root[name] = (
            normalized
            if isinstance(normalized, (str, int, float, bool, datetime, Path))
            else str(normalized)
        )
        return flext_core.FlextResult.ok(value=True)

    @staticmethod
    def push_context(
        state: flext_core.p.HandlerRuntimeState,
        ctx: flext_core.t.JsonMapping | flext_core.p.ExecutionContext,
    ) -> flext_core.p.Result[flext_core.p.HandlerRuntimeState]:
        """Validate a context and return state with an extended stack.

        Returns:
            The resulting ``p.Result[p.HandlerRuntimeState]``.

        """
        if not isinstance(ctx, Mapping):
            pushed_context = ctx.model_copy()
        else:
            validated = flext_core.FlextResult.from_validation(
                ctx, flext_core.m.ExecutionContext
            )
            if validated.failure:
                return flext_core.FlextResult[flext_core.p.HandlerRuntimeState].fail_op(
                    "push handler context",
                    validated.error,
                )
            execution_context: flext_core.p.ExecutionContext = validated.unwrap()
            pushed_context = execution_context
        return flext_core.FlextResult.ok(
            state.model_copy(
                update={"context_stack": (*state.context_stack, pushed_context)},
            ),
        )

    @staticmethod
    def pop_context(
        state: flext_core.p.HandlerRuntimeState,
    ) -> flext_core.p.Result[
        tuple[
            flext_core.p.HandlerRuntimeState,
            flext_core.p.RootDict[flext_core.t.JsonPayload],
        ]
    ]:
        """Return state without the top context plus its validated identity.

        Returns:
            State without the top context plus its validated identity.

        """
        if not state.context_stack:
            empty_context: flext_core.p.RootDict[flext_core.t.JsonPayload] = (
                flext_core.m.ConfigMap(root={})
            )
            return flext_core.FlextResult.ok((state, empty_context))
        popped = state.context_stack[-1]
        next_state = state.model_copy(
            update={"context_stack": state.context_stack[:-1]},
        )
        popped_context: flext_core.p.RootDict[flext_core.t.JsonPayload] = (
            flext_core.m.ConfigMap(
                root={
                    "handler_name": popped.handler_name,
                    flext_core.c.FIELD_HANDLER_MODE: popped.handler_mode.value,
                },
            )
        )
        return flext_core.FlextResult.ok((next_state, popped_context))


__all__: MutableSequence[str] = ["FlextUtilitiesHandler"]
