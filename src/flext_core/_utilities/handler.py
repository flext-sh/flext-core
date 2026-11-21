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
from collections.abc import Mapping, MutableMapping, MutableSequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from flext_core import c, m, p, r, t
from flext_core._utilities import FlextUtilitiesPydantic


class FlextUtilitiesHandler:
    """Stateless orchestration helpers for handler pipelines."""

    @staticmethod
    def create_runtime_state(
        handler_name: str,
        handler_mode: c.HandlerType,
    ) -> p.HandlerRuntimeState:
        """Build runtime state with a fresh execution context.

        Returns:
            The resulting ``p.HandlerRuntimeState``.

        """
        return m.HandlerRuntimeState(
            execution_context=m.ExecutionContext(
                handler_name=handler_name,
                handler_mode=handler_mode,
            ),
        )

    @staticmethod
    def start_execution(
        state: p.HandlerRuntimeState,
    ) -> p.HandlerRuntimeState:
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
        ctx: p.ExecutionContext,
        name: str,
        value: t.JsonPayload,
    ) -> p.Result[bool]:
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
        return r.ok(value=True)

    @staticmethod
    def push_context(
        state: p.HandlerRuntimeState,
        ctx: t.JsonMapping | p.ExecutionContext,
    ) -> p.Result[p.HandlerRuntimeState]:
        """Validate a context and return state with an extended stack.

        Returns:
            The resulting ``p.Result[p.HandlerRuntimeState]``.

        """
        if not isinstance(ctx, Mapping):
            pushed_context = ctx.model_copy()
        else:
            validated = r.from_validation(ctx, m.ExecutionContext)
            if validated.failure:
                return r[p.HandlerRuntimeState].fail_op(
                    "push handler context",
                    validated.error,
                )
            execution_context: p.ExecutionContext = validated.unwrap()
            pushed_context = execution_context
        return r.ok(
            state.model_copy(
                update={"context_stack": (*state.context_stack, pushed_context)},
            ),
        )

    @staticmethod
    def pop_context(
        state: p.HandlerRuntimeState,
    ) -> p.Result[
        tuple[
            p.HandlerRuntimeState,
            p.RootDict[t.JsonPayload],
        ]
    ]:
        """Return state without the top context plus its validated identity.

        Returns:
            State without the top context plus its validated identity.

        """
        if not state.context_stack:
            empty_context: p.RootDict[t.JsonPayload] = m.ConfigMap(root={})
            return r.ok((state, empty_context))
        popped = state.context_stack[-1]
        next_state = state.model_copy(
            update={"context_stack": state.context_stack[:-1]},
        )
        popped_context: p.RootDict[t.JsonPayload] = m.ConfigMap(
            root={
                "handler_name": popped.handler_name,
                c.FIELD_HANDLER_MODE: popped.handler_mode.value,
            },
        )
        return r.ok((next_state, popped_context))

    class PerformanceTracker:
        """Context manager for handler timing with automatic logging."""

        def __init__(self, logger: p.Logger, operation_name: str) -> None:
            """Initialize with logger and operation name."""
            super().__init__()
            self.logger = logger
            self._operation_name = operation_name
            self._start_time: float = 0.0

        def __enter__(self) -> Self:
            """Start timing this operation.

            Returns:
                This active performance tracker.

            """
            self._start_time = time.time()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Log timing while preserving the original business exception."""
            elapsed = time.time() - self._start_time
            success = exc_type is None
            status = c.Status.SUCCESS if success else c.Status.FAILED
            context: MutableMapping[str, p.LogValue] = {
                c.MetadataKey.DURATION_SECONDS: elapsed,
                c.HandlerType.OPERATION: self._operation_name,
                c.FIELD_STATUS: status,
            }
            if not success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""
            emit = self.logger.info if success else self.logger.error
            event = f"{self._operation_name} {status}"
            try:
                _ = emit(event, **context)
            except BaseException as logging_error:
                if exc_val is not None and exc_val is not logging_error:
                    exc_val.add_note(
                        f"Logging also failed: {type(logging_error).__name__}"
                    )
                    raise exc_val.with_traceback(exc_tb) from exc_val.__cause__
                raise


__all__: MutableSequence[str] = ["FlextUtilitiesHandler"]
