"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    c,
    m,
    p,
    r,
    t,
    u,
)
from flext_core._utilities.handler import FlextUtilitiesHandler

from .flexthandlers_part_03 import (
    FlextHandlers as FlextHandlersPart03,
)


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersPart03[MessageT_contra, ResultT],
):
    def handle(self, message: MessageT_contra) -> p.Result[ResultT]:
        """Handle the message - abstract method to be implemented by subclasses.

        This is the core business logic method that must be implemented by all
        concrete handler subclasses. It contains the actual command/query/event
        processing logic specific to each handler implementation.

        Args:
            message: The message (command, query, or event) to handle

        Returns:
            r[ResultT]: Success with result or failure with error details

        Note:
            This method should focus on business logic only. Validation should
            be handled separately in the validate() method and executed via execute().

        """
        _ = message
        raise NotImplementedError

    def pop_context(self) -> p.Result[m.ConfigMap]:
        """Pop execution context from the local handler stack."""
        result = FlextUtilitiesHandler.pop_context(self._runtime_state)
        if result.failure:
            return r[m.ConfigMap].fail_op(
                "pop handler context",
                result.error or c.ERR_HANDLER_FAILED,
            )
        return r[m.ConfigMap].ok(m.ConfigMap.model_validate(result.value))

    def push_context(
        self,
        ctx: t.JsonMapping | m.ExecutionContext,
    ) -> p.Result[bool]:
        """Push execution context onto the local handler stack."""
        return FlextUtilitiesHandler.push_context(self._runtime_state, ctx)

    def record_metric(self, name: str, value: t.JsonPayload) -> p.Result[bool]:
        """Record a metric value in the current handler state."""
        return FlextUtilitiesHandler.record_metric(
            self._runtime_state.execution_context,
            name,
            value,
        )

    def validate_message(self, data: MessageT_contra) -> p.Result[bool]:
        """Validate input data using extensible validation pipeline.

        Base validation method that can be overridden by subclasses to implement
        custom validation logic. By default, performs basic type checking and
        returns success. Subclasses should extend this method for domain-specific
        validation rules.

        The validation follows railway-oriented programming principles, returning
        r[bool] to allow for detailed error reporting and chaining.

        Args:
            data: Input data to validate (message, command, query, or event)

        Returns:
            r[bool]: Success (True) if valid, failure with error details if invalid

        Example:
            >>> handler = UserHandler()
            >>> result = handler.validate_message(invalid_data)
            >>> if result.failure:
            ...     print(f"Validation error: {result.error}")

        Note: self is required for subclass override compatibility, even though
        this base implementation doesn't use instance state.

        """
        if data is None:
            return r[bool].fail_op(
                "validate handler message", c.ERR_MESSAGE_CANNOT_BE_NONE,
            )
        return r[bool].ok(True)

    def _record_execution_metrics(
        self,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record execution metrics (helper to reduce locals in _run_pipeline)."""
        raw_time = self._runtime_state.execution_context.execution_time_ms
        exec_time = u.to_float(raw_time() if callable(raw_time) else raw_time)
        _ = self.record_metric("execution_time_ms", exec_time)
        _ = self.record_metric("success", success)
        if error is not None:
            _ = self.record_metric(c.WarningLevel.ERROR, error)


__all__: list[str] = ["FlextHandlers"]
