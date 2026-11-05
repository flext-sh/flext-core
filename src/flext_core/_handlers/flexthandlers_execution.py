"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import flext_core
from flext_core._handlers import FlextHandlersDispatch


class FlextHandlersExecution[MessageT_contra, ResultT](
    FlextHandlersDispatch[MessageT_contra, ResultT],
):
    def handle(self, message: MessageT_contra) -> flext_core.p.Result[ResultT]:
        """Handle the message - abstract method to be implemented by subclasses.

        This is the core business logic method that must be implemented by all
        concrete handler subclasses. It contains the actual command/query/event
        processing logic specific to each handler implementation.

        Args:
            message: The message (command, query, or event) to handle

        Raises:
            NotImplementedError: Always; concrete subclasses must override.

        Note:
            This method should focus on business logic only. Validation should
            be handled separately in the validate() method and executed via execute().

        """
        _ = self, message
        raise NotImplementedError

    def pop_context(
        self,
    ) -> flext_core.p.Result[flext_core.p.RootDict[flext_core.t.JsonPayload]]:
        """Pop execution context from the local handler stack.

        Returns:
            The resulting ``p.Result[p.RootDict[t.JsonPayload]]``.

        """
        result = flext_core.u.pop_context(self._runtime_state)
        if result.failure:
            return flext_core.r[
                flext_core.p.RootDict[flext_core.t.JsonPayload]
            ].fail_op(
                "pop handler context",
                result.error,
            )
        self._runtime_state, context = result.unwrap()
        return flext_core.r.ok(context)

    def push_context(
        self, ctx: flext_core.t.JsonMapping | flext_core.p.ExecutionContext
    ) -> flext_core.p.Result[bool]:
        """Push execution context onto the local handler stack.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        result = flext_core.u.push_context(self._runtime_state, ctx)
        if result.failure:
            return flext_core.r[bool].fail_op("push handler context", result.error)
        self._runtime_state = result.unwrap()
        return flext_core.r.ok(value=True)

    def record_metric(
        self, name: str, value: flext_core.t.JsonPayload
    ) -> flext_core.p.Result[bool]:
        """Record a metric value in the current handler state.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        return flext_core.u.record_metric(
            self._runtime_state.execution_context,
            name,
            value,
        )

    @staticmethod
    def validate_message(data: MessageT_contra) -> flext_core.p.Result[bool]:
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
            ...     u.Cli.print(f"Validation error: {result.error}")

        Note: the base check reads no handler state, so it is a static method;
        subclasses override it with their own static domain validation.

        """
        if data is None:
            return flext_core.r[bool].fail_op(
                "validate handler message",
                flext_core.c.ERR_MESSAGE_CANNOT_BE_NONE,
            )
        return flext_core.r[bool].ok(value=True)

    def _record_execution_metrics(
        self,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record execution metrics (helper to reduce locals in _run_pipeline)."""
        exec_time = flext_core.u.to_float(
            self._runtime_state.execution_context.execution_time_ms
        )
        _ = self.record_metric("execution_time_ms", exec_time)
        _ = self.record_metric("success", success)
        if error is not None:
            _ = self.record_metric(flext_core.c.WarningLevel.ERROR, error)


__all__: list[str] = ["FlextHandlersExecution"]
