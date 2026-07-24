"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, p, r
from flext_core._utilities.handler import FlextUtilitiesHandler

from .flexthandlers_part_04 import FlextHandlers as FlextHandlersPart04


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersPart04[MessageT_contra, ResultT]
):
    def _run_pipeline(
        self, message: MessageT_contra, operation: str = c.DEFAULT_HANDLER_MODE
    ) -> p.Result[ResultT]:
        """Run the handler execution pipeline (internal).

        Internal implementation that executes the full handler pipeline including
        mode validation, can_handle check, message validation, execution,
        context tracking, and metrics recording.

        Args:
            message: The message to process
            operation: Operation type (command, query, event)

        Returns:
            r[ResultT]: Handler execution result

        """
        handler_mode = getattr(
            self._config_model.handler_mode, "value", self._config_model.handler_mode
        )
        valid_operations = {
            c.DEFAULT_HANDLER_MODE,
            c.HandlerMode.QUERY,
            c.HandlerType.EVENT.value,
        }
        if operation != handler_mode and operation in valid_operations:
            error_msg = c.ERR_HANDLER_INCOMPATIBLE_PIPELINE_MODE.format(
                handler_mode=handler_mode, operation=operation
            )
            return r[ResultT].fail_op("validate handler pipeline mode", error_msg)
        message_type = message.__class__
        if not self.can_handle(message_type):
            type_name = message_type.__name__
            error_msg = c.ERR_HANDLER_CANNOT_HANDLE_MESSAGE_TYPE.format(
                type_name=type_name
            )
            return r[ResultT].fail_op("validate handler message type", error_msg)
        validation = self.validate_message(message)
        if validation.failure:
            return r[ResultT].from_failure(validation)
        self._runtime_state = FlextUtilitiesHandler.start_execution(self._runtime_state)
        _ = self.push_context(self._runtime_state.execution_context)
        try:
            result = self.handle(message)
            self._record_execution_metrics(success=result.success)
            return result
        except c.EXC_BROAD_RUNTIME as exc:
            self.logger.warning(c.LOG_HANDLER_PIPELINE_FAILURE, exc_info=exc)
            self._record_execution_metrics(success=False, error=str(exc))
            return r[ResultT].fail_op(
                "run handler pipeline",
                c.ERR_HANDLER_CRITICAL_FAILURE.format(error=str(exc)),
            )
        finally:
            _ = self.pop_context()

    def dispatch_message(
        self, message: MessageT_contra, operation: str = c.DEFAULT_HANDLER_MODE
    ) -> p.Result[ResultT]:
        """Dispatch message through the handler execution pipeline.

        Public method that executes the full handler pipeline including
        mode validation, can_handle check, message validation, execution,
        context tracking, and metrics recording.

        This method is the primary entry point for external systems (like
        FlextDispatcher) to execute handlers with full CQRS support.

        Args:
            message: The message to process
            operation: Operation type (command, query, event)

        Returns:
            r[ResultT]: Handler execution result

        """
        return self._run_pipeline(message, operation)

    def execute(self, message: MessageT_contra) -> p.Result[ResultT]:
        """Execute handler with complete validation and error handling pipeline.

        Implements the railway-oriented programming pattern by first validating
        the input message, then executing the business logic if validation passes.
        Uses r for consistent error handling without exceptions.

        Execution Pipeline:
        1. Validate input message using validate() method
        2. If validation fails, return failure result with error details
        3. If validation passes, execute handle() method with business logic
        4. Return result from handle() method (success or failure)

        Args:
            message: The message to execute handler for

        Returns:
            r[ResultT]: Success with handler result or failure with validation/business error

        Example:
            >>> handler = UserHandler()
            >>> result = handler.execute(UserCommand(user_id="123", action="create"))
            >>> if result.success:
            ...     u.Cli.print(f"Success: {result.value}")
            ... else:
            ...     u.Cli.print(f"Failed: {result.error}")

        """
        validation = self.validate_message(message)
        if validation.failure:
            return r[ResultT].from_failure(validation)
        return self.handle(message)


__all__: list[str] = ["FlextHandlers"]
