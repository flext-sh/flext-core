"""Dispatcher facade delivering the Phase 1 unified dispatcher charter.

The façade wraps ``FlextBus`` so handler registration, context propagation, and
metadata-aware dispatch all match the expectations documented in ``README.md``
and ``docs/architecture.md`` for the 1.0.0 modernization programme.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import Token
from dataclasses import dataclass
from typing import overload

from flext_core.bus import FlextBus
from flext_core.context import FlextContext
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, MessageT, ResultT


class FlextDispatcher:
    """Orchestrates CQRS execution while enforcing context-first observability.

    The dispatcher is the front door promoted across the ecosystem: all handler
    registration flows, context scoping, and dispatch telemetry align with the
    modernization plan so downstream packages can adopt a consistent runtime
    contract without bespoke buses.
    """

    @dataclass(slots=True)
    class Registration[MessageT, ResultT]:
        """Registration payload returned to callers for tracking."""

        message_type: type[MessageT] | None
        handler: FlextHandlers[MessageT, ResultT]

    def __init__(
        self,
        *,
        bus: FlextBus | None = None,
        auto_context: bool = True,
        bus_config: FlextModels.CqrsConfig.Bus | dict[str, object] | None = None,
    ) -> None:
        """Initialise dispatcher with optional bus configuration."""
        self._bus = bus or FlextBus.create_command_bus(bus_config=bus_config)
        self._auto_context = auto_context
        self._logger = FlextLogger(self.__class__.__name__)

    @property
    def bus(self) -> FlextBus:
        """Access the underlying bus implementation."""
        return self._bus

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_handler(
        self,
        handler: FlextHandlers[MessageT, ResultT],
    ) -> FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]:
        """Register a fully constructed handler instance using railway pattern.
        
        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        return self._bus.register_handler(handler).flat_map(
            lambda _: self._create_registration_and_log(handler, None)
        )

    def _create_registration_and_log(
        self,
        handler: FlextHandlers[MessageT, ResultT],
        message_type: type[MessageT] | None,
    ) -> FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]:
        """Create registration and log success.
        
        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """

    def register_command(
        self,
        command_type: type[MessageT],
        handler: FlextHandlers[MessageT, ResultT],
    ) -> FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]:
        """Register handler bound to a specific command type using railway pattern.
        
        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        return self._bus.register_handler(command_type, handler).flat_map(
            lambda _: self._create_registration_and_log(handler, command_type)
        )

    def register_query(
        self,
        query_type: type[MessageT],
        handler: FlextHandlers[MessageT, ResultT],
    ) -> FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]:
        """Register query handler by delegating to the underlying bus.
        
        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        return self.register_command(query_type, handler)

    def register_function(
        self,
        message_type: type[MessageT],
        handler_func: Callable[[MessageT], ResultT | FlextResult[ResultT]],
        *,
        handler_config: FlextModels.CqrsConfig.Handler
        | dict[str, object]
        | None = None,
        mode: str = "command",
    ) -> FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]:
        """Register a simple function as a handler using railway pattern.
        
        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """

    def _validate_handler_mode(self, mode: str) -> FlextResult[None]:
        """Validate handler mode parameter.
        
        Returns:
            FlextResult[None]: Success result if mode is valid, failure result otherwise.

        """
        if mode not in {"command", "query"}:
            return FlextResult[None].fail("mode must be 'command' or 'query'")
        return FlextResult[None].ok(None)

    def _create_handler_from_function(
        self,
        handler_func: Callable[[MessageT], ResultT | FlextResult[ResultT]],
        handler_config: FlextModels.CqrsConfig.Handler | dict[str, object] | None,
        mode: str,
    ) -> FlextResult[FlextHandlers[MessageT, ResultT]]:
        """Create handler from function using appropriate factory.
        
        Returns:
            FlextResult[FlextHandlers[MessageT, ResultT]]: Success result with handler instance.

        """

    # ------------------------------------------------------------------
    # Dispatch execution
    # ------------------------------------------------------------------
    @overload
    def dispatch(self, message: object) -> FlextResult[object]: ...

    @overload
    def dispatch(
        self,
        message: object,
        *,
        metadata: FlextTypes.Core.Dict | None,
    ) -> FlextResult[object]: ...

    def dispatch(
        self,
        message: object,
        *,
        metadata: FlextTypes.Core.Dict | None = None,
    ) -> FlextResult[object]:
        """Dispatch commands or queries through the underlying bus using railway pattern.

        Args:
            message: The command or query message to dispatch.
            metadata: Optional metadata to include in the execution context.

        Returns:
            A FlextResult containing the execution result or error details.

        """
        with self._context_scope(metadata):
            result = self._bus.execute(message)
            if result.is_success:
                self._log_dispatch_success(message, result.value)
                return result
            self._log_dispatch_failure(message, result.error or "Unknown error")
            return result

    def _log_dispatch_success(self, message: object, result: object) -> object:
        """Log successful dispatch and return result.
        
        Returns:
            object: The result that was logged.

        """
        self._logger.debug(
            "dispatch_succeeded",
            message_type=type(message).__name__,
        )
        return result

    def _log_dispatch_failure(self, message: object, error: str) -> str:
        """Log failed dispatch and return error.
        
        Returns:
            str: The error message that was logged.

        """
        self._logger.error(
            "dispatch_failed",
            message_type=type(message).__name__,
            error=error,
        )
        return error

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    @contextmanager
    def _context_scope(
        self,
        metadata: FlextTypes.Core.Dict | None = None,
    ) -> Generator[None]:
        if not self._auto_context:
            yield
            return

        metadata_token: Token[FlextTypes.Core.Dict | None] | None = None
        metadata_var = FlextContext.Variables.Performance.OPERATION_METADATA
        with FlextContext.Correlation.inherit_correlation() as correlation_id:
            if metadata:
                metadata_token = metadata_var.set(metadata)
            self._logger.debug(
                "dispatch_context_entered",
                correlation_id=correlation_id,
            )
            try:
                yield
            finally:
                if metadata_token is not None:
                    metadata_var.reset(metadata_token)
                self._logger.debug(
                    "dispatch_context_exited",
                    correlation_id=correlation_id,
                )


__all__ = ["FlextDispatcher"]
