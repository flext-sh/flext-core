"""Unified dispatcher facade built on top of FlextBus.

This module introduces a small orchestration layer that coordinates
handler registration, contextual execution, and command/query dispatching
while preserving the existing FlextBus semantics. It will be adopted
incrementally across CLI and connector packages as part of the
modernization roadmap.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import Token
from dataclasses import dataclass
from typing import cast, overload

from flext_core.bus import FlextBus
from flext_core.context import FlextContext
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, MessageT, ResultT


class FlextDispatcher:
    """Dispatcher facade coordinating bus execution and context propagation."""

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
        """Register a fully constructed handler instance."""
        register_result = self._bus.register_handler(handler)
        if register_result.is_failure:
            return FlextResult[FlextDispatcher.Registration[MessageT, ResultT]].fail(
                register_result.error or "Failed to register handler with dispatcher",
            )
        self._logger.debug(
            "handler_registered",
            handler=handler.__class__.__name__,
        )
        return FlextResult[FlextDispatcher.Registration[MessageT, ResultT]].ok(
            FlextDispatcher.Registration[MessageT, ResultT](None, handler),
        )

    def register_command(
        self,
        command_type: type[MessageT],
        handler: FlextHandlers[MessageT, ResultT],
    ) -> FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]:
        """Register handler bound to a specific command type."""
        register_result = self._bus.register_handler(command_type, handler)
        if register_result.is_failure:
            return FlextResult[FlextDispatcher.Registration[MessageT, ResultT]].fail(
                register_result.error or "Failed to register command handler",
            )
        self._logger.debug(
            "command_handler_registered",
            command_type=command_type.__name__,
            handler=handler.__class__.__name__,
        )
        return FlextResult[FlextDispatcher.Registration[MessageT, ResultT]].ok(
            FlextDispatcher.Registration(command_type, handler)
        )

    def register_query(
        self,
        query_type: type[MessageT],
        handler: FlextHandlers[MessageT, ResultT],
    ) -> FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]:
        """Register query handler by delegating to the underlying bus."""
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
        """Register a simple function as a handler using bus helpers."""
        if mode not in {"command", "query"}:
            return FlextResult[FlextDispatcher.Registration[MessageT, ResultT]].fail(
                "mode must be 'command' or 'query'",
            )
        factory = (
            FlextBus.create_simple_handler
            if mode == "command"
            else FlextBus.create_query_handler
        )
        # Cast handler function to match factory signature
        typed_handler_func = cast("Callable[[object], object]", handler_func)
        generic_handler = factory(typed_handler_func, handler_config=handler_config)
        # Cast to correct type since we know the factory returns the right type
        typed_handler = cast("FlextHandlers[MessageT, ResultT]", generic_handler)
        return self.register_command(message_type, typed_handler)

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
        """Dispatch commands or queries through the underlying bus."""
        with self._context_scope(metadata):
            result = self._bus.execute(message)
            if result.is_failure:
                self._logger.error(
                    "dispatch_failed",
                    message_type=type(message).__name__,
                    error=result.error,
                )
            else:
                self._logger.debug(
                    "dispatch_succeeded",
                    message_type=type(message).__name__,
                )
            return result

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
