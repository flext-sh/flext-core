"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import c, r, t

from .flexthandlers_part_05 import FlextHandlers as FlextHandlersPart05

from collections.abc import Callable

from flext_core import p


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersPart05[MessageT_contra, ResultT]
):
    @staticmethod
    def create_from_callable(
        handler_callable: Callable[[t.Scalar], t.Scalar],
        handler_name: str | None = None,
        handler_type: c.HandlerType | None = None,
        handler_config: p.HandlerConfig | None = None,
    ) -> p.Handler[t.Scalar, t.Scalar]:
        """Create a handler instance from a callable function.

        Factory method that wraps a callable function in a h instance,
        enabling the use of simple functions as CQRS handlers.

        Args:
            handler_callable: Callable that takes a message and returns result
            handler_name: Optional handler name (defaults to function name)
            handler_type: Optional handler type (command, query, event)
            handler_config: Optional validated handler configuration

        Returns:
            p.Handler[t.Scalar, t.Scalar]: Handler instance wrapping the callable

        Example:
            >>> def my_handler(msg: str) -> p.Result[str]:
            ...     return r[str].ok(f"processed_{msg}")
            >>> handler = FlextHandlers.create_from_callable(my_handler)
            >>> result = handler.handle("test")

        """

        class CallableHandler(FlextHandlers[t.Scalar, t.Scalar]):
            """Dynamic handler created from callable."""

            _handler_fn: Callable[[t.Scalar], t.Scalar]

            def __init__(
                self,
                handler_fn: Callable[[t.Scalar], t.Scalar],
                settings: p.HandlerConfig | None = None,
            ) -> None:
                super().__init__(settings=settings)
                self._handler_fn = handler_fn

            @override
            def handle(self, message: t.Scalar) -> p.Result[t.Scalar]:
                """Execute the wrapped callable."""
                try:
                    result = self._handler_fn(message)
                    if isinstance(result, r):
                        return result
                    return r[t.Scalar].ok(result)
                except c.EXC_BROAD_RUNTIME as exc:
                    self.logger.debug("Callable handler execution failed", exc_info=exc)
                    return r[t.Scalar].fail_op("execute callable handler", exc)

        if handler_config is not None:
            return CallableHandler(handler_fn=handler_callable, settings=handler_config)
        resolved_type = handler_type or c.HandlerType.COMMAND
        resolved_name: str = handler_name or str(
            getattr(handler_callable, "__name__", "unknown_handler")
            or "unknown_handler"
        )
        from flext_core import m

        settings = m.Handler(
            handler_id=f"callable_{id(handler_callable)}",
            handler_name=resolved_name,
            handler_type=resolved_type,
            handler_mode=resolved_type,
        )
        return CallableHandler(handler_fn=handler_callable, settings=settings)

    def __call__(self, message: MessageT_contra) -> p.Result[ResultT]:
        """Callable interface — auto-scopes correlation ID when _auto_context_scope=True."""
        if not self._auto_context_scope:
            return self.handle(message)
        operation_name = f"{self.__class__.__qualname__}.handle"
        with self._context_type.new_correlation():
            self._context_type.apply_operation_name(operation_name)
            return self.handle(message)


__all__: list[str] = ["FlextHandlers"]
