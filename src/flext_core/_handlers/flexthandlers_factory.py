"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._handlers import FlextHandlersCqrsBase, FlextHandlersPipeline

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextHandlersFactory[MessageT_contra, ResultT](
    FlextHandlersPipeline[MessageT_contra, ResultT],
):
    def __init__(self, *, settings: flext_core.p.HandlerConfig | None = None) -> None:
        """Initialize the handler, forwarding configuration to its owner.

        The base constructor is invoked explicitly so the pydantic model
        synthesis pyright applies to every intermediate subclass keeps exposing
        the owner-declared ``settings`` keyword at the boundary.

        Args:
            settings: Optional handler configuration model.

        """
        FlextHandlersCqrsBase[MessageT_contra, ResultT].__init__(
            self, settings=settings
        )

    @staticmethod
    def create_from_callable(
        handler_callable: Callable[[flext_core.t.Scalar], flext_core.t.Scalar],
        handler_name: str | None = None,
        handler_type: flext_core.c.HandlerType | None = None,
        handler_config: flext_core.p.HandlerConfig | None = None,
    ) -> flext_core.p.Handler[flext_core.t.Scalar, flext_core.t.Scalar]:
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

        class CallableHandler(
            FlextHandlersFactory[flext_core.t.Scalar, flext_core.t.Scalar]
        ):
            """Dynamic handler created from callable."""

            _handler_fn: Callable[[flext_core.t.Scalar], flext_core.t.Scalar]

            def __init__(
                self,
                handler_fn: Callable[[flext_core.t.Scalar], flext_core.t.Scalar],
                settings: flext_core.p.HandlerConfig | None = None,
            ) -> None:
                super().__init__(settings=settings)
                self._handler_fn = handler_fn

            @override
            def handle(
                self, message: flext_core.t.Scalar
            ) -> flext_core.p.Result[flext_core.t.Scalar]:
                """Execute the wrapped callable.

                Returns:
                    The resulting ``p.Result[t.Scalar]``.

                """
                try:
                    result = self._handler_fn(message)
                    if isinstance(result, flext_core.r):
                        return cast("flext_core.p.Result[flext_core.t.Scalar]", result)
                    return flext_core.r[flext_core.t.Scalar].ok(result)
                except flext_core.c.EXC_BROAD_RUNTIME as exc:
                    self.logger.debug("Callable handler execution failed", exc_info=exc)
                    return flext_core.r[flext_core.t.Scalar].fail_op(
                        "execute callable handler", exc
                    )

        if handler_config is not None:
            return CallableHandler(handler_fn=handler_callable, settings=handler_config)
        resolved_type = handler_type or flext_core.c.HandlerType.COMMAND
        resolved_name: str = handler_name or str(
            getattr(handler_callable, "__name__", "unknown_handler"),
        )
        settings = flext_core.m.HandlerConfig(
            handler_id=f"callable_{id(handler_callable)}",
            handler_name=resolved_name,
            handler_type=resolved_type,
            handler_mode=resolved_type,
        )
        return CallableHandler(handler_fn=handler_callable, settings=settings)

    def __call__(self, message: MessageT_contra) -> flext_core.p.Result[ResultT]:
        """Callable interface that auto-scopes the correlation ID.

        Scoping is active when ``_auto_context_scope=True``.

        Returns:
            The resulting ``p.Result[ResultT]``.

        """
        if not self._auto_context_scope:
            return self.handle(message)
        operation_name = f"{self.__class__.__qualname__}.handle"
        with flext_core.FlextContext.new_correlation():
            flext_core.FlextContext.apply_operation_name(operation_name)
            return self.handle(message)


__all__: list[str] = ["FlextHandlersFactory"]
