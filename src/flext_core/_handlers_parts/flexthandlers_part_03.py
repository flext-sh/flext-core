"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c, m
from flext_core._handlers_parts import flexthandlers_part_02

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core import p, t


class FlextHandlersPart03[MessageT_contra, ResultT](
    flexthandlers_part_02.FlextHandlersPart02[MessageT_contra, ResultT],
):
    @staticmethod
    def handler[
        **PHandler,
        TResult: t.JsonPayload | p.ResultView[t.JsonPayload] | None,
    ](
        command: type,
        *,
        priority: int = c.DEFAULT_MAX_COMMAND_RETRIES,
        timeout: float | None = None,
        middleware: t.SequenceOf[type[p.Middleware]] | None = None,
    ) -> Callable[[Callable[PHandler, TResult]], Callable[PHandler, TResult]]:
        """Mark methods as handlers for commands.

        Stores handler configuration as metadata on the decorated method,
        enabling auto-discovery by FlextService and handler registries.
        The result bound describes the synchronous executor's supported domain;
        metadata attachment does not validate a callable's signature or output.

        Args:
            command: The command type this handler processes
            priority: Higher priorities are discovered first; the model owns
                the default.
            timeout: Non-None policies are rejected until the executor supports them.
            middleware: Nonempty policies are rejected until the executor supports them.

        Returns:
            A declaration decorator that returns the original callable unchanged.

        Raises:
            ValueError: If timeout or middleware execution is requested.

        Example:
            >>> @FlextHandlers.handler(command=m.Command)
            ... def handle_command(request: p.CommandRoutable) -> p.Result[str]:
            ...     return r[str].ok(request.command_type)

        """
        if timeout is not None:
            msg = "Handler timeout execution is not supported"
            raise ValueError(msg)
        if middleware:
            msg = "Handler middleware execution is not supported"
            raise ValueError(msg)

        def decorator(func: Callable[PHandler, TResult]) -> Callable[PHandler, TResult]:
            """Apply handler configuration metadata to function.

            A callable has one declaration; stacked declarations are rejected.

            Returns:
                The resulting ``Callable[PHandler, TResult]``.

            Raises:
                ValueError: If the callable already declares handler metadata.

            """
            if hasattr(func, c.HANDLER_ATTR):
                msg = "Handler metadata is already declared on this callable"
                raise ValueError(msg)
            settings = m.DecoratorConfig(
                command=command,
                priority=priority,
                timeout=timeout,
                middleware=(),
            )
            setattr(func, c.HANDLER_ATTR, settings)
            return func

        return decorator

    def can_handle(self, message_type: type) -> bool:
        """Check if handler can handle the specified message type.

        Determines message type compatibility using duck typing and class hierarchy.
        If _expected_message_type is set, checks if the message_type is a subclass
        of the expected type. If not set, accepts any message type (flexible handler).

        This method enables handler registration and routing in dispatcher systems,
        allowing handlers to declare their capabilities through configuration.

        Args:
            message_type: The message type to check compatibility for

        Returns:
            bool: True if handler can handle this message type, False otherwise

        Example:
            >>> class UserCommand:
            ...     pass
            >>> class AdminCommand:
            ...     pass
            >>> handler = UserHandler()
            >>> handler.can_handle(UserCommand)  # True
            >>> handler.can_handle(AdminCommand)  # Depends on _expected_message_type

        """
        if self._expected_message_type is None:
            return True
        return issubclass(message_type, self._expected_message_type)


__all__: list[str] = ["FlextHandlersPart03"]
