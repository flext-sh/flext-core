"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


from flext_core import c

from .flexthandlers_part_02 import FlextHandlers as FlextHandlersPart02

from collections.abc import Callable

from flext_core import p, t


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersPart02[MessageT_contra, ResultT]
):
    @staticmethod
    def handler[**PHandler, TResult](
        command: type,
        *,
        priority: int = c.DEFAULT_MAX_COMMAND_RETRIES,
        timeout: float | None = c.DEFAULT_TIMEOUT_SECONDS,
        middleware: t.SequenceOf[type[p.Middleware]] | None = None,
    ) -> Callable[[Callable[PHandler, TResult]], Callable[PHandler, TResult]]:
        """Mark methods as handlers for commands.

        Stores handler configuration as metadata on the decorated method,
        enabling auto-discovery by FlextService and handler registries.

        Args:
            command: The command type this handler processes
            priority: Handler priority (higher = processed first). Default: 0
            timeout: Handler execution timeout in seconds. Default: None
            middleware: List of middleware types to apply to this handler

        Returns:
            Decorator function for marking handler methods

        Example:
            >>> @FlextHandlers.handler(command=CreateUserCommand, priority=10)
            ... def handle_create_user(self, cmd: CreateUserCommand) -> p.Result[User]:
            ...     return r[User].ok(self._create(cmd))

        """

        def decorator(func: Callable[PHandler, TResult]) -> Callable[PHandler, TResult]:
            """Apply handler configuration metadata to function.

            Only sets the attribute if not already set - innermost decorator wins.
            When multiple @h.handler() decorators are stacked, the first (innermost)
            one to run takes precedence.
            """
            if not hasattr(func, c.HANDLER_ATTR):
                from flext_core import m

                settings = m.DecoratorConfig(
                    command=command,
                    priority=priority,
                    timeout=timeout,
                    middleware=middleware or (),
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


__all__: list[str] = ["FlextHandlers"]
