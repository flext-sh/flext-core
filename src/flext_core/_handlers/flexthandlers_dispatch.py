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
from flext_core._handlers import FlextHandlersSettings

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextHandlersDispatch[MessageT_contra, ResultT](
    FlextHandlersSettings[MessageT_contra, ResultT],
):
    @staticmethod
    def handler[
        **PHandler,
        TResult,
    ](
        command: type | None = None,
        *,
        options: m.DecoratorConfig | None = None,
    ) -> Callable[[Callable[PHandler, TResult]], Callable[PHandler, TResult]]:
        """Mark methods as command handlers.

        Stores handler configuration as metadata on the decorated method,
        enabling auto-discovery by FlextService and handler registries.
        Metadata attachment does not execute or validate a callable's signature
        or output; each consuming executor owns its result contract.

        Args:
            command: Request model this handler processes, or None when the
                operation takes no request.
            options: Validated declaration metadata (priority, scope,
                execution mode, ticket policy and default flag).

        Returns:
            A declaration decorator that returns the original callable unchanged.

        """
        return FlextHandlersDispatch._declare(
            (options or m.DecoratorConfig()).model_copy(
                update={
                    "command": command,
                    "handler_type": c.HandlerType.COMMAND,
                }
            )
        )

    @staticmethod
    def query[
        **PHandler,
        TResult,
    ](
        command: type | None = None,
        *,
        options: m.DecoratorConfig | None = None,
    ) -> Callable[[Callable[PHandler, TResult]], Callable[PHandler, TResult]]:
        """Mark methods as read-only query handlers.

        Uses the same declaration contract as :meth:`handler`, recorded as
        ``c.HandlerType.QUERY`` so a CLI or transport can project the operation
        as a read instead of a command.

        Args:
            command: Request model this handler processes, or None when the
                operation takes no request.
            options: Validated declaration metadata (priority, scope,
                execution mode, ticket policy and default flag).

        Returns:
            A declaration decorator that returns the original callable unchanged.

        """
        return FlextHandlersDispatch._declare(
            (options or m.DecoratorConfig()).model_copy(
                update={
                    "command": command,
                    "handler_type": c.HandlerType.QUERY,
                }
            )
        )

    @staticmethod
    def _declare[
        **PHandler,
        TResult,
    ](
        declaration: m.DecoratorConfig,
    ) -> Callable[[Callable[PHandler, TResult]], Callable[PHandler, TResult]]:
        """Build the declaration decorator shared by handler and query.

        Returns:
            A declaration decorator that returns the original callable unchanged.

        Raises:
            ValueError: If the declaration requests an unsupported timeout or
                middleware policy, or the callable already declares metadata.

        """
        if declaration.timeout is not None or declaration.middleware:
            raise ValueError(c.ERR_HANDLER_POLICY_UNSUPPORTED)

        def decorator(func: Callable[PHandler, TResult]) -> Callable[PHandler, TResult]:
            """Apply handler configuration metadata to the function.

            A callable has one declaration; stacked declarations are rejected.

            Returns:
                The original callable, unchanged.

            Raises:
                ValueError: If the callable already declares handler metadata.

            """
            if hasattr(func, c.HANDLER_ATTR):
                raise ValueError(c.ERR_HANDLER_METADATA_DUPLICATE)
            setattr(func, c.HANDLER_ATTR, declaration)
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


__all__: list[str] = ["FlextHandlersDispatch"]
