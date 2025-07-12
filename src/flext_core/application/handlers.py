"""Base handlers for CQRS pattern - v0.7.0.

Copyright (c) 2024 FLEXT Contributors
SPDX-License-Identifier: MIT

FLEXT-Core foundation for command and query handlers.
Uses modern Python 3.13 patterns with ServiceResult.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from flext_core.domain.types import ServiceResult

# Type variables for generic handlers
TCommand = TypeVar("TCommand")
TQuery = TypeVar("TQuery")
TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")


class CommandHandler[TCommand, TResult](ABC):
    """Base command handler with type-safe results.

    Commands change system state and return results.
    """

    @abstractmethod
    async def handle(self, command: TCommand) -> ServiceResult[TResult]:
        """Handle a command and return a service result.

        Args:
            command: The command to handle

        Returns:
            Service result with the operation outcome

        """
        ...


class QueryHandler[TQuery, TResult](ABC):
    """Base query handler with type-safe results.

    Queries read system state without changes.
    """

    @abstractmethod
    async def handle(self, query: TQuery) -> ServiceResult[TResult]:
        """Handle a query and return a service result.

        Args:
            query: The query to handle

        Returns:
            Service result with the query data

        """
        ...


class EventHandler[TEvent, TResult](ABC):
    """Base event handler with type-safe results.

    Events are notifications of things that happened.
    """

    @abstractmethod
    async def handle(self, event: TEvent) -> ServiceResult[TResult]:
        """Handle an event and return a service result.

        Args:
            event: The event to handle

        Returns:
            Service result with the operation outcome

        """
        ...


# Convenience type aliases for common patterns
class VoidCommandHandler(CommandHandler[TCommand, None]):
    """Command handler that returns no data."""

    @abstractmethod
    async def handle(self, command: TCommand) -> ServiceResult[None]:
        """Handle a command that returns no data.

        Args:
            command: The command to handle

        Returns:
            Service result indicating success or failure

        """
        ...


class SimpleQueryHandler(QueryHandler[TQuery, dict[str, Any]]):
    """Query handler that returns dict data."""

    @abstractmethod
    async def handle(self, query: TQuery) -> ServiceResult[dict[str, Any]]:
        """Handle a query that returns dictionary data.

        Args:
            query: The query to handle

        Returns:
            Service result with dictionary data

        """
        ...


__all__ = [
    "CommandHandler",
    "EventHandler",
    "QueryHandler",
    "SimpleQueryHandler",
    "VoidCommandHandler",
]
