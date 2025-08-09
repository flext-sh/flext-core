"""FLEXT Base handlers module.

Backward compatibility module that re-exports handlers from handlers.py.
This module exists to maintain compatibility with existing tests.
"""

# Re-export specific handlers for backward compatibility following SOLID principles
# Generic type variables for handlers
from abc import ABC, abstractmethod
from typing import TypeVar

from flext_core.handlers import (
    FlextAuthorizingHandler,
    FlextBaseHandler,
    FlextEventHandler,
    FlextHandlerChain,
    FlextHandlerRegistry,
    FlextMetricsHandler,
    FlextValidatingHandler,
)
from flext_core.result import FlextResult

TCommand = TypeVar("TCommand")
TResult = TypeVar("TResult")
TQuery = TypeVar("TQuery")
TEvent = TypeVar("TEvent")


class FlextCommandHandler[TCommand, TResult](FlextBaseHandler, ABC):
    """Generic command handler for backward compatibility."""

    @abstractmethod
    def handle_command(self, command: TCommand) -> FlextResult[TResult]:
        """Handle command - must be implemented in subclasses."""


class FlextQueryHandler[TQuery, TResult](FlextBaseHandler, ABC):
    """Generic query handler for backward compatibility."""

    @abstractmethod
    def handle_query(self, query: TQuery) -> FlextResult[TResult]:
        """Handle query - must be implemented in subclasses."""


# Legacy aliases for backward compatibility
FlextHandlers = FlextBaseHandler

__all__ = [
    "FlextAuthorizingHandler",
    "FlextBaseHandler",
    "FlextCommandHandler",  # Legacy alias
    "FlextEventHandler",
    "FlextHandlerChain",
    "FlextHandlerRegistry",
    "FlextHandlers",  # Legacy alias
    "FlextMetricsHandler",
    "FlextQueryHandler",  # Legacy alias
    "FlextValidatingHandler",
]
