"""FLEXT Base handlers module.

Backward compatibility module that provides handler abstractions.
This module exists to maintain compatibility with existing tests.
"""

from __future__ import annotations

# Re-export specific handlers for backward compatibility following SOLID principles
# Generic type variables for handlers
from abc import ABC, abstractmethod
from typing import TypeVar

# Import abstractions directly from handlers_base to avoid circular imports
from flext_core.handlers_base import FlextAbstractHandler
from flext_core.result import FlextResult

TCommand = TypeVar("TCommand")
TResult = TypeVar("TResult")
TQuery = TypeVar("TQuery")
TEvent = TypeVar("TEvent")


# Create concrete base handler first
class FlextBaseHandler(FlextAbstractHandler[object, object]):
    """Concrete base handler for backward compatibility."""

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return self.__class__.__name__

    def can_handle(self, request: object) -> bool:  # noqa: ARG002
        """Check if handler can handle request."""
        return True

    def handle(self, request: object) -> FlextResult[object]:
        """Handle request - delegate to process_request."""
        return self.process_request(request)

    def validate_request(self, request: object) -> FlextResult[None]:  # noqa: ARG002
        """Validate request - default implementation allows all."""
        return FlextResult.ok(None)

    def process_request(self, request: object) -> FlextResult[object]:
        """Process request - default implementation."""
        return FlextResult.ok(request)


class FlextCommandHandler[TCommand, TResult](
    FlextAbstractHandler[TCommand, TResult],
    ABC,
):
    """Generic command handler for backward compatibility."""

    @abstractmethod
    def handle_command(self, command: TCommand) -> FlextResult[TResult]:
        """Handle command - must be implemented in subclasses."""

    def can_handle(self, request: TCommand) -> bool:  # noqa: ARG002
        """Check if handler can handle request."""
        return True

    def handle(self, request: TCommand) -> FlextResult[TResult]:
        """Handle request - delegate to handle_command."""
        return self.handle_command(request)

    def validate_request(self, request: TCommand) -> FlextResult[None]:  # noqa: ARG002
        """Validate request - default implementation allows all."""
        return FlextResult.ok(None)

    def process_request(self, request: TCommand) -> FlextResult[TResult]:
        """Process request - delegate to handle_command."""
        return self.handle_command(request)

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return self.__class__.__name__


class FlextQueryHandler[TQuery, TResult](FlextAbstractHandler[TQuery, TResult], ABC):
    """Generic query handler for backward compatibility."""

    @abstractmethod
    def handle_query(self, query: TQuery) -> FlextResult[TResult]:
        """Handle query - must be implemented in subclasses."""

    def can_handle(self, request: TQuery) -> bool:  # noqa: ARG002
        """Check if handler can handle request."""
        return True

    def handle(self, request: TQuery) -> FlextResult[TResult]:
        """Handle request - delegate to handle_query."""
        return self.handle_query(request)

    def validate_request(self, request: TQuery) -> FlextResult[None]:  # noqa: ARG002
        """Validate request - default implementation allows all."""
        return FlextResult.ok(None)

    def process_request(self, request: TQuery) -> FlextResult[TResult]:
        """Process request - delegate to handle_query."""
        return self.handle_query(request)

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return self.__class__.__name__


# Create simple aliases for other handlers to avoid circular imports
class FlextValidatingHandler(FlextBaseHandler):
    """Validating handler for backward compatibility."""


class FlextAuthorizingHandler(FlextBaseHandler):
    """Authorizing handler for backward compatibility."""


class FlextEventHandler(FlextBaseHandler):
    """Event handler for backward compatibility."""


class FlextMetricsHandler(FlextBaseHandler):
    """Metrics handler for backward compatibility."""


class FlextHandlerChain(FlextBaseHandler):
    """Handler chain for backward compatibility."""


class FlextHandlerRegistry:
    """Handler registry for backward compatibility."""

    def __init__(self) -> None:
        """Initialize handler registry."""
        self._handlers: list[FlextBaseHandler] = []

    def register(self, handler: FlextAbstractHandler[object, object] | FlextBaseHandler) -> None:
        """Register a handler."""
        # Normalize to base handler type for storage
        if isinstance(handler, FlextBaseHandler):
            self._handlers.append(handler)
        else:
            # Wrap unknown abstract handler types into a simple adapter
            class _Adapter(FlextBaseHandler):
                def process_request(self, request: object) -> FlextResult[object]:
                    return handler.handle(request)

            self._handlers.append(_Adapter())

    def get_handlers(self) -> list[FlextBaseHandler]:
        """Get registered handlers."""
        return self._handlers.copy()


# Legacy aliases for backward compatibility
FlextHandlers = FlextBaseHandler

__all__: list[str] = [
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
