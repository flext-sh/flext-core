"""FLEXT Core Handler Pattern - Unified Handler System.

Enterprise-grade handler pattern implementation for processing
messages, events, and requests with standardized interfaces.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Protocol
from typing import TypeVar

from flext_core.patterns.typedefs import FlextHandlerId
from flext_core.patterns.typedefs import FlextHandlerName
from flext_core.patterns.typedefs import FlextMessageType
from flext_core.result import FlextResult

# =============================================================================
# TYPE VARIABLES - Generic handler typing
# =============================================================================

TMessage = TypeVar("TMessage")
TEvent = TypeVar("TEvent")
TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")

# =============================================================================
# BASE HANDLER - Abstract foundation for all handlers
# =============================================================================


class FlextHandler(ABC):
    """Base class for all FLEXT handlers.

    Provides common functionality for message processing,
    error handling, and metadata management.
    """

    def __init__(
        self,
        handler_id: FlextHandlerId | None = None,
        handler_name: FlextHandlerName | None = None,
    ) -> None:
        """Initialize handler with optional ID and name."""
        self.handler_id = handler_id or FlextHandlerId(
            f"{self.__class__.__name__}_{id(self)}",
        )
        self.handler_name = handler_name or FlextHandlerName(
            self.__class__.__name__,
        )

    @abstractmethod
    def can_handle(self, message: object) -> bool:
        """Check if handler can process the given message.

        Args:
            message: Message to check

        Returns:
            True if handler can process message

        """

    def get_handler_metadata(self) -> dict[str, Any]:
        """Get handler metadata for logging and debugging."""
        return {
            "handler_id": self.handler_id,
            "handler_name": self.handler_name,
            "handler_class": self.__class__.__name__,
        }

    def validate_message(self, message: object) -> FlextResult[None]:
        """Validate message before processing.

        Args:
            message: Message to validate

        Returns:
            FlextResult indicating validation success or failure

        """
        if message is None:
            return FlextResult.fail("Message cannot be None")
        return FlextResult.ok(None)


# =============================================================================
# MESSAGE HANDLER - For general message processing
# =============================================================================


class FlextMessageHandler[TMessage, TResponse](FlextHandler):
    """Handler for processing messages with typed input/output."""

    @abstractmethod
    def handle_message(self, message: TMessage) -> FlextResult[TResponse]:
        """Process a message and return result.

        Args:
            message: Message to process

        Returns:
            FlextResult with processed data or error

        """

    def process(self, message: TMessage) -> FlextResult[TResponse]:
        """Process message with validation and error handling.

        Args:
            message: Message to process

        Returns:
            FlextResult with processing results

        """
        # Validate message
        validation_result = self.validate_message(message)
        if validation_result.is_failure:
            return FlextResult.fail(
                f"Message validation failed: {validation_result.error}",
            )

        # Check if we can handle this message
        if not self.can_handle(message):
            return FlextResult.fail(
                f"Handler {self.handler_name} cannot process this message",
            )

        # Process message
        try:
            return self.handle_message(message)
        except (ValueError, TypeError, AttributeError) as e:
            return FlextResult.fail(
                f"Message processing failed: {e!s}",
            )
        except Exception as e:
            return FlextResult.fail(
                f"Unexpected message processing error: {e!s}",
            )


# =============================================================================
# EVENT HANDLER - For event-driven processing
# =============================================================================


class FlextEventHandler[TEvent](FlextHandler):
    """Handler for processing domain events."""

    @abstractmethod
    def handle_event(self, event: TEvent) -> FlextResult[None]:
        """Process an event.

        Args:
            event: Event to process

        Returns:
            FlextResult indicating processing success or failure

        """

    @abstractmethod
    def get_event_type(self) -> FlextMessageType:
        """Get the event type this handler processes."""

    def can_handle(self, message: object) -> bool:
        """Check if this is an event we can handle."""
        if not hasattr(message, "event_type"):
            return False
        return bool(message.event_type == self.get_event_type())

    def process_event(self, event: TEvent) -> FlextResult[None]:
        """Process event with validation and error handling.

        Args:
            event: Event to process

        Returns:
            FlextResult indicating processing success

        """
        # Validate event
        validation_result = self.validate_message(event)
        if validation_result.is_failure:
            return FlextResult.fail(
                f"Event validation failed: {validation_result.error}",
            )

        # Check if we can handle this event
        if not self.can_handle(event):
            return FlextResult.fail(
                f"Handler {self.handler_name} cannot process this event",
            )

        # Process event
        try:
            return self.handle_event(event)
        except (ValueError, TypeError, AttributeError) as e:
            return FlextResult.fail(
                f"Event processing failed: {e!s}",
            )
        except Exception as e:
            return FlextResult.fail(
                f"Unexpected event processing error: {e!s}",
            )


# =============================================================================
# REQUEST HANDLER - For request/response processing
# =============================================================================


class FlextRequestHandler[TRequest, TResponse](FlextHandler):
    """Handler for processing requests with responses."""

    @abstractmethod
    def handle_request(self, request: TRequest) -> FlextResult[TResponse]:
        """Process a request and return response.

        Args:
            request: Request to process

        Returns:
            FlextResult with response data or error

        """

    @abstractmethod
    def get_request_type(self) -> type[TRequest]:
        """Get the request type this handler processes."""

    def can_handle(self, message: object) -> bool:
        """Check if this is a request we can handle."""
        return isinstance(message, self.get_request_type())

    def process_request(self, request: TRequest) -> FlextResult[TResponse]:
        """Process request with validation and error handling.

        Args:
            request: Request to process

        Returns:
            FlextResult with response or error

        """
        # Validate request
        validation_result = self.validate_message(request)
        if validation_result.is_failure:
            return FlextResult.fail(
                f"Request validation failed: {validation_result.error}",
            )

        # Check if we can handle this request
        if not self.can_handle(request):
            return FlextResult.fail(
                f"Handler {self.handler_name} cannot process this request",
            )

        # Process request
        try:
            return self.handle_request(request)
        except (ValueError, TypeError, AttributeError) as e:
            return FlextResult.fail(
                f"Request processing failed: {e!s}",
            )
        except Exception as e:
            return FlextResult.fail(
                f"Unexpected request processing error: {e!s}",
            )


# =============================================================================
# HANDLER REGISTRY - Central handler management
# =============================================================================


class FlextHandlerRegistry:
    """Registry for managing and discovering handlers."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._handlers: list[FlextHandler] = []

    def register(self, handler: object) -> FlextResult[None]:
        """Register a handler.

        Args:
            handler: Handler to register

        Returns:
            FlextResult indicating registration success

        """
        if not isinstance(handler, FlextHandler):
            return FlextResult.fail(
                "Handler must be instance of FlextHandler",
            )

        self._handlers.append(handler)
        return FlextResult.ok(None)

    def find_handlers(self, message: object) -> list[FlextHandler]:
        """Find all handlers that can process a message.

        Args:
            message: Message to find handlers for

        Returns:
            List of capable handlers

        """
        return [handler for handler in self._handlers if handler.can_handle(message)]

    def get_handler_by_id(
        self,
        handler_id: FlextHandlerId,
    ) -> FlextHandler | None:
        """Get handler by ID.

        Args:
            handler_id: ID of handler to find

        Returns:
            Handler if found, None otherwise

        """
        for handler in self._handlers:
            if handler.handler_id == handler_id:
                return handler
        return None

    def get_all_handlers(self) -> list[FlextHandler]:
        """Get all registered handlers."""
        return list(self._handlers)

    def get_handler_info(self) -> list[dict[str, Any]]:
        """Get information about all registered handlers."""
        return [handler.get_handler_metadata() for handler in self._handlers]


# =============================================================================
# HANDLER PROTOCOLS - For type checking
# =============================================================================


class MessageProcessor(Protocol):
    """Protocol for message processing."""

    def process(self, message: object) -> FlextResult[object]:
        """Process a message."""
        ...


class EventProcessor(Protocol):
    """Protocol for event processing."""

    def process_event(self, event: object) -> FlextResult[None]:
        """Process an event."""
        ...


class RequestProcessor(Protocol):
    """Protocol for request processing."""

    def process_request(self, request: object) -> FlextResult[object]:
        """Process a request."""
        ...


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "EventProcessor",
    "FlextEventHandler",
    "FlextHandler",
    "FlextHandlerRegistry",
    "FlextMessageHandler",
    "FlextRequestHandler",
    "MessageProcessor",
    "RequestProcessor",
    "TEvent",
    "TMessage",
    "TRequest",
    "TResponse",
]
