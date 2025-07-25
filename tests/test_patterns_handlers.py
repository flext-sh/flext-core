"""Comprehensive tests for FLEXT handler pattern."""

from __future__ import annotations

from typing import Any

from flext_core.patterns.handlers import FlextEventHandler
from flext_core.patterns.handlers import FlextHandlerRegistry
from flext_core.patterns.handlers import FlextMessageHandler
from flext_core.patterns.handlers import FlextRequestHandler
from flext_core.patterns.typedefs import FlextHandlerId
from flext_core.patterns.typedefs import FlextHandlerName
from flext_core.patterns.typedefs import FlextMessageType
from flext_core.result import FlextResult

# ===================================================================
# TEST DATA CLASSES
# ===================================================================


class SampleMessage:
    """Test message for handler testing."""

    def __init__(self, content: str) -> None:
        """Initialize test message with content."""
        self.content = content


class SampleEvent:
    """Test event for handler testing."""

    def __init__(self, event_type: str, data: dict[str, Any]) -> None:
        """Initialize test event with type and data."""
        self.event_type = event_type
        self.data = data


class SampleRequest:
    """Test request for handler testing."""

    def __init__(self, action: str, params: dict[str, Any]) -> None:
        """Initialize test request with action and parameters."""
        self.action = action
        self.params = params


class SampleResponse:
    """Test response for handler testing."""

    def __init__(
        self,
        result: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize test response with result and optional data."""
        self.result = result
        self.data = data or {}


# ===================================================================
# TEST HANDLER IMPLEMENTATIONS
# ===================================================================


class SampleMessageHandler(FlextMessageHandler[SampleMessage, str]):
    """Test implementation of message handler."""

    def __init__(self, handled_content: str = "test") -> None:
        """Initialize message handler with content to handle."""
        super().__init__()
        self.handled_content = handled_content

    def can_handle(self, message: Any) -> bool:
        """Check if can handle message."""
        if not isinstance(message, SampleMessage):
            return False
        return message.content == self.handled_content

    def handle_message(self, message: SampleMessage) -> FlextResult[str]:
        """Handle the message."""
        return FlextResult.ok(f"Handled: {message.content}")


class SampleEventHandler(FlextEventHandler[SampleEvent]):
    """Test implementation of event handler."""

    def __init__(self, event_type: str = "test.event") -> None:
        """Initialize event handler with event type to handle."""
        super().__init__()
        self.event_type_value = event_type

    def get_event_type(self) -> FlextMessageType:
        """Get event type this handler processes."""
        return FlextMessageType(self.event_type_value)

    def handle_event(self, event: SampleEvent) -> FlextResult[None]:
        """Handle the event."""
        if event.data.get("should_fail"):
            return FlextResult.fail("Event processing failed")
        return FlextResult.ok(None)


class SampleRequestHandler(FlextRequestHandler[SampleRequest, SampleResponse]):
    """Test implementation of request handler."""

    def __init__(self, handled_action: str = "test") -> None:
        """Initialize request handler with action to handle."""
        super().__init__()
        self.handled_action = handled_action

    def get_request_type(self) -> type[SampleRequest]:
        """Get request type this handler processes."""
        return SampleRequest

    def can_handle(self, message: Any) -> bool:
        """Check if can handle message."""
        if not isinstance(message, SampleRequest):
            return False
        return message.action == self.handled_action

    def handle_request(
        self,
        request: SampleRequest,
    ) -> FlextResult[SampleResponse]:
        """Handle the request."""
        if request.action != self.handled_action:
            return FlextResult.fail(f"Cannot handle action: {request.action}")

        response = SampleResponse("success", {"processed": True})
        return FlextResult.ok(response)


# ===================================================================
# TEST BASE HANDLER
# ===================================================================


class TestFlextHandler:
    """Test base FlextHandler functionality."""

    def test_handler_creation(self) -> None:
        """Test creating handler with default values."""
        handler = SampleMessageHandler()

        assert handler.handler_id is not None
        assert handler.handler_name == FlextHandlerName("SampleMessageHandler")

    def test_handler_creation_with_custom_values(self) -> None:
        """Test creating handler with custom ID and name."""
        handler_id = FlextHandlerId("custom_handler")
        handler_name = FlextHandlerName("CustomHandler")

        handler = SampleMessageHandler()
        handler.handler_id = handler_id
        handler.handler_name = handler_name

        assert handler.handler_id == handler_id
        assert handler.handler_name == handler_name

    def test_get_handler_metadata(self) -> None:
        """Test getting handler metadata."""
        handler = SampleMessageHandler()
        metadata = handler.get_handler_metadata()

        assert "handler_id" in metadata
        assert "handler_name" in metadata
        assert "handler_class" in metadata
        assert metadata["handler_class"] == "SampleMessageHandler"

    def test_validate_message_none(self) -> None:
        """Test validating None message."""
        handler = SampleMessageHandler()
        result = handler.validate_message(None)

        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "message cannot be none" in result.error.lower()

    def test_validate_message_valid(self) -> None:
        """Test validating valid message."""
        handler = SampleMessageHandler()
        message = SampleMessage("test")
        result = handler.validate_message(message)

        assert result.is_success is True


# ===================================================================
# TEST MESSAGE HANDLER
# ===================================================================


class TestFlextMessageHandler:
    """Test FlextMessageHandler functionality."""

    def test_handle_message_success(self) -> None:
        """Test successful message handling."""
        handler = SampleMessageHandler("hello")
        message = SampleMessage("hello")

        result = handler.handle_message(message)
        assert result.is_success is True
        assert result.data == "Handled: hello"

    def test_process_message_success(self) -> None:
        """Test complete message processing flow."""
        handler = SampleMessageHandler("test")
        message = SampleMessage("test")

        result = handler.process(message)
        assert result.is_success is True
        assert result.data == "Handled: test"

    def test_process_message_validation_failure(self) -> None:
        """Test processing with validation failure."""
        handler = SampleMessageHandler()

        result = handler.process(None)
        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "validation failed" in result.error.lower()

    def test_process_message_cannot_handle(self) -> None:
        """Test processing message that cannot be handled."""
        handler = SampleMessageHandler("specific")
        message = SampleMessage("different")

        result = handler.process(message)
        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "cannot process" in result.error.lower()

    def test_process_message_handling_exception(self) -> None:
        """Test handling exceptions during message processing."""

        class FailingHandler(FlextMessageHandler[SampleMessage, str]):
            def can_handle(self, message: Any) -> bool:
                return isinstance(message, SampleMessage)

            def handle_message(
                self,
                message: SampleMessage,  # noqa: ARG002
            ) -> FlextResult[str]:
                msg = "Handler failed"
                raise ValueError(msg)

        handler = FailingHandler()
        message = SampleMessage("test")

        result = handler.process(message)
        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "processing failed" in result.error.lower()


# ===================================================================
# TEST EVENT HANDLER
# ===================================================================


class TestFlextEventHandler:
    """Test FlextEventHandler functionality."""

    def test_handle_event_success(self) -> None:
        """Test successful event handling."""
        handler = SampleEventHandler("user.created")
        event = SampleEvent("user.created", {"user_id": "123"})

        result = handler.handle_event(event)
        assert result.is_success is True

    def test_handle_event_failure(self) -> None:
        """Test event handling failure."""
        handler = SampleEventHandler("user.created")
        event = SampleEvent("user.created", {"should_fail": True})

        result = handler.handle_event(event)
        assert result.is_failure is True

    def test_can_handle_correct_event_type(self) -> None:
        """Test can_handle with correct event type."""
        handler = SampleEventHandler("order.placed")
        event = SampleEvent("order.placed", {})

        assert handler.can_handle(event) is True

    def test_can_handle_wrong_event_type(self) -> None:
        """Test can_handle with wrong event type."""
        handler = SampleEventHandler("order.placed")
        event = SampleEvent("order.cancelled", {})

        assert handler.can_handle(event) is False

    def test_can_handle_no_event_type_attribute(self) -> None:
        """Test can_handle with object without event_type."""
        handler = SampleEventHandler()
        message = SampleMessage("test")

        assert handler.can_handle(message) is False

    def test_process_event_success(self) -> None:
        """Test complete event processing flow."""
        handler = SampleEventHandler("payment.completed")
        event = SampleEvent("payment.completed", {"amount": 100})

        result = handler.process_event(event)
        assert result.is_success is True

    def test_process_event_validation_failure(self) -> None:
        """Test event processing with validation failure."""
        handler = SampleEventHandler()

        result = handler.process_event(None)
        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "validation failed" in result.error.lower()

    def test_process_event_cannot_handle(self) -> None:
        """Test processing event that cannot be handled."""
        handler = SampleEventHandler("specific.event")
        event = SampleEvent("different.event", {})

        result = handler.process_event(event)
        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "cannot process" in result.error.lower()


# ===================================================================
# TEST REQUEST HANDLER
# ===================================================================


class TestFlextRequestHandler:
    """Test FlextRequestHandler functionality."""

    def test_handle_request_success(self) -> None:
        """Test successful request handling."""
        handler = SampleRequestHandler("create_user")
        request = SampleRequest("create_user", {"name": "John"})

        result = handler.handle_request(request)
        assert result.is_success is True
        assert result.data is not None
        assert result.data.result == "success"

    def test_handle_request_failure(self) -> None:
        """Test request handling failure."""
        handler = SampleRequestHandler("create_user")
        request = SampleRequest("delete_user", {"id": "123"})

        result = handler.handle_request(request)
        assert result.is_failure is True

    def test_can_handle_correct_request_type(self) -> None:
        """Test can_handle with correct request type."""
        handler = SampleRequestHandler()
        request = SampleRequest("test", {})

        assert handler.can_handle(request) is True

    def test_can_handle_wrong_request_type(self) -> None:
        """Test can_handle with wrong request type."""
        handler = SampleRequestHandler()
        message = SampleMessage("test")

        assert handler.can_handle(message) is False

    def test_process_request_success(self) -> None:
        """Test complete request processing flow."""
        handler = SampleRequestHandler("update_profile")
        request = SampleRequest("update_profile", {"name": "Jane"})

        result = handler.process_request(request)
        assert result.is_success is True

    def test_process_request_validation_failure(self) -> None:
        """Test request processing with validation failure."""
        handler = SampleRequestHandler()

        result = handler.process_request(None)
        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "validation failed" in result.error.lower()

    def test_process_request_cannot_handle(self) -> None:
        """Test processing request that cannot be handled."""
        handler = SampleRequestHandler("specific_action")
        request = SampleRequest("different_action", {})

        result = handler.process_request(request)
        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "cannot process" in result.error.lower()


# ===================================================================
# TEST HANDLER REGISTRY
# ===================================================================


class TestFlextHandlerRegistry:
    """Test FlextHandlerRegistry functionality."""

    def test_registry_creation(self) -> None:
        """Test creating empty registry."""
        registry = FlextHandlerRegistry()
        assert len(registry.get_all_handlers()) == 0

    def test_register_handler_success(self) -> None:
        """Test successful handler registration."""
        registry = FlextHandlerRegistry()
        handler = SampleMessageHandler()

        result = registry.register(handler)
        assert result.is_success is True
        assert len(registry.get_all_handlers()) == 1

    def test_register_invalid_handler(self) -> None:
        """Test registering invalid handler."""
        registry = FlextHandlerRegistry()

        result = registry.register("not_a_handler")
        assert result.is_failure is True

    def test_find_handlers_for_message(self) -> None:
        """Test finding handlers for specific message."""
        registry = FlextHandlerRegistry()
        handler1 = SampleMessageHandler("hello")
        handler2 = SampleMessageHandler("goodbye")

        registry.register(handler1)
        registry.register(handler2)

        message = SampleMessage("hello")
        handlers = registry.find_handlers(message)

        assert len(handlers) == 1
        assert handlers[0] == handler1

    def test_find_handlers_no_matches(self) -> None:
        """Test finding handlers when none match."""
        registry = FlextHandlerRegistry()
        handler = SampleMessageHandler("hello")
        registry.register(handler)

        message = SampleMessage("different")
        handlers = registry.find_handlers(message)

        assert len(handlers) == 0

    def test_get_handler_by_id(self) -> None:
        """Test getting handler by ID."""
        registry = FlextHandlerRegistry()
        handler = SampleMessageHandler()
        handler_id = handler.handler_id

        registry.register(handler)

        found_handler = registry.get_handler_by_id(handler_id)
        assert found_handler == handler

    def test_get_handler_by_id_not_found(self) -> None:
        """Test getting handler by non-existent ID."""
        registry = FlextHandlerRegistry()

        found_handler = registry.get_handler_by_id(
            FlextHandlerId("nonexistent"),
        )
        assert found_handler is None

    def test_get_all_handlers(self) -> None:
        """Test getting all registered handlers."""
        registry = FlextHandlerRegistry()
        handler1 = SampleMessageHandler("test1")
        handler2 = SampleEventHandler("test.event")

        registry.register(handler1)
        registry.register(handler2)

        all_handlers = registry.get_all_handlers()
        assert len(all_handlers) == 2
        assert handler1 in all_handlers
        assert handler2 in all_handlers

    def test_get_handler_info(self) -> None:
        """Test getting handler information."""
        registry = FlextHandlerRegistry()
        handler = SampleMessageHandler()
        registry.register(handler)

        info = registry.get_handler_info()
        assert len(info) == 1
        assert "handler_id" in info[0]
        assert "handler_class" in info[0]


# ===================================================================
# INTEGRATION TESTS
# ===================================================================


class TestHandlerPatternIntegration:
    """Integration tests for handler pattern."""

    def test_multi_handler_message_processing(self) -> None:
        """Test processing messages with multiple handlers."""
        registry = FlextHandlerRegistry()

        # Register multiple handlers for different message types
        hello_handler = SampleMessageHandler("hello")
        goodbye_handler = SampleMessageHandler("goodbye")

        registry.register(hello_handler)
        registry.register(goodbye_handler)

        # Test hello message
        hello_message = SampleMessage("hello")
        hello_handlers = registry.find_handlers(hello_message)
        assert len(hello_handlers) == 1

        message_handler = hello_handlers[0]
        assert isinstance(message_handler, FlextMessageHandler)
        result = message_handler.process(hello_message)
        assert result.is_success is True

        # Test goodbye message
        goodbye_message = SampleMessage("goodbye")
        goodbye_handlers = registry.find_handlers(goodbye_message)
        assert len(goodbye_handlers) == 1

        message_handler = goodbye_handlers[0]
        assert isinstance(message_handler, FlextMessageHandler)
        result = message_handler.process(goodbye_message)
        assert result.is_success is True

    def test_event_driven_processing_flow(self) -> None:
        """Test complete event-driven processing flow."""
        registry = FlextHandlerRegistry()

        # Register event handlers
        user_handler = SampleEventHandler("user.created")
        order_handler = SampleEventHandler("order.placed")

        registry.register(user_handler)
        registry.register(order_handler)

        # Process user created event
        user_event = SampleEvent("user.created", {"user_id": "123"})
        handlers = registry.find_handlers(user_event)
        assert len(handlers) == 1

        event_handler = handlers[0]
        assert isinstance(event_handler, FlextEventHandler)
        result = event_handler.process_event(user_event)
        assert result.is_success is True

        # Process order placed event
        order_event = SampleEvent("order.placed", {"order_id": "456"})
        handlers = registry.find_handlers(order_event)
        assert len(handlers) == 1

        event_handler = handlers[0]
        assert isinstance(event_handler, FlextEventHandler)
        result = event_handler.process_event(order_event)
        assert result.is_success is True

    def test_request_response_processing_flow(self) -> None:
        """Test complete request/response processing flow."""
        registry = FlextHandlerRegistry()

        # Register request handlers
        user_handler = SampleRequestHandler("create_user")
        profile_handler = SampleRequestHandler("update_profile")

        registry.register(user_handler)
        registry.register(profile_handler)

        # Process create user request
        create_request = SampleRequest("create_user", {"name": "John"})
        handlers = registry.find_handlers(create_request)
        assert len(handlers) == 1

        request_handler = handlers[0]
        assert isinstance(request_handler, FlextRequestHandler)
        result = request_handler.process_request(create_request)
        assert result.is_success is True
        assert result.data is not None
        assert result.data.result == "success"

        # Process update profile request
        update_request = SampleRequest("update_profile", {"bio": "Developer"})
        handlers = registry.find_handlers(update_request)
        assert len(handlers) == 1

        request_handler = handlers[0]
        assert isinstance(request_handler, FlextRequestHandler)
        result = request_handler.process_request(update_request)
        assert result.is_success is True

    def test_mixed_handler_types_in_registry(self) -> None:
        """Test registry with mixed handler types."""
        registry = FlextHandlerRegistry()

        # Register different types of handlers
        message_handler = SampleMessageHandler("process")
        event_handler = SampleEventHandler("data.updated")
        request_handler = SampleRequestHandler("execute")

        registry.register(message_handler)
        registry.register(event_handler)
        registry.register(request_handler)

        # Verify all handlers are registered
        all_handlers = registry.get_all_handlers()
        assert len(all_handlers) == 3

        # Test each handler type can be found
        message = SampleMessage("process")
        assert len(registry.find_handlers(message)) == 1

        event = SampleEvent("data.updated", {})
        assert len(registry.find_handlers(event)) == 1

        request = SampleRequest("execute", {})
        assert len(registry.find_handlers(request)) == 1

    def test_handler_error_recovery(self) -> None:
        """Test error handling and recovery in handlers."""
        # Test message handler error recovery
        handler = SampleMessageHandler()
        message = SampleMessage("unhandled")

        result = handler.process(message)
        assert result.is_failure is True

        # Test event handler error recovery
        event_handler = SampleEventHandler("test.event")
        failing_event = SampleEvent("test.event", {"should_fail": True})

        event_result = event_handler.process_event(failing_event)
        assert event_result.is_failure is True

        # Verify handler is still functional for valid inputs
        valid_event = SampleEvent("test.event", {"valid": True})
        valid_event_result = event_handler.process_event(valid_event)
        assert valid_event_result.is_success is True
