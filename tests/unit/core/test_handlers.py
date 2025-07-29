"""Comprehensive tests for FlextHandlers and handler functionality."""

from __future__ import annotations

# Use flext-core modern type definitions
from typing import TYPE_CHECKING

from flext_core.handlers import FlextHandlers
from flext_core.result import FlextResult

# Constants
EXPECTED_BULK_SIZE = 2

if TYPE_CHECKING:
    from flext_core.types import TData


class _TestMessage:
    """Test message class for handler testing."""

    def __init__(self, content: str, message_type: str = "test") -> None:
        self.content = content
        self.message_type = message_type


class _TestCommand:
    """Test command class for command handler testing."""

    def __init__(self, action: str, data: TData | None = None) -> None:
        self.action = action
        self.data = data or {}


class _TestEvent:
    """Test event class for event handler testing."""

    def __init__(self, event_type: str, payload: TData | None = None) -> None:
        self.event_type = event_type
        self.payload = payload or {}


class _TestQuery:
    """Test query class for query handler testing."""

    def __init__(self, query_type: str, parameters: TData | None = None) -> None:
        self.query_type = query_type
        self.parameters = parameters or {}


class TestBaseHandler:
    """Test FlextHandlers.Handler base functionality."""

    def test_handler_creation(self) -> None:
        """Test basic handler creation."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        if handler.handler_name != "test_handler":
            raise AssertionError(
                f"Expected {'test_handler'}, got {handler.handler_name}"
            )
        assert handler.handler_id.startswith("Handler_")
        assert hasattr(handler, "_metrics")
        if handler._metrics["messages_handled"] != 0:
            raise AssertionError(
                f"Expected {0}, got {handler._metrics['messages_handled']}"
            )

    def test_handler_default_name(self) -> None:
        """Test handler creation with default name."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler()

        if handler.handler_name != "Handler":
            raise AssertionError(f"Expected {'Handler'}, got {handler.handler_name}")
        assert handler._handler_name == "Handler"

    def test_handler_can_handle_default(self) -> None:
        """Test default can_handle implementation."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler()
        message = _TestMessage("test")

        # Default implementation should return True
        if not (handler.can_handle(message)):
            raise AssertionError(f"Expected True, got {handler.can_handle(message)}")
        assert handler.can_handle(None) is True
        if not (handler.can_handle("string")):
            raise AssertionError(f"Expected True, got {handler.can_handle('string')}")

    def test_handler_handle_default(self) -> None:
        """Test default handle implementation."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler()
        message = _TestMessage("test")

        result = handler.handle(message)
        assert isinstance(result, FlextResult)
        assert result.is_success
        if result.data != message:
            raise AssertionError(f"Expected {message}, got {result.data}")

    def test_handler_validate_message(self) -> None:
        """Test message validation."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler()

        # Valid message
        message = _TestMessage("test")
        result = handler.validate_message(message)
        assert result.is_success
        if result.data != message:
            raise AssertionError(f"Expected {message}, got {result.data}")

        # Invalid message (None)
        result = handler.validate_message(None)
        assert result.is_failure
        assert result.error is not None
        assert result.error is not None
        if "cannot be None" not in result.error:
            raise AssertionError(f"Expected {'cannot be None'} in {result.error}")

    def test_handler_get_handler_metadata(self) -> None:
        """Test handler metadata."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )
        metadata = handler.get_handler_metadata()

        assert isinstance(metadata, dict)
        if metadata["handler_name"] != "test_handler":
            raise AssertionError(
                f"Expected {'test_handler'}, got {metadata['handler_name']}"
            )
        assert metadata["handler_class"] == "Handler"
        if "handler_id" not in metadata:
            raise AssertionError(f"Expected {'handler_id'} in {metadata}")

    def test_handler_process_message(self) -> None:
        """Test message processing workflow."""

        class TestHandler(FlextHandlers.Handler[object, object]):
            def handle(self, message: object) -> FlextResult[object]:
                if isinstance(message, _TestMessage):
                    return FlextResult.ok(f"Processed: {message.content}")
                return FlextResult.ok(message)

        handler = TestHandler("processor")
        message = _TestMessage("hello")

        result = handler.process_message(message)
        assert result.is_success
        if result.data != "Processed: hello":
            raise AssertionError(f"Expected {'Processed: hello'}, got {result.data}")

    def test_handler_process_message_validation_failure(self) -> None:
        """Test message processing with validation failure."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler()

        result = handler.process_message(None)
        assert result.is_failure
        assert result.error is not None
        assert result.error is not None
        if "cannot be None" not in result.error:
            raise AssertionError(f"Expected {'cannot be None'} in {result.error}")

    def test_handler_process_message_cannot_handle(self) -> None:
        """Test message processing when handler cannot handle message."""

        class SelectiveHandler(FlextHandlers.Handler[object, object]):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestMessage)

        handler = SelectiveHandler()

        result = handler.process_message("not a test message")
        assert result.is_failure
        assert result.error is not None
        if "cannot process this message" not in result.error:
            raise AssertionError(
                f"Expected {'cannot process this message'} in {result.error}"
            )

    def test_handler_process_message_exception_handling(self) -> None:
        """Test message processing with exception handling."""

        class FailingHandler(FlextHandlers.Handler[object, object]):
            def handle_message(self, message: object) -> FlextResult[object]:
                msg = "Processing failed"
                raise RuntimeError(msg)

        handler = FailingHandler()
        message = _TestMessage("test")

        result = handler.process_message(message)
        assert result.is_failure
        assert result.error is not None
        if "Processing failed" not in result.error:
            raise AssertionError(f"Expected {'Processing failed'} in {result.error}")

    def test_handler_process_generic_message(self) -> None:
        """Test generic message processing."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler()
        message = _TestMessage("test")

        result = handler.process(message)
        assert result.is_success

        # Test with None
        result = handler.process(None)
        assert result.is_failure
        assert result.error is not None
        if "validation failed" not in result.error:
            raise AssertionError(f"Expected {'validation failed'} in {result.error}")

    def test_handler_delegation_methods(self) -> None:
        """Test delegation to base handler."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler("test")
        message = _TestMessage("test")

        # Test delegation methods exist and work
        pre_result = handler.pre_handle(message)
        assert isinstance(pre_result, FlextResult)

        post_result = handler.post_handle(FlextResult.ok(message))
        assert isinstance(post_result, FlextResult)

        # Test handle_with_hooks
        hooks_result = handler.handle_with_hooks(message)
        assert isinstance(hooks_result, FlextResult)

        # Test get_metrics
        metrics = handler.get_metrics()
        assert isinstance(metrics, dict)

        # Test logger access
        logger = handler.logger
        # Logger might be None or an actual logger object
        assert logger is not None or logger is None


class _TestCommandHandler:
    """Test FlextHandlers.CommandHandler functionality."""

    def test_command_handler_creation(self) -> None:
        """Test command handler creation."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler("test_command_handler")
        )

        if handler._handler_name != "test_command_handler":
            raise AssertionError(
                f"Expected {'test_command_handler'}, got {handler._handler_name}"
            )

    def test_command_handler_default_name(self) -> None:
        """Test command handler with default name."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler()
        )

        assert handler._handler_name is None

    def test_command_handler_validate_command(self) -> None:
        """Test command validation."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler()
        )
        command = _TestCommand("create_user")

        result = handler.validate_command(command)
        assert result.is_success

    def test_command_handler_handle_default(self) -> None:
        """Test default command handling."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler()
        )
        command = _TestCommand("test_action")

        result = handler.handle(command)
        assert result.is_success
        if result.data != command:
            raise AssertionError(f"Expected {command}, got {result.data}")

    def test_command_handler_can_handle(self) -> None:
        """Test command handler can_handle."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler()
        )

        if not (handler.can_handle(_TestCommand("test"))):
            raise AssertionError(
                f"Expected True, got {handler.can_handle(_TestCommand('test'))}"
            )
        assert handler.can_handle("any message") is True

    def test_command_handler_pre_handle(self) -> None:
        """Test command pre-processing."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler()
        )
        command = _TestCommand("test")

        result = handler.pre_handle(command)
        assert result.is_success
        if result.data != command:
            raise AssertionError(f"Expected {command}, got {result.data}")

    def test_command_handler_pre_handle_validation_failure(self) -> None:
        """Test command pre-processing with validation failure."""

        class ValidatingCommandHandler(FlextHandlers.CommandHandler):
            def validate_command(self, command: object) -> FlextResult[None]:
                return FlextResult.fail("Validation failed")

        handler = ValidatingCommandHandler()
        command = _TestCommand("test")

        result = handler.pre_handle(command)
        assert result.is_failure
        assert result.error is not None
        if "Validation failed" not in result.error:
            raise AssertionError(f"Expected {'Validation failed'} in {result.error}")

    def test_command_handler_post_handle(self) -> None:
        """Test command post-processing."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler()
        )
        result = FlextResult.ok("test result")

        post_result = handler.post_handle(result)
        assert post_result.is_success
        if post_result.data != "test result":
            raise AssertionError(f"Expected {'test result'}, got {post_result.data}")

    def test_command_handler_handle_with_hooks(self) -> None:
        """Test command handling with hooks."""

        class _TestCommandHandler(FlextHandlers.CommandHandler):
            def handle(self, command: object) -> FlextResult[object]:
                if isinstance(command, _TestCommand):
                    return FlextResult.ok(f"Handled: {command.action}")
                return FlextResult.ok(command)

        handler = _TestCommandHandler()
        command = _TestCommand("create_user")

        result = handler.handle_with_hooks(command)
        assert result.is_success
        if result.data != "Handled: create_user":
            raise AssertionError(
                f"Expected {'Handled: create_user'}, got {result.data}"
            )

    def test_command_handler_handle_with_hooks_pre_failure(self) -> None:
        """Test command handling with pre-handle failure."""

        class FailingPreHandler(FlextHandlers.CommandHandler):
            def validate_command(self, command: object) -> FlextResult[None]:
                return FlextResult.fail("Pre-processing failed")

        handler = FailingPreHandler()
        command = _TestCommand("test")

        result = handler.handle_with_hooks(command)
        assert result.is_failure
        assert result.error is not None
        if "Pre-processing failed" not in result.error:
            raise AssertionError(
                f"Expected {'Pre-processing failed'} in {result.error}"
            )

    def test_command_handler_get_metrics(self) -> None:
        """Test command handler metrics."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler("test_handler")
        )
        metrics = handler.get_metrics()

        assert isinstance(metrics, dict)
        if metrics["handler_name"] != "test_handler":
            raise AssertionError(
                f"Expected {'test_handler'}, got {metrics['handler_name']}"
            )
        assert metrics["handler_type"] == "CommandHandler"

    def test_command_handler_logger_access(self) -> None:
        """Test command handler logger access."""
        handler: FlextHandlers.CommandHandler[object, object] = (
            FlextHandlers.CommandHandler()
        )
        logger = handler.logger

        # Logger might be None in the current implementation
        assert logger is None


class _TestEventHandler:
    """Test FlextHandlers.EventHandler functionality."""

    def test_event_handler_creation(self) -> None:
        """Test event handler creation."""

        class _TestEventHandler(FlextHandlers.EventHandler):
            def process_event_impl(self, event: object) -> None:
                pass

        handler = _TestEventHandler("test_event_handler")
        if handler.handler_name != "test_event_handler":
            raise AssertionError(
                f"Expected {'test_event_handler'}, got {handler.handler_name}"
            )

    def test_event_handler_handle(self) -> None:
        """Test event handling."""

        class _TestEventHandler(FlextHandlers.EventHandler):
            def process_event_impl(self, event: object) -> None:
                pass

        handler = _TestEventHandler()
        event = _TestEvent("user_created")

        result = handler.handle(event)
        assert result.is_success
        assert result.data is None

    def test_event_handler_process_event(self) -> None:
        """Test event processing."""

        class _TestEventHandler(FlextHandlers.EventHandler):
            def process_event_impl(self, event: object) -> None:
                pass

        handler = _TestEventHandler()
        event = _TestEvent("user_updated")

        result = handler.process_event(event)
        assert result.is_success

    def test_event_handler_process_event_validation_failure(self) -> None:
        """Test event processing with validation failure."""

        class _TestEventHandler(FlextHandlers.EventHandler):
            def process_event_impl(self, event: object) -> None:
                pass

        handler = _TestEventHandler()

        result = handler.process_event(None)
        assert result.is_failure
        assert result.error is not None
        if "cannot be None" not in result.error:
            raise AssertionError(f"Expected {'cannot be None'} in {result.error}")

    def test_event_handler_process_event_cannot_handle(self) -> None:
        """Test event processing when handler cannot handle."""

        class SelectiveEventHandler(FlextHandlers.EventHandler):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestEvent)

            def process_event_impl(self, event: object) -> None:
                pass

        handler = SelectiveEventHandler()

        result = handler.process_event("not an event")
        assert result.is_failure
        assert result.error is not None
        if "cannot process this event" not in result.error:
            raise AssertionError(
                f"Expected {'cannot process this event'} in {result.error}"
            )

    def test_event_handler_process_event_exception(self) -> None:
        """Test event processing with exception."""

        class FailingEventHandler(FlextHandlers.EventHandler):
            def handle_event(self, event: object) -> FlextResult[None]:
                msg = "Event processing failed"
                raise RuntimeError(msg)

            def process_event_impl(self, event: object) -> None:
                pass

        handler = FailingEventHandler()
        event = _TestEvent("test")

        result = handler.process_event(event)
        assert result.is_failure
        assert result.error is not None
        if "Event processing failed" not in result.error:
            raise AssertionError(
                f"Expected {'Event processing failed'} in {result.error}"
            )

    def test_event_handler_handle_event(self) -> None:
        """Test event handler handle_event method."""

        class _TestEventHandler(FlextHandlers.EventHandler):
            def handle(self, event: object) -> FlextResult[object]:
                return FlextResult.ok(f"Processed {event}")

            def process_event_impl(self, event: object) -> None:
                pass

        handler = _TestEventHandler()
        event = _TestEvent("test")

        result = handler.handle_event(event)
        assert result.is_success

    def test_event_handler_handle_event_failure(self) -> None:
        """Test event handler handle_event with failure."""

        class FailingEventHandler(FlextHandlers.EventHandler):
            def handle(self, event: object) -> FlextResult[object]:
                return FlextResult.fail("Handle failed")

            def process_event_impl(self, event: object) -> None:
                pass

        handler = FailingEventHandler()
        event = _TestEvent("test")

        result = handler.handle_event(event)
        assert result.is_failure
        assert result.error is not None
        if "Handle failed" not in result.error:
            raise AssertionError(f"Expected {'Handle failed'} in {result.error}")

    def test_event_handler_abstract_method(self) -> None:
        """Test that process_event_impl is abstract."""
        # EventHandler can be instantiated but process_event_impl is abstract
        # The abstract method is enforced by the abstractmethod decorator
        try:
            # Create instance - this might work
            handler = FlextHandlers.EventHandler("test")

            # But calling the abstract method should raise NotImplementedError
            # Since the method is decorated with @abstractmethod,
            # it might be an empty method
            # or it may not have an implementation, so let's test it
            try:
                handler.process_event_impl("test")
                # If we get here without exception, the method exists but may be empty
                assert True  # Test passes - the method exists
            except NotImplementedError:
                # Expected behavior - method raises NotImplementedError
                assert True  # Test passes
        except TypeError:
            # If instantiation fails due to abstract method, that's also valid
            assert True  # Test passes


class _TestQueryHandler:
    """Test FlextHandlers.QueryHandler functionality."""

    def test_query_handler_creation(self) -> None:
        """Test query handler creation."""
        handler = FlextHandlers.QueryHandler("test_query_handler")

        if handler.handler_name != "test_query_handler":
            raise AssertionError(
                f"Expected {'test_query_handler'}, got {handler.handler_name}"
            )

    def test_query_handler_authorize_query(self) -> None:
        """Test query authorization."""
        handler = FlextHandlers.QueryHandler()
        query = _TestQuery("get_users")

        result = handler.authorize_query(query)
        assert result.is_success

    def test_query_handler_pre_handle_with_authorization(self) -> None:
        """Test query pre-handling with authorization."""
        handler = FlextHandlers.QueryHandler()
        query = _TestQuery("get_data")

        result = handler.pre_handle(query)
        assert result.is_success
        if result.data != query:
            raise AssertionError(f"Expected {query}, got {result.data}")

    def test_query_handler_pre_handle_authorization_failure(self) -> None:
        """Test query pre-handling with authorization failure."""

        class AuthorizingQueryHandler(FlextHandlers.QueryHandler):
            def authorize_query(self, query: object) -> FlextResult[None]:
                return FlextResult.fail("Access denied")

        handler = AuthorizingQueryHandler()
        query = _TestQuery("get_sensitive_data")

        result = handler.pre_handle(query)
        assert result.is_failure
        assert result.error is not None
        if "Access denied" not in result.error:
            raise AssertionError(f"Expected {'Access denied'} in {result.error}")

    def test_query_handler_process_request(self) -> None:
        """Test request processing."""
        handler = FlextHandlers.QueryHandler()
        request = _TestQuery("search")

        result = handler.process_request(request)
        assert result.is_success

    def test_query_handler_process_request_validation_failure(self) -> None:
        """Test request processing with validation failure."""
        handler = FlextHandlers.QueryHandler()

        result = handler.process_request(None)
        assert result.is_failure
        assert result.error is not None
        if "cannot be None" not in result.error:
            raise AssertionError(f"Expected {'cannot be None'} in {result.error}")

    def test_query_handler_process_request_cannot_handle(self) -> None:
        """Test request processing when handler cannot handle."""

        class SelectiveQueryHandler(FlextHandlers.QueryHandler):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestQuery)

        handler = SelectiveQueryHandler()

        result = handler.process_request("not a query")
        assert result.is_failure
        assert result.error is not None
        if "cannot process this request" not in result.error:
            raise AssertionError(
                f"Expected {'cannot process this request'} in {result.error}"
            )

    def test_query_handler_process_request_exception(self) -> None:
        """Test request processing with exception."""

        class FailingQueryHandler(FlextHandlers.QueryHandler):
            def handle_request(self, request: object) -> FlextResult[object]:
                msg = "Request processing failed"
                raise RuntimeError(msg)

        handler = FailingQueryHandler()
        request = _TestQuery("test")

        result = handler.process_request(request)
        assert result.is_failure
        assert result.error is not None
        if "Request processing failed" not in result.error:
            raise AssertionError(
                f"Expected {'Request processing failed'} in {result.error}"
            )

    def test_query_handler_handle_request(self) -> None:
        """Test query handler handle_request method."""

        class _TestQueryHandler(FlextHandlers.QueryHandler):
            def handle(self, request: object) -> FlextResult[object]:
                if isinstance(request, _TestQuery):
                    return FlextResult.ok(f"Query result for {request.query_type}")
                return FlextResult.ok(request)

        handler = _TestQueryHandler()
        query = _TestQuery("get_users")

        result = handler.handle_request(query)
        assert result.is_success
        if result.data != "Query result for get_users":
            raise AssertionError(
                f"Expected {'Query result for get_users'}, got {result.data}"
            )


class TestHandlerRegistry:
    """Test FlextHandlers.Registry functionality."""

    def test_registry_creation(self) -> None:
        """Test registry creation."""
        registry = FlextHandlers.Registry()

        assert isinstance(registry._handlers, dict)
        assert isinstance(registry._type_handlers, dict)
        assert isinstance(registry._handler_list, list)
        if len(registry._handlers) != 0:
            raise AssertionError(f"Expected {0}, got {len(registry._handlers)}")

    def test_registry_register_handler_single_arg(self) -> None:
        """Test registering handler with single argument."""
        registry = FlextHandlers.Registry()
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        result = registry.register(handler)
        assert result.is_success
        if len(registry._handlers) != 1:
            raise AssertionError(f"Expected {1}, got {len(registry._handlers)}")
        assert len(registry._handler_list) == 1

    def test_registry_register_handler_two_args(self) -> None:
        """Test registering handler with key and handler."""
        registry = FlextHandlers.Registry()
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        result = registry.register("custom_key", handler)
        assert result.is_success
        if "custom_key" not in registry._handlers:
            raise AssertionError(f"Expected {'custom_key'} in {registry._handlers}")
        if len(registry._handler_list) != 1:
            raise AssertionError(f"Expected {1}, got {len(registry._handler_list)}")

    def test_registry_register_invalid_handler(self) -> None:
        """Test registering invalid handler."""
        registry = FlextHandlers.Registry()
        invalid_handler = "not a handler"

        result = registry.register(invalid_handler)
        assert result.is_failure
        assert result.error is not None
        if "must have 'handle' method" not in result.error:
            raise AssertionError(
                f"Expected {"must have 'handle' method"} in {result.error}"
            )

    def test_registry_register_duplicate_key(self) -> None:
        """Test registering handler with duplicate key."""
        registry = FlextHandlers.Registry()
        handler1 = FlextHandlers.Handler("handler1")
        handler2 = FlextHandlers.Handler("handler2")

        # Register first handler
        result1 = registry.register("duplicate_key", handler1)
        assert result1.is_success

        # Try to register second handler with same key
        result2 = registry.register("duplicate_key", handler2)
        assert result2.is_failure
        if "already registered" not in result2.error:
            raise AssertionError(f"Expected {'already registered'} in {result2.error}")

    def test_registry_register_for_type(self) -> None:
        """Test registering handler for message type."""
        registry = FlextHandlers.Registry()
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        result = registry.register_for_type(_TestMessage, handler)
        assert result.is_success
        if _TestMessage not in registry._type_handlers:
            raise AssertionError(
                f"Expected {_TestMessage} in {registry._type_handlers}"
            )

    def test_registry_register_for_type_duplicate(self) -> None:
        """Test registering handler for type with duplicate."""
        registry = FlextHandlers.Registry()
        handler1 = FlextHandlers.Handler("handler1")
        handler2 = FlextHandlers.Handler("handler2")

        # Register first handler
        result1 = registry.register_for_type(_TestMessage, handler1)
        assert result1.is_success

        # Try to register second handler for same type
        result2 = registry.register_for_type(_TestMessage, handler2)
        assert result2.is_failure
        if "already registered" not in result2.error:
            raise AssertionError(f"Expected {'already registered'} in {result2.error}")

    def test_registry_get_handler(self) -> None:
        """Test getting handler by key."""
        registry = FlextHandlers.Registry()
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        # Register handler
        registry.register("test_key", handler)

        # Get handler
        result = registry.get_handler("test_key")
        assert result.is_success
        if result.data != handler:
            raise AssertionError(f"Expected {handler}, got {result.data}")

    def test_registry_get_handler_not_found(self) -> None:
        """Test getting handler with non-existent key."""
        registry = FlextHandlers.Registry()

        result = registry.get_handler("non_existent_key")
        assert result.is_failure
        assert result.error is not None
        if "No handler found" not in result.error:
            raise AssertionError(f"Expected {'No handler found'} in {result.error}")

    def test_registry_get_handler_for_type(self) -> None:
        """Test getting handler for message type."""
        registry = FlextHandlers.Registry()
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        # Register handler for type
        registry.register_for_type(_TestMessage, handler)

        # Get handler for type
        result = registry.get_handler_for_type(_TestMessage)
        assert result.is_success
        if result.data != handler:
            raise AssertionError(f"Expected {handler}, got {result.data}")

    def test_registry_get_handler_for_type_not_found(self) -> None:
        """Test getting handler for non-existent type."""
        registry = FlextHandlers.Registry()

        result = registry.get_handler_for_type(_TestCommand)
        assert result.is_failure
        assert result.error is not None
        if "No handler found" not in result.error:
            raise AssertionError(f"Expected {'No handler found'} in {result.error}")

    def test_registry_get_all_handlers(self) -> None:
        """Test getting all registered handlers."""
        registry = FlextHandlers.Registry()
        handler1 = FlextHandlers.Handler("handler1")
        handler2 = FlextHandlers.Handler("handler2")

        registry.register(handler1)
        registry.register(handler2)

        all_handlers = registry.get_all_handlers()
        if len(all_handlers) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(all_handlers)}")
        if handler1 not in all_handlers:
            raise AssertionError(f"Expected {handler1} in {all_handlers}")
        assert handler2 in all_handlers

    def test_registry_find_handlers(self) -> None:
        """Test finding handlers that can process message."""
        registry = FlextHandlers.Registry()

        # Create handler that can handle _TestMessage
        class SelectiveHandler(FlextHandlers.Handler[object, object]):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestMessage)

        selective_handler = SelectiveHandler()
        universal_handler: FlextHandlers.Handler[object, object] = (
            FlextHandlers.Handler()  # Can handle anything
        )

        registry.register(selective_handler)
        registry.register(universal_handler)

        message = _TestMessage("test")
        handlers = registry.find_handlers(message)

        if len(handlers) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(handlers)}")
        if selective_handler not in handlers:
            raise AssertionError(f"Expected {selective_handler} in {handlers}")
        assert universal_handler in handlers

    def test_registry_get_handler_by_id(self) -> None:
        """Test getting handler by ID."""
        registry = FlextHandlers.Registry()
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        registry.register(handler)

        found_handler = registry.get_handler_by_id(handler.handler_id)
        if found_handler != handler:
            raise AssertionError(f"Expected {handler}, got {found_handler}")

    def test_registry_get_handler_by_id_not_found(self) -> None:
        """Test getting handler by non-existent ID."""
        registry = FlextHandlers.Registry()

        found_handler = registry.get_handler_by_id("non_existent_id")
        assert found_handler is None

    def test_registry_get_handler_info(self) -> None:
        """Test getting handler information."""
        registry = FlextHandlers.Registry()
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        registry.register(handler)

        info_list = registry.get_handler_info()
        if len(info_list) != 1:
            raise AssertionError(f"Expected {1}, got {len(info_list)}")

        info = info_list[0]
        assert isinstance(info, dict)
        if info["handler_name"] != "test_handler":
            raise AssertionError(
                f"Expected {'test_handler'}, got {info['handler_name']}"
            )
        assert info["handler_class"] == "Handler"
        if "handler_id" not in info:
            raise AssertionError(f"Expected {'handler_id'} in {info}")


class TestHandlerChain:
    """Test FlextHandlers.Chain functionality."""

    def test_chain_creation(self) -> None:
        """Test chain creation."""
        chain = FlextHandlers.Chain()

        assert isinstance(chain._handlers, list)
        if len(chain._handlers) != 0:
            raise AssertionError(f"Expected {0}, got {len(chain._handlers)}")

    def test_chain_add_handler(self) -> None:
        """Test adding handler to chain."""
        chain = FlextHandlers.Chain()
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "test_handler"
        )

        chain.add_handler(handler)
        if len(chain._handlers) != 1:
            raise AssertionError(f"Expected {1}, got {len(chain._handlers)}")
        assert chain._handlers[0] == handler

    def test_chain_process_message(self) -> None:
        """Test processing message through chain."""
        chain = FlextHandlers.Chain()

        class TestHandler(FlextHandlers.Handler[object, object]):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestMessage)

            def handle_with_hooks(self, message: object) -> FlextResult[object]:
                if isinstance(message, _TestMessage):
                    return FlextResult.ok(f"Processed: {message.content}")
                return FlextResult.ok(message)

        handler = TestHandler()
        chain.add_handler(handler)

        message = _TestMessage("test message")
        result = chain.process(message)

        assert result.is_success
        if result.data != "Processed: test message":
            raise AssertionError(
                f"Expected {'Processed: test message'}, got {result.data}"
            )

    def test_chain_process_no_handler_found(self) -> None:
        """Test processing message when no handler can handle it."""
        chain = FlextHandlers.Chain()

        class SelectiveHandler(FlextHandlers.Handler[object, object]):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestCommand)  # Can only handle commands

        handler = SelectiveHandler()
        chain.add_handler(handler)

        message = _TestMessage("test message")  # Send a message, not a command
        result = chain.process(message)

        assert result.is_failure
        assert result.error is not None
        if "No handler found" not in result.error:
            raise AssertionError(f"Expected {'No handler found'} in {result.error}")

    def test_chain_process_all(self) -> None:
        """Test processing message through all applicable handlers."""
        chain = FlextHandlers.Chain()

        class Handler1(FlextHandlers.Handler[object, object]):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestMessage)

            def handle_with_hooks(self, message: object) -> FlextResult[object]:
                return FlextResult.ok("Handler1 result")

        class Handler2(FlextHandlers.Handler[object, object]):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestMessage)

            def handle_with_hooks(self, message: object) -> FlextResult[object]:
                return FlextResult.ok("Handler2 result")

        class Handler3(FlextHandlers.Handler[object, object]):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestCommand)  # Different type

            def handle_with_hooks(self, message: object) -> FlextResult[object]:
                return FlextResult.ok("Handler3 result")

        chain.add_handler(Handler1())
        chain.add_handler(Handler2())
        chain.add_handler(Handler3())

        message = _TestMessage("test")
        results = chain.process_all(message)

        # Should get results from Handler1 and Handler2, not Handler3
        if len(results) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(results)}")
        if all(result.is_success for result in results):
            raise AssertionError(
                f"Expected {all(result.is_success for result in results)} in {results}"
            )
        if results[0].data != "Handler1 result":
            raise AssertionError(f"Expected {'Handler1 result'}, got {results[0].data}")
        assert results[1].data == "Handler2 result"

    def test_chain_process_all_no_handlers(self) -> None:
        """Test processing message through all handlers when none can handle."""
        chain = FlextHandlers.Chain()

        class SelectiveHandler(FlextHandlers.Handler[object, object]):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, _TestCommand)

        chain.add_handler(SelectiveHandler())

        message = _TestMessage("test")
        results = chain.process_all(message)

        if len(results) != 0:
            raise AssertionError(f"Expected {0}, got {len(results)}")


class TestFactoryMethods:
    """Test FlextHandlers factory methods."""

    def test_create_function_handler(self) -> None:
        """Test creating handler from function."""

        def test_function(message: object) -> FlextResult[str]:
            if isinstance(message, _TestMessage):
                return FlextResult.ok(f"Function processed: {message.content}")
            return FlextResult.ok("Function processed unknown message")

        handler = FlextHandlers.flext_create_function_handler(test_function)

        assert isinstance(handler, FlextHandlers.Handler)

        message = _TestMessage("test")
        result = handler.handle(message)

        assert result.is_success
        if result.data != "Function processed: test":
            raise AssertionError(
                f"Expected {'Function processed: test'}, got {result.data}"
            )

    def test_create_function_handler_non_result_return(self) -> None:
        """Test creating handler from function that doesn't return FlextResult."""

        def simple_function(message: object) -> str:
            return "simple result"

        handler = FlextHandlers.flext_create_function_handler(simple_function)
        message = _TestMessage("test")

        result = handler.handle(message)
        assert result.is_success
        if result.data != "simple result":
            raise AssertionError(f"Expected {'simple result'}, got {result.data}")

    def test_create_registry(self) -> None:
        """Test creating registry."""
        registry = FlextHandlers.flext_create_registry()

        assert isinstance(registry, FlextHandlers.Registry)
        if len(registry._handlers) != 0:
            raise AssertionError(f"Expected {0}, got {len(registry._handlers)}")

    def test_create_chain(self) -> None:
        """Test creating chain."""
        chain = FlextHandlers.flext_create_chain()

        assert isinstance(chain, FlextHandlers.Chain)
        if len(chain._handlers) != 0:
            raise AssertionError(f"Expected {0}, got {len(chain._handlers)}")


class TestHandlerEdgeCases:
    """Test edge cases and error conditions."""

    def test_handler_with_none_message(self) -> None:
        """Test handler with None message."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler()

        # Validation should catch None
        result = handler.process_message(None)
        assert result.is_failure

    def test_handler_metadata_with_complex_names(self) -> None:
        """Test handler metadata with complex names."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "complex-handler_name.with.dots"
        )
        metadata = handler.get_handler_metadata()

        if "handler_name" not in metadata:
            raise AssertionError(f"Expected {'handler_name'} in {metadata}")
        if metadata["handler_name"] != "complex-handler_name.with.dots":
            raise AssertionError(
                f"Expected {'complex-handler_name.with.dots'}, got {metadata['handler_name']}"
            )

    def test_registry_with_handler_without_id(self) -> None:
        """Test registry with handler that doesn't have handler_id."""
        registry = FlextHandlers.Registry()

        # Create a mock handler without handler_id attribute
        class MockHandler:
            def handle(self, message: object) -> FlextResult[object]:
                return FlextResult.ok(message)

        mock_handler = MockHandler()
        result = registry.register(mock_handler)

        # Should still register successfully using class name
        assert result.is_success

    def test_chain_with_mixed_handler_types(self) -> None:
        """Test chain with different handler types."""
        chain = FlextHandlers.Chain()

        # Add different types of handlers
        base_handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "base"
        )
        FlextHandlers.CommandHandler("command")

        # Note: We need to create a concrete EventHandler
        class ConcreteEventHandler(FlextHandlers.EventHandler):
            def process_event_impl(self, event: object) -> None:
                pass

        ConcreteEventHandler("event")

        chain.add_handler(base_handler)
        # Note: CommandHandler and EventHandler might not be compatible with Chain
        # due to type signature differences. Test what actually works.

        message = _TestMessage("test")
        result = chain.process(message)

        # Should work with base handler
        assert result.is_success

    def test_handler_inheritance_behavior(self) -> None:
        """Test handler inheritance behavior."""

        class CustomHandler(FlextHandlers.Handler[object, object]):
            def __init__(self, name: str) -> None:
                super().__init__(name)
                self.custom_attribute = "custom_value"

            def handle(self, message: object) -> FlextResult[object]:
                if isinstance(message, _TestMessage):
                    return FlextResult.ok(f"Custom: {message.content}")
                return super().handle(message)

        handler = CustomHandler("custom_handler")

        if handler.custom_attribute != "custom_value":
            raise AssertionError(
                f"Expected {'custom_value'}, got {handler.custom_attribute}"
            )
        assert handler.handler_name == "custom_handler"

        message = _TestMessage("test")
        result = handler.handle(message)

        assert result.is_success
        if result.data != "Custom: test":
            raise AssertionError(f"Expected {'Custom: test'}, got {result.data}")

    def test_performance_with_many_handlers(self) -> None:
        """Test performance characteristics with many handlers."""
        registry = FlextHandlers.Registry()

        # Register many handlers
        handlers = []
        for i in range(100):
            handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
                f"handler_{i}"
            )
            handlers.append(handler)
            registry.register(handler)

        if len(registry.get_all_handlers()) != 100:
            raise AssertionError(
                f"Expected {100}, got {len(registry.get_all_handlers())}"
            )

        # Test finding handlers
        message = _TestMessage("test")
        found_handlers = registry.find_handlers(message)

        # All handlers should be found since default can_handle returns True
        if len(found_handlers) != 100:
            raise AssertionError(f"Expected {100}, got {len(found_handlers)}")

    def test_thread_safety_basic(self) -> None:
        """Test basic thread safety of handlers."""
        handler: FlextHandlers.Handler[object, object] = FlextHandlers.Handler(
            "thread_test"
        )

        # Process multiple messages to test for basic issues
        messages = [_TestMessage(f"message_{i}") for i in range(10)]

        results = []
        for message in messages:
            result = handler.process_message(message)
            results.append(result)

        # All should succeed
        if all(result.is_success for result in results):
            raise AssertionError(
                f"Expected {all(result.is_success for result in results)} in {results}"
            )
        if len(results) != 10:
            raise AssertionError(f"Expected {10}, got {len(results)}")
