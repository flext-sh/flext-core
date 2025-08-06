"""Focused tests for handlers.py module targeting coverage gaps.

This test suite focuses on testing the handler functionality gaps identified
through coverage analysis to achieve high coverage of the handlers.py module.

Coverage Target: handlers.py 68% → 95%+

Key areas targeted:
- Error handling paths in can_handle method
- CommandBus registration and pipeline processing
- QueryBus caching and query processing
- EventBus publish/subscribe functionality
- Pipeline behaviors (Validation, Logging, Metrics)
- Factory method edge cases
"""

from __future__ import annotations

import time

# Removed explicit Any usage - using object instead
from unittest.mock import Mock, patch

import pytest

from flext_core.handlers import FlextHandlers
from flext_core.result import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


# =============================================================================
# TEST HANDLERS - Concrete implementations for testing coverage gaps
# =============================================================================


class TestCommand:
    """Test command class for command bus testing."""

    def __init__(self, data: str) -> None:
        self.data = data
        self.executed = False

    def execute(self) -> str:
        self.executed = True
        return f"executed: {self.data}"


class TestQuery:
    """Test query class for query bus testing."""

    def __init__(self, query_id: str) -> None:
        self.query_id = query_id


class TestEvent:
    """Test event class for event bus testing."""

    def __init__(self, event_type: str, data: dict[str, object]) -> None:
        self.event_type = event_type
        self.data = data


class ValidatingTestCommand:
    """Test command with validation method."""

    def __init__(self, data: str) -> None:
        self.data = data

    def validate(self) -> FlextResult[None]:
        if not self.data:
            return FlextResult.fail("Data cannot be empty")
        return FlextResult.ok(None)


class TestCommandHandler(FlextHandlers.CommandHandler):
    """Test command handler for testing."""

    def handle(self, command: object) -> FlextResult[object]:
        if isinstance(command, TestCommand):
            result = command.execute()
            return FlextResult.ok(result)
        return FlextResult.fail("Invalid command type")


class TestQueryHandler(FlextHandlers.QueryHandler[TestQuery, str]):
    """Test query handler for testing."""

    def handle(self, query: TestQuery) -> FlextResult[str]:
        return FlextResult.ok(f"result for {query.query_id}")


class TestEventHandler(FlextHandlers.EventHandler):
    """Test event handler for testing."""

    def __init__(self, handler_name: str | None = None) -> None:
        super().__init__(handler_name)
        self.processed_events: list[TestEvent] = []

    def handle(self, event: TestEvent) -> FlextResult[None]:
        self.processed_events.append(event)
        return FlextResult.ok(None)

    def process_event_impl(self, event: TestEvent) -> None:
        """Process event with side effects."""
        self.processed_events.append(event)


# =============================================================================
# COVERAGE GAP TESTS - Handler Error Paths and Edge Cases
# =============================================================================


class TestHandlerErrorPaths:
    """Test error paths and edge cases in handlers."""

    def test_handler_can_handle_type_analysis_failure(self) -> None:
        """Test can_handle method when type analysis fails."""
        handler = FlextHandlers.Handler[str, str]()

        # Mock get_type_hints to raise TypeError
        with patch(
            "flext_core.handlers.get_type_hints", side_effect=TypeError("Type error")
        ):
            result = handler.can_handle("test message")
            assert result is False

    def test_handler_can_handle_import_error(self) -> None:
        """Test can_handle method when import fails."""
        handler = FlextHandlers.Handler[str, str]()

        # Mock get_type_hints to raise ImportError
        with patch(
            "flext_core.handlers.get_type_hints",
            side_effect=ImportError("Import error"),
        ):
            result = handler.can_handle("test message")
            assert result is False

    def test_handler_can_handle_attribute_error(self) -> None:
        """Test can_handle method when attribute access fails."""
        handler = FlextHandlers.Handler[str, str]()

        # Mock get_type_hints to raise AttributeError
        with patch(
            "flext_core.handlers.get_type_hints",
            side_effect=AttributeError("Attribute error"),
        ):
            result = handler.can_handle("test message")
            assert result is False

    def test_handler_can_handle_no_type_hints(self) -> None:
        """Test can_handle method when handler has no type hints."""

        class HandlerNoHints(FlextHandlers.Handler[str, str]):
            def handle(self, message):  # No type hints deliberately
                return FlextResult.ok(message)

        handler = HandlerNoHints()
        result = handler.can_handle("test message")
        assert result is False

    def test_command_handler_validate_with_injected_validator_failure(self) -> None:
        """Test command handler validation with injected validator that fails."""
        mock_validator = Mock()
        mock_validator.validate_message.return_value = FlextResult.fail(
            "Validation failed"
        )

        handler = FlextHandlers.CommandHandler(validator=mock_validator)
        result = handler.validate_command("test command")

        assert result.is_failure
        assert "Validation failed" in result.error

    def test_command_handler_can_handle_complex_scenarios(self) -> None:
        """Test command handler can_handle with various message types."""
        handler = FlextHandlers.CommandHandler()

        # Test with command-like object
        test_cmd = TestCommand("test")
        assert handler.can_handle(test_cmd) is True

        # Test with callable object
        assert handler.can_handle(lambda: "test") is True

        # Test with object having execute method
        class ExecutableObject:
            def execute(self) -> str:
                return "executed"

        executable = ExecutableObject()
        assert handler.can_handle(executable) is True

        # Test with dict (structured message)
        assert handler.can_handle({"command": "test"}) is True

        # Test with simple string (no __dict__)
        assert handler.can_handle("simple string") is False


# =============================================================================
# COMMAND BUS COVERAGE TESTS - Registration and Pipeline Processing
# =============================================================================


class TestCommandBusCoverage:
    """Test CommandBus functionality for coverage gaps."""

    def test_command_bus_register_handler_duplicate(self) -> None:
        """Test registering duplicate command handler."""
        bus = FlextHandlers.CommandBus()
        handler = TestCommandHandler()

        # Register first time - should succeed
        result1 = bus.register_handler(TestCommand, handler)
        assert result1.success

        # Register same type again - should fail
        result2 = bus.register_handler(TestCommand, handler)
        assert result2.is_failure
        assert "Handler already registered" in result2.error

    def test_command_bus_add_behavior(self) -> None:
        """Test adding pipeline behavior to command bus."""
        bus = FlextHandlers.CommandBus()
        behavior = FlextHandlers.ValidationBehavior()

        result = bus.add_behavior(behavior)
        assert result.success

    def test_command_bus_send_no_handler(self) -> None:
        """Test sending command with no registered handler."""
        bus = FlextHandlers.CommandBus()
        command = TestCommand("test")

        result = bus.send(command)
        assert result.is_failure
        assert "No handler registered for command type" in result.error

    def test_command_bus_send_with_behaviors(self) -> None:
        """Test sending command through behavior pipeline."""
        bus = FlextHandlers.CommandBus()
        handler = TestCommandHandler()
        validation_behavior = FlextHandlers.ValidationBehavior()

        # Register handler and behavior
        bus.register_handler(TestCommand, handler)
        bus.add_behavior(validation_behavior)

        command = TestCommand("test data")
        result = bus.send(command)

        assert result.success
        assert "executed: test data" in str(result.data)

    def test_command_bus_pipeline_processing(self) -> None:
        """Test complex pipeline processing with multiple behaviors."""
        bus = FlextHandlers.CommandBus()
        handler = TestCommandHandler()

        # Add multiple behaviors
        bus.add_behavior(FlextHandlers.ValidationBehavior())
        bus.add_behavior(FlextHandlers.LoggingBehavior())
        bus.add_behavior(FlextHandlers.MetricsBehavior())

        bus.register_handler(TestCommand, handler)

        command = TestCommand("pipeline test")
        result = bus.send(command)

        assert result.success

    def test_command_bus_metrics(self) -> None:
        """Test command bus metrics collection."""
        bus = FlextHandlers.CommandBus()
        handler = TestCommandHandler()
        bus.register_handler(TestCommand, handler)

        # Send successful command
        command1 = TestCommand("success")
        result1 = bus.send(command1)
        assert result1.success

        # Send command that will fail (no handler)
        command2 = object()  # Invalid command type
        result2 = bus.send(command2)
        assert result2.is_failure

        metrics = bus.get_metrics()
        assert metrics["commands_processed"] == 2
        assert metrics["successful_commands"] == 1
        assert metrics["failed_commands"] == 1
        assert metrics["success_rate"] == 0.5

    def test_command_bus_audit_trail(self) -> None:
        """Test command bus audit trail functionality."""
        bus = FlextHandlers.CommandBus()
        handler = TestCommandHandler()
        bus.register_handler(TestCommand, handler)

        command = TestCommand("audit test")
        bus.send(command)

        audit_trail = bus.get_audit_trail()
        assert len(audit_trail) == 1
        assert audit_trail[0]["command_type"] == "TestCommand"
        assert audit_trail[0]["command_id"] == id(command)
        assert "timestamp" in audit_trail[0]


# =============================================================================
# QUERY BUS COVERAGE TESTS - Caching and Query Processing
# =============================================================================


class TestQueryBusCoverage:
    """Test QueryBus functionality for coverage gaps."""

    def test_query_bus_register_handler_duplicate(self) -> None:
        """Test registering duplicate query handler."""
        bus = FlextHandlers.QueryBus()
        handler = TestQueryHandler()

        # Register first time - should succeed
        result1 = bus.register_handler(TestQuery, handler)
        assert result1.success

        # Register same type again - should fail
        result2 = bus.register_handler(TestQuery, handler)
        assert result2.is_failure
        assert "Handler already registered" in result2.error

    def test_query_bus_enable_caching(self) -> None:
        """Test enabling query result caching."""
        bus = FlextHandlers.QueryBus()

        result = bus.enable_caching(cache_ttl_seconds=600)
        assert result.success

    def test_query_bus_set_authorizer(self) -> None:
        """Test setting query authorizer."""
        bus = FlextHandlers.QueryBus()
        mock_authorizer = Mock()
        mock_authorizer.authorize_query.return_value = FlextResult.ok(None)

        result = bus.set_authorizer(mock_authorizer)
        assert result.success

    def test_query_bus_send_with_authorization_failure(self) -> None:
        """Test sending query with authorization failure."""
        bus = FlextHandlers.QueryBus()
        handler = TestQueryHandler()

        mock_authorizer = Mock()
        mock_authorizer.authorize_query.return_value = FlextResult.fail("Unauthorized")

        bus.register_handler(TestQuery, handler)
        bus.set_authorizer(mock_authorizer)

        query = TestQuery("test_id")
        result = bus.send(query)

        assert result.is_failure
        assert "Unauthorized" in result.error

    def test_query_bus_caching_functionality(self) -> None:
        """Test query bus caching with cache hit and miss."""
        bus = FlextHandlers.QueryBus()
        handler = TestQueryHandler()

        bus.register_handler(TestQuery, handler)
        bus.enable_caching(cache_ttl_seconds=300)

        query = TestQuery("cached_query")

        # First call - cache miss
        result1 = bus.send(query)
        assert result1.success

        # Second call - cache hit
        result2 = bus.send(query)
        assert result2.success
        assert result1.data == result2.data

        metrics = bus.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1

    def test_query_bus_cache_expiration(self) -> None:
        """Test query cache expiration functionality."""
        bus = FlextHandlers.QueryBus()
        handler = TestQueryHandler()

        bus.register_handler(TestQuery, handler)
        bus.enable_caching(cache_ttl_seconds=1)  # 1 second TTL

        query = TestQuery("expiring_query")

        # First call
        result1 = bus.send(query)
        assert result1.success

        # Wait for cache to expire
        time.sleep(1.1)

        # Second call - should be cache miss due to expiration
        result2 = bus.send(query)
        assert result2.success

        metrics = bus.get_metrics()
        assert metrics["cache_misses"] == 2  # Both calls were misses

    def test_query_bus_clear_cache(self) -> None:
        """Test clearing query cache."""
        bus = FlextHandlers.QueryBus()
        handler = TestQueryHandler()

        bus.register_handler(TestQuery, handler)
        bus.enable_caching()

        # Add item to cache
        query = TestQuery("cached_item")
        bus.send(query)

        # Clear cache
        result = bus.clear_cache()
        assert result.success

        # Verify cache is empty by checking metrics
        metrics = bus.get_metrics()
        assert metrics["cached_items"] == 0

    def test_query_bus_no_handler_found(self) -> None:
        """Test query bus with no handler registered."""
        bus = FlextHandlers.QueryBus()
        query = TestQuery("no_handler")

        result = bus.send(query)
        assert result.is_failure
        assert "No handler registered" in result.error

    def test_query_bus_handler_without_hooks(self) -> None:
        """Test query bus with handler that doesn't support handle_with_hooks."""
        bus = FlextHandlers.QueryBus()

        # Mock handler without handle_with_hooks
        mock_handler = Mock()
        del mock_handler.handle_with_hooks  # Remove the method

        bus.register_handler(TestQuery, mock_handler)
        query = TestQuery("no_hooks")

        result = bus.send(query)
        assert result.is_failure
        assert "does not support handle_with_hooks" in result.error

    def test_query_bus_comprehensive_metrics(self) -> None:
        """Test query bus comprehensive metrics collection."""
        bus = FlextHandlers.QueryBus()
        handler = TestQueryHandler()

        bus.register_handler(TestQuery, handler)
        bus.enable_caching()

        # Send successful queries
        for i in range(3):
            query = TestQuery(f"query_{i}")
            result = bus.send(query)
            assert result.success

        # Send query that will fail (no handler)
        bad_query = object()
        result = bus.send(bad_query)
        assert result.is_failure

        metrics = bus.get_metrics()
        assert metrics["queries_processed"] == 4
        assert metrics["successful_queries"] == 3
        assert metrics["failed_queries"] == 1
        assert metrics["success_rate"] == 0.75
        assert metrics["cache_misses"] == 4  # All were misses since different queries


# =============================================================================
# EVENT BUS COVERAGE TESTS - Publish/Subscribe Functionality
# =============================================================================


class TestEventBusCoverage:
    """Test EventBus functionality for coverage gaps."""

    def test_event_bus_subscribe(self) -> None:
        """Test event bus subscription functionality."""
        bus = FlextHandlers.EventBus()
        handler = TestEventHandler()

        result = bus.subscribe(TestEvent, handler)
        assert result.success

        # Subscribe same handler again - should not duplicate
        result2 = bus.subscribe(TestEvent, handler)
        assert result2.success

    def test_event_bus_publish_no_subscribers(self) -> None:
        """Test publishing event with no subscribers."""
        bus = FlextHandlers.EventBus()
        event = TestEvent("test_type", {"data": "test"})

        result = bus.publish(event)
        assert result.success

    def test_event_bus_publish_with_handlers(self) -> None:
        """Test publishing event to subscribed handlers."""
        bus = FlextHandlers.EventBus()
        handler1 = TestEventHandler()
        handler2 = TestEventHandler()

        bus.subscribe(TestEvent, handler1)
        bus.subscribe(TestEvent, handler2)

        event = TestEvent("test_type", {"data": "test"})
        result = bus.publish(event)

        assert result.success
        assert len(handler1.processed_events) == 1
        assert len(handler2.processed_events) == 1

    def test_event_bus_handler_failure(self) -> None:
        """Test event bus with handler that raises exception."""
        bus = FlextHandlers.EventBus()

        # Mock handler that raises exception
        mock_handler = Mock()
        mock_handler.handle.side_effect = Exception("Handler error")

        bus.subscribe(TestEvent, mock_handler)

        event = TestEvent("test_type", {"data": "test"})
        result = bus.publish(event)

        # Should still succeed even if handler fails
        assert result.success

        metrics = bus.get_metrics()
        assert metrics["events_published"] == 1
        # events_processed only increments in the try block, not on exception
        assert metrics["events_processed"] == 0
        assert metrics["failed_events"] == 1

    def test_event_bus_handler_success_metrics(self) -> None:
        """Test event bus metrics with successful handler."""
        bus = FlextHandlers.EventBus()
        handler = TestEventHandler()

        bus.subscribe(TestEvent, handler)

        event = TestEvent("success_type", {"data": "success"})
        result = bus.publish(event)

        assert result.success

        metrics = bus.get_metrics()
        assert metrics["events_published"] == 1
        assert metrics["events_processed"] == 1
        assert metrics["successful_events"] == 1

    def test_event_bus_metrics_collection(self) -> None:
        """Test comprehensive event bus metrics collection."""
        bus = FlextHandlers.EventBus()

        # Add successful handler
        success_handler = TestEventHandler()
        bus.subscribe(TestEvent, success_handler)

        # Add failing handler
        fail_handler = Mock()
        fail_handler.handle.side_effect = Exception("Fail")
        bus.subscribe(TestEvent, fail_handler)

        # Publish multiple events
        for i in range(3):
            event = TestEvent(f"type_{i}", {"data": f"test_{i}"})
            bus.publish(event)

        metrics = bus.get_metrics()
        assert metrics["events_published"] == 3
        # Only successful handlers increment events_processed (in try block)
        assert metrics["events_processed"] == 3  # 3 events x 1 successful handler
        assert metrics["successful_events"] == 3  # Only success_handler succeeds
        assert metrics["failed_events"] == 3  # fail_handler always fails


# =============================================================================
# PIPELINE BEHAVIOR COVERAGE TESTS - Validation, Logging, Metrics
# =============================================================================


class TestPipelineBehaviorsCoverage:
    """Test pipeline behaviors for coverage gaps."""

    def test_validation_behavior_strict_mode(self) -> None:
        """Test validation behavior with strict validation."""
        behavior = FlextHandlers.ValidationBehavior(strict_validation=True)

        def next_handler() -> FlextResult[object]:
            return FlextResult.ok("processed")

        # Test with None message
        result = behavior.process(None, next_handler)
        assert result.is_failure
        assert "Message cannot be None" in result.error

    def test_validation_behavior_with_validating_message(self) -> None:
        """Test validation behavior with message that has validate method."""
        behavior = FlextHandlers.ValidationBehavior()

        def next_handler() -> FlextResult[object]:
            return FlextResult.ok("processed")

        # Test with validating command that fails
        invalid_command = ValidatingTestCommand("")  # Empty data should fail
        result = behavior.process(invalid_command, next_handler)
        assert result.is_failure
        assert "Data cannot be empty" in result.error

        # Test with validating command that succeeds
        valid_command = ValidatingTestCommand("valid data")
        result = behavior.process(valid_command, next_handler)
        assert result.success

    def test_validation_behavior_with_validate_returning_non_result(self) -> None:
        """Test validation behavior when validate returns non-FlextResult."""
        behavior = FlextHandlers.ValidationBehavior()

        def next_handler() -> FlextResult[object]:
            return FlextResult.ok("processed")

        # Mock message with validate that returns something without is_failure
        mock_message = Mock()
        mock_message.validate.return_value = True  # Not a FlextResult

        result = behavior.process(mock_message, next_handler)
        assert result.success  # Should proceed since validation didn't fail

    def test_logging_behavior_functionality(self) -> None:
        """Test logging behavior with different log levels."""
        behavior = FlextHandlers.LoggingBehavior(log_level="DEBUG")

        def next_handler_success() -> FlextResult[object]:
            return FlextResult.ok("success result")

        def next_handler_failure() -> FlextResult[object]:
            return FlextResult.fail("processing failed")

        message = TestCommand("log test")

        # Test successful processing with logging
        with (
            patch.object(behavior._logger, "info") as mock_info,
            patch.object(behavior._logger, "error") as mock_error,
        ):
            result = behavior.process(message, next_handler_success)
            assert result.success

            # Verify logging calls
            assert mock_info.call_count == 2  # Before and after
            mock_error.assert_not_called()

        # Test failed processing with logging
        with (
            patch.object(behavior._logger, "info") as mock_info,
            patch.object(behavior._logger, "error") as mock_error,
        ):
            result = behavior.process(message, next_handler_failure)
            assert result.is_failure

            # Verify logging calls
            mock_info.assert_called_once()  # Only before processing
            mock_error.assert_called_once()  # Error logging

    def test_metrics_behavior_comprehensive(self) -> None:
        """Test metrics behavior with comprehensive scenarios."""
        behavior = FlextHandlers.MetricsBehavior()

        def next_handler_success() -> FlextResult[object]:
            time.sleep(0.01)  # Simulate processing time
            return FlextResult.ok("success")

        def next_handler_failure() -> FlextResult[object]:
            time.sleep(0.01)  # Simulate processing time
            return FlextResult.fail("failure")

        message1 = TestCommand("metrics test 1")
        message2 = TestQuery("metrics test 2")
        message3 = TestEvent("metrics_type", {"test": "data"})

        # Process successful messages
        result1 = behavior.process(message1, next_handler_success)
        result2 = behavior.process(message2, next_handler_success)
        assert result1.success
        assert result2.success

        # Process failed message
        result3 = behavior.process(message3, next_handler_failure)
        assert result3.is_failure

        metrics = behavior.get_metrics()
        assert metrics["messages_processed"] == 3
        assert metrics["successful_messages"] == 2
        assert metrics["failed_messages"] == 1
        assert metrics["success_rate"] == 2 / 3
        assert metrics["average_processing_time_ms"] > 0

        # Check message type tracking
        message_types = metrics["message_types"]
        assert message_types["TestCommand"] == 1
        assert message_types["TestQuery"] == 1
        assert message_types["TestEvent"] == 1

    def test_metrics_behavior_zero_messages(self) -> None:
        """Test metrics behavior with no messages processed."""
        behavior = FlextHandlers.MetricsBehavior()
        metrics = behavior.get_metrics()

        assert metrics["messages_processed"] == 0
        assert metrics["successful_messages"] == 0
        assert metrics["failed_messages"] == 0
        assert metrics["success_rate"] == 0.0
        assert metrics["average_processing_time_ms"] == 0.0


# =============================================================================
# FACTORY METHOD COVERAGE TESTS - Edge Cases and Error Paths
# =============================================================================


class TestFactoryMethodsCoverage:
    """Test factory methods for coverage gaps."""

    def test_create_command_bus_factory(self) -> None:
        """Test command bus factory method."""
        bus = FlextHandlers.flext_create_command_bus()
        assert isinstance(bus, FlextHandlers.CommandBus)

    def test_create_query_bus_factory(self) -> None:
        """Test query bus factory method."""
        bus = FlextHandlers.flext_create_query_bus()
        assert isinstance(bus, FlextHandlers.QueryBus)

    def test_create_event_bus_factory(self) -> None:
        """Test event bus factory method."""
        bus = FlextHandlers.flext_create_event_bus()
        assert isinstance(bus, FlextHandlers.EventBus)

    def test_create_function_handler_factory(self) -> None:
        """Test function handler factory method."""

        def test_function(message: object) -> FlextResult[str]:
            return FlextResult.ok(f"processed: {message}")

        handler = FlextHandlers.flext_create_function_handler(test_function)
        result = handler.handle("test message")

        assert result.success
        assert "processed: test message" in str(result.data)

    def test_create_function_handler_non_result_return(self) -> None:
        """Test function handler factory with non-FlextResult return."""

        def test_function(message: object) -> str:
            return f"processed: {message}"

        handler = FlextHandlers.flext_create_function_handler(test_function)
        result = handler.handle("test message")

        assert result.success
        assert "processed: test message" in str(result.data)

    def test_create_registry_factory(self) -> None:
        """Test registry factory method."""
        registry = FlextHandlers.flext_create_registry()
        assert isinstance(registry, FlextHandlers.Registry)

    def test_create_chain_factory(self) -> None:
        """Test chain factory method."""
        chain = FlextHandlers.flext_create_chain()
        assert isinstance(chain, FlextHandlers.Chain)


# =============================================================================
# INTEGRATION TESTS - Complex Scenarios
# =============================================================================


class TestHandlerIntegration:
    """Test complex integration scenarios."""

    def test_full_cqrs_pipeline(self) -> None:
        """Test complete CQRS pipeline with commands and queries."""
        # Setup command bus
        command_bus = FlextHandlers.CommandBus()
        command_handler = TestCommandHandler()
        command_bus.register_handler(TestCommand, command_handler)
        command_bus.add_behavior(FlextHandlers.ValidationBehavior())
        command_bus.add_behavior(FlextHandlers.LoggingBehavior())

        # Setup query bus
        query_bus = FlextHandlers.QueryBus()
        query_handler = TestQueryHandler()
        query_bus.register_handler(TestQuery, query_handler)
        query_bus.enable_caching()

        # Setup event bus
        event_bus = FlextHandlers.EventBus()
        event_handler = TestEventHandler()
        event_bus.subscribe(TestEvent, event_handler)

        # Execute command
        command = TestCommand("integration test")
        command_result = command_bus.send(command)
        assert command_result.success

        # Execute query
        query = TestQuery("integration_query")
        query_result = query_bus.send(query)
        assert query_result.success

        # Publish event
        event = TestEvent("integration_event", {"command_id": id(command)})
        event_result = event_bus.publish(event)
        assert event_result.success

        # Verify all systems worked together
        assert len(event_handler.processed_events) == 1
        assert command_result.data
        assert query_result.data

    def test_error_propagation_through_pipeline(self) -> None:
        """Test error propagation through complex handler pipeline."""

        class FailingBehavior(FlextHandlers.PipelineBehavior):
            def process(self, message: object, next_handler) -> FlextResult[object]:
                return FlextResult.fail("Behavior failure")

        command_bus = FlextHandlers.CommandBus()
        command_handler = TestCommandHandler()

        command_bus.register_handler(TestCommand, command_handler)
        command_bus.add_behavior(FailingBehavior())

        command = TestCommand("error test")
        result = command_bus.send(command)

        assert result.is_failure
        assert "Behavior failure" in result.error

    def test_performance_metrics_integration(self) -> None:
        """Test performance metrics across all handler types."""
        # Create handlers with metrics behavior
        metrics_behavior = FlextHandlers.MetricsBehavior()

        command_bus = FlextHandlers.CommandBus()
        command_handler = TestCommandHandler()
        command_bus.register_handler(TestCommand, command_handler)
        command_bus.add_behavior(metrics_behavior)

        # Process multiple commands
        for i in range(5):
            command = TestCommand(f"perf_test_{i}")
            result = command_bus.send(command)
            assert result.success

        # Check metrics
        metrics = metrics_behavior.get_metrics()
        assert metrics["messages_processed"] == 5
        assert metrics["successful_messages"] == 5
        assert metrics["success_rate"] == 1.0
        # Processing time should be >= 0 (could be very small)
        assert metrics["average_processing_time_ms"] >= 0
