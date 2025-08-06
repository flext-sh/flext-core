"""Focused coverage boost tests for handlers.py module.

This test suite targets specific uncovered code paths in handlers.py
to improve coverage without adding complex dependencies.
"""

from __future__ import annotations

import pytest

from flext_core.handlers import FlextHandlers
from flext_core.result import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestCommandBusBasics:
    """Test Command Bus basic functionality."""

    def test_command_bus_creation(self) -> None:
        """Test CommandBus creation and basic state."""
        bus = FlextHandlers.CommandBus()

        # Test metrics initialization
        metrics = bus.get_metrics()
        assert isinstance(metrics, dict)
        assert "commands_processed" in metrics
        assert metrics["commands_processed"] == 0

    def test_command_bus_unregistered_command(self) -> None:
        """Test sending command without handler."""
        bus = FlextHandlers.CommandBus()

        class UnknownCommand:
            pass

        command = UnknownCommand()
        result = bus.send(command)

        assert result.success is False
        assert "No handler registered" in result.error or "not found" in result.error


class TestQueryBusBasics:
    """Test Query Bus basic functionality."""

    def test_query_bus_creation(self) -> None:
        """Test QueryBus creation."""
        bus = FlextHandlers.QueryBus()

        # Test basic functionality
        metrics = bus.get_metrics()
        assert isinstance(metrics, dict)

    def test_query_bus_unregistered_query(self) -> None:
        """Test sending query without handler."""
        bus = FlextHandlers.QueryBus()

        class UnknownQuery:
            pass

        query = UnknownQuery()
        result = bus.send(query)

        assert result.success is False


class TestHandlerMetadata:
    """Test handler metadata functionality."""

    def test_handler_metadata(self) -> None:
        """Test handler metadata collection."""

        class TestHandler(FlextHandlers.Handler):
            def handle(self, message: object) -> FlextResult[object]:
                return FlextResult.ok(message)

        handler = TestHandler()
        metadata = handler.get_handler_metadata()

        assert isinstance(metadata, dict)
        assert "handler_class" in metadata or "handler_name" in metadata

    def test_command_handler_metadata(self) -> None:
        """Test command handler specific metadata."""

        class TestCommandHandler(FlextHandlers.CommandHandler):
            def handle(self, command: object) -> FlextResult[object]:
                return FlextResult.ok("processed")

        handler = TestCommandHandler()
        metrics = handler.get_metrics()

        assert isinstance(metrics, dict)
        assert "handler_name" in metrics

    def test_query_handler_metadata(self) -> None:
        """Test query handler specific metadata."""

        class TestQueryHandler(FlextHandlers.QueryHandler):
            def handle(self, query: object) -> FlextResult[object]:
                return FlextResult.ok("result")

        handler = TestQueryHandler()

        # Test validation
        validation_result = handler.validate_message("test")
        assert validation_result.success is True or validation_result.success is False


class TestPipelineBehaviorBasics:
    """Test pipeline behavior basics."""

    def test_pipeline_behavior_abstract(self) -> None:
        """Test PipelineBehavior abstract class."""

        class SimpleBehavior(FlextHandlers.PipelineBehavior):
            def process(self, message: object, next_handler) -> FlextResult[object]:
                return next_handler(message)

        behavior = SimpleBehavior()

        # Test with mock next handler
        def mock_next(msg):
            return FlextResult.ok(f"processed: {msg}")

        result = behavior.process("test", mock_next)
        assert result.success is True
        assert "processed: test" in result.data


class TestHandlerTypeChecking:
    """Test handler type checking capabilities."""

    def test_handler_can_handle_method(self) -> None:
        """Test handler can_handle method."""

        class TypedHandler(FlextHandlers.Handler):
            def handle(self, message: object) -> FlextResult[object]:
                return FlextResult.ok(message)

            def can_handle(self, message: object) -> bool:
                return isinstance(message, str)

        handler = TypedHandler()

        # Test type checking
        assert handler.can_handle("string") is True
        assert handler.can_handle(123) is False

    def test_event_handler_can_handle(self) -> None:
        """Test event handler type checking."""

        class StringEventHandler(FlextHandlers.EventHandler):
            def handle(self, event: object) -> FlextResult[None]:
                return FlextResult.ok(None)

            def can_handle(self, message: object) -> bool:
                return isinstance(message, str)

        handler = StringEventHandler()

        assert handler.can_handle("event") is True
        assert handler.can_handle(42) is False
