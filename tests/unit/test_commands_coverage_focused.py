"""Focused FlextCommands coverage tests targeting specific uncovered lines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from unittest.mock import Mock

from flext_core import FlextCommands, FlextResult
from flext_tests import FlextTestsMatchers


class TestCommandModel:
    """Test command model for coverage testing."""

    def __init__(self, command_type: str = "test_command") -> None:
        """Initialize test command model."""
        self.command_type = command_type

    def to_payload(self) -> dict:
        """Convert to payload dictionary."""
        return {"command_type": self.command_type}

    @classmethod
    def from_payload(cls, payload: dict) -> TestCommandModel:
        """Create from payload dictionary."""
        return cls(payload.get("command_type", "test_command"))


class TestQueryModel:
    """Test query model for coverage testing."""

    def __init__(self, query_type: str = "test_query") -> None:
        """Initialize test query model."""
        self.query_type = query_type

    def to_payload(self) -> dict:
        """Convert to payload dictionary."""
        return {"query_type": self.query_type}

    @classmethod
    def from_payload(cls, payload: dict) -> TestQueryModel:
        """Create from payload dictionary."""
        return cls(payload.get("query_type", "test_query"))


class TestCommandHandler(FlextCommands.Handlers.CommandHandler):
    """Test command handler for coverage testing."""

    def can_handle(self, command: object) -> bool:
        """Check if this handler can handle the given command."""
        # Handle both instances and types
        if isinstance(command, type):
            return issubclass(command, TestCommandModel)
        return isinstance(command, TestCommandModel)

    def handle(self, command: object) -> FlextResult[dict]:
        """Handle the given command and return result."""
        if isinstance(command, TestCommandModel):
            return FlextResult[dict].ok(
                {"handled": True, "command_type": command.command_type}
            )
        return FlextResult[dict].fail("Cannot handle command")


class TestQueryHandler(FlextCommands.Handlers.QueryHandler):
    """Test query handler for coverage testing."""

    def can_handle(self, query: object) -> bool:
        """Check if this handler can handle the given query."""
        # Handle both instances and types
        if isinstance(query, type):
            return issubclass(query, TestQueryModel)
        return isinstance(query, TestQueryModel)

    def handle_query(self, query: object) -> FlextResult[dict]:
        """Handle the given query and return result."""
        if isinstance(query, TestQueryModel):
            return FlextResult[dict].ok(
                {"handled": True, "query_type": query.query_type}
            )
        return FlextResult[dict].fail("Cannot handle query")

    def handle(self, query: object) -> FlextResult[dict]:
        """Handle the given query by delegating to handle_query."""
        # Delegate to handle_query
        return self.handle_query(query)


class TestFlextCommandsCoverageFocused:
    """Focused tests for FlextCommands coverage improvement."""

    def test_command_model_creation_and_payload(self) -> None:
        """Test Command model creation and payload methods."""
        command_class = FlextCommands.Models.Command

        # Test basic command creation
        command = command_class(command_type="test", payload={"key": "value"})
        assert command.command_type == "test"
        assert command.payload == {"key": "value"}

        # Test to_payload method
        payload = command.to_payload()
        assert payload.message_type == "Command"
        assert "payload" in payload.data
        assert payload.data["payload"] == {"key": "value"}

        # Test from_payload method
        recreated_result = command_class.from_payload(payload)
        assert recreated_result.is_success
        recreated = recreated_result.value
        assert recreated.command_type == "test"
        assert recreated.payload == {"key": "value"}

    def test_query_model_creation_and_payload(self) -> None:
        """Test Query model creation and payload methods."""
        query_class = FlextCommands.Models.Query

        # Test basic query creation
        query = query_class(query_type="search", filters={"term": "test"})
        assert query.query_type == "search"
        assert query.filters == {"term": "test"}

        # Test to_payload method
        payload = query.to_payload()
        assert payload.data["query_type"] == "search"
        assert payload.data["filters"] == {"term": "test"}

        # Test from_payload method
        recreated_result = query_class.from_payload(payload)
        assert recreated_result.is_success
        recreated = recreated_result.value
        assert recreated.query_type == "search"
        assert recreated.filters == {"term": "test"}

    def test_command_handler_abstract_methods(self) -> None:
        """Test CommandHandler abstract methods implementation."""
        handler = TestCommandHandler()
        command = TestCommandModel("test_cmd")

        # Test can_handle method
        assert handler.can_handle(command) is True
        assert handler.can_handle("not_a_command") is False

        # Test handle method
        result = handler.handle(command)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value["handled"] is True

        # Test execute method
        execute_result = handler.execute(command)
        FlextTestsMatchers.assert_result_success(execute_result)

    def test_query_handler_abstract_methods(self) -> None:
        """Test QueryHandler abstract methods implementation."""
        handler = TestQueryHandler()
        query = TestQueryModel("test_query")

        # Test can_handle method
        assert handler.can_handle(query) is True
        assert handler.can_handle("not_a_query") is False

        # Test handle_query method
        result = handler.handle_query(query)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value["handled"] is True

        # Test handle method (QueryHandler doesn't have execute method)
        handle_result = handler.handle(query)
        FlextTestsMatchers.assert_result_success(handle_result)

    def test_command_bus_initialization(self) -> None:
        """Test CommandBus initialization and configuration."""
        bus = FlextCommands.Bus()

        # Test default initialization
        assert bus._handlers == {}
        assert bus._middleware == []
        assert bus._execution_count == 0
        assert hasattr(bus, "logger")

        # Test initialization with config
        config = {"max_handlers": 100, "enable_metrics": True}
        bus_with_config = FlextCommands.Bus(bus_config=config)
        assert bus_with_config._config == config

    def test_command_bus_handler_registration(self) -> None:
        """Test handler registration and management."""
        bus = FlextCommands.Bus()
        handler = TestCommandHandler()
        command = TestCommandModel("register_test")

        # Test handler registration
        register_result = bus.register_handler(TestCommandModel, handler)
        FlextTestsMatchers.assert_result_success(register_result)

        # Test finding registered handler
        found_handler = bus.find_handler(command)
        assert found_handler is handler

        # Test getting all handlers
        all_handlers = bus.get_all_handlers()
        assert handler in all_handlers  # Check that our handler instance is in the list

        # Test getting registered handlers
        registered = bus.get_registered_handlers()
        assert (
            "TestCommandModel" in registered
        )  # Check that the command type key is registered

    def test_command_bus_execution(self) -> None:
        """Test command execution through bus."""
        bus = FlextCommands.Bus()
        handler = TestCommandHandler()
        command = TestCommandModel("execute_test")

        # Register handler
        bus.register_handler(TestCommandModel, handler)

        # Test successful execution
        result = bus.execute(command)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value["handled"] is True

        # Test send_command alias
        send_result = bus.send_command(command)
        FlextTestsMatchers.assert_result_success(send_result)

    def test_command_bus_no_handler_found(self) -> None:
        """Test bus behavior when no handler is found."""
        bus = FlextCommands.Bus()
        unknown_command = TestCommandModel("unknown")

        # Test execution with no handler
        result = bus.execute(unknown_command)
        FlextTestsMatchers.assert_result_failure(result)

    def test_command_bus_middleware(self) -> None:
        """Test middleware functionality."""
        bus = FlextCommands.Bus()
        handler = TestCommandHandler()
        command = TestCommandModel("middleware_test")

        # Create mock middleware
        middleware = Mock()
        middleware.process = Mock(
            return_value=FlextResult[dict].ok({"processed": True})
        )

        # Register handler and middleware
        bus.register_handler(TestCommandModel, handler)
        add_result = bus.add_middleware(middleware)
        FlextTestsMatchers.assert_result_success(add_result)

        # Test execution with middleware
        result = bus.execute(command)
        FlextTestsMatchers.assert_result_success(result)

    def test_command_bus_unregister_handler(self) -> None:
        """Test handler unregistration."""
        bus = FlextCommands.Bus()
        handler = TestCommandHandler()

        # Register and then unregister
        bus.register_handler(TestCommandModel, handler)
        unregister_result = bus.unregister_handler("TestCommandModel")
        assert unregister_result is True

        # Verify handler is removed
        assert TestCommandModel not in bus.get_all_handlers()

    def test_command_decorators(self) -> None:
        """Test command decorators functionality."""
        decorators = FlextCommands.Decorators()

        @decorators.command_handler("test_command")
        def test_handler_func(command: object) -> dict[str, bool]:
            # Use the command parameter to avoid unused argument warning
            _ = command  # Acknowledge parameter usage
            return {"handled_by_decorator": True}

        # Test that decorator works
        assert callable(test_handler_func)
        result = test_handler_func(TestCommandModel("test"))
        assert result["handled_by_decorator"] is True

    def test_command_results_factory(self) -> None:
        """Test Results factory methods."""
        results = FlextCommands.Results()

        # Test success factory method
        success_result = results.success({"data": "success"})
        FlextTestsMatchers.assert_result_success(success_result)
        assert success_result.value["data"] == "success"

        # Test failure factory method
        failure_result = results.failure("Test error")
        FlextTestsMatchers.assert_result_failure(failure_result)

    def test_command_factories(self) -> None:
        """Test Factories class methods."""
        factories = FlextCommands.Factories()

        # Test create_command_bus
        bus = factories.create_command_bus()
        assert isinstance(bus, FlextCommands.Bus)

        # Test create_simple_handler
        simple_handler = factories.create_simple_handler(lambda _: {"simple": True})
        assert callable(simple_handler)

        # Test create_query_handler
        query_handler = factories.create_query_handler(lambda _: {"query": True})
        assert callable(query_handler)

    def test_command_handler_logger_property(self) -> None:
        """Test CommandHandler logger property."""
        handler = TestCommandHandler()
        assert hasattr(handler, "logger")
        assert handler.logger is not None

    def test_query_handler_logger_property(self) -> None:
        """Test QueryHandler logger property."""
        handler = TestQueryHandler()
        assert hasattr(handler, "logger")
        assert handler.logger is not None

    def test_command_handler_cannot_handle(self) -> None:
        """Test CommandHandler when it cannot handle command."""
        handler = TestCommandHandler()

        # Test with invalid command
        result = handler.execute("invalid_command")
        FlextTestsMatchers.assert_result_failure(result)

    def test_query_handler_cannot_handle(self) -> None:
        """Test QueryHandler when it cannot handle query."""
        handler = TestQueryHandler()

        # Test with invalid query
        result = handler.execute("invalid_query")
        FlextTestsMatchers.assert_result_failure(result)

    def test_command_bus_middleware_rejection(self) -> None:
        """Test middleware that rejects processing."""
        bus = FlextCommands.Bus()
        handler = TestCommandHandler()
        command = TestCommandModel("reject_test")

        # Create middleware that rejects
        rejecting_middleware = Mock()
        rejecting_middleware.process = Mock(
            return_value=FlextResult[dict].fail("Rejected by middleware")
        )

        # Register handler and middleware
        bus.register_handler(TestCommandModel, handler)
        bus.add_middleware(rejecting_middleware)

        # Test execution with rejecting middleware
        result = bus.execute(command)
        FlextTestsMatchers.assert_result_failure(result)

    def test_command_bus_execution_count(self) -> None:
        """Test that execution count is properly tracked."""
        bus = FlextCommands.Bus()
        handler = TestCommandHandler()
        command = TestCommandModel("count_test")

        # Register handler
        bus.register_handler(TestCommandModel, handler)

        initial_count = bus._execution_count
        bus.execute(command)
        assert bus._execution_count == initial_count + 1

    def test_command_bus_private_methods(self) -> None:
        """Test private methods of command bus."""
        bus = FlextCommands.Bus()
        handler = TestCommandHandler()
        command = TestCommandModel("private_test")

        # Test _execute_handler method
        execute_result = bus._execute_handler(handler, command)
        FlextTestsMatchers.assert_result_success(execute_result)

        # Test _apply_middleware method
        middleware_result = bus._apply_middleware(
            command, lambda _: FlextResult[dict].ok({"processed": True})
        )
        FlextTestsMatchers.assert_result_success(middleware_result)

    def test_command_model_validation_edge_cases(self) -> None:
        """Test Command model validation and edge cases."""
        command_class = FlextCommands.Models.Command

        # Test with minimal data
        minimal_command = command_class(command_type="minimal")
        assert minimal_command.command_type == "minimal"

        # Test payload with missing data
        payload = {"command_type": "partial"}
        recreated_result = command_class.from_payload(payload)
        FlextTestsMatchers.assert_result_success(recreated_result)
        recreated = recreated_result.unwrap()
        assert recreated.command_type == "partial"

    def test_query_model_validation_edge_cases(self) -> None:
        """Test Query model validation and edge cases."""
        query_class = FlextCommands.Models.Query

        # Test with minimal data
        minimal_query = query_class(query_type="minimal")
        assert minimal_query.query_type == "minimal"

        # Test payload with missing data
        payload = {"query_type": "partial"}
        recreated_result = query_class.from_payload(payload)
        FlextTestsMatchers.assert_result_success(recreated_result)
        recreated = recreated_result.unwrap()
        assert recreated.query_type == "partial"
