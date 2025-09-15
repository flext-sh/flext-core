"""Tests for CQRS Commands functionality and compatibility.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from flext_core import (
    FlextCommands,
    FlextModels,
    FlextResult,
    FlextTypes,
    FlextUtilities,
)


class TestCommandsCompatibility:
    """Test CQRS Commands functionality and basic compatibility."""

    def test_command_model_creation(self) -> None:
        """Test basic command model creation and validation."""
        command = FlextCommands.Models.Command(
            command_type="test_command", user_id="user123"
        )

        assert command.command_id is not None
        assert command.command_type == "test_command"
        assert command.user_id == "user123"

        # Test validation
        validation_result = command.validate_command()
        assert validation_result.success

    def test_query_model_creation(self) -> None:
        """Test basic query model creation."""
        query = FlextCommands.Models.Query(query_type="test_query", user_id="user123")

        assert query.query_id is not None
        assert query.query_type == "test_query"
        assert query.user_id == "user123"

    def test_command_bus_creation(self) -> None:
        """Test command bus instantiation."""
        bus = FlextCommands.Bus()

        assert bus is not None
        assert hasattr(bus, "register_handler")
        assert hasattr(bus, "execute")

    def test_command_handler_creation(self) -> None:
        """Test command handler creation."""

        class TestCommand(FlextCommands.Models.Command):
            def __init__(
                self, command_type: str = "test", user_id: str | None = None
            ) -> None:
                super().__init__(command_type=command_type, user_id=user_id)

        class TestHandler(FlextCommands.Handlers.CommandHandler[TestCommand, str]):
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Handled: {command.command_type}")

        handler = TestHandler()
        assert handler is not None
        assert hasattr(handler, "handle")

    def test_command_bus_registration_and_execution(self) -> None:
        """Test command bus handler registration and command execution."""

        # Create test command
        class TestCommand(FlextCommands.Models.Command):
            def __init__(
                self, command_type: str = "test_execution", user_id: str | None = None
            ) -> None:
                super().__init__(command_type=command_type, user_id=user_id)

        # Create test handler
        class TestHandler(FlextCommands.Handlers.CommandHandler[TestCommand, str]):
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Executed: {command.command_type}")

        # Create bus and register handler
        bus = FlextCommands.Bus()
        bus.register_handler(TestHandler())

        # Execute command
        command = TestCommand(user_id="user123")
        result = bus.execute(command)

        assert result.success
        assert result.value == "Executed: test_execution"

    def test_command_decorator_functionality(self) -> None:
        """Test command decorator for function-based handlers."""

        class TestCommand(FlextCommands.Models.Command):
            def __init__(
                self, command_type: str = "test_decorator", user_id: str | None = None
            ) -> None:
                super().__init__(command_type=command_type, user_id=user_id)

        # Create decorated handler function
        @FlextCommands.Decorators.command_handler(TestCommand)
        def handle_test_command(command: TestCommand) -> str:
            return f"Decorated: {command.command_type}"

        # Test the decorated function
        command = TestCommand()
        result = handle_test_command(command)

        assert result == "Decorated: test_decorator"

    def test_query_handler_creation(self) -> None:
        """Test query handler creation."""

        class TestQuery(FlextCommands.Models.Query):
            def __init__(
                self, query_type: str = "test_query", user_id: str | None = None
            ) -> None:
                super().__init__(query_type=query_type, user_id=user_id)

        class TestQueryHandler(FlextCommands.Handlers.QueryHandler[TestQuery, str]):
            def handle(self, query: TestQuery) -> FlextResult[str]:
                return FlextResult[str].ok(
                    f"Query result for {query.query_id if hasattr(query, 'query_id') else 'unknown'}"
                )

        handler = TestQueryHandler()
        assert handler is not None
        assert hasattr(handler, "handle")

    def test_command_payload_conversion(self) -> None:
        """Test command to payload conversion."""
        command = FlextCommands.Models.Command(
            command_type="test_payload", user_id="user123"
        )

        # Test payload conversion
        payload_result = command.to_payload()
        assert isinstance(payload_result, FlextModels.Payload)

        # Add command_type to payload data for reconstruction
        payload_data = payload_result.data.copy()
        payload_data["command_type"] = command.command_type

        # Create new payload with command_type included
        test_payload = FlextModels.Payload[FlextTypes.Core.Dict](
            data=payload_data,
            message_type=command.__class__.__name__,
            source_service="command_service",
            correlation_id=command.correlation_id,
            message_id=FlextUtilities.Generators.generate_uuid(),
        )

        # Test command reconstruction from payload
        reconstructed_result = FlextCommands.Models.Command.from_payload(test_payload)
        assert reconstructed_result.success
        assert reconstructed_result.value.command_type == "test_payload"

    def test_bus_middleware_support(self) -> None:
        """Test command bus middleware functionality."""
        bus = FlextCommands.Bus()

        # Create simple middleware
        class TestMiddleware:
            def process(self, command: object, handler: object) -> FlextResult[None]:
                # Process middleware with validation
                if command is None or handler is None:
                    return FlextResult[None].fail("Command and handler cannot be None")
                return FlextResult[None].ok(None)

        # Add middleware to bus
        middleware_config = FlextModels.SystemConfigs.MiddlewareConfig(
            middleware_id="test_mw", middleware_type="test", enabled=True, order=0
        )

        bus.add_middleware(TestMiddleware(), middleware_config.model_dump())
        assert len(bus._middleware) == 1

    def test_factories_functionality(self) -> None:
        """Test factory methods for creating CQRS components."""
        # Test command bus factory
        bus = FlextCommands.Factories.create_command_bus()
        assert isinstance(bus, FlextCommands.Bus)

        # Test simple handler factory
        def test_handler_func(command: object) -> str:
            return f"Factory handler result for {type(command).__name__}"

        handler = FlextCommands.Factories.create_simple_handler(test_handler_func)
        assert isinstance(handler, FlextCommands.Handlers.CommandHandler)

        # Test query handler factory
        query_handler = FlextCommands.Factories.create_query_handler(test_handler_func)
        assert isinstance(query_handler, FlextCommands.Handlers.QueryHandler)

    def test_result_helpers(self) -> None:
        """Test result helper methods."""
        # Test success result creation
        success_result = FlextCommands.Results.success("test data")
        assert success_result.success
        assert success_result.value == "test data"

        # Test failure result creation
        failure_result = FlextCommands.Results.failure("test error", "TEST_ERROR")
        assert not failure_result.success
        assert failure_result.error == "test error"
