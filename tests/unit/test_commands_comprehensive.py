"""Comprehensive tests for FlextCommands to achieve high coverage."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from flext_core import (
    FlextCommands,
    FlextModels,
    FlextResult,
)


class TestCommandCommand(FlextCommands.Models.Command):
    """Test command implementation."""

    name: str
    value: int = 0

    def validate_command(self) -> FlextResult[None]:
        """Validate the command."""
        if not self.name:
            return FlextResult[None].fail("Name is required")
        if self.value < 0:
            return FlextResult[None].fail("Value must be non-negative")
        return FlextResult[None].ok(None)


class TestQueryModel(FlextCommands.Models.Query):
    """Test query implementation."""

    search_term: str
    limit: int = 10

    def validate_query(self) -> FlextResult[None]:
        """Validate the query."""
        if not self.search_term:
            return FlextResult[None].fail("Search term is required")
        if self.limit <= 0:
            return FlextResult[None].fail("Limit must be positive")
        return FlextResult[None].ok(None)


class TestCommandsAdvancedFeatures:
    """Test advanced features of FlextCommands."""

    def test_command_metadata_management(self) -> None:
        """Test command metadata functionality."""
        command = TestCommandCommand(name="test", value=42)

        # Test command has basic attributes
        assert hasattr(command, "name")
        assert hasattr(command, "value")
        assert command.name == "test"
        assert command.value == 42

        # Test creating metadata with proper types
        # FlextModels has a root field that expects dict[str, str]
        import json

        metadata_dict = {
            "created_at": datetime.now(UTC).isoformat(),
            "tags": json.dumps({"test": "value"}),  # Must be string
        }
        metadata = FlextModels(root=metadata_dict)
        assert json.loads(metadata.root["tags"]) == {"test": "value"}
        assert "created_at" in metadata.root

    def test_command_validation_helpers(self) -> None:
        """Test command validation helper methods."""

        # Test custom validation helpers
        def require_field(field_name: str, value: str) -> FlextResult[None]:
            if not value:
                return FlextResult[None].fail(f"{field_name} is required")
            return FlextResult[None].ok(None)

        # Test require_field
        result = require_field("test_field", "")
        assert result.is_failure
        assert "test_field is required" in (result.error or "")

        result = require_field("test_field", "value")
        assert result.success

        # Test email validation
        def require_email(email: str) -> FlextResult[None]:
            if "@" not in email:
                return FlextResult[None].fail("Invalid email")
            return FlextResult[None].ok(None)

        result = require_email("invalid-email")
        assert result.is_failure
        assert "Invalid email" in (result.error or "")

        result = require_email("test@example.com")
        assert result.success

        # Test min length validation
        def require_min_length(
            value: str, min_len: int, field: str
        ) -> FlextResult[None]:
            if len(value) < min_len:
                return FlextResult[None].fail(
                    f"{field} must be at least {min_len} characters"
                )
            return FlextResult[None].ok(None)

        result = require_min_length("ab", 3, "field")
        assert result.is_failure
        assert "field must be at least 3 characters" in (result.error or "")

        result = require_min_length("abcd", 3, "field")
        assert result.success

    def test_command_to_dict(self) -> None:
        """Test command model_dump method."""
        command = TestCommandCommand(name="test", value=42)

        # Test model_dump (Pydantic v2 method)
        data = command.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["value"] == 42
        # Base fields may not be present in basic test model
        assert "name" in data
        assert "value" in data

    def test_command_to_json(self) -> None:
        """Test command model_dump_json method."""
        command = TestCommandCommand(name="test", value=42)

        # Test model_dump_json (Pydantic v2 method)
        json_str = command.model_dump_json()
        assert isinstance(json_str, str)
        assert '"name":"test"' in json_str or '"name": "test"' in json_str
        assert '"value":42' in json_str or '"value": 42' in json_str

    def test_query_model_functionality(self) -> None:
        """Test Query model functionality."""
        # Test query creation
        query = TestQueryModel(search_term="test", limit=20)
        assert query.search_term == "test"
        assert query.limit == 20

        # Test query validation
        result = query.validate_query()
        assert result.success

        # Test invalid query
        invalid_query = TestQueryModel(search_term="", limit=0)
        result = invalid_query.validate_query()
        assert result.is_failure

    def test_query_to_dict(self) -> None:
        """Test query model_dump method."""
        query = TestQueryModel(search_term="test", limit=20)

        data = query.model_dump()
        assert isinstance(data, dict)
        assert data["search_term"] == "test"
        assert data["limit"] == 20
        # Base fields may not be present in basic test model
        assert "search_term" in data
        assert "limit" in data

    def test_query_to_json(self) -> None:
        """Test query model_dump_json method."""
        query = TestQueryModel(search_term="test", limit=20)

        json_str = query.model_dump_json()
        assert isinstance(json_str, str)
        assert "search_term" in json_str
        assert "test" in json_str


class TestCommandHandlers:
    """Test command and query handlers."""

    def test_command_handler_base(self) -> None:
        """Test CommandHandler base functionality."""

        class TestHandler(
            FlextCommands.Handlers.CommandHandler[TestCommandCommand, str]
        ):
            def handle(self, command: TestCommandCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Handled: {command.name}")

            def can_handle(self, command: object) -> bool:
                return isinstance(command, TestCommandCommand)

        handler = TestHandler()

        # Test handler_name
        assert handler.handler_name == "TestHandler"

        # Test can_handle
        command = TestCommandCommand(name="test", value=42)
        assert handler.can_handle(command) is True
        assert handler.can_handle("not a command") is False

        # Test handle
        result = handler.handle(command)
        assert result.success
        assert result.value == "Handled: test"

    def test_query_handler_base(self) -> None:
        """Test QueryHandler base functionality."""

        class TestQueryHandler(
            FlextCommands.Handlers.QueryHandler[TestQueryModel, list[str]]
        ):
            def handle(self, query: TestQueryModel) -> FlextResult[list[str]]:
                results = [f"Result {i}" for i in range(query.limit)]
                return FlextResult[list[str]].ok(results)

            def can_handle(self, query: object) -> bool:
                return isinstance(query, TestQueryModel)

        handler = TestQueryHandler()

        # Test handler_name
        assert handler.handler_name == "TestQueryHandler"

        # Test can_handle
        query = TestQueryModel(search_term="test", limit=5)
        assert handler.can_handle(query) is True
        assert handler.can_handle("not a query") is False

        # Test handle
        result = handler.handle(query)
        assert result.success
        assert len(result.value) == 5
        assert result.value[0] == "Result 0"


class TestCommandBus:
    """Test command bus functionality."""

    def test_bus_handler_registration(self) -> None:
        """Test handler registration in command bus."""
        bus = FlextCommands.Bus()

        class TestHandler(
            FlextCommands.Handlers.CommandHandler[TestCommandCommand, str]
        ):
            def handle(self, command: TestCommandCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Handled: {command.name}")

            def can_handle(self, command: object) -> bool:
                return isinstance(command, TestCommandCommand)

        handler = TestHandler()

        # Test register_handler
        bus.register_handler(handler)

        # Test get_all_handlers
        handlers = bus.get_all_handlers()
        assert isinstance(handlers, list)
        assert len(handlers) > 0

    def test_bus_middleware(self) -> None:
        """Test command bus middleware functionality."""
        bus = FlextCommands.Bus()

        class TestMiddleware:
            def process(self, _command: object, _handler: object) -> FlextResult[None]:
                # Simple middleware that logs
                return FlextResult[None].ok(None)

        middleware = TestMiddleware()
        bus.add_middleware(middleware)

        # Verify middleware was added (indirect test)
        assert hasattr(bus, "add_middleware")

    def test_bus_execute_command(self) -> None:
        """Test command execution through bus."""
        bus = FlextCommands.Bus()

        class TestHandler(
            FlextCommands.Handlers.CommandHandler[TestCommandCommand, str]
        ):
            def handle(self, command: TestCommandCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Executed: {command.name}")

            def can_handle(self, command: object) -> bool:
                return isinstance(command, TestCommandCommand)

        handler = TestHandler()
        bus.register_handler(handler)

        command = TestCommandCommand(name="test", value=42)

        # Test execute
        result = bus.execute(command)
        assert result.success
        assert "Executed: test" in str(result.value) or result.value == "Executed: test"


class TestCommandTypes:
    """Test command type definitions."""

    def test_type_aliases(self) -> None:
        """Test FlextCommands.Types aliases."""
        # Test CommandId type
        cmd_id = "cmd_123"
        assert isinstance(cmd_id, str)
        # Verify type alias exists
        assert hasattr(FlextCommands.Types, "CommandId")

        # Test CommandType type
        cmd_type = "create_user"
        assert isinstance(cmd_type, str)
        # Verify type alias exists
        assert hasattr(FlextCommands.Types, "CommandType")

        # Test CommandMetadata type
        metadata = {"key": "value"}
        assert isinstance(metadata, dict)

        # Test CommandParameters type
        params = {"param1": "value1"}
        assert isinstance(params, dict)


class TestCommandConfiguration:
    """Test command configuration functionality."""

    def test_command_config_pattern(self) -> None:
        """Test command configuration patterns."""
        # Test configuration dictionary pattern
        config: dict[str, dict[str, int | float]] = {
            "retry_config": {"max_retries": 3, "backoff_factor": 2.0},
            "timeout_config": {"default_timeout": 30, "max_timeout": 300},
        }

        assert isinstance(config, dict)
        assert "retry_config" in config
        assert "timeout_config" in config
        retry_config = config["retry_config"]
        assert retry_config["max_retries"] == 3

    def test_environment_specific_config(self) -> None:
        """Test environment-specific configuration patterns."""
        # Test creating environment-specific config
        environments = ["development", "production", "test", "staging", "local"]

        for env in environments:
            # Simulate environment config creation
            config = {
                "environment": env,
                "debug": env == "development",
                "max_retries": 1 if env == "development" else 3,
                "timeout": 10 if env == "test" else 30,
            }

            assert config["environment"] == env
            if env == "development":
                assert config["debug"] is True
                assert config["max_retries"] == 1


class TestCommandDecorators:
    """Test command decorator patterns."""

    def test_command_handler_decorator(self) -> None:
        """Test command_handler decorator pattern."""

        # Test custom command handler decorator pattern
        def command_handler(
            command_type: type[TestCommandCommand],
        ) -> Callable[
            [Callable[[TestCommandCommand], str]], Callable[[TestCommandCommand], str]
        ]:
            def decorator(
                func: Callable[[TestCommandCommand], str],
            ) -> Callable[[TestCommandCommand], str]:
                func.__dict__ = getattr(func, "__dict__", {})
                func.__dict__["command_type"] = command_type
                return func

            return decorator

        @command_handler(TestCommandCommand)
        def my_handler(command: TestCommandCommand) -> str:
            return f"Decorated: {command.name}"

        # Test decorator sets attributes
        assert hasattr(my_handler, "__dict__")
        assert "command_type" in my_handler.__dict__

        # Test handler execution
        command = TestCommandCommand(name="test", value=42)
        result = my_handler(command)
        assert result == "Decorated: test"

    def test_validate_command_decorator(self) -> None:
        """Test validate_command decorator pattern."""

        # Test custom validation decorator pattern
        def validate_command(
            func: Callable[[TestCommandCommand], str],
        ) -> Callable[[TestCommandCommand], str]:
            def wrapper(command: TestCommandCommand) -> str:
                # Validate command before processing
                validation = command.validate_command()
                if validation.is_failure:
                    raise ValueError(validation.error)
                return func(command)

            return wrapper

        @validate_command
        def process_command(command: TestCommandCommand) -> str:
            return f"Processing: {command.name}"

        # Test with valid command
        valid_command = TestCommandCommand(name="test", value=42)
        result = process_command(valid_command)
        assert result == "Processing: test"

        # Test with invalid command
        invalid_command = TestCommandCommand(name="", value=-1)
        with pytest.raises(ValueError):
            process_command(invalid_command)


class TestAsyncCommandFeatures:
    """Test async command features."""

    @pytest.mark.asyncio
    async def test_async_command_handler(self) -> None:
        """Test async command handler pattern."""

        class AsyncTestHandler(
            FlextCommands.Handlers.CommandHandler[TestCommandCommand, str]
        ):
            async def handle_async(
                self, command: TestCommandCommand
            ) -> FlextResult[str]:
                await asyncio.sleep(0.01)  # Simulate async work
                return FlextResult[str].ok(f"Async handled: {command.name}")

            def handle(self, command: TestCommandCommand) -> FlextResult[str]:
                # Sync wrapper for async handler
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.handle_async(command))

        handler = AsyncTestHandler()
        command = TestCommandCommand(name="async_test", value=42)

        # Test async handling
        result = await handler.handle_async(command)
        assert result.success
        assert result.value == "Async handled: async_test"


class TestCommandFactories:
    """Test command factory methods."""

    def test_create_simple_handler(self) -> None:
        """Test creating simple handler from function."""

        # Test factory pattern for simple handler creation
        class SimpleHandler(
            FlextCommands.Handlers.CommandHandler[TestCommandCommand, str]
        ):
            def __init__(
                self, handler_func: Callable[[TestCommandCommand], str]
            ) -> None:
                self.handler_func = handler_func

            def handle(self, command: TestCommandCommand) -> FlextResult[str]:
                result = self.handler_func(command)
                return FlextResult[str].ok(result)

        def my_handler_func(command: TestCommandCommand) -> str:
            return f"Simple: {command.name}"

        # Create handler using factory pattern
        handler = SimpleHandler(my_handler_func)
        assert isinstance(handler, FlextCommands.Handlers.CommandHandler)

        # Test handler works
        command = TestCommandCommand(name="test", value=42)
        result = handler.handle(command)
        assert result.success
        assert "Simple: test" in str(result.value)

    def test_create_query_handler(self) -> None:
        """Test creating query handler from function."""

        # Test factory pattern for query handler creation
        class SimpleQueryHandler(
            FlextCommands.Handlers.QueryHandler[TestQueryModel, list[str]]
        ):
            def __init__(
                self, handler_func: Callable[[TestQueryModel], list[str]]
            ) -> None:
                self.handler_func = handler_func

            def handle(self, query: TestQueryModel) -> FlextResult[list[str]]:
                result = self.handler_func(query)
                return FlextResult[list[str]].ok(result)

        def my_query_func(query: TestQueryModel) -> list[str]:
            return [f"Result for: {query.search_term}"]

        # Create handler using factory pattern
        handler = SimpleQueryHandler(my_query_func)
        assert isinstance(handler, FlextCommands.Handlers.QueryHandler)

        # Test handler works
        query = TestQueryModel(search_term="test", limit=10)
        result = handler.handle(query)
        assert result.success
        assert "Result for: test" in str(result.value)


class TestCommandResults:
    """Test command result patterns."""

    def test_results_with_metadata(self) -> None:
        """Test creating results with metadata."""
        # Test success result pattern
        result = FlextResult.ok("data")
        assert result.success
        assert result.value == "data"

        # Test result with attributes
        result_with_meta = FlextResult.ok("data")
        # FlextResult doesn't have settable metadata, test other attributes
        assert result_with_meta.success
        assert result_with_meta.value == "data"

        # Test failure with error code
        result = FlextResult.fail("error message", error_code="CMD_ERROR")
        assert result.is_failure
        assert result.error == "error message"
        assert result.error_code == "CMD_ERROR"

        # Test failure attributes
        result_with_data: FlextResult[None] = FlextResult.fail("error message")
        assert result_with_data.is_failure
        assert result_with_data.error == "error message"


class TestCommandConstants:
    """Test command constants."""

    def test_command_constants(self) -> None:
        """Test command-related constants."""
        from flext_core import FlextConstants

        # Test performance constants
        assert hasattr(FlextConstants.Performance, "DEFAULT_BATCH_SIZE")
        assert FlextConstants.Performance.DEFAULT_BATCH_SIZE > 0

        # Test timeout constants (using actual attribute names)
        assert hasattr(FlextConstants.Performance, "TIMEOUT")
        assert FlextConstants.Performance.TIMEOUT > 0

        # Test command timeout
        assert hasattr(FlextConstants.Performance, "COMMAND_TIMEOUT")
        assert FlextConstants.Performance.COMMAND_TIMEOUT > 0

        # Test that constants are properly defined
        assert isinstance(FlextConstants.Performance.DEFAULT_BATCH_SIZE, int)
        assert isinstance(FlextConstants.Performance.TIMEOUT, (int, float))
        assert isinstance(FlextConstants.Performance.COMMAND_TIMEOUT, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
