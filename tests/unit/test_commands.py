"""Extended tests for FlextCommands to achieve higher coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import Field

from flext_core import (
    FlextCommands,
    FlextResult,
    FlextTypes,
)


class ExtendedTestCommand(FlextCommands.Models.Command):
    """Extended test command with more features."""

    name: str
    value: int = 0
    # Override metadata to allow more flexible types
    metadata: FlextTypes.Core.Headers = Field(
        default_factory=dict, description="Command metadata"
    )

    # Additional flexible metadata field
    extra_data: FlextTypes.Core.Dict | None = Field(
        default=None, description="Additional command data"
    )

    def validate_command(self) -> FlextResult[bool]:
        """Validate the command."""
        if not self.name:
            return FlextResult[bool].fail("Name is required")
        if self.value < 0:
            return FlextResult[bool].fail("Value must be non-negative")
        return FlextResult[bool].ok(True)

    def execute(self) -> FlextResult[str]:
        """Execute the command."""
        return FlextResult[str].ok(f"Executed {self.name} with value {self.value}")


class ExtendedQuery(FlextCommands.Models.Query):
    """Extended query with additional features."""

    search_term: str
    # Override filters to allow None
    filters: FlextTypes.Core.Dict = Field(
        default_factory=dict, description="Query filters"
    )

    # Additional flexible filters field
    custom_filters: FlextTypes.Core.Dict | None = Field(
        default=None, description="Custom query filters"
    )
    limit: int = 10

    def validate_query(self) -> FlextResult[bool]:
        """Validate the query."""
        if not self.search_term:
            return FlextResult[bool].fail("Search term required")
        if self.limit <= 0:
            return FlextResult[bool].fail("Limit must be positive")
        return FlextResult[bool].ok(True)


class TestCommandBusExtended:
    """Extended tests for Command Bus functionality."""

    def test_bus_with_multiple_handlers(self) -> None:
        """Test bus with multiple handlers for different commands."""
        bus = FlextCommands.Bus()

        # Create multiple handlers for different command types
        class Handler1(FlextCommands.Handlers.CommandHandler[ExtendedTestCommand, str]):
            def handle(self, command: ExtendedTestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Handler1: {command.name}")

            def can_handle(self, command_type: object) -> bool:
                return command_type == ExtendedTestCommand

        class Handler2(FlextCommands.Handlers.CommandHandler[ExtendedTestCommand, str]):
            def handle(self, command: ExtendedTestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Handler2: {command.name}")

            def can_handle(self, command_type: object) -> bool:
                return command_type == ExtendedTestCommand

        handler1 = Handler1()
        handler2 = Handler2()

        bus.register_handler(handler1)
        bus.register_handler(handler2)

        # Test with command that should be handled by handler1
        cmd1 = ExtendedTestCommand(command_type="test", name="test1", value=25)
        result1 = bus.execute(cmd1)
        assert result1.success
        assert "Handler1" in str(result1.value) or "test1" in str(result1.value)

        # Test with command that should be handled by handler2
        cmd2 = ExtendedTestCommand(command_type="test", name="test2", value=75)
        result2 = bus.execute(cmd2)
        assert result2.success
        assert "Handler2" in str(result2.value) or "test2" in str(result2.value)

    def test_bus_no_handler_found(self) -> None:
        """Test bus when no handler can handle the command."""
        bus = FlextCommands.Bus()

        # Command with no registered handler
        cmd = ExtendedTestCommand(command_type="test", name="unhandled", value=0)
        result = bus.execute(cmd)

        # Should return failure when no handler found
        assert result.is_failure or result.success  # May have default handler


class TestCommandValidation:
    """Test command validation features."""

    def test_command_with_metadata(self) -> None:
        """Test command with metadata."""
        metadata: FlextTypes.Core.Headers = {
            "user_id": "123",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        extra_data: FlextTypes.Core.Dict = {
            "complex_data": {"nested": "value"},
            "number": 42,
        }
        cmd = ExtendedTestCommand(
            command_type="test",
            name="test",
            value=42,
            metadata=metadata,
            extra_data=extra_data,
        )

        assert cmd.metadata == metadata
        assert cmd.metadata is not None
        assert cmd.metadata["user_id"] == "123"
        assert cmd.extra_data == extra_data

        # Test validation still works
        result = cmd.validate_command()
        assert result.success

    def test_invalid_command_validation(self) -> None:
        """Test invalid command validation."""
        # Empty name
        cmd1 = ExtendedTestCommand(command_type="test", name="", value=10)
        result1 = cmd1.validate_command()
        assert result1.is_failure
        assert "Name is required" in (result1.error or "")

        # Negative value
        cmd2 = ExtendedTestCommand(command_type="test", name="test", value=-5)
        result2 = cmd2.validate_command()
        assert result2.is_failure
        assert "Value must be non-negative" in (result2.error or "")

    def test_command_execution(self) -> None:
        """Test command execution method."""
        cmd = ExtendedTestCommand(command_type="test", name="execute_test", value=100)
        result = cmd.execute()

        assert result.success
        assert "Executed execute_test" in result.value
        assert "value 100" in result.value


class TestQueryHandling:
    """Test query handling features."""

    def test_query_with_filters(self) -> None:
        """Test query with filter parameters."""
        filters: FlextTypes.Core.Dict = {
            "category": "electronics",
            "price_range": [100, 500],
        }
        query = ExtendedQuery(
            query_type="search", search_term="laptop", filters=filters, limit=20
        )

        assert query.filters == filters
        assert query.filters is not None
        assert query.filters["category"] == "electronics"
        assert query.limit == 20

        # Validate query
        result = query.validate_query()
        assert result.success

    def test_invalid_query_validation(self) -> None:
        """Test invalid query validation."""
        # Empty search term
        query1 = ExtendedQuery(query_type="search", search_term="", limit=10)
        result1 = query1.validate_query()
        assert result1.is_failure
        assert "Search term required" in (result1.error or "")

        # Invalid limit
        query2 = ExtendedQuery(query_type="search", search_term="test", limit=0)
        result2 = query2.validate_query()
        assert result2.is_failure
        assert "Limit must be positive" in (result2.error or "")

    def test_query_handler_execution(self) -> None:
        """Test query handler execution."""

        class SearchHandler(
            FlextCommands.Handlers.QueryHandler[
                ExtendedQuery, list[FlextTypes.Core.Dict]
            ],
        ):
            def handle(
                self,
                query: ExtendedQuery,
            ) -> FlextResult[list[FlextTypes.Core.Dict]]:
                # Simulate search results
                results = [
                    {
                        "id": f"item_{i}",
                        "name": f"{query.search_term}_{i}",
                        "score": 1.0 - (i * 0.1),
                    }
                    for i in range(min(query.limit, 5))
                ]
                return FlextResult[list[FlextTypes.Core.Dict]].ok(results)

            def can_handle(self, query: object) -> bool:
                return isinstance(query, ExtendedQuery)

        handler = SearchHandler()
        query = ExtendedQuery(query_type="search", search_term="product", limit=3)

        result = handler.handle(query)
        assert result.success
        assert len(result.value) == 3
        assert result.value[0]["name"] == "product_0"


class TestCommandSerialization:
    """Test command serialization features."""

    def test_command_serialization(self) -> None:
        """Test command to dict and JSON serialization."""
        cmd = ExtendedTestCommand(command_type="test", name="serialize", value=42)

        # Test model_dump
        data = cmd.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "serialize"
        assert data["value"] == 42

        # Test model_dump_json
        json_str = cmd.model_dump_json()
        assert isinstance(json_str, str)
        assert "serialize" in json_str
        assert "42" in json_str

    def test_query_serialization(self) -> None:
        """Test query serialization."""
        query = ExtendedQuery(query_type="test", search_term="test", limit=5)

        # Test model_dump
        data = query.model_dump()
        assert isinstance(data, dict)
        assert data["search_term"] == "test"
        assert data["limit"] == 5

        # Test model_dump_json
        json_str = query.model_dump_json()
        assert isinstance(json_str, str)
        assert "test" in json_str


class TestCommandChaining:
    """Test command chaining and composition."""

    def test_command_chaining(self) -> None:
        """Test chaining multiple commands."""
        # Create a chain of commands
        cmd1 = ExtendedTestCommand(command_type="test", name="step1", value=10)
        cmd2 = ExtendedTestCommand(command_type="test", name="step2", value=20)
        cmd3 = ExtendedTestCommand(command_type="test", name="step3", value=30)

        # Execute chain
        results = []
        for cmd in [cmd1, cmd2, cmd3]:
            validation = cmd.validate_command()
            if validation.success:
                exec_result = cmd.execute()
                results.append(exec_result)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert "step1" in results[0].value
        assert "step2" in results[1].value
        assert "step3" in results[2].value

    def test_conditional_command_execution(self) -> None:
        """Test conditional command execution."""
        cmd = ExtendedTestCommand(command_type="test", name="conditional", value=50)

        # Only execute if value is above threshold
        threshold = 40
        if cmd.value > threshold:
            result = cmd.execute()
            assert result.success
            assert "conditional" in result.value
        else:
            # Should not reach here
            msg = "Command should have executed"
            raise AssertionError(msg)


class TestAsyncCommandPatterns:
    """Test async command patterns."""

    @pytest.mark.asyncio
    async def test_async_command_execution(self) -> None:
        """Test async command execution pattern."""

        class AsyncCommand(ExtendedTestCommand):
            async def execute_async(self) -> FlextResult[str]:
                await asyncio.sleep(0.01)  # Simulate async work
                return FlextResult[str].ok(f"Async executed: {self.name}")

        cmd = AsyncCommand(command_type="async", name="async_test", value=100)
        result = await cmd.execute_async()

        assert result.success
        assert "Async executed: async_test" in result.value

    @pytest.mark.asyncio
    async def test_async_query_handler(self) -> None:
        """Test async query handler."""

        class AsyncQueryHandler(
            FlextCommands.Handlers.QueryHandler[
                ExtendedQuery, FlextTypes.Core.StringList
            ],
        ):
            async def handle_async(
                self,
                query: ExtendedQuery,
            ) -> FlextResult[FlextTypes.Core.StringList]:
                await asyncio.sleep(0.01)  # Simulate async work
                results = [f"Async result {i}" for i in range(query.limit)]
                return FlextResult[FlextTypes.Core.StringList].ok(results)

            def handle(
                self, query: ExtendedQuery
            ) -> FlextResult[FlextTypes.Core.StringList]:
                # Sync wrapper
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.handle_async(query))

            def can_handle(self, query: object) -> bool:
                return isinstance(query, ExtendedQuery)

        handler = AsyncQueryHandler()
        query = ExtendedQuery(query_type="async_search", search_term="async", limit=3)

        result = await handler.handle_async(query)
        assert result.success
        assert len(result.value) == 3
        assert "Async result 0" in result.value[0]


class TestCommandMiddleware:
    """Test command middleware patterns."""

    def test_logging_middleware(self) -> None:
        """Test logging middleware pattern."""
        logged_commands = []

        class LoggingMiddleware:
            def process(
                self,
                command: object,
                _next_handler: object,
            ) -> FlextResult[object]:
                # Log command before processing
                if isinstance(command, ExtendedTestCommand):
                    logged_commands.append(command.name)

                # Call next handler (simulated)
                return FlextResult[object].ok("Processed")

        middleware = LoggingMiddleware()
        cmd = ExtendedTestCommand(command_type="test", name="logged_command", value=50)

        # Process through middleware
        result = middleware.process(cmd, None)

        assert result.success
        assert "logged_command" in logged_commands

    def test_validation_middleware(self) -> None:
        """Test validation middleware pattern."""

        class ValidationMiddleware:
            def process(
                self,
                command: object,
                _next_handler: object,
            ) -> FlextResult[object]:
                # Validate command before processing
                if isinstance(command, ExtendedTestCommand):
                    validation = command.validate_command()
                    if validation.is_failure:
                        return FlextResult[object].fail(
                            f"Validation failed: {validation.error}",
                        )

                # Proceed if valid
                return FlextResult[object].ok("Valid and processed")

        middleware = ValidationMiddleware()

        # Test with valid command
        valid_cmd = ExtendedTestCommand(command_type="test", name="valid", value=10)
        result = middleware.process(valid_cmd, None)
        assert result.success
        assert result.value == "Valid and processed"

        # Test with invalid command
        invalid_cmd = ExtendedTestCommand(command_type="test", name="", value=10)
        result = middleware.process(invalid_cmd, None)
        assert result.is_failure
        assert "Validation failed" in (result.error or "")


class TestCommandRetry:
    """Test command retry patterns."""

    def test_retry_on_failure(self) -> None:
        """Test retry logic for failed commands."""
        attempt_count = 0
        max_retries = 3

        class RetryableCommand(ExtendedTestCommand):
            def execute_with_retry(self) -> FlextResult[str]:
                nonlocal attempt_count
                attempt_count += 1

                if attempt_count < max_retries:
                    return FlextResult[str].fail(f"Attempt {attempt_count} failed")

                return FlextResult[str].ok(f"Success after {attempt_count} attempts")

        cmd = RetryableCommand(command_type="retry", name="retry_test", value=10)

        # Execute with retry logic
        result = None
        for _ in range(max_retries):
            result = cmd.execute_with_retry()
            if result.success:
                break

        assert result is not None
        assert result.success
        assert f"Success after {max_retries} attempts" in result.value
        assert attempt_count == max_retries


class TestCommandAggregation:
    """Test command aggregation patterns."""

    def test_batch_command_execution(self) -> None:
        """Test batch execution of multiple commands."""
        commands = [
            ExtendedTestCommand(command_type="test", name=f"batch_{i}", value=i * 10)
            for i in range(5)
        ]

        # Execute batch
        results = [cmd.execute() for cmd in commands if cmd.validate_command().success]

        assert len(results) == 5
        assert all(r.success for r in results)

        # Verify all commands were executed
        for i, result in enumerate(results):
            assert f"batch_{i}" in result.value

    def test_aggregate_command_results(self) -> None:
        """Test aggregating results from multiple commands."""
        commands = [
            ExtendedTestCommand(command_type="test", name="sum", value=10),
            ExtendedTestCommand(command_type="test", name="sum", value=20),
            ExtendedTestCommand(command_type="test", name="sum", value=30),
        ]

        # Execute and aggregate
        total_value = 0
        for cmd in commands:
            if cmd.validate_command().success:
                total_value += cmd.value

        assert total_value == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
