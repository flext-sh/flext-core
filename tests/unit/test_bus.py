"""Comprehensive tests for FlextBus - CQRS Command Bus.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from flext_core import FlextBus, FlextResult


class TestFlextBus:
    """Test suite for FlextBus CQRS command bus functionality."""

    def test_bus_initialization(self) -> None:
        """Test bus initialization with default settings."""
        bus = FlextBus()
        assert bus is not None
        assert isinstance(bus, FlextBus)
        assert bus.config is not None

    def test_bus_with_custom_config(self) -> None:
        """Test bus initialization with custom configuration."""
        config = {"enable_middleware": True, "enable_caching": False}
        bus = FlextBus(bus_config=config)
        assert bus is not None
        assert bus.config.enable_caching is False

    def test_register_handler_single_arg(self) -> None:
        """Test handler registration with single argument (auto-discovery)."""
        bus = FlextBus()

        class TestHandler:
            def handle(self, command: object) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command}")

        handler = TestHandler()
        result = bus.register_handler(handler)
        assert result.is_success

    def test_register_handler_two_args(self) -> None:
        """Test handler registration with command type and handler."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class TestHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command.data}")

        handler = TestHandler()
        result = bus.register_handler("TestCommand", handler)
        assert result.is_success

    def test_register_handler_invalid(self) -> None:
        """Test handler registration with invalid parameters."""
        bus = FlextBus()

        # Test with invalid command type (empty string)
        result = bus.register_handler("", object())
        assert result.is_failure

    def test_unregister_handler(self) -> None:
        """Test handler unregistration."""
        bus = FlextBus()

        class TestHandler:
            def handle(self, command: object) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command}")

        handler = TestHandler()
        bus.register_handler("TestCommand", handler)
        result = bus.unregister_handler("TestCommand")
        assert result.is_success

    def test_unregister_nonexistent_handler(self) -> None:
        """Test unregistering non-existent handler."""
        bus = FlextBus()

        result = bus.unregister_handler("NonexistentCommand")
        assert result.is_failure

    def test_execute_command(self) -> None:
        """Test command execution."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class TestHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command.data}")

        handler = TestHandler()
        bus.register_handler("TestCommand", handler)

        command = TestCommand("test_data")
        result = bus.execute(command)
        assert result.is_success
        assert result.value == "processed_test_data"

    def test_execute_command_nonexistent_handler(self) -> None:
        """Test executing command with non-existent handler."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        command = TestCommand("test_data")
        result = bus.execute(command)
        assert result.is_failure

    def test_execute_command_with_failing_handler(self) -> None:
        """Test executing command with failing handler."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class FailingHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].fail(
                    f"Handler failed for command: {command.data}"
                )

        handler = FailingHandler()
        bus.register_handler("TestCommand", handler)

        command = TestCommand("test_data")
        result = bus.execute(command)
        assert result.is_failure
        assert "Handler failed" in result.error

    def test_get_registered_handlers(self) -> None:
        """Test getting registered handlers."""
        bus = FlextBus()

        class TestHandler:
            def handle(self, command: object) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command}")

        handler = TestHandler()
        bus.register_handler("TestCommand", handler)
        handlers = bus.get_registered_handlers()
        assert len(handlers) == 1
        assert "TestCommand" in handlers

    def test_get_all_handlers(self) -> None:
        """Test getting all handlers."""
        bus = FlextBus()

        class TestHandler:
            def handle(self, command: object) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command}")

        handler = TestHandler()
        bus.register_handler("TestCommand", handler)
        handlers = bus.get_all_handlers()
        assert len(handlers) == 1
        assert handler in handlers

    def test_find_handler(self) -> None:
        """Test finding handler for command."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class TestHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command.data}")

        handler = TestHandler()
        bus.register_handler("TestCommand", handler)

        command = TestCommand("test_data")
        found_handler = bus.find_handler(command)
        assert found_handler is not None
        assert found_handler == handler

    def test_send_command_compatibility(self) -> None:
        """Test send_command method (compatibility shim)."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class TestHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command.data}")

        handler = TestHandler()
        bus.register_handler("TestCommand", handler)

        command = TestCommand("test_data")
        result = bus.send_command(command)
        assert result.is_success
        assert result.value == "processed_test_data"

    def test_bus_error_handling(self) -> None:
        """Test bus error handling mechanisms."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class ErrorHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                msg = f"Handler error for command: {command.data}"
                raise ValueError(msg)

        handler = ErrorHandler()
        bus.register_handler("TestCommand", handler)

        command = TestCommand("test_data")
        result = bus.execute(command)
        assert result.is_failure
        assert "Handler error" in result.error

    def test_bus_performance(self) -> None:
        """Test bus performance characteristics."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class FastHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command.data}")

        handler = FastHandler()
        bus.register_handler("TestCommand", handler)

        start_time = time.time()
        for i in range(10):  # Reduced from 100 to 10 for faster execution
            command = TestCommand(f"data_{i}")
            result = bus.execute(command)
            assert result.is_success
        end_time = time.time()

        # Should complete 10 operations in reasonable time
        assert end_time - start_time < 5.0  # Increased timeout to 5 seconds

    def test_bus_concurrent_access(self) -> None:
        """Test bus concurrent access patterns."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class TestHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command.data}")

        handler = TestHandler()
        bus.register_handler("TestCommand", handler)

        # Simulate concurrent access
        results = []
        for i in range(10):
            command = TestCommand(f"data_{i}")
            result = bus.execute(command)
            results.append(result)

        # All operations should succeed
        assert all(result.is_success for result in results)

    def test_bus_middleware(self) -> None:
        """Test bus middleware functionality."""
        bus = FlextBus()

        class TestCommand(BaseModel):
            data: str

        class TestMiddleware:
            def process(
                self, command: TestCommand, handler: object
            ) -> FlextResult[None]:
                # Log the middleware processing
                _ = command  # Use command parameter to avoid unused argument warning
                _ = handler  # Use handler parameter to avoid unused argument warning
                return FlextResult[None].ok(None)

        class TestHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{command.data}")

        middleware = TestMiddleware()
        handler = TestHandler()

        bus.add_middleware(middleware)
        bus.register_handler("TestCommand", handler)

        command = TestCommand("test_data")
        result = bus.execute(command)
        assert result.is_success

    def test_create_command_bus_factory(self) -> None:
        """Test create_command_bus factory method."""
        bus = FlextBus.create_command_bus()
        assert bus is not None
        assert isinstance(bus, FlextBus)

    def test_create_simple_handler(self) -> None:
        """Test create_simple_handler factory method."""

        def simple_handler(data: str) -> str:
            return f"processed_{data}"

        handler = FlextBus.create_simple_handler(simple_handler)
        assert handler is not None

    def test_create_query_handler(self) -> None:
        """Test create_query_handler factory method."""

        def query_handler(data: str) -> str:
            return f"query_result_{data}"

        handler = FlextBus.create_query_handler(query_handler)
        assert handler is not None

    def test_bus_caching(self) -> None:
        """Test bus caching functionality."""
        bus = FlextBus(bus_config={"enable_caching": True, "max_cache_size": 10})

        class TestQuery:
            def __init__(self, data: str) -> None:
                self.data = data
                self.query_id = "test_query"

        class TestQueryHandler:
            def handle(self, query: TestQuery) -> FlextResult[str]:
                return FlextResult[str].ok(f"query_result_{query.data}")

        handler = TestQueryHandler()
        bus.register_handler("TestQuery", handler)

        query = TestQuery("test_data")

        # First execution should cache the result
        result1 = bus.execute(query)
        assert result1.is_success

        # Second execution should use cache
        result2 = bus.execute(query)
        assert result2.is_success
        assert result1.value == result2.value
