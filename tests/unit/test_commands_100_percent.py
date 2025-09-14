"""Final push for 100% commands.py coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextCommands, FlextResult
from flext_tests import FlextTestsMatchers


class TestAbstractMethodCoverage:
    """Test coverage for abstract method NotImplementedError paths."""

    def test_command_handler_abstract_handle_method(self) -> None:
        """Test CommandHandler abstract handle method (lines 260-261)."""
        # Create handler without implementing handle method
        handler = FlextCommands.Handlers.CommandHandler()

        with pytest.raises(
            NotImplementedError, match="Subclasses must implement handle method"
        ):
            handler.handle({})

    def test_query_handler_abstract_handle_method(self) -> None:
        """Test QueryHandler abstract handle method (lines 449-450)."""
        # Create handler without implementing handle method
        handler = FlextCommands.Handlers.QueryHandler()

        with pytest.raises(
            NotImplementedError, match="Subclasses must implement handle method"
        ):
            handler.handle({})


class TestBusMiddlewareOrderEdgeCases:
    """Test coverage for Bus middleware order edge cases."""

    def test_bus_middleware_order_type_variations(self) -> None:
        """Test Bus _apply_middleware with various order types (line 683)."""
        bus = FlextCommands.Bus()

        # Create middleware with different order types
        middleware_configs = [
            {"middleware_id": "no_order", "enabled": True},  # No order field
            {
                "middleware_id": "none_order",
                "order": None,
                "enabled": True,
            },  # None order
            {
                "middleware_id": "valid_order",
                "order": 1,
                "enabled": True,
            },  # Valid order
        ]

        bus._middleware = middleware_configs

        # Should handle various order types gracefully (line 683)
        result = bus._apply_middleware({}, None)
        FlextTestsMatchers.assert_result_success(result)

    def test_bus_middleware_disabled_middleware(self) -> None:
        """Test Bus _apply_middleware with disabled middleware (line 689)."""
        bus = FlextCommands.Bus()

        # Create disabled middleware config
        middleware_config = {
            "middleware_id": "disabled-middleware",
            "enabled": False,  # Disabled
            "order": 1,
        }

        bus._middleware.append(middleware_config)

        # Should skip disabled middleware (line 689 - continue statement)
        result = bus._apply_middleware({}, None)
        FlextTestsMatchers.assert_result_success(result)


class TestBusUnregisterEdgeCasesFinal:
    """Test coverage for Bus unregister final edge cases."""

    def test_bus_unregister_handler_no_name_attribute(self) -> None:
        """Test Bus unregister_handler with key that has no __name__ (lines 842-845)."""
        bus = FlextCommands.Bus()

        # Create a key object without __name__ attribute
        class KeyWithoutName:
            def __str__(self) -> str:
                return "KeyWithoutName"

        key_obj = KeyWithoutName()

        def handler(cmd: object) -> str:
            # Use the cmd parameter to avoid unused argument warning
            _ = cmd  # Acknowledge parameter usage
            return "handled"

        # Manually add to handlers dict with object key
        bus._handlers[key_obj] = handler

        # Try to unregister - should match by str() comparison (lines 842-845)
        result = bus.unregister_handler("KeyWithoutName")
        assert result is True
        assert key_obj not in bus._handlers


class TestBusExecutionEdgeCasesRemaining:
    """Test remaining execution edge cases."""

    def test_bus_execute_metrics_disabled_no_cache(self) -> None:
        """Test Bus execute when metrics are disabled (lines 598-603)."""
        config = {"enable_metrics": False}  # Disable metrics
        bus = FlextCommands.Bus(bus_config=config)

        class TestQuery:
            query_id = "test-query"

        class TestHandler:
            handler_id = "test-handler"

            def handle(self, query: TestQuery) -> FlextResult[str]:
                return FlextResult[str].ok("query result")

            def execute(self, query: TestQuery) -> FlextResult[str]:
                return self.handle(query)

            def can_handle(self, query_type: type) -> bool:
                return True

        handler = TestHandler()
        bus.register_handler(handler)

        query = TestQuery()

        # Execute - should not use caching logic when metrics disabled
        result = bus.execute(query)
        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "query result"


# Add a comprehensive integration test that exercises multiple paths
class TestCommandsIntegrationFinal:
    """Final integration test to catch any remaining uncovered lines."""

    def test_complete_commands_workflow_with_edge_cases(self) -> None:
        """Comprehensive test of commands workflow with all edge cases."""

        # Test command with custom validation
        class CustomCommand(FlextCommands.Models.Command):
            name: str = "test"

            def validate_command(self) -> FlextResult[bool]:
                return (
                    FlextResult[bool].ok(True)
                    if self.name
                    else FlextResult[bool].fail("Name required")
                )

        # Test handler with edge case handling
        class CustomHandler(FlextCommands.Handlers.CommandHandler[CustomCommand, str]):
            def handle(self, command: CustomCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Handled: {command.name}")

            def can_handle(self, command_type: object) -> bool:
                return True

        # Test middleware with edge cases
        class TestMiddleware:
            def process(self, command: object, handler: object) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        # Setup bus with all configurations
        bus_config = {"enable_middleware": True, "enable_metrics": True}
        bus = FlextCommands.Bus(bus_config=bus_config)

        # Register handler and middleware
        handler = CustomHandler()
        bus.register_handler(handler)

        middleware = TestMiddleware()
        bus.add_middleware(
            middleware,
            {"middleware_id": "test-middleware", "order": 1, "enabled": True},
        )

        # Execute command
        command = CustomCommand(command_type="custom", name="integration_test")
        result = bus.execute(command)

        FlextTestsMatchers.assert_result_success(result)
        assert "Handled: integration_test" in result.unwrap()

        # Test factories
        FlextCommands.Factories.create_simple_handler(lambda cmd: f"Factory: {cmd}")
        FlextCommands.Factories.create_query_handler(lambda q: [f"Query: {q}"])

        # Test results helpers
        success_result = FlextCommands.Results.success("test_data")
        FlextTestsMatchers.assert_result_success(success_result)

        failure_result = FlextCommands.Results.failure("test_error")
        FlextTestsMatchers.assert_result_failure(failure_result)

        # Test decorators
        @FlextCommands.Decorators.command_handler(CustomCommand)
        def decorated_handler(cmd: CustomCommand) -> str:
            return f"Decorated: {cmd.name}"

        result = decorated_handler(command)
        assert "Decorated: integration_test" in result
