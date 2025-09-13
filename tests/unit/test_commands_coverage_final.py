"""Final coverage tests for FlextCommands to reach 100% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from unittest.mock import patch

from flext_core import (
    FlextCommands,
    FlextResult,
)
from flext_tests import FlextTestsMatchers


class TestCommandsPayloadException:
    """Test coverage for payload creation exception path (lines 129-130)."""

    def test_command_to_payload_exception(self) -> None:
        """Test command to_payload method exception path."""
        command = FlextCommands.Models.Command(command_type="test")

        # Mock FlextUtilities.Generators.generate_uuid to raise an exception
        with patch(
            "flext_core.commands.FlextUtilities.Generators.generate_uuid"
        ) as mock_uuid:
            mock_uuid.side_effect = Exception("UUID generation failed")

            result = command.to_payload()

            # Should return FlextResult.fail on exception (lines 129-130)
            FlextTestsMatchers.assert_result_failure(result)
            assert "Failed to create payload" in result.error
            assert "UUID generation failed" in result.error


class TestQueryTypeValidatorNonDict:
    """Test coverage for Query._ensure_query_type with non-dict data (line 172)."""

    def test_query_ensure_query_type_non_dict_data(self) -> None:
        """Test _ensure_query_type validator with non-dict data."""

        class TestQuery(FlextCommands.Models.Query):
            pass

        # Test with non-dict data (should return unchanged - line 172)
        result = TestQuery._ensure_query_type("not_a_dict")
        assert result == "not_a_dict"

        # Test with None (should return unchanged)
        result = TestQuery._ensure_query_type(None)
        assert result is None


class TestCommandHandlerTimingEdgeCases:
    """Test coverage for CommandHandler timing edge cases (lines 260-261)."""

    def test_command_handler_timing_without_start_time(self) -> None:
        """Test CommandHandler when _start_time is not set."""

        class TestCommand:
            command_id = "test-command"

            def validate_command(self) -> FlextResult[bool]:
                return FlextResult[bool].ok(True)

        class TestHandler(FlextCommands.Handlers.CommandHandler[TestCommand, str]):
            def handle(self, command: TestCommand) -> FlextResult[str]:
                # Test with missing _start_time attribute entirely
                _ = command  # Acknowledge parameter usage
                if hasattr(self, "_start_time"):
                    delattr(self, "_start_time")
                return FlextResult[str].ok("handled")

            def can_handle(self, command_type: object) -> bool:
                _ = command_type  # Acknowledge parameter usage
                return True

        handler = TestHandler()
        command = TestCommand()

        # Execute command - should handle missing _start_time gracefully
        result = handler.execute(command)

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "handled"


class TestQueryHandlerEdgeCases:
    """Test coverage for QueryHandler edge cases (lines 449-450)."""

    def test_query_handler_validate_query_result_not_flext_result(self) -> None:
        """Test QueryHandler when validate_query returns non-FlextResult."""

        class TestQuery:
            def validate_query(self) -> bool:  # Returns bool, not FlextResult
                return True

        class TestHandler(FlextCommands.Handlers.QueryHandler[TestQuery, str]):
            def handle(self, query: TestQuery) -> FlextResult[str]:
                _ = query  # Acknowledge parameter usage
                return FlextResult[str].ok("handled")

        handler = TestHandler()
        query = TestQuery()

        # Should handle non-FlextResult validation return gracefully
        result = handler.validate_query(query)
        FlextTestsMatchers.assert_result_success(result)


class TestBusMiddlewareEdgeCases:
    """Test coverage for Bus middleware edge cases (lines 598-603, 677-683, 689)."""

    def test_bus_execute_middleware_disabled_but_configured(self) -> None:
        """Test Bus execute when middleware is disabled but middleware is configured."""
        config = {"enable_middleware": False}
        bus = FlextCommands.Bus(bus_config=config)

        # Add middleware config even though disabled
        bus._middleware.append({"middleware_id": "test-middleware", "enabled": True})

        class TestCommand:
            pass

        command = TestCommand()
        result = bus.execute(command)

        # Should fail with middleware disabled error (lines 598-603)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Middleware pipeline is disabled" in result.error

    def test_bus_apply_middleware_sorting_with_string_order(self) -> None:
        """Test Bus _apply_middleware with string order values."""
        bus = FlextCommands.Bus()

        # Create middleware with string order values
        middleware_configs = [
            {"middleware_id": "third", "order": "3", "enabled": True},
            {"middleware_id": "first", "order": "1", "enabled": True},
            {"middleware_id": "second", "order": "2", "enabled": True},
            {
                "middleware_id": "invalid",
                "order": "invalid",
                "enabled": True,
            },  # Invalid order
        ]

        bus._middleware = middleware_configs

        # Should handle string order values and invalid orders (lines 677-683)
        result = bus._apply_middleware({}, None)
        FlextTestsMatchers.assert_result_success(result)

    def test_bus_apply_middleware_no_instances(self) -> None:
        """Test Bus _apply_middleware when middleware config has no instance."""
        bus = FlextCommands.Bus()

        # Create middleware config without corresponding instance
        middleware_config = {
            "middleware_id": "missing-middleware",
            "enabled": True,
            "order": 1,
        }

        bus._middleware.append(middleware_config)
        # Don't add to _middleware_instances

        # Should skip middleware without instance (line 689)
        result = bus._apply_middleware({}, None)
        FlextTestsMatchers.assert_result_success(result)


class TestBusUnregisterEdgeCases:
    """Test coverage for Bus unregister edge cases (lines 842-845)."""

    def test_bus_unregister_handler_by_string_key(self) -> None:
        """Test Bus unregister_handler with string key matching."""
        bus = FlextCommands.Bus()

        class TestCommand:
            pass

        class TestHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                _ = command  # Acknowledge parameter usage
                return FlextResult[str].ok("handled")

        handler = TestHandler()

        # Register with type
        bus.register_handler(TestCommand, handler)

        # Unregister by string name - should find by string comparison (lines 842-845)
        result = bus.unregister_handler("TestCommand")
        assert result is True


class TestFactoriesEdgeCases:
    """Test coverage for Factories edge cases (line 912, 929)."""

    def test_simple_handler_with_non_flext_result_return(self) -> None:
        """Test create_simple_handler when function returns non-FlextResult."""

        def handler_function(command: dict) -> str:
            _ = command  # Acknowledge parameter usage
            return "simple result"  # Returns string, not FlextResult

        handler = FlextCommands.Factories.create_simple_handler(handler_function)
        command = {"test": "data"}

        result = handler.handle(command)

        # Should wrap non-FlextResult return in FlextResult.ok (line 912)
        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "simple result"

    def test_query_handler_with_non_flext_result_return(self) -> None:
        """Test create_query_handler when function returns non-FlextResult."""

        def query_function(query: dict) -> list[str]:
            _ = query  # Acknowledge parameter usage
            return ["query result"]  # Returns list, not FlextResult

        handler = FlextCommands.Factories.create_query_handler(query_function)
        query = {"search": "test"}

        result = handler.handle(query)

        # Should wrap non-FlextResult return in FlextResult.ok (line 929)
        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == ["query result"]
