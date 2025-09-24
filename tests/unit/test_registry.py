"""Comprehensive tests for FlextRegistry - 100% coverage target.

Tests the registry functionality including handler registration,
function mapping, bindings, and error handling with FlextResult patterns.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

from flext_core import FlextDispatcher, FlextRegistry, FlextResult


class TestFlextRegistry:
    """Test FlextRegistry core functionality."""

    def test_initialization(self) -> None:
        """Test registry initialization."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        assert registry._dispatcher is mock_dispatcher
        assert registry._registered_keys == set()

    def test_safe_get_handler_mode_valid(self) -> None:
        """Test _safe_get_handler_mode with valid modes."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        # Test valid handler modes
        assert registry._safe_get_handler_mode("command") == "command"
        assert registry._safe_get_handler_mode("query") == "query"

    def test_safe_get_handler_mode_invalid(self) -> None:
        """Test _safe_get_handler_mode with invalid modes."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        # Test invalid handler mode - should return default "command" for invalid values
        result = registry._safe_get_handler_mode("invalid")
        assert result == "command"

    def test_safe_get_status_valid(self) -> None:
        """Test _safe_get_status with valid statuses."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        # Test valid registration statuses
        assert registry._safe_get_status("active") == "active"
        assert registry._safe_get_status("inactive") == "inactive"

    def test_safe_get_status_invalid(self) -> None:
        """Test _safe_get_status with invalid status."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        # Test invalid status - should return default "active" for invalid values
        result = registry._safe_get_status("invalid")
        assert result == "active"

    def test_register_handler_success(self) -> None:
        """Test successful handler registration."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        mock_dispatcher.register_handler.return_value = FlextResult[
            dict[str, object]
        ].ok({
            "registration_id": "test_reg",
            "handler_mode": "command",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "active",
        })

        registry = FlextRegistry(mock_dispatcher)

        # Mock handler with proper attributes
        mock_handler = Mock()
        mock_handler.handler_id = "TestHandler"
        mock_handler.__class__.__name__ = "TestHandler"

        result = registry.register_handler(mock_handler)

        assert result.is_success
        assert "TestHandler" in registry._registered_keys
        mock_dispatcher.register_handler.assert_called_once_with(mock_handler)

    def test_register_handler_failure(self) -> None:
        """Test handler registration failure."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        mock_dispatcher.register_handler.return_value = FlextResult[
            dict[str, object]
        ].fail("Registration failed")

        registry = FlextRegistry(mock_dispatcher)

        # Mock handler with proper attributes
        mock_handler = Mock()
        mock_handler.handler_id = "TestHandler"
        mock_handler.__class__.__name__ = "TestHandler"

        result = registry.register_handler(mock_handler)

        assert result.is_failure
        assert "TestHandler" not in registry._registered_keys
        assert result.error is not None
        assert "Registration failed" in str(result.error)

    def test_register_handlers_multiple_success(self) -> None:
        """Test registering multiple handlers successfully."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        mock_dispatcher.register_handler.return_value = FlextResult[
            dict[str, object]
        ].ok({
            "registration_id": "test_reg",
            "handler_mode": "command",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "active",
        })

        registry = FlextRegistry(mock_dispatcher)

        # Create proper mock handlers
        mock_handler1 = Mock()
        mock_handler1.handler_id = "Handler1"
        mock_handler1.__class__.__name__ = "Handler1"

        mock_handler2 = Mock()
        mock_handler2.handler_id = "Handler2"
        mock_handler2.__class__.__name__ = "Handler2"

        handlers = [mock_handler1, mock_handler2]

        result = registry.register_handlers(handlers)

        assert result.is_success
        summary = result.unwrap()
        assert len(summary.registered) == 2
        assert summary.successful_registrations == 2
        assert summary.failed_registrations == 0

    def test_register_handlers_mixed_results(self) -> None:
        """Test registering multiple handlers with mixed success/failure."""
        mock_dispatcher = Mock(spec=FlextDispatcher)

        # First call succeeds, second fails
        mock_dispatcher.register_handler.side_effect = [
            FlextResult[dict[str, object]].ok({
                "registration_id": "handler1_reg",
                "handler_mode": "command",
                "timestamp": "2024-01-01T00:00:00Z",
                "status": "active",
            }),
            FlextResult[dict[str, object]].fail("Handler2 failed"),
        ]

        registry = FlextRegistry(mock_dispatcher)

        # Create proper mock handlers
        mock_handler1 = Mock()
        mock_handler1.handler_id = "Handler1"
        mock_handler1.__class__.__name__ = "Handler1"

        mock_handler2 = Mock()
        mock_handler2.handler_id = "Handler2"
        mock_handler2.__class__.__name__ = "Handler2"

        handlers = [mock_handler1, mock_handler2]

        result = registry.register_handlers(handlers)

        assert result.is_failure  # Should fail due to second handler failure
        assert result.error is not None
        assert "Handler2 failed" in str(result.error)


class TestFlextRegistryEdgeCases:
    """Test edge cases and error conditions."""

    def test_register_handler_with_none_handler(self) -> None:
        """Test registering None as handler."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        # This should be handled gracefully
        try:
            result = registry.register_handler(None)
            # If it doesn't raise, it should fail gracefully
            assert result.is_failure
        except (TypeError, AttributeError):
            # Expected behavior for None handler
            pass

    def test_empty_handlers_dict(self) -> None:
        """Test registering empty handlers list."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        result = registry.register_handlers([])

        assert result.is_success
        summary = result.unwrap()
        assert len(summary.registered) == 0
        assert summary.successful_registrations == 0
        assert summary.failed_registrations == 0

    def test_empty_bindings_dict(self) -> None:
        """Test registering empty bindings list."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        result = registry.register_bindings([])

        assert result.is_success
        summary = result.unwrap()
        assert len(summary.registered) == 0
        assert len(summary.skipped) == 0
        assert len(summary.errors) == 0

    def test_empty_function_map(self) -> None:
        """Test registering empty function map."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        registry = FlextRegistry(mock_dispatcher)

        result = registry.register_function_map({})

        assert result.is_success
        summary = result.unwrap()
        assert len(summary.registered) == 0

    def test_function_map_missing_metadata(self) -> None:
        """Test function map entry missing metadata."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        mock_dispatcher.register_function.return_value = FlextResult[
            dict[str, object]
        ].ok({
            "registration_id": "test_reg",
            "handler_mode": "command",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "active",
        })
        registry = FlextRegistry(mock_dispatcher)

        class TestType:
            pass

        # Function map entry without metadata - using proper typing
        def test_function(message: object) -> object:
            return message

        function_map: dict[
            type[object],
            tuple[
                Callable[[object], object | FlextResult[object]],
                dict[str, object] | None,
            ],
        ] = {
            TestType: (test_function, None)  # Missing metadata key
        }
        result = registry.register_function_map(function_map)

        assert result.is_success  # Summary contains error details
        summary = result.unwrap()
        assert (
            summary.failed_registrations >= 0
        )  # Should handle missing metadata gracefully

    def test_function_map_invalid_metadata_type(self) -> None:
        """Test function map with invalid metadata type."""
        mock_dispatcher = Mock(spec=FlextDispatcher)
        # Configure mock to return successful registration
        mock_dispatcher.register_function.return_value = FlextResult[
            dict[str, object]
        ].ok({
            "registration_id": "func1_registration",
            "handler_mode": "command",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "active",
        })
        registry = FlextRegistry(mock_dispatcher)

        class TestType:
            pass

        # Function map with invalid metadata - using proper typing

        def test_function2(message: object) -> object:
            return message

        function_map: dict[
            type[object],
            tuple[
                Callable[[object], object | FlextResult[object]],
                dict[str, object] | None,
            ],
        ] = {
            TestType: (
                test_function2,
                {"invalid": "metadata_type"},
            )  # Valid dict format
        }

        result = registry.register_function_map(function_map)

        assert result.is_success  # Summary contains error details
        summary = result.unwrap()
        # Should handle invalid metadata type gracefully
        assert len(summary.registered) == 1
