"""Comprehensive tests for FlextRegistry - Service Registry.

Tests the actual FlextRegistry API with real functionality testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextDispatcher,
    FlextHandlers,
    FlextModels,
    FlextRegistry,
    FlextResult,
)


class ConcreteTestHandler(FlextHandlers[str, str]):
    """Concrete implementation of FlextHandlers for testing."""

    def handle(self, message: str) -> FlextResult[str]:
        """Handle the message."""
        return FlextResult[str].ok(f"processed_{message}")


class TestFlextRegistry:
    """Test suite for FlextRegistry service registry functionality."""

    def test_registry_initialization(self) -> None:
        """Test registry initialization."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)
        assert registry is not None
        assert isinstance(registry, FlextRegistry)

    def test_registry_register_handler(self) -> None:
        """Test handler registration."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_1", handler_name="Test Handler 1"
        )
        handler = ConcreteTestHandler(config=config)

        result = registry.register_handler(handler)
        assert result.is_success
        assert isinstance(result.value, FlextModels.RegistrationDetails)

    def test_registry_register_handler_none(self) -> None:
        """Test handler registration with None handler."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        result = registry.register_handler(None)
        assert result.is_failure
        assert "Handler cannot be None" in result.error

    def test_registry_register_handlers(self) -> None:
        """Test multiple handler registration."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_1", handler_name="Test Handler 1"
        )
        handler1 = ConcreteTestHandler(config=config)
        handler2 = ConcreteTestHandler(config=config)

        handlers = [handler1, handler2]
        result = registry.register_handlers(handlers)
        assert result.is_success
        assert isinstance(result.value, FlextRegistry.Summary)

    def test_registry_register_handlers_empty(self) -> None:
        """Test multiple handler registration with empty list."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        result = registry.register_handlers([])
        assert result.is_success
        assert isinstance(result.value, FlextRegistry.Summary)

    def test_registry_register_bindings(self) -> None:
        """Test binding registration."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_binding_handler", handler_name="Test Binding Handler"
        )
        handler = ConcreteTestHandler(config=config)
        bindings = [(str, handler)]

        result = registry.register_bindings(bindings)
        assert result.is_success
        assert isinstance(result.value, FlextRegistry.Summary)

    def test_registry_register_bindings_empty(self) -> None:
        """Test binding registration with empty bindings."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        result = registry.register_bindings({})
        assert result.is_success
        assert isinstance(result.value, FlextRegistry.Summary)

    def test_registry_register_function_map(self) -> None:
        """Test function map registration."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        def test_function(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{data}")

        function_map = {"test_function": test_function}

        result = registry.register_function_map(function_map)
        assert result.is_success
        assert isinstance(result.value, FlextRegistry.Summary)

    def test_registry_register_function_map_empty(self) -> None:
        """Test function map registration with empty map."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        result = registry.register_function_map({})
        assert result.is_success
        assert isinstance(result.value, FlextRegistry.Summary)

    def test_registry_summary_properties(self) -> None:
        """Test registry summary properties."""
        summary = FlextRegistry.Summary()

        # Test initial state
        assert summary.is_success is True
        assert summary.successful_registrations == 0
        assert summary.failed_registrations == 0

        # Add some test data
        summary.registered.append(
            FlextModels.RegistrationDetails(
                registration_id="test1",
                handler_mode="command",
                timestamp="2025-01-01T00:00:00Z",
                status="active",
            )
        )
        summary.errors.append("test_error")

        # Test updated state
        assert summary.is_success is False
        assert summary.successful_registrations == 1
        assert summary.failed_registrations == 1

    def test_registry_idempotent_registration(self) -> None:
        """Test that re-registering the same handler is idempotent."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_1", handler_name="Test Handler 1"
        )
        handler = ConcreteTestHandler(config=config)

        # First registration
        result1 = registry.register_handler(handler)
        assert result1.is_success

        # Second registration (should be idempotent)
        result2 = registry.register_handler(handler)
        assert result2.is_success

        # Both should return success
        assert result1.is_success == result2.is_success

    def test_registry_safe_get_handler_mode(self) -> None:
        """Test safe handler mode extraction."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        # Test valid modes
        assert registry._safe_get_handler_mode("command") == "command"
        assert registry._safe_get_handler_mode("query") == "query"

        # Test invalid mode (should default to command)
        assert registry._safe_get_handler_mode("invalid") == "command"
        assert registry._safe_get_handler_mode(None) == "command"

    def test_registry_safe_get_status(self) -> None:
        """Test safe status extraction."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        # Test valid statuses
        assert registry._safe_get_status("active") == "active"
        assert registry._safe_get_status("inactive") == "inactive"

        # Test invalid status (should default to active)
        assert registry._safe_get_status("invalid") == "active"
        assert registry._safe_get_status(None) == "active"

    def test_registry_resolve_handler_key(self) -> None:
        """Test handler key resolution."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_1", handler_name="Test Handler 1"
        )
        handler = ConcreteTestHandler(config=config)

        key = registry._resolve_handler_key(handler)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_registry_resolve_binding_key(self) -> None:
        """Test binding key resolution."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        binding_entry = {"handler_mode": "command", "status": "active"}

        key = registry._resolve_binding_key("test_binding", binding_entry)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_registry_resolve_binding_key_from_entry(self) -> None:
        """Test binding key resolution from entry."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        binding_entry = {"handler_mode": "command", "status": "active"}

        key = registry._resolve_binding_key_from_entry(binding_entry, str)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_registry_with_real_dispatcher(self) -> None:
        """Test registry with real dispatcher functionality."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_1", handler_name="Test Handler 1"
        )
        handler = ConcreteTestHandler(config=config)

        # Register handler
        result = registry.register_handler(handler)
        assert result.is_success

        # Verify registration details
        reg_details = result.value
        assert reg_details.registration_id is not None
        assert reg_details.handler_mode in {"command", "query"}
        assert reg_details.status in {"active", "inactive"}

    def test_registry_error_handling(self) -> None:
        """Test registry error handling."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        # Test with None handler
        result = registry.register_handler(None)
        assert result.is_failure
        assert "Handler cannot be None" in result.error

    def test_registry_summary_creation(self) -> None:
        """Test registry summary creation."""
        summary = FlextRegistry.Summary()

        # Test initial state
        assert len(summary.registered) == 0
        assert len(summary.skipped) == 0
        assert len(summary.errors) == 0

        # Test properties
        assert summary.is_success is True
        assert summary.successful_registrations == 0
        assert summary.failed_registrations == 0

    def test_registry_summary_with_data(self) -> None:
        """Test registry summary with data."""
        summary = FlextRegistry.Summary()

        # Add registered handler
        summary.registered.append(
            FlextModels.RegistrationDetails(
                registration_id="test1",
                handler_mode="command",
                timestamp="2025-01-01T00:00:00Z",
                status="active",
            )
        )

        # Add skipped handler
        summary.skipped.append("skipped_handler")

        # Add error
        summary.errors.append("registration_error")

        # Test properties
        assert summary.is_success is False  # Has errors
        assert summary.successful_registrations == 1
        assert summary.failed_registrations == 1
        assert len(summary.skipped) == 1
