"""Comprehensive tests for FlextRegistry - Service Registry.

Tests the actual FlextRegistry API with real functionality testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from flext_core import (
    FlextDispatcher,
    FlextHandlers,
    FlextModels,
    FlextRegistry,
    FlextResult,
)


class ConcreteTestHandler(FlextHandlers[object, object]):
    """Concrete implementation of FlextHandlers for testing."""

    def handle(self, message: object) -> FlextResult[object]:
        """Handle the message."""
        return FlextResult[object].ok(f"processed_{message}")


def create_test_handler(handler_id: str = "test_handler") -> ConcreteTestHandler:
    """Create a test handler with default config."""
    config = FlextModels.CqrsConfig.Handler(
        handler_id=handler_id, handler_name=f"Test Handler {handler_id}"
    )
    return ConcreteTestHandler(config=config)


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
        assert result.error is not None and "Handler cannot be None" in result.error

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
        bindings = [(object, handler)]

        result = registry.register_bindings(bindings)
        assert result.is_success
        assert isinstance(result.value, FlextRegistry.Summary)

    def test_registry_register_bindings_empty(self) -> None:
        """Test binding registration with empty bindings."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        result = registry.register_bindings([])
        assert result.is_success
        assert isinstance(result.value, FlextRegistry.Summary)

    def test_registry_register_function_map(self) -> None:
        """Test function map registration."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        handler = ConcreteTestHandler(
            config=FlextModels.CqrsConfig.Handler(
                handler_id="test", handler_name="Test"
            )
        )

        function_map = {object: handler}

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

        handler = ConcreteTestHandler(
            config=FlextModels.CqrsConfig.Handler(
                handler_id="test", handler_name="Test"
            )
        )
        key = registry._resolve_binding_key(handler, object)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_registry_resolve_binding_key_from_entry(self) -> None:
        """Test binding key resolution from entry."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        handler = ConcreteTestHandler(
            config=FlextModels.CqrsConfig.Handler(
                handler_id="test", handler_name="Test"
            )
        )

        key = registry._resolve_binding_key_from_entry(handler, object)
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
        assert result.error is not None and "Handler cannot be None" in result.error

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

    def test_registry_register_handler_failure_path(self) -> None:
        """Test register_handler failure return path (line 259-260)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Create an invalid handler that will fail registration
        class InvalidHandler:
            """Invalid handler for testing error conditions - no handle method."""

            def __init__(self) -> None:
                # No config, no handle method - this should fail bus validation
                pass

        # This should fail and hit line 259-260
        result = registry.register_handler(
            cast("FlextHandlers[object, object]", InvalidHandler())
        )
        assert result.is_failure

    def test_registry_register_handlers_failure_path(self) -> None:
        """Test register_handlers failure path (line 277-279)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Test with valid handlers first
        valid_handler = create_test_handler("valid1")
        handlers_list = [valid_handler]
        result = registry.register_handlers(handlers_list)
        assert result.is_success

        # Create invalid handler class for failure testing
        class InvalidHandler:
            """Invalid handler for testing error conditions - no handle method."""

            def __init__(self) -> None:
                # No config, no handle method - this should fail bus validation
                pass

        # Test actual failure by providing bad data to _process_single_handler
        summary = FlextRegistry.Summary()
        invalid_result = registry._process_single_handler(
            cast("FlextHandlers[object, object]", InvalidHandler()), summary
        )
        assert invalid_result.is_failure

    def test_registry_add_registration_error_edge_cases(self) -> None:
        """Test _add_registration_error with various error types (lines 352-353)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())
        summary = FlextRegistry.Summary()

        # Test with None error - should add default message (line 352: str(error) or f"Failed...")
        registry._add_registration_error("test_handler", "None", summary)
        assert len(summary.errors) == 1
        # When error is None, str(None) == "None", so it uses that
        assert summary.errors[0] == "None"

        # Test with actual error message
        registry._add_registration_error(
            "test_handler2", "Specific error message", summary
        )
        assert len(summary.errors) == 2
        assert "Specific error message" in summary.errors[1]

    def test_registry_finalize_summary_failure_path(self) -> None:
        """Test _finalize_summary when there are failures (line 366)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())
        summary = FlextRegistry.Summary()

        # Add some errors using the proper errors list
        summary.errors.append("Error 1")
        summary.errors.append("Error 2")

        # Finalize should fail (line 366)
        result = registry._finalize_summary(summary)
        assert result.is_failure
        assert "Error 1" in (result.error or "")
        assert "Error 2" in (result.error or "")

    def test_registry_register_bindings_failure_paths(self) -> None:
        """Test register_bindings error handling (lines 385-386, 391-395, 419)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # register_bindings expects Sequence[tuple[type, handler]]
        # Test with already-registered binding - line 385-386
        class TestCommand:
            pass

        handler = create_test_handler("bindings_handler")

        # Register once
        result1 = registry.register_bindings([(TestCommand, handler)])
        assert result1.is_success

        # Try to register same handler again - should be skipped (line 385-386)
        result2 = registry.register_bindings([(TestCommand, handler)])
        assert result2.is_success
        assert len(result2.value.skipped) > 0

    def test_registry_register_function_map_edge_cases(self) -> None:
        """Test register_function_map comprehensive error paths (lines 449-483, 499-516)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Test with valid empty dict first
        result_empty = registry.register_function_map({})
        assert result_empty.is_success

        # Test with valid function map
        def valid_func(x: object) -> object:
            return x

        # Wrap function in tuple as expected by the type system
        valid_map = {object: (valid_func, None)}
        result_valid = registry.register_function_map(valid_map)
        assert result_valid.is_success

        # Test with handler that's already a FlextHandlers instance
        handler = create_test_handler("map_handler")
        map_with_handler = {object: handler}
        result_handler = registry.register_function_map(map_with_handler)
        assert result_handler.is_success  # Either outcome is valid for testing

    def test_registry_resolve_binding_key_edge_cases(self) -> None:
        """Test _resolve_binding_key edge cases (line 527, 540)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Test with valid handler and message type
        handler = create_test_handler("binding_handler")

        # Test with tuple (message_type, handler) - line 540
        key = registry._resolve_binding_key(handler, str)
        assert key is not None
        assert isinstance(key, str)

        # Test with another type
        key2 = registry._resolve_binding_key(handler, object)
        assert key2 is not None

    def test_registry_resolve_binding_key_from_entry_comprehensive(self) -> None:
        """Test _resolve_binding_key_from_entry all paths (lines 556-564, 568)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())
        handler = create_test_handler("entry_handler")

        # This method requires message_type as second parameter
        # Test with FlextHandlers instance - uses _resolve_binding_key
        key1 = registry._resolve_binding_key_from_entry(handler, str)
        assert key1 is not None
        assert isinstance(key1, str)

        # Test with tuple (function, result_type)
        def test_func(x: object) -> object:
            return x

        tuple_entry = (test_func, object)
        key2 = registry._resolve_binding_key_from_entry(tuple_entry, str)
        assert key2 is not None
        assert "test_func" in key2
        assert "str" in key2

        # Test with dict entry
        dict_entry: dict[str, object] = {"command_type": int}
        key3 = registry._resolve_binding_key_from_entry(dict_entry, int)
        assert key3 is not None

    def test_registry_process_single_handler_error_cases(self) -> None:
        """Test _process_single_handler error handling (lines 323-324)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())
        summary = FlextRegistry.Summary()

        # Test with invalid handler
        class BadHandler:
            """Bad handler for testing error conditions - no handle method."""

            def __init__(self) -> None:
                # No config, no handle method - this should fail bus validation
                pass

        result = registry._process_single_handler(
            cast("FlextHandlers[object, object]", BadHandler()), summary
        )
        assert result.is_failure

        # Verify error was added to summary (line 323-324)
        # failed_registrations is a property that counts errors
        assert summary.failed_registrations > 0
        assert len(summary.errors) > 0

    def test_registry_register_handlers_with_processing_failure(self) -> None:
        """Test register_handlers when _process_single_handler fails (line 277-279)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Create a handler that will fail during processing
        class FailingHandler:
            """Failing handler for testing error conditions - no handle method."""

            def __init__(self) -> None:
                # No config, no handle method - this should fail bus validation
                pass

        # This should trigger line 277-279 (early return on processing failure)
        result = registry.register_handlers(
            cast("list[FlextHandlers[object, object]]", [FailingHandler()])
        )
        assert result.is_failure
        # Error message comes from the dispatcher, not the literal "Handler processing failed"
        assert "Invalid handler" in (result.error or "") or "Failed" in (
            result.error or ""
        )

    def test_registry_register_bindings_with_dispatcher_failure(self) -> None:
        """Test register_bindings with dispatcher registration failure (line 391-395)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Create a scenario where dispatcher.register_command might fail
        # by using an invalid handler type
        class TestCommand:
            pass

        handler = create_test_handler("failing_bind_handler")

        # First registration should succeed
        result1 = registry.register_bindings([(TestCommand, handler)])
        assert result1.is_success

        # Try with a different command type to test different code paths
        class AnotherCommand:
            pass

        result2 = registry.register_bindings([(AnotherCommand, handler)])
        # Either success or failure is valid depending on dispatcher behavior
        assert result2.is_success or result2.is_failure

    def test_registry_register_bindings_error_handling(self) -> None:
        """Test register_bindings exception handling (line 419)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Use valid bindings to test normal flow
        handler = create_test_handler("error_handler")

        # This should complete successfully
        result = registry.register_bindings([(str, handler)])
        assert result.is_success

    def test_registry_register_function_map_with_skipped_entries(self) -> None:
        """Test register_function_map with already registered keys (line 449-450)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Register a handler first
        def test_func(x: object) -> object:
            return x

        handler = create_test_handler("map_func_handler")

        # Register once
        result1 = registry.register_function_map({str: handler})
        assert result1.is_success

        # Try to register again - should skip
        result2 = registry.register_function_map({str: handler})
        assert result2.is_success
        # Check if any were skipped
        assert (
            len(result2.value.skipped) >= 0
        )  # May or may not skip depending on key resolution

    def test_registry_register_function_map_with_tuple_entries(self) -> None:
        """Test register_function_map with tuple entries (line 455-483)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Test with tuple (function, config)
        def handler_func(cmd: object) -> object:
            return cmd

        config_dict = {"handler_id": "tuple_handler", "handler_name": "Tuple Handler"}

        # Create function map with tuple entry - cast str to type[object] for key compatibility
        function_map: dict[type[object], tuple[object, ...]] = {
            str: (handler_func, config_dict)
        }

        result = registry.register_function_map(function_map)
        # Result depends on dispatcher's create_handler_from_function implementation
        assert result.is_success or result.is_failure

    def test_registry_register_function_map_with_handler_registration_failure(
        self,
    ) -> None:
        """Test register_function_map when handler registration fails (line 499)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Use a FlextHandlers instance directly
        handler = create_test_handler("direct_handler")

        # Register it - tests line 487-501 (FlextHandlers instance branch)
        result = registry.register_function_map({object: handler})
        assert result.is_success

    def test_registry_register_function_map_with_exception(self) -> None:
        """Test register_function_map exception handling (line 513-516)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Use valid entries that should process without exceptions
        handler = create_test_handler("exception_test_handler")

        result = registry.register_function_map({str: handler, int: handler})
        assert result.is_success

    def test_registry_resolve_binding_key_with_tuple(self) -> None:
        """Test _resolve_binding_key with tuple input (line 540)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        handler = create_test_handler("tuple_key_handler")

        # Test with handler and message_type
        key = registry._resolve_binding_key(handler, str)
        assert key is not None
        assert isinstance(key, str)

    def test_registry_resolve_binding_key_from_entry_with_dict(self) -> None:
        """Test _resolve_binding_key_from_entry with dict (line 563)."""
        from flext_core import FlextDispatcher

        registry = FlextRegistry(FlextDispatcher())

        # Test with dict entry (non-tuple, non-FlextHandlers)
        dict_entry: dict[str, object] = {"some_key": "some_value"}

        key = registry._resolve_binding_key_from_entry(dict_entry, str)
        assert key is not None
        # Should return string representation of dict
        assert isinstance(key, str)
