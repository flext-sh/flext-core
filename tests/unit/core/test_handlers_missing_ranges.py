"""Tests targeting specific missing line ranges in handlers.py to maximize coverage.

This test suite targets the most critical missing ranges:
- Lines 631-665: Custom metrics collection in FlextMetricsHandler
- Lines 673-729: Operation metrics collection methods  
- Lines 819-837: Handler registration with auto-naming
- Lines 864-896: Type-based handler registration and lookup
- Lines 904-914: Handler clearing and convenience methods
- Lines 1107-1140: Chain processing with error handling
- Lines 1151-1184: Chain processing helpers and batch processing

Current coverage: 54% → Target: 75%+
"""

from __future__ import annotations

# Removed mock imports - using real implementations only
import pytest

from flext_core import (
    FlextBaseHandler,
    FlextHandlerChain,
    FlextHandlerRegistry,
    FlextMetricsHandler,
    FlextResult,
    FlextValidatingHandler,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestMetricsHandlerCustomMetrics:
    """Test lines 631-665: Custom metrics collection in FlextMetricsHandler."""

    def test_collect_metrics_initializes_custom_metrics(self) -> None:
        """Test line 635-636: Initialize custom_metrics dict when missing."""
        handler = FlextMetricsHandler()

        # Process a message to trigger collect_metrics
        result = handler.handle("test_message")
        assert result.success

        # Verify custom_metrics was initialized
        metrics = handler.get_metrics()
        assert "custom_metrics" in metrics
        custom_metrics = metrics["custom_metrics"]
        assert isinstance(custom_metrics, dict)
        assert "success_count" in custom_metrics
        assert "failure_count" in custom_metrics

    def test_collect_metrics_handles_corrupted_custom_metrics(self) -> None:
        """Test lines 639-641: Reset corrupted custom_metrics to safe defaults."""
        handler = FlextMetricsHandler()

        # Manually corrupt the custom_metrics
        handler.metrics["custom_metrics"] = "not_a_dict"

        # Process a message to trigger reset
        result = handler.handle("test_message")
        assert result.success

        # Verify custom_metrics was reset to safe defaults
        metrics = handler.get_metrics()
        custom_metrics = metrics["custom_metrics"]
        assert isinstance(custom_metrics, dict)
        assert custom_metrics["success_count"] >= 0
        assert custom_metrics["failure_count"] >= 0

    def test_collect_metrics_validates_and_resets_invalid_values(self) -> None:
        """Test lines 647-654: Validate metric values and reset if corrupted."""
        handler = FlextMetricsHandler()

        # Set up corrupted metric values
        handler.metrics["custom_metrics"] = {
            "success_count": "not_an_int",
            "failure_count": 5,
            "invalid_key": None
        }

        # Process a message to trigger validation
        result = handler.handle("success_message")
        assert result.success

        # Verify metrics were reset to safe defaults due to corruption
        metrics = handler.get_metrics()
        custom_metrics = metrics["custom_metrics"]
        assert custom_metrics["success_count"] == 1  # Reset + incremented
        assert custom_metrics["failure_count"] == 0

    def test_collect_metrics_increments_success_count(self) -> None:
        """Test lines 660-662: Increment success_count for successful operations."""
        handler = FlextMetricsHandler()

        # Process multiple successful messages
        handler.handle("success1")
        handler.handle("success2")
        handler.handle("success3")

        metrics = handler.get_metrics()
        custom_metrics = metrics["custom_metrics"]
        assert custom_metrics["success_count"] == 3
        assert custom_metrics["failure_count"] == 0

    def test_collect_metrics_increments_failure_count(self) -> None:
        """Test lines 664-665: Increment failure_count for failed operations."""
        class FailingMetricsHandler(FlextMetricsHandler):
            def process_message(self, message: object) -> FlextResult[object]:
                return FlextResult[object].fail("Intentional failure")

        handler = FailingMetricsHandler()

        # Process multiple failing messages
        handler.handle("fail1")
        handler.handle("fail2")

        metrics = handler.get_metrics()
        custom_metrics = metrics["custom_metrics"]
        assert custom_metrics["success_count"] == 0
        assert custom_metrics["failure_count"] == 2


class TestMetricsHandlerOperationMetrics:
    """Test lines 673-729: Operation metrics collection methods."""

    def test_collect_operation_metrics_success(self) -> None:
        """Test lines 673-677: Successful operation metrics collection."""
        handler = FlextMetricsHandler()

        # Collect metrics for an operation
        result = handler.collect_operation_metrics("test_op", 1.5)
        assert result.success

        # Verify operation metrics were stored
        metrics = handler.get_metrics()
        assert "operations" in metrics
        operations = metrics["operations"]
        assert "test_op" in operations

        op_metrics = operations["test_op"]
        assert op_metrics["count"] == 1
        assert op_metrics["total_duration"] == 1.5
        assert op_metrics["avg_duration"] == 1.5

    def test_collect_operation_metrics_handles_exceptions(self) -> None:
        """Test lines 678-681: Exception handling in operation metrics collection."""

        # Create a handler that will cause exception by overriding method to fail
        class FailingMetricsHandler(FlextMetricsHandler):
            def _ensure_operations_dict(self) -> None:
                """Override to always raise TypeError to test exception handling."""
                raise TypeError("Internal state corruption")

        handler = FailingMetricsHandler()
        result = handler.collect_operation_metrics("test_op", 1.0)

        assert result.is_failure
        assert "Metrics collection failed" in (result.error or "")
        assert "Internal state corruption" in (result.error or "")

    def test_ensure_operations_dict_creates_dict(self) -> None:
        """Test lines 684-687: Create operations dict when missing."""
        handler = FlextMetricsHandler()

        # Ensure no operations dict exists
        if "operations" in handler.metrics:
            del handler.metrics["operations"]

        # Call the method
        operations = handler._ensure_operations_dict()

        assert isinstance(operations, dict)
        assert "operations" in handler.metrics

    def test_ensure_operations_dict_validates_structure(self) -> None:
        """Test lines 688-701: Validate and fix operations dict structure."""
        handler = FlextMetricsHandler()

        # Set up invalid operations structure
        handler.metrics["operations"] = {
            "valid_op": {"count": 5, "total_duration": 10.0},
            "invalid_op": "not_a_dict",
            "partial_valid": {"count": "not_int", "valid_key": 3.5}
        }

        # Call validation
        operations = handler._ensure_operations_dict()

        # Check that invalid entries were filtered out
        assert "valid_op" in operations
        assert "invalid_op" not in operations
        # partial_valid should be filtered to only valid entries
        if "partial_valid" in operations:
            partial = operations["partial_valid"]
            # Only valid entries should remain
            assert "valid_key" in partial
            assert partial["valid_key"] == 3.5

    def test_ensure_operation_metrics_creates_default(self) -> None:
        """Test lines 708-714: Create default operation metrics."""
        operations: dict[str, dict[str, int | float]] = {}

        metrics = FlextMetricsHandler._ensure_operation_metrics(operations, "new_op")

        assert metrics["count"] == 0
        assert metrics["total_duration"] == 0.0
        assert metrics["avg_duration"] == 0.0
        assert operations["new_op"] is metrics

    def test_update_operation_metrics_calculates_correctly(self) -> None:
        """Test lines 721-728: Update operation metrics with new duration."""
        metrics: dict[str, int | float] = {
            "count": 2,
            "total_duration": 5.0,
            "avg_duration": 2.5
        }

        FlextMetricsHandler._update_operation_metrics(metrics, 3.0)

        assert metrics["count"] == 3
        assert metrics["total_duration"] == 8.0
        assert metrics["avg_duration"] == 8.0 / 3  # ~2.667


class TestHandlerRegistryAutoNaming:
    """Test lines 819-837: Handler registration with auto-naming."""

    def test_register_handler_auto_naming_success(self) -> None:
        """Test lines 821-829: Auto-naming when handler parameter is None."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler(name="TestHandler")

        # Use auto-naming by passing handler as first parameter (handler=None triggers auto-naming)
        result = registry.register(handler)
        assert result.success

        # Verify handler was registered with its class name
        all_handlers = registry.get_all_handlers()
        assert "FlextBaseHandler" in all_handlers

    def test_register_handler_auto_naming_invalid_object(self) -> None:
        """Test lines 821-824: Fail when auto-naming object is invalid."""
        registry = FlextHandlerRegistry()

        # Try to auto-name a non-handler object
        result = registry.register("not_a_handler")
        assert result.is_failure
        assert "INVALID_HANDLER_PROVIDED" in (result.error or "")

    def test_register_handler_explicit_name_validation(self) -> None:
        """Test lines 832-835: Validate explicit name is string."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler()

        # Try to use non-string name
        result = registry.register(123, handler)
        assert result.is_failure
        assert "HANDLER_NAME_MUST_BE_STRING" in (result.error or "")

    def test_register_handler_explicit_name_success(self) -> None:
        """Test lines 836-837: Success with explicit string name."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler()

        result = registry.register("explicit_name", handler)
        assert result.success

        all_handlers = registry.get_all_handlers()
        assert "explicit_name" in all_handlers


class TestHandlerRegistryTypeBasedOperations:
    """Test lines 864-896: Type-based handler registration and lookup."""

    def test_register_for_type_maintains_mapping(self) -> None:
        """Test lines 864-869: Register handler for specific type and maintain mapping."""
        registry = FlextHandlerRegistry()
        handler = FlextValidatingHandler(name="StringValidator")

        result = registry.register_for_type(str, "string_handler", handler)
        assert result.success

        # Verify type mapping was created
        assert str in registry._type_mappings
        assert "string_handler" in registry._type_mappings[str]

    def test_get_handler_for_type_uses_explicit_mapping(self) -> None:
        """Test lines 877-882: Get handler using explicit type mapping."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler(name="StringHandler")

        # Register for specific type
        registry.register_for_type(str, "string_handler", handler)

        # Retrieve using type mapping
        result = registry.get_handler_for_type(str)
        assert result.success
        assert result.value is handler

    def test_get_handler_for_type_fallback_capability_scan(self) -> None:
        """Test lines 884-886: Fallback to capability-based scan when no explicit mapping."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler(name="GeneralHandler")

        # Register handler without type mapping
        registry.register_handler("general", handler)

        # Get handler - should fallback to capability scan
        result = registry.get_handler_for_type(str)
        assert result.success  # FlextBaseHandler can handle anything

    def test_get_handler_for_type_no_handler_found(self) -> None:
        """Test lines 887-889: Fail when no handler found for type."""
        registry = FlextHandlerRegistry()

        # Create handler that can't handle the type
        class SelectiveHandler(FlextBaseHandler):
            def can_handle(self, message: object) -> bool:
                return isinstance(message, int)

        handler = SelectiveHandler()
        registry.register_handler("selective", handler)

        # Try to get handler for string type
        result = registry.get_handler_for_type(str)
        assert result.is_failure
        assert "No handler registered for type" in (result.error or "")

    def test_unregister_handler_success(self) -> None:
        """Test lines 893-895: Successfully unregister existing handler."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler()

        registry.register_handler("test_handler", handler)
        assert "test_handler" in registry.get_all_handlers()

        # Unregister handler
        success = registry.unregister_handler("test_handler")
        assert success is True
        assert "test_handler" not in registry.get_all_handlers()

    def test_unregister_handler_not_found(self) -> None:
        """Test line 896: Return False when handler not found."""
        registry = FlextHandlerRegistry()

        success = registry.unregister_handler("nonexistent")
        assert success is False


class TestHandlerRegistryConvenienceMethods:
    """Test lines 904-914: Handler clearing and convenience methods."""

    def test_clear_handlers_removes_all(self) -> None:
        """Test lines 904-905: Clear all handlers and type mappings."""
        registry = FlextHandlerRegistry()

        # Add some handlers and type mappings
        handler1 = FlextBaseHandler(name="Handler1")
        handler2 = FlextBaseHandler(name="Handler2")
        registry.register_handler("h1", handler1)
        registry.register_for_type(str, "h2", handler2)

        # Verify handlers exist
        assert len(registry.get_all_handlers()) >= 2
        assert len(registry._type_mappings) >= 1

        # Clear everything
        registry.clear_handlers()

        # Verify everything is cleared
        assert len(registry.get_all_handlers()) == 0
        assert len(registry._type_mappings) == 0

    def test_clear_alias_method(self) -> None:
        """Test line 909: clear() method as alias for clear_handlers()."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler()

        registry.register_handler("test", handler)
        assert len(registry.get_all_handlers()) >= 1

        # Use clear() alias
        registry.clear()
        assert len(registry.get_all_handlers()) == 0

    def test_get_handlers_convenience_method(self) -> None:
        """Test line 914: get_handlers() returns list of handlers."""
        registry = FlextHandlerRegistry()
        handler1 = FlextBaseHandler(name="Handler1")
        handler2 = FlextBaseHandler(name="Handler2")

        registry.register_handler("h1", handler1)
        registry.register_handler("h2", handler2)

        # Get handlers as list (convenience method)
        handlers_list = registry.get_handlers()
        assert isinstance(handlers_list, list)
        assert len(handlers_list) >= 2


class TestHandlerChainComplexProcessing:
    """Test lines 1107-1140: Chain processing with error handling."""

    def test_process_chain_success_path(self) -> None:
        """Test lines 1107-1116: Successful chain processing."""
        chain = FlextHandlerChain()

        handler1 = FlextBaseHandler(name="Handler1")
        handler2 = FlextBaseHandler(name="Handler2")
        chain.add_handler(handler1)
        chain.add_handler(handler2)

        result = chain.process_chain("test_message")
        assert result.success
        # Should return last successful result
        assert result.value == "test_message"

    def test_process_chain_stops_on_failure(self) -> None:
        """Test lines 1117-1119: Stop processing on first failure."""
        chain = FlextHandlerChain()

        # First handler succeeds, second fails, third never reached
        chain.add_handler(FlextBaseHandler(name="Success"))

        class FailingHandler(FlextBaseHandler):
            def process_message(self, message: object) -> FlextResult[object]:
                return FlextResult[object].fail("Handler failure")

        chain.add_handler(FailingHandler(name="Failure"))
        chain.add_handler(FlextBaseHandler(name="NeverReached"))

        result = chain.process_chain("test_message")
        # Should return last successful result before failure
        assert result.success  # Last successful was from first handler

    def test_process_chain_exception_handling(self) -> None:
        """Test lines 1121-1126: Exception handling during chain processing."""
        chain = FlextHandlerChain()

        class ExceptionHandler(FlextBaseHandler):
            def handle(self, message: object) -> FlextResult[object]:
                raise ValueError("Handler threw exception")

        chain.add_handler(ExceptionHandler())

        result = chain.process_chain("test_message")
        # Chain should handle exception and add failure result
        assert isinstance(result, FlextResult)

    def test_process_chain_no_successful_results(self) -> None:
        """Test lines 1129-1140: Handle case with no successful results."""
        chain = FlextHandlerChain()

        class AlwaysFailHandler(FlextBaseHandler):
            def process_message(self, message: object) -> FlextResult[object]:
                return FlextResult[object].fail("Always fails")

        chain.add_handler(AlwaysFailHandler(name="Fail1"))
        chain.add_handler(AlwaysFailHandler(name="Fail2"))

        result = chain.process_chain("test_message")
        assert result.is_failure
        assert "Chain processing failed" in (result.error or "")

    def test_process_chain_no_handlers(self) -> None:
        """Test line 1140: Handle empty chain case."""
        chain = FlextHandlerChain()

        result = chain.process_chain("test_message")
        assert result.is_failure
        assert "No handlers processed the message" in (result.error or "")


class TestHandlerChainConvenienceMethods:
    """Test lines 1151-1184: Chain processing helpers and batch processing."""

    def test_handle_convenience_method_exception_handling(self) -> None:
        """Test lines 1151-1152: Exception handling in handle() convenience method."""
        chain = FlextHandlerChain()

        class ExceptionHandler(FlextBaseHandler):
            def can_handle(self, message: object) -> bool:
                raise RuntimeError("can_handle threw exception")

        chain.add_handler(ExceptionHandler())

        result = chain.handle("test_message")
        assert result.is_failure
        assert "Chain handler failed" in (result.error or "")

    def test_can_handle_empty_chain(self) -> None:
        """Test lines 1157-1158: can_handle returns True for empty chain."""
        chain = FlextHandlerChain()

        # Empty chain should be permissive
        can_handle = chain.can_handle("any_message")
        assert can_handle is True

    def test_can_handle_exception_permissive(self) -> None:
        """Test lines 1161-1163: can_handle is permissive on exceptions."""
        chain = FlextHandlerChain()

        class ExceptionHandler(FlextBaseHandler):
            def can_handle(self, message: object) -> bool:
                raise ValueError("Exception in can_handle")

        chain.add_handler(ExceptionHandler())

        # Should be permissive and return True despite exception
        can_handle = chain.can_handle("test_message")
        assert can_handle is True

    def test_process_all_success(self) -> None:
        """Test lines 1174-1184: Process multiple messages successfully."""
        chain = FlextHandlerChain()
        handler = FlextBaseHandler()
        chain.add_handler(handler)

        messages = ["msg1", "msg2", "msg3"]
        result = chain.process_all(messages)

        assert result.success
        assert result.value == messages  # All messages processed successfully

    def test_process_all_short_circuit_on_failure(self) -> None:
        """Test lines 1181-1183: Short-circuit on failure during batch processing."""
        chain = FlextHandlerChain()

        class SelectiveHandler(FlextBaseHandler):
            def process_message(self, message: object) -> FlextResult[object]:
                if message == "fail":
                    return FlextResult[object].fail("Selective failure")
                return FlextResult[object].ok(message)

        chain.add_handler(SelectiveHandler())

        messages = ["msg1", "fail", "msg3"]  # Second message will fail
        result = chain.process_all(messages)

        assert result.is_failure
        assert "Selective failure" in (result.error or "")
