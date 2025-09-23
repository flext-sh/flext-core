"""Test suite for FlextContext.HandlerExecutionContext companion module.

Extracted during FlextHandlers refactoring to ensure 100% coverage
of execution context management and timing logic.
"""

from __future__ import annotations

import math
import time
from unittest.mock import patch

from flext_core import FlextContext


class TestHandlerExecutionContext:
    """Test suite for FlextContext.HandlerExecutionContext companion module."""

    def test_initialization_with_handler_info(self) -> None:
        """Test HandlerExecutionContext initialization with handler information."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        assert context.handler_name == "TestHandler"
        assert context.handler_mode == "command"
        assert context.get_execution_time_ms() == 0.0  # Not started yet
        assert context.get_metrics_state() == {}  # Empty initial state

    def test_initialization_with_query_mode(self) -> None:
        """Test HandlerExecutionContext initialization with query mode."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="QueryHandler", handler_mode="query"
        )

        assert context.handler_name == "QueryHandler"
        assert context.handler_mode == "query"

    def test_start_execution_sets_start_time(self) -> None:
        """Test that start_execution sets the start time."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        with patch("time.time") as mock_time:
            mock_time.return_value = 1234567890.123
            context.start_execution()

        # Verify execution started by checking time is not 0
        execution_time = context.get_execution_time_ms()
        assert execution_time >= 0.0

    def test_get_execution_time_ms_before_start(self) -> None:
        """Test get_execution_time_ms returns 0 when not started."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        execution_time = context.get_execution_time_ms()
        assert execution_time == 0.0

    def test_get_execution_time_ms_after_start(self) -> None:
        """Test get_execution_time_ms calculates correct time after start."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        with patch("time.time") as mock_time:
            # Mock start time
            mock_time.return_value = 1000.0
            context.start_execution()

            # Mock current time (100ms later)
            mock_time.return_value = 1000.1
            execution_time = context.get_execution_time_ms()

        assert execution_time == 100.0

    def test_get_execution_time_ms_with_fractional_seconds(self) -> None:
        """Test get_execution_time_ms with fractional seconds."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        with patch("time.time") as mock_time:
            # Mock start time
            mock_time.return_value = 1000.0
            context.start_execution()

            # Mock current time (42.567ms later)
            mock_time.return_value = 1000.042567
            execution_time = context.get_execution_time_ms()

        assert execution_time == 42.57  # Rounded to 2 decimal places

    def test_get_execution_time_ms_rounding(self) -> None:
        """Test that get_execution_time_ms rounds to 2 decimal places."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        with patch("time.time") as mock_time:
            # Mock start time
            mock_time.return_value = 1000.0
            context.start_execution()

            # Mock current time (1.23456789ms later)
            mock_time.return_value = 1000.00123456789
            execution_time = context.get_execution_time_ms()

        assert execution_time == 1.23  # Rounded to 2 decimal places

    def test_get_metrics_state_initial_empty(self) -> None:
        """Test that get_metrics_state returns empty dict initially."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        metrics_state = context.get_metrics_state()
        assert metrics_state == {}
        assert context.get_metrics_state() == {}

    def test_get_metrics_state_lazy_initialization(self) -> None:
        """Test that get_metrics_state initializes metrics_state lazily."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        # Initially empty dict (lazy initialization)
        metrics_state = context.get_metrics_state()
        assert metrics_state == {}

        # Should return the same instance
        metrics_state2 = context.get_metrics_state()
        assert metrics_state is metrics_state2

    def test_get_metrics_state_returns_same_instance(self) -> None:
        """Test that get_metrics_state returns the same instance."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        metrics_state1 = context.get_metrics_state()
        metrics_state2 = context.get_metrics_state()
        assert metrics_state1 is metrics_state2

    def test_set_metrics_state(self) -> None:
        """Test setting metrics state."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        test_state: dict[str, object] = {
            "counter": 42,
            "status": "processing",
            "data": [1, 2, 3],
        }
        context.set_metrics_state(test_state)

        assert context.get_metrics_state() == test_state

    def test_set_metrics_state_overwrites_existing(self) -> None:
        """Test that set_metrics_state overwrites existing state."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        # Set initial state
        initial_state: dict[str, object] = {"key1": "value1"}
        context.set_metrics_state(initial_state)
        assert context.get_metrics_state() == initial_state

        # Overwrite with new state
        new_state: dict[str, object] = {"key2": "value2", "key3": "value3"}
        context.set_metrics_state(new_state)
        assert context.get_metrics_state() == new_state

    def test_reset_clears_start_time(self) -> None:
        """Test that reset clears start time."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        with patch("time.time") as mock_time:
            mock_time.return_value = 1234567890.123
            context.start_execution()

        # Verify execution started
        execution_time = context.get_execution_time_ms()
        assert execution_time >= 0.0

        context.reset()
        assert context.get_execution_time_ms() == 0.0

    def test_reset_clears_metrics_state(self) -> None:
        """Test that reset clears metrics state."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        # Set some metrics state
        test_state: dict[str, object] = {"key": "value"}
        context.set_metrics_state(test_state)
        assert context.get_metrics_state() == test_state

        context.reset()
        assert context.get_metrics_state() == {}

    def test_reset_complete_state_clearing(self) -> None:
        """Test that reset clears all state completely."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        # Set up context with state
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            context.start_execution()

        test_data: dict[str, object] = {"test": "data"}
        context.set_metrics_state(test_data)

        # Verify state is set
        execution_time = context.get_execution_time_ms()
        assert execution_time >= 0.0
        assert context.get_metrics_state() == {"test": "data"}

        # Reset and verify all state is cleared
        context.reset()
        assert context.get_execution_time_ms() == 0.0
        assert context.get_metrics_state() == {}

    def test_create_for_handler_class_method(self) -> None:
        """Test the create_for_handler class method."""
        context = FlextContext.HandlerExecutionContext.create_for_handler(
            handler_name="CreatedHandler", handler_mode="query"
        )

        assert isinstance(context, FlextContext.HandlerExecutionContext)
        assert context.handler_name == "CreatedHandler"
        assert context.handler_mode == "query"
        assert context.get_execution_time_ms() == 0.0
        assert context.get_metrics_state() == {}

    def test_create_for_handler_command_mode(self) -> None:
        """Test create_for_handler with command mode."""
        context = FlextContext.HandlerExecutionContext.create_for_handler(
            handler_name="CommandHandler", handler_mode="command"
        )

        assert context.handler_name == "CommandHandler"
        assert context.handler_mode == "command"

    def test_create_for_handler_returns_new_instance(self) -> None:
        """Test that create_for_handler returns new instances."""
        context1 = FlextContext.HandlerExecutionContext.create_for_handler(
            handler_name="Handler1", handler_mode="command"
        )
        context2 = FlextContext.HandlerExecutionContext.create_for_handler(
            handler_name="Handler2", handler_mode="query"
        )

        assert context1 is not context2
        assert context1.handler_name != context2.handler_name
        assert context1.handler_mode != context2.handler_mode


class TestHandlerExecutionContextIntegration:
    """Integration tests for HandlerExecutionContext with realistic scenarios."""

    def test_full_execution_lifecycle(self) -> None:
        """Test complete execution lifecycle with timing."""
        context = FlextContext.HandlerExecutionContext.create_for_handler(
            handler_name="IntegrationHandler", handler_mode="command"
        )

        # Start execution
        context.start_execution()

        # Simulate some processing time
        time.sleep(0.01)  # Sleep for 10ms

        # Get execution time
        execution_time = context.get_execution_time_ms()
        assert execution_time >= 10.0  # Should be at least 10ms
        assert (
            execution_time < 100.0
        )  # But not too much more (allowing for test variance)

        # Reset and verify
        context.reset()
        assert context.get_execution_time_ms() == 0.0

    def test_metrics_state_management_workflow(self) -> None:
        """Test realistic metrics state management workflow."""
        context = FlextContext.HandlerExecutionContext.create_for_handler(
            handler_name="MetricsHandler", handler_mode="query"
        )

        # Initialize with processing state
        initial_state: dict[str, object] = {
            "phase": "validation",
            "items_processed": 0,
            "errors": [],
        }
        context.set_metrics_state(initial_state)

        # Update state during processing
        state = context.get_metrics_state()
        state["phase"] = "execution"
        state["items_processed"] = 42
        state["errors"] = ["validation_error"]

        # Verify state is preserved
        updated_state = context.get_metrics_state()
        assert updated_state["phase"] == "execution"
        assert updated_state["items_processed"] == 42
        assert updated_state["errors"] == ["validation_error"]

    def test_concurrent_context_isolation(self) -> None:
        """Test that different contexts are isolated from each other."""
        context1 = FlextContext.HandlerExecutionContext.create_for_handler(
            handler_name="Handler1", handler_mode="command"
        )
        context2 = FlextContext.HandlerExecutionContext.create_for_handler(
            handler_name="Handler2", handler_mode="query"
        )

        # Set different states
        state1: dict[str, object] = {"handler": "first"}
        state2: dict[str, object] = {"handler": "second"}
        context1.set_metrics_state(state1)
        context2.set_metrics_state(state2)

        # Start timing at different times
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            context1.start_execution()

            mock_time.return_value = 2000.0
            context2.start_execution()

            # Check isolated timing
            mock_time.return_value = 1000.05  # 50ms for context1
            context1_time = context1.get_execution_time_ms()

            mock_time.return_value = 2000.03  # 30ms for context2
            context2_time = context2.get_execution_time_ms()

        assert context1_time == 50.0
        assert context2_time == 30.0

        # Check isolated state
        assert context1.get_metrics_state()["handler"] == "first"
        assert context2.get_metrics_state()["handler"] == "second"


class TestHandlerExecutionContextEdgeCases:
    """Test edge cases and error conditions for HandlerExecutionContext."""

    def test_empty_handler_name(self) -> None:
        """Test initialization with empty handler name."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="", handler_mode="command"
        )

        assert not context.handler_name
        assert context.handler_mode == "command"

    def test_empty_handler_mode(self) -> None:
        """Test initialization with empty handler mode."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode=""
        )

        assert context.handler_name == "TestHandler"
        assert not context.handler_mode

    def test_special_characters_in_names(self) -> None:
        """Test initialization with special characters in names."""
        special_chars = "!@#$%^&*()_+{}|:<>?[]\\;'\",./"
        context = FlextContext.HandlerExecutionContext(
            handler_name=f"Handler_{special_chars}",
            handler_mode=f"mode_{special_chars}",
        )

        assert context.handler_name == f"Handler_{special_chars}"
        assert context.handler_mode == f"mode_{special_chars}"

    def test_multiple_start_execution_calls(self) -> None:
        """Test calling start_execution multiple times."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        with patch("time.time") as mock_time:
            # First start
            mock_time.return_value = 1000.0
            context.start_execution()
            first_execution_time = context.get_execution_time_ms()

            # Second start (should overwrite)
            mock_time.return_value = 2000.0
            context.start_execution()
            second_execution_time = context.get_execution_time_ms()

        # Both should show some execution time
        assert first_execution_time >= 0.0
        assert second_execution_time >= 0.0

    def test_set_metrics_state_with_none(self) -> None:
        """Test setting metrics state to None."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        # Set initial state
        initial_state: dict[str, object] = {"key": "value"}
        context.set_metrics_state(initial_state)
        assert context.get_metrics_state() == {"key": "value"}

        # Set to empty dict (equivalent to None for lazy init)
        empty_state: dict[str, object] = {}
        context.set_metrics_state(empty_state)
        assert context.get_metrics_state() == {}

    def test_set_metrics_state_with_complex_data(self) -> None:
        """Test setting metrics state with complex data structures."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        complex_state: dict[str, object] = {
            "nested": {"deep": {"value": 42}},
            "list": [1, 2, {"nested_in_list": True}],
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
            "none_value": None,
            "bool_value": True,
            "float_value": math.pi,
        }

        context.set_metrics_state(complex_state)
        retrieved_state = context.get_metrics_state()

        assert retrieved_state == complex_state
        # Access nested values safely with proper type checking
        # retrieved_state is already typed as dict[str, object] from get_metrics_state()
        nested_dict = retrieved_state.get("nested")
        if isinstance(nested_dict, dict):
            deep_dict = nested_dict.get("deep")
            if isinstance(deep_dict, dict):
                nested_value = deep_dict.get("value")
                assert nested_value == 42

        list_item = retrieved_state.get("list")
        if isinstance(list_item, list) and len(list_item) > 2:
            third_item = list_item[2]
            if isinstance(third_item, dict):
                assert third_item.get("nested_in_list") is True

    def test_execution_time_with_zero_start_time(self) -> None:
        """Test execution time calculation when start time is 0."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        with patch("time.time") as mock_time:
            # Start at time 0
            mock_time.return_value = 0.0
            context.start_execution()

            # Current time 1.5 seconds
            mock_time.return_value = 1.5
            execution_time = context.get_execution_time_ms()

        assert execution_time == 1500.0

    def test_reset_multiple_times(self) -> None:
        """Test calling reset multiple times in a row."""
        context = FlextContext.HandlerExecutionContext(
            handler_name="TestHandler", handler_mode="command"
        )

        # Set up state
        context.start_execution()
        test_state: dict[str, object] = {"key": "value"}
        context.set_metrics_state(test_state)

        # Reset multiple times
        context.reset()
        context.reset()
        context.reset()

        # Should still be in clean state
        assert context.get_execution_time_ms() == 0.0
        assert context.get_metrics_state() == {}
