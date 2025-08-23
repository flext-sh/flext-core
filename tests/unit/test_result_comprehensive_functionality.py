"""Comprehensive tests for FlextResult advanced functionality - Real functional testing."""

from __future__ import annotations

import operator

import pytest

from flext_core import FlextResult

# Test markers
pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextResultAdvancedOperations:
    """Test advanced FlextResult operations with comprehensive real functionality."""

    def test_chain_results_empty_list(self) -> None:
        """Test chain_results with empty input returns empty success."""
        result = FlextResult.chain_results()
        assert result.is_success
        assert result.value == []

    def test_chain_results_multiple_success(self) -> None:
        """Test chain_results with multiple successful results."""
        result1 = FlextResult[str].ok("first")
        result2 = FlextResult[int].ok(42)
        result3 = FlextResult[bool].ok(True)

        chained = FlextResult.chain_results(result1, result2, result3)

        assert chained.is_success
        assert chained.value == ["first", 42, True]

    def test_chain_results_failure_short_circuits(self) -> None:
        """Test chain_results fails fast on first failure."""
        result1 = FlextResult[str].ok("success")
        result2 = FlextResult[int].fail("middle failure")
        result3 = FlextResult[bool].ok(True)  # Should not be reached

        chained = FlextResult.chain_results(result1, result2, result3)

        assert chained.is_failure
        assert chained.error == "middle failure"

    def test_chain_results_first_failure(self) -> None:
        """Test chain_results with first result failing."""
        result1 = FlextResult[str].fail("first error")
        result2 = FlextResult[int].ok(42)

        chained = FlextResult.chain_results(result1, result2)

        assert chained.is_failure
        assert chained.error == "first error"

    def test_map_with_working_transformation(self) -> None:
        """Test map successfully transforms data."""
        result = FlextResult[int].ok(42)

        # Working transformation
        mapped = result.map(str)
        assert mapped.is_success
        assert mapped.value == "42"

    def test_map_on_failure_propagates_error(self) -> None:
        """Test map on failure result propagates the error."""
        result = FlextResult[int].fail("original error")

        # Map should not execute on failure and propagate error
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_failure
        assert mapped.error == "original error"

    def test_map_complex_transformation(self) -> None:
        """Test map with complex data transformation."""
        result = FlextResult[dict[str, int]].ok({"a": 1, "b": 2})

        # Transform dictionary to list of tuples
        mapped = result.map(lambda d: [(k, v) for k, v in d.items()])
        assert mapped.is_success
        assert mapped.value == [("a", 1), ("b", 2)]

    def test_flat_map_successful_chaining(self) -> None:
        """Test flat_map successfully chains operations."""
        result = FlextResult[str].ok("10")

        def parse_int(s: str) -> FlextResult[int]:
            try:
                return FlextResult[int].ok(int(s))
            except ValueError:
                return FlextResult[int].fail("Not a valid integer")

        mapped = result.flat_map(parse_int)
        assert mapped.is_success
        assert mapped.value == 10

    def test_flat_map_chain_failure_propagation(self) -> None:
        """Test flat_map propagates failure from chained operation."""
        result = FlextResult[str].ok("not_a_number")

        def parse_int(s: str) -> FlextResult[int]:
            try:
                return FlextResult[int].ok(int(s))
            except ValueError:
                return FlextResult[int].fail("Not a valid integer")

        mapped = result.flat_map(parse_int)
        assert mapped.is_failure
        assert mapped.error == "Not a valid integer"

    def test_flat_map_on_failure_propagates_error(self) -> None:
        """Test flat_map on failure result propagates the original error."""
        result = FlextResult[str].fail("original error")

        def parse_int(s: str) -> FlextResult[int]:
            return FlextResult[int].ok(int(s))

        mapped = result.flat_map(parse_int)
        assert mapped.is_failure
        assert mapped.error == "original error"


class TestFlextResultRecoveryOperations:
    """Test FlextResult recovery and error handling operations."""

    def test_recover_from_failure_success(self) -> None:
        """Test recover successfully transforms error to success."""
        result = FlextResult[int].fail("calculation error")

        recovered = result.recover(lambda error: 42)  # Default value

        assert recovered.is_success
        assert recovered.value == 42

    def test_recover_on_success_returns_self(self) -> None:
        """Test recover on successful result returns original result."""
        result = FlextResult[int].ok(10)

        recovered = result.recover(lambda error: 999)

        assert recovered.is_success
        assert recovered.value == 10  # Original value preserved

    def test_recover_exception_handling(self) -> None:
        """Test recover handles exceptions in recovery function."""
        result = FlextResult[int].fail("error")

        def failing_recovery(error: str) -> int:
            msg = "Recovery failed"
            raise ValueError(msg)

        recovered = result.recover(failing_recovery)

        assert recovered.is_failure
        assert "Recovery failed" in recovered.error

    def test_recover_with_success_flow(self) -> None:
        """Test recover_with with successful recovery result."""
        result = FlextResult[int].fail("database error")

        def recovery_with_result(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(100)  # Recovery value

        recovered = result.recover_with(recovery_with_result)

        assert recovered.is_success
        assert recovered.value == 100

    def test_recover_with_failure_flow(self) -> None:
        """Test recover_with with failing recovery result."""
        result = FlextResult[int].fail("original error")

        def failing_recovery(error: str) -> FlextResult[int]:
            return FlextResult[int].fail("recovery also failed")

        recovered = result.recover_with(failing_recovery)

        assert recovered.is_failure
        assert recovered.error == "recovery also failed"

    def test_recover_with_exception_handling(self) -> None:
        """Test recover_with handles exceptions in recovery function."""
        result = FlextResult[int].fail("error")

        def failing_recovery_with_exception(error: str) -> FlextResult[int]:
            msg = "Recovery function crashed"
            raise TypeError(msg)

        recovered = result.recover_with(failing_recovery_with_exception)

        assert recovered.is_failure
        assert "Recovery function crashed" in recovered.error


class TestFlextResultSideEffectOperations:
    """Test FlextResult side-effect operations (tap, tap_error)."""

    def test_tap_executes_on_success_with_data(self) -> None:
        """Test tap executes side effect on successful result."""
        result = FlextResult[str].ok("test data")
        side_effect_executed = []

        def side_effect(data: str) -> None:
            side_effect_executed.append(data.upper())

        tapped = result.tap(side_effect)

        assert tapped.is_success
        assert tapped.value == "test data"  # Original result preserved
        assert side_effect_executed == ["TEST DATA"]

    def test_tap_ignores_on_failure(self) -> None:
        """Test tap does not execute side effect on failure."""
        result = FlextResult[str].fail("error occurred")
        side_effect_executed = []

        def side_effect(data: str) -> None:
            side_effect_executed.append(data)

        tapped = result.tap(side_effect)

        assert tapped.is_failure
        assert tapped.error == "error occurred"
        assert side_effect_executed == []  # No side effect executed

    def test_tap_handles_side_effect_exceptions(self) -> None:
        """Test tap suppresses exceptions in side effect function."""
        result = FlextResult[str].ok("test")

        def failing_side_effect(data: str) -> None:
            msg = "Side effect failed"
            raise ValueError(msg)

        tapped = result.tap(failing_side_effect)

        # Result should be preserved even if side effect fails
        assert tapped.is_success
        assert tapped.value == "test"

    def test_tap_error_executes_on_failure(self) -> None:
        """Test tap_error executes side effect on failure."""
        result = FlextResult[str].fail("critical error")
        error_log = []

        def log_error(error: str) -> None:
            error_log.append(f"LOGGED: {error}")

        tapped = result.tap_error(log_error)

        assert tapped.is_failure
        assert tapped.error == "critical error"  # Original error preserved
        assert error_log == ["LOGGED: critical error"]

    def test_tap_error_ignores_on_success(self) -> None:
        """Test tap_error does not execute on success."""
        result = FlextResult[str].ok("success data")
        error_log = []

        def log_error(error: str) -> None:
            error_log.append(error)

        tapped = result.tap_error(log_error)

        assert tapped.is_success
        assert tapped.value == "success data"
        assert error_log == []  # No error logging

    def test_tap_error_handles_exceptions(self) -> None:
        """Test tap_error suppresses specific exceptions in error handler."""
        result = FlextResult[str].fail("error")

        def failing_error_handler_with_type_error(error: str) -> None:
            msg = "Error handler failed"
            raise TypeError(msg)

        tapped = result.tap_error(failing_error_handler_with_type_error)

        # Result should be preserved when TypeError is suppressed
        assert tapped.is_failure
        assert tapped.error == "error"

    def test_tap_error_runtime_error_propagation(self) -> None:
        """Test tap_error allows RuntimeError to propagate (not in suppress list)."""
        result = FlextResult[str].fail("error")

        def failing_error_handler_with_runtime_error(error: str) -> None:
            msg = "Error handler failed"
            raise RuntimeError(msg)

        # RuntimeError should propagate since it's not in contextlib.suppress list
        with pytest.raises(RuntimeError, match="Error handler failed"):
            result.tap_error(failing_error_handler_with_runtime_error)


class TestFlextResultFilterOperations:
    """Test FlextResult filtering operations."""

    def test_filter_success_passes_predicate(self) -> None:
        """Test filter passes when predicate is true."""
        result = FlextResult[int].ok(42)

        filtered = result.filter(lambda x: x > 40, "Number too small")

        assert filtered.is_success
        assert filtered.value == 42

    def test_filter_success_fails_predicate(self) -> None:
        """Test filter fails when predicate is false."""
        result = FlextResult[int].ok(10)

        filtered = result.filter(lambda x: x > 40, "Number too small")

        assert filtered.is_failure
        assert filtered.error == "Number too small"

    def test_filter_failure_propagates(self) -> None:
        """Test filter propagates failure without applying predicate."""
        result = FlextResult[int].fail("original error")
        predicate_called = []

        def tracking_predicate(x: int) -> bool:
            predicate_called.append(x)
            return True

        filtered = result.filter(tracking_predicate, "should not see this")

        assert filtered.is_failure
        assert filtered.error == "original error"
        assert predicate_called == []  # Predicate not called on failure

    def test_filter_predicate_exception_handling(self) -> None:
        """Test filter handles exceptions in predicate function."""
        result = FlextResult[str].ok("test")

        def failing_predicate(x: str) -> bool:
            msg = "Predicate failed"
            raise ValueError(msg)

        filtered = result.filter(failing_predicate, "filter error")

        assert filtered.is_failure
        assert "Predicate failed" in filtered.error


class TestFlextResultCombinationOperations:
    """Test FlextResult combination operations."""

    def test_zip_with_both_success(self) -> None:
        """Test zip_with combines two successful results."""
        result1 = FlextResult[int].ok(10)
        result2 = FlextResult[int].ok(5)

        combined = result1.zip_with(result2, operator.add)

        assert combined.is_success
        assert combined.value == 15

    def test_zip_with_first_failure(self) -> None:
        """Test zip_with propagates first result failure."""
        result1 = FlextResult[int].fail("first error")
        result2 = FlextResult[int].ok(5)

        combined = result1.zip_with(result2, operator.add)

        assert combined.is_failure
        assert combined.error == "first error"

    def test_zip_with_second_failure(self) -> None:
        """Test zip_with propagates second result failure."""
        result1 = FlextResult[int].ok(10)
        result2 = FlextResult[int].fail("second error")

        combined = result1.zip_with(result2, operator.add)

        assert combined.is_failure
        assert combined.error == "second error"

    def test_zip_with_none_data_handling(self) -> None:
        """Test zip_with handles None data appropriately."""
        # Create a result that has success state but None data
        result1 = FlextResult[int | None].ok(None)
        result2 = FlextResult[int].ok(5)

        combined = result1.zip_with(result2, lambda x, y: (x, y))

        assert combined.is_failure
        assert "Missing data for zip operation" in combined.error

    def test_zip_with_function_exception(self) -> None:
        """Test zip_with handles exceptions in combination function."""
        result1 = FlextResult[int].ok(10)
        result2 = FlextResult[int].ok(0)

        def failing_function(x: int, y: int) -> float:
            return x / y  # Division by zero

        combined = result1.zip_with(result2, failing_function)

        assert combined.is_failure
        assert (
            "ZeroDivisionError" in combined.error
            or "division by zero" in combined.error
        )


class TestFlextResultUtilityOperations:
    """Test FlextResult utility operations."""

    def test_to_either_success(self) -> None:
        """Test to_either returns (data, None) for success."""
        result = FlextResult[str].ok("success data")

        data, error = result.to_either()

        assert data == "success data"
        assert error is None

    def test_to_either_failure(self) -> None:
        """Test to_either returns (None, error) for failure."""
        result = FlextResult[str].fail("error message")

        data, error = result.to_either()

        assert data is None
        assert error == "error message"

    def test_to_exception_success_returns_none(self) -> None:
        """Test to_exception returns None for successful result."""
        result = FlextResult[str].ok("success")

        exception = result.to_exception()

        assert exception is None

    def test_unwrap_or_success_returns_value(self) -> None:
        """Test unwrap_or returns value for successful result."""
        result = FlextResult[int].ok(42)

        value = result.unwrap_or(999)

        assert value == 42

    def test_unwrap_or_failure_returns_default(self) -> None:
        """Test unwrap_or returns default for failed result."""
        result = FlextResult[int].fail("error")

        value = result.unwrap_or(999)

        assert value == 999

    def test_unwrap_or_with_none_success_data(self) -> None:
        """Test unwrap_or handles None success data correctly."""
        result = FlextResult[int | None].ok(None)

        value = result.unwrap_or(999)

        # Should return None (the success data), not the default
        assert value is None


class TestFlextResultEdgeCases:
    """Test FlextResult edge cases and type narrowing scenarios."""

    def test_none_success_data_handling(self) -> None:
        """Test FlextResult correctly handles None as valid success data."""
        result = FlextResult[str | None].ok(None)

        assert result.is_success
        assert result.value is None

    def test_empty_string_as_success_data(self) -> None:
        """Test FlextResult handles empty string as valid success data."""
        result = FlextResult[str].ok("")

        assert result.is_success
        assert result.value == ""

    def test_zero_as_success_data(self) -> None:
        """Test FlextResult handles zero as valid success data."""
        result = FlextResult[int].ok(0)

        assert result.is_success
        assert result.value == 0

    def test_false_as_success_data(self) -> None:
        """Test FlextResult handles False as valid success data."""
        result = FlextResult[bool].ok(False)

        assert result.is_success
        assert result.value is False

    def test_empty_collection_as_success_data(self) -> None:
        """Test FlextResult handles empty collections as valid success data."""
        result = FlextResult[list[int]].ok([])

        assert result.is_success
        assert result.value == []

    def test_complex_data_structures(self) -> None:
        """Test FlextResult with complex nested data structures."""
        complex_data = {
            "users": [
                {"id": 1, "name": "Alice", "active": True},
                {"id": 2, "name": "Bob", "active": False},
            ],
            "metadata": {"total": 2, "page": 1},
        }

        result = FlextResult[dict[str, object]].ok(complex_data)

        assert result.is_success
        assert result.value == complex_data
        assert isinstance(result.value["users"], list)


class TestFlextResultTypeNarrowing:
    """Test FlextResult type narrowing and type guard functionality."""

    def test_type_guard_success_state(self) -> None:
        """Test type guard correctly identifies success state."""
        result = FlextResult[str].ok("test")

        # Type guard should confirm success state
        assert result._is_success_state(result._data)

    def test_type_guard_failure_state(self) -> None:
        """Test type guard correctly identifies failure state."""
        result = FlextResult[str].fail("error")

        # Type guard should reject failure state
        assert not result._is_success_state(result._data)

    def test_value_access_after_success_check(self) -> None:
        """Test type-safe value access after success check."""
        result = FlextResult[str].ok("typed value")

        if result.is_success:
            # This should be type-safe access
            value = result.value
            assert isinstance(value, str)
            assert value == "typed value"

    def test_value_access_on_failure_raises_exception(self) -> None:
        """Test value access on failure raises appropriate exception."""
        result = FlextResult[str].fail("error occurred")

        with pytest.raises(
            TypeError, match="Attempted to access value on failed result"
        ):
            _ = result.value


class TestFlextResultRailwayOrientedComposition:
    """Test complex railway-oriented programming compositions."""

    def test_complex_railway_composition_success(self) -> None:
        """Test complex railway-oriented programming chain with all successes."""

        def validate_positive(x: int) -> FlextResult[int]:
            return (
                FlextResult[int].ok(x)
                if x > 0
                else FlextResult[int].fail("Not positive")
            )

        def double_value(x: int) -> int:
            return x * 2

        def format_result(x: int) -> str:
            return f"Result: {x}"

        result = (
            FlextResult[int]
            .ok(5)
            .flat_map(validate_positive)
            .map(double_value)
            .map(format_result)
        )

        assert result.is_success
        assert result.value == "Result: 10"

    def test_complex_railway_composition_with_failure(self) -> None:
        """Test complex railway-oriented programming chain with failure."""

        def validate_positive(x: int) -> FlextResult[int]:
            return (
                FlextResult[int].ok(x)
                if x > 0
                else FlextResult[int].fail("Not positive")
            )

        def double_value(x: int) -> int:
            return x * 2

        def format_result(x: int) -> str:
            return f"Result: {x}"

        result = (
            FlextResult[int]
            .ok(-3)  # Negative number
            .flat_map(validate_positive)  # This will fail
            .map(double_value)  # Should be skipped
            .map(format_result)  # Should be skipped
        )

        assert result.is_failure
        assert result.error == "Not positive"

    def test_railway_with_recovery_composition(self) -> None:
        """Test railway composition with recovery operations."""

        def risky_operation(x: int) -> FlextResult[int]:
            return (
                FlextResult[int].fail("Operation failed")
                if x == 0
                else FlextResult[int].ok(x * 10)
            )

        result = (
            FlextResult[int]
            .ok(0)  # Will cause failure
            .flat_map(risky_operation)
            .recover(lambda error: 42)  # Recovery value
            .map(lambda x: x + 8)  # Should work on recovered value
        )

        assert result.is_success
        assert result.value == 50  # 42 + 8

    def test_railway_with_side_effects_and_filtering(self) -> None:
        """Test railway composition with side effects and filtering."""
        processed_values = []

        def log_processing(x: int) -> None:
            processed_values.append(f"Processing {x}")

        result = (
            FlextResult[int]
            .ok(15)
            .tap(log_processing)  # Side effect
            .filter(lambda x: x > 10, "Value too small")  # Filter
            .map(lambda x: x * 2)  # Transform
            .tap(lambda x: processed_values.append(f"Final: {x}"))  # Final side effect
        )

        assert result.is_success
        assert result.value == 30
        assert processed_values == ["Processing 15", "Final: 30"]

    def test_complex_error_propagation_chain(self) -> None:
        """Test complex error propagation through multiple operations."""

        def step_1(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def step_2(x: int) -> FlextResult[int]:
            return FlextResult[int].fail("Step 2 failed")  # Always fails

        def step_3(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 3)  # Should not execute

        result = (
            FlextResult[int]
            .ok(10)
            .flat_map(step_1)  # Should succeed: 10 -> 11
            .flat_map(step_2)  # Should fail
            .flat_map(step_3)  # Should be skipped due to failure
            .map(lambda x: x * 100)  # Should be skipped
        )

        assert result.is_failure
        assert result.error == "Step 2 failed"
