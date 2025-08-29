# ruff: noqa: ARG001, ARG002, PLR0904, PLR0913
"""Comprehensive tests for FlextResult with 100% coverage using advanced pytest features.

Tests all FlextResult functionality including:
- Factory methods and properties
- Railway-oriented programming patterns
- Error handling and recovery
- Collection operations and combinators
- Context manager protocol
- Performance characteristics
- Async operations
- Advanced matchers and builders
"""

from __future__ import annotations

import operator
from typing import cast

import pytest
from hypothesis import given, strategies as st

from flext_core import FlextResult

from ..support import (
    AsyncTestUtils,
    FlextMatchers,
    FlextResultFactory,
    MemoryProfiler,
    TestBuilders,
    UserDataFactory,
)

# Test markers
pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextResultFactoryMethods:
    """Test FlextResult factory methods and core creation patterns."""

    def test_ok_factory_method(self) -> None:
        """Test FlextResult.ok creates successful result using FlextMatchers."""
        # Using TestBuilders for consistent test data
        test_data = UserDataFactory.create(name="John Doe", email="john@example.com")
        result = FlextResult[dict].ok(test_data)

        # Using FlextMatchers for assertions
        FlextMatchers.assert_result_success(result, expected_data=test_data)

        # Traditional assertions for aliases
        assert result.success  # Alias
        assert not result.is_failure
        assert not result.is_fail  # Alias
        assert result.value == test_data
        assert result.data == test_data  # Alias
        assert result.error is None

    def test_fail_factory_method(self) -> None:
        """Test FlextResult.fail creates failure result using FlextMatchers."""
        # Using FlextResultFactory for consistent error creation
        result = FlextResultFactory.create_failure(
            error="Operation failed due to invalid input", error_code="VALIDATION_ERROR"
        )

        # Using FlextMatchers for sophisticated assertions
        FlextMatchers.assert_result_failure(
            result,
            expected_error="Operation failed",
            expected_error_code="VALIDATION_ERROR",
        )

        # Test value access raises exception
        with pytest.raises(TypeError, match="Attempted to access value on failed result"):
            _ = result.value

    def test_fail_with_error_code_and_data(self) -> None:
        """Test FlextResult.fail with error codes and metadata."""
        result = FlextResult[str].fail(
            "validation failed",
            error_code="VALIDATION_ERROR",
            error_data={"operation": "user_creation", "field": "email"},
        )
        assert result.error_code == "VALIDATION_ERROR"
        assert result.error_data["operation"] == "user_creation"
        assert result.error_data["field"] == "email"

    def test_failure_alias(self) -> None:
        """Test FlextResult.failure as alias for fail."""
        result = FlextResult[str].failure("error")
        assert result.is_failure
        assert result.error == "error"

    @given(st.text())
    def test_ok_with_hypothesis(self, value: str) -> None:
        """Property-based test for ok factory method."""
        result = FlextResult[str].ok(value)
        assert result.success
        assert result.value == value

    def test_from_exception_success(self) -> None:
        """Test from_exception with successful function."""

        def safe_function() -> str:
            return "success"

        result = FlextResult.from_exception(safe_function)
        assert result.is_success
        assert result.value == "success"

    def test_from_exception_with_exception(self) -> None:
        """Test from_exception catches exceptions."""

        def failing_function() -> str:
            error_msg = "test error"
            raise ValueError(error_msg)

        result = FlextResult.from_exception(failing_function)
        assert result.is_failure
        assert result.error is not None and "test error" in result.error

    def test_safe_call_success(self) -> None:
        """Test safe_call with successful function."""

        def safe_func() -> int:
            return 42

        result = FlextResult.safe_call(safe_func)
        assert result.is_success
        assert result.value == 42

    def test_safe_call_with_exception(self) -> None:
        """Test safe_call handles exceptions."""

        def unsafe_func() -> int:
            error_msg = "runtime error"
            raise RuntimeError(error_msg)

        result = FlextResult.safe_call(unsafe_func)
        assert result.is_failure
        assert result.error is not None and "runtime error" in result.error


class TestFlextResultProperties:
    """Test FlextResult properties and accessor methods."""

    def test_value_or_none_success(self) -> None:
        """Test value_or_none returns value for success."""
        result = FlextResult[int].ok(42)
        assert result.value_or_none == 42

    def test_value_or_none_failure(self) -> None:
        """Test value_or_none returns None for failure."""
        result = FlextResult[int].fail("error")
        assert result.value_or_none is None

    def test_legacy_aliases(self) -> None:
        """Test legacy property aliases."""
        success_result = FlextResult[str].ok("test")
        assert success_result.is_valid  # Legacy alias

        failure_result = FlextResult[str].fail("error", error_code="CODE_001")
        assert failure_result.error_message == "error"  # Legacy alias

    def test_error_properties_on_success(self) -> None:
        """Test error properties return None for successful results."""
        result = FlextResult[str].ok("success")
        assert result.error is None
        assert result.error_code is None
        assert result.error_data == {}
        assert result.metadata == {}  # Alias


class TestFlextResultRailwayOperations:
    """Test railway-oriented programming operations."""

    def test_map_success(self) -> None:
        """Test map transforms success values."""
        result = FlextResult[int].ok(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_success
        assert mapped.value == 10

    def test_map_failure_passthrough(self) -> None:
        """Test map passes through failures unchanged."""
        result = FlextResult[int].fail("error")
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_failure
        assert mapped.error == "error"

    def test_flat_map_success(self) -> None:
        """Test flat_map chains successful operations."""

        def divide_by_two(x: int) -> FlextResult[float]:
            return FlextResult[float].ok(x / 2)

        result = FlextResult[int].ok(10)
        chained = result.flat_map(divide_by_two)
        assert chained.is_success
        assert chained.value == 5.0

    def test_flat_map_failure_in_chain(self) -> None:
        """Test flat_map propagates failures in chain."""

        def failing_operation(x: int) -> FlextResult[int]:
            return FlextResult[int].fail("chain error")

        result = FlextResult[int].ok(5)
        chained = result.flat_map(failing_operation)
        assert chained.is_failure
        assert chained.error == "chain error"

    def test_bind_alias_for_flat_map(self) -> None:
        """Test bind as alias for flat_map."""

        def increment(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        result = FlextResult[int].ok(5)
        bound = result.bind(increment)
        assert bound.is_success
        assert bound.value == 6

    def test_tap_error_inspects_failures(self) -> None:
        """Test tap_error allows error inspection without transformation."""
        result = FlextResult[int].fail("original error")
        errors_seen = []
        transformed = result.tap_error(lambda e: errors_seen.append(e))
        assert transformed.is_failure
        assert transformed.error == "original error"
        assert errors_seen == ["original error"]

    def test_tap_error_passthrough_success(self) -> None:
        """Test tap_error ignores successful results."""
        result = FlextResult[int].ok(42)
        errors_seen = []
        transformed = result.tap_error(lambda e: errors_seen.append(e))
        assert transformed.is_success
        assert transformed.value == 42
        assert errors_seen == []


class TestFlextResultErrorRecovery:
    """Test error recovery and alternative value patterns."""

    def test_unwrap_success(self) -> None:
        """Test unwrap returns value for success."""
        result = FlextResult[str].ok("success")
        assert result.unwrap() == "success"

    def test_unwrap_failure_raises(self) -> None:
        """Test unwrap raises for failure."""
        result = FlextResult[str].fail("error")
        with pytest.raises(RuntimeError, match="error"):
            result.unwrap()

    def test_unwrap_or_with_success(self) -> None:
        """Test unwrap_or returns value for success."""
        result = FlextResult[str].ok("success")
        assert result.unwrap_or("default") == "success"

    def test_unwrap_or_with_failure(self) -> None:
        """Test unwrap_or returns default for failure."""
        result = FlextResult[str].fail("error")
        assert result.unwrap_or("default") == "default"

    def test_or_else_get_with_success(self) -> None:
        """Test or_else_get returns original successful result."""
        result = FlextResult[str].ok("success")
        alternative = result.or_else_get(lambda: FlextResult.ok("computed"))
        assert alternative.is_success
        assert alternative.value == "success"

    def test_or_else_get_with_failure(self) -> None:
        """Test or_else_get computes alternative for failure."""
        result = FlextResult[str].fail("error")
        alternative = result.or_else_get(lambda: FlextResult.ok("computed"))
        assert alternative.is_success
        assert alternative.value == "computed"

    def test_or_else_recovers_from_failure(self) -> None:
        """Test or_else provides alternative result for failures."""
        failed_result = FlextResult[int].fail("error")
        recovery_result = FlextResult[int].ok(999)

        recovered = failed_result.or_else(recovery_result)
        assert recovered.is_success
        assert recovered.value == 999

    def test_or_else_preserves_success(self) -> None:
        """Test or_else preserves successful results."""
        success_result = FlextResult[int].ok(42)
        recovery_result = FlextResult[int].ok(999)

        result = success_result.or_else(recovery_result)
        assert result.is_success
        assert result.value == 42


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

        chained = FlextResult.chain_results(
            cast("FlextResult[object]", result1),
            cast("FlextResult[object]", result2),
            cast("FlextResult[object]", result3)
        )

        assert chained.is_success
        assert chained.value == ["first", 42, True]

    def test_chain_results_single_result(self) -> None:
        """Test chain_results with single result."""
        result = FlextResult[str].ok("single")
        chained = FlextResult.chain_results(cast("FlextResult[object]", result))
        assert chained.is_success
        assert chained.value == ["single"]

    def test_combine_all_success(self) -> None:
        """Test combine with all successful results."""
        result1 = FlextResult[int].ok(1)
        result2 = FlextResult[int].ok(2)
        result3 = FlextResult[int].ok(3)

        collected = FlextResult.combine(
            cast("FlextResult[object]", result1),
            cast("FlextResult[object]", result2),
            cast("FlextResult[object]", result3)
        )
        assert collected.is_success
        assert collected.value == [1, 2, 3]

    def test_combine_with_failure(self) -> None:
        """Test combine fails on any failure."""
        result1 = FlextResult[int].ok(1)
        result2 = FlextResult[int].fail("error")
        result3 = FlextResult[int].ok(3)

        collected = FlextResult.combine(
            cast("FlextResult[object]", result1),
            cast("FlextResult[object]", result2),
            cast("FlextResult[object]", result3)
        )
        assert collected.is_failure
        assert collected.error is not None and "error" in collected.error

    def test_combine_empty_varargs(self) -> None:
        """Test combine with no arguments returns empty success."""
        collected = FlextResult.combine()
        assert collected.is_success
        assert collected.value == []

    def test_chain_results_failure_short_circuits(self) -> None:
        """Test chain_results fails fast on first failure."""
        result1 = FlextResult[str].ok("success")
        result2 = FlextResult[int].fail("middle failure")
        result3 = FlextResult[bool].ok(True)  # Should not be reached

        chained = FlextResult.chain_results(
            cast("FlextResult[object]", result1),
            cast("FlextResult[object]", result2),
            cast("FlextResult[object]", result3)
        )

        assert chained.is_failure
        assert chained.error == "middle failure"

    def test_batch_process_all_success(self) -> None:
        """Test batch_process applies function to all values and collects results."""
        values = [1, 2, 3]

        def safe_double(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        successes, failures = FlextResult.batch_process(values, safe_double)
        assert len(failures) == 0  # No failures
        assert successes == [2, 4, 6]

    def test_batch_process_with_failure(self) -> None:
        """Test batch_process separates successes from failures."""
        values = [1, 2, 3]

        def failing_on_two(x: int) -> FlextResult[int]:
            if x == 2:
                return FlextResult[int].fail("failed on 2")
            return FlextResult[int].ok(x * 2)

        successes, failures = FlextResult.batch_process(values, failing_on_two)
        assert successes == [2, 6]  # 1*2=2, 3*2=6
        assert len(failures) == 1  # One failure
        assert "failed on 2" in failures[0]

    def test_chain_results_first_failure(self) -> None:
        """Test chain_results with first result failing."""
        result1 = FlextResult[str].fail("first error")
        result2 = FlextResult[int].ok(42)

        chained = FlextResult.chain_results(
            cast("FlextResult[object]", result1),
            cast("FlextResult[object]", result2)
        )

        assert chained.is_failure
        assert chained.error == "first error"

    @given(st.lists(st.integers(), min_size=0, max_size=10))
    def test_chain_results_hypothesis(self, values: list[int]) -> None:
        """Property-based test for chain_results."""
        results = [FlextResult[int].ok(v) for v in values]
        chained = FlextResult.chain_results(*[cast("FlextResult[object]", r) for r in results])

        assert chained.is_success
        assert chained.value == values

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

        recovered = result.recover(lambda _: 42)  # Default value

        assert recovered.is_success
        assert recovered.value == 42

    def test_recover_on_success_returns_self(self) -> None:
        """Test recover on successful result returns original result."""
        result = FlextResult[int].ok(10)

        recovered = result.recover(lambda _: 999)

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
        assert recovered.error is not None and "Recovery failed" in recovered.error

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
        assert recovered.error is not None and "Recovery function crashed" in recovered.error


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
        assert filtered.error is not None and "Predicate failed" in filtered.error


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
        assert combined.error is not None and "Missing data for zip operation" in combined.error

    def test_zip_with_function_exception(self) -> None:
        """Test zip_with handles exceptions in combination function."""
        result1 = FlextResult[int].ok(10)
        result2 = FlextResult[int].ok(0)

        def failing_function(x: int, y: int) -> float:
            return x / y  # Division by zero

        combined = result1.zip_with(result2, failing_function)

        assert combined.is_failure
        assert combined.error is not None and (
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
            .recover(lambda _: 42)  # Recovery value
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


class TestFlextResultAdvancedFeatures:
    """Advanced FlextResult tests using all tests/support/ capabilities."""

    def test_performance_benchmarking(self, benchmark) -> None:
        """Test FlextResult performance using pytest-benchmark."""

        def create_and_chain_results():
            return (
                FlextResult[int]
                .ok(42)
                .map(lambda x: x * 2)
                .flat_map(lambda x: FlextResult[int].ok(x + 1))
                .unwrap()
            )

        # Using FlextMatchers for performance assertions
        result = FlextMatchers.assert_performance_within_limit(
            benchmark, create_and_chain_results, max_time_seconds=0.001
        )
        assert result == 85

    def test_memory_profiling(self) -> None:
        """Test memory efficiency using MemoryProfiler."""
        with MemoryProfiler.track_memory_leaks(max_increase_mb=5.0):
            # Create large chain of operations
            results = []
            for i in range(1000):
                result = (
                    FlextResult[int]
                    .ok(i)
                    .map(lambda x: x * 2)
                    .flat_map(lambda x: FlextResult[int].ok(x + 1))
                )
                results.append(result.value)

        # Test completed without memory leak assertion error

    @pytest.mark.asyncio
    async def test_async_operations(self) -> None:
        """Test FlextResult with async operations."""

        async def async_transformation(x: int) -> FlextResult[str]:
            await AsyncTestUtils.simulate_delay(0.01)
            return FlextResult[str].ok(f"async_{x}")

        # Test async chaining
        result = FlextResult[int].ok(42)
        async_result = await async_transformation(result.value)

        FlextMatchers.assert_result_success(async_result, expected_data="async_42")

    @pytest.mark.asyncio
    async def test_concurrent_result_operations(self) -> None:
        """Test concurrent FlextResult operations."""

        async def async_compute(value: int) -> FlextResult[int]:
            await AsyncTestUtils.simulate_delay(0.01)
            return FlextResult[int].ok(value * 2)

        # Run multiple operations concurrently
        tasks = [async_compute(i) for i in range(5)]
        results = await AsyncTestUtils.run_concurrently(*tasks)

        # Assert all operations succeeded
        for i, result in enumerate(results):
            FlextMatchers.assert_result_success(result, expected_data=i * 2)

    def test_builder_pattern_integration(self, mocker) -> None:
        """Test FlextResult with TestBuilders integration."""
        # Build complex test scenario
        container = TestBuilders.container().with_database_service().build()

        # Create result from container operation
        db_result = container.get("database")
        FlextMatchers.assert_result_success(db_result)

        # Test with mock integration
        mock_service = (
            TestBuilders.mock(mocker).returns_result_success({"status": "ok"}).build()
        )
        mock_result = mock_service()

        FlextMatchers.assert_result_success(mock_result, expected_data={"status": "ok"})

    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20))
    def test_hypothesis_result_collections(self, values: list[int]) -> None:
        """Property-based testing with result collections."""
        # Create results from input values
        results = [FlextResult[int].ok(v) for v in values]

        # Test individual results
        for i, result in enumerate(results):
            FlextMatchers.assert_result_success(result, expected_data=values[i])

        # Test that all results are successful
        all_successful = all(r.is_success for r in results)
        assert all_successful, "All results should be successful"

        # Test mathematical properties
        actual_values = [r.value for r in results]
        assert actual_values == values

    def test_real_world_user_workflow(self) -> None:
        """Test real-world user processing workflow."""
        # Create realistic user data
        user_data = UserDataFactory.create(
            name="Alice Johnson", email="alice.johnson@company.com", age=28
        )

        def validate_user(data: dict) -> FlextResult[dict]:
            """Realistic user validation."""
            if not data.get("email"):
                return cast("FlextResult[dict]", FlextResultFactory.validation_error("email", None))
            if data.get("age", 0) < 18:
                return cast("FlextResult[dict]", FlextResultFactory.validation_error("age", data.get("age")))
            return FlextResult[dict].ok(data)

        def enrich_user(data: dict) -> FlextResult[dict]:
            """Add computed fields."""
            enriched = data.copy()
            enriched["display_name"] = f"{data['name']} <{data['email']}>"
            enriched["is_adult"] = data["age"] >= 18
            return FlextResult[dict].ok(enriched)

        # Process user through pipeline
        result = (
            FlextResult[dict]
            .ok(user_data)
            .flat_map(validate_user)
            .flat_map(enrich_user)
        )

        # Assert successful processing
        FlextMatchers.assert_result_success(result)
        processed_user = result.value

        assert (
            processed_user["display_name"]
            == "Alice Johnson <alice.johnson@company.com>"
        )
        assert processed_user["is_adult"] is True

        # Test JSON structure
        FlextMatchers.assert_json_structure(
            processed_user,
            ["name", "email", "age", "display_name", "is_adult"],
            exact_match=False,
        )

    def test_error_recovery_patterns(self) -> None:
        """Test error recovery and fallback patterns."""

        def risky_operation(value: int) -> FlextResult[str]:
            if value < 0:
                return cast("FlextResult[str]", FlextResultFactory.create_failure(
                    "Negative values not allowed", "VALIDATION_ERROR"
                ))
            return FlextResult[str].ok(f"processed_{value}")

        def fallback_operation(error: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"fallback_for_{error}")

        # Test successful path
        success_result = risky_operation(42)
        FlextMatchers.assert_result_success(success_result, "processed_42")

        # Test error recovery
        failed_result = risky_operation(-1)
        FlextMatchers.assert_result_failure(
            failed_result,
            expected_error="Negative values",
            expected_error_code="VALIDATION_ERROR",
        )

        # Test recovery with fallback
        fallback_result = fallback_operation(failed_result.error or "unknown")
        FlextMatchers.assert_result_success(fallback_result)
        assert "fallback_for_" in fallback_result.value

    def test_complex_data_transformations(self) -> None:
        """Test complex data transformation pipelines."""
        # Start with realistic data
        orders = [
            {"id": i, "amount": i * 10.0, "status": "pending"} for i in range(1, 6)
        ]

        def process_orders(orders_data: list[dict]) -> FlextResult[dict]:
            total_amount = sum(order["amount"] for order in orders_data)
            processed = {
                "order_count": len(orders_data),
                "total_amount": total_amount,
                "average_order": total_amount / len(orders_data),
                "summary": f"{len(orders_data)} orders worth ${total_amount}",
            }
            return FlextResult[dict].ok(processed)

        result = FlextResult[list].ok(orders).flat_map(process_orders)

        FlextMatchers.assert_result_success(result)
        summary = result.value

        assert summary["order_count"] == 5
        assert summary["total_amount"] == 150.0  # 10+20+30+40+50
        assert summary["average_order"] == 30.0
