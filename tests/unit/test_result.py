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
from pytest_mock import MockerFixture

from flext_core import FlextResult, FlextTypes
from tests.support.asyncs import AsyncTestUtils
from tests.support.builders import TestBuilders
from tests.support.factories import FlextResultFactory
from tests.support.matchers import FlextMatchers
from tests.support.performance import BenchmarkProtocol, MemoryProfiler

# Test markers
pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextResultFactoryMethods:
    """Test FlextResult factory methods and core creation patterns."""

    def test_ok_factory_method(self) -> None:
        """Test FlextResult.ok creates successful result using FlextMatchers."""
        # Using test data for FlextResult validation
        test_data: FlextTypes.Core.JsonObject = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
            "is_active": True,
        }
        result = FlextResult[FlextTypes.Core.JsonObject].ok(test_data)

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
        with pytest.raises(
            TypeError, match="Attempted to access value on failed result"
        ):
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
        assert result.error is not None
        assert "test error" in result.error

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
        assert result.error is not None
        assert "runtime error" in result.error


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

        def failing_operation(_: int) -> FlextResult[int]:
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
            cast("FlextResult[object]", result3),
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
            cast("FlextResult[object]", result3),
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
            cast("FlextResult[object]", result3),
        )
        assert collected.is_failure
        assert collected.error is not None
        assert "error" in collected.error

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
            cast("FlextResult[object]", result3),
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
            cast("FlextResult[object]", result1), cast("FlextResult[object]", result2)
        )

        assert chained.is_failure
        assert chained.error == "first error"

    @given(st.lists(st.integers(), min_size=0, max_size=10))
    def test_chain_results_hypothesis(self, values: list[int]) -> None:
        """Property-based test for chain_results."""
        results = [FlextResult[int].ok(v) for v in values]
        chained = FlextResult.chain_results(
            *[cast("FlextResult[object]", r) for r in results]
        )

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

        def failing_recovery(_: str) -> int:
            msg = "Recovery failed"
            raise ValueError(msg)

        recovered = result.recover(failing_recovery)

        assert recovered.is_failure
        assert recovered.error is not None
        assert "Recovery failed" in recovered.error

    def test_recover_with_success_flow(self) -> None:
        """Test recover_with with successful recovery result."""
        result = FlextResult[int].fail("database error")

        def recovery_with_result(_: str) -> FlextResult[int]:
            return FlextResult[int].ok(100)  # Recovery value

        recovered = result.recover_with(recovery_with_result)

        assert recovered.is_success
        assert recovered.value == 100

    def test_recover_with_failure_flow(self) -> None:
        """Test recover_with with failing recovery result."""
        result = FlextResult[int].fail("original error")

        def failing_recovery(_: str) -> FlextResult[int]:
            return FlextResult[int].fail("recovery also failed")

        recovered = result.recover_with(failing_recovery)

        assert recovered.is_failure
        assert recovered.error == "recovery also failed"

    def test_recover_with_exception_handling(self) -> None:
        """Test recover_with handles exceptions in recovery function."""
        result = FlextResult[int].fail("error")

        def failing_recovery_with_exception(_: str) -> FlextResult[int]:
            msg = "Recovery function crashed"
            raise TypeError(msg)

        recovered = result.recover_with(failing_recovery_with_exception)

        assert recovered.is_failure
        assert recovered.error is not None
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

        def failing_side_effect(_: str) -> None:
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

        def failing_error_handler_with_type_error(_: str) -> None:
            msg = "Error handler failed"
            raise TypeError(msg)

        tapped = result.tap_error(failing_error_handler_with_type_error)

        # Result should be preserved when TypeError is suppressed
        assert tapped.is_failure
        assert tapped.error == "error"

    def test_tap_error_runtime_error_propagation(self) -> None:
        """Test tap_error allows RuntimeError to propagate (not in suppress list)."""
        result = FlextResult[str].fail("error")

        def failing_error_handler_with_runtime_error(_: str) -> None:
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

        def failing_predicate(_: str) -> bool:
            msg = "Predicate failed"
            raise ValueError(msg)

        filtered = result.filter(failing_predicate, "filter error")

        assert filtered.is_failure
        assert filtered.error is not None
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
        assert combined.error is not None
        assert "Missing data for zip operation" in combined.error

    def test_zip_with_function_exception(self) -> None:
        """Test zip_with handles exceptions in combination function."""
        result1 = FlextResult[int].ok(10)
        result2 = FlextResult[int].ok(0)

        def failing_function(x: int, y: int) -> float:
            return x / y  # Division by zero

        combined = result1.zip_with(result2, failing_function)

        assert combined.is_failure
        assert combined.error is not None
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

        result = FlextResult[FlextTypes.Core.JsonObject].ok(
            cast("FlextTypes.Core.JsonObject", complex_data)
        )

        assert result.is_success
        assert result.value == complex_data
        users = result.value.get("users")
        assert isinstance(users, list)


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

        def step_2(_: int) -> FlextResult[int]:
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

    def test_performance_benchmarking(self, benchmark: BenchmarkProtocol) -> None:
        """Test FlextResult performance using pytest-benchmark."""

        def create_and_chain_results() -> int:
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

    def test_builder_pattern_integration(self, mocker: MockerFixture) -> None:
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
        user_data: FlextTypes.Core.JsonObject = {
            "name": "Alice Johnson",
            "email": "alice.johnson@company.com",
            "age": 28,
        }

        def validate_user(
            data: FlextTypes.Core.JsonObject,
        ) -> FlextResult[FlextTypes.Core.JsonObject]:
            """Realistic user validation."""
            if not data.get("email"):
                return FlextResult[FlextTypes.Core.JsonObject].fail(
                    "Email is required", error_code="VALIDATION_ERROR"
                )
            age = data.get("age")
            if isinstance(age, int) and age < 18:
                return FlextResult[FlextTypes.Core.JsonObject].fail(
                    "Age must be 18 or older", error_code="VALIDATION_ERROR"
                )
            return FlextResult[FlextTypes.Core.JsonObject].ok(data)

        def enrich_user(
            data: FlextTypes.Core.JsonObject,
        ) -> FlextResult[FlextTypes.Core.JsonObject]:
            """Add computed fields."""
            enriched = data.copy()
            enriched["display_name"] = f"{data['name']} <{data['email']}>"
            age = data["age"]
            enriched["is_adult"] = isinstance(age, int) and age >= 18
            return FlextResult[FlextTypes.Core.JsonObject].ok(enriched)

        # Process user through pipeline
        result = (
            FlextResult[FlextTypes.Core.JsonObject]
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
                return FlextResult[str].fail(
                    "Negative values not allowed", error_code="VALIDATION_ERROR"
                )
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

        def process_orders(
            orders_data: list[FlextTypes.Core.JsonObject],
        ) -> FlextResult[FlextTypes.Core.JsonObject]:
            total_amount = 0.0
            for order in orders_data:
                amount = order.get("amount")
                if isinstance(amount, (int, float)):
                    total_amount += float(amount)
            processed = {
                "order_count": len(orders_data),
                "total_amount": total_amount,
                "average_order": total_amount / len(orders_data),
                "summary": f"{len(orders_data)} orders worth ${total_amount}",
            }
            return FlextResult[FlextTypes.Core.JsonObject].ok(processed)

        result = (
            FlextResult[list[FlextTypes.Core.JsonObject]]
            .ok(orders)
            .flat_map(process_orders)
        )

        FlextMatchers.assert_result_success(result)
        summary = result.value

        assert summary["order_count"] == 5
        assert summary["total_amount"] == 150.0  # 10+20+30+40+50
        assert summary["average_order"] == 30.0


class TestFlextResultUtilityMethods:
    """Test FlextResult utility class methods for 100% coverage."""

    def test_safe_unwrap_or_none_success(self) -> None:
        """Test safe_unwrap_or_none returns value for success."""
        result = FlextResult[str].ok("success_value")
        value = FlextResult.safe_unwrap_or_none(result)
        assert value == "success_value"

    def test_safe_unwrap_or_none_failure(self) -> None:
        """Test safe_unwrap_or_none returns None for failure."""
        result = FlextResult[str].fail("error")
        value = FlextResult.safe_unwrap_or_none(result)
        assert value is None

    def test_unwrap_or_raise_success(self) -> None:
        """Test unwrap_or_raise returns value for success."""
        result = FlextResult[int].ok(42)
        value = FlextResult.unwrap_or_raise(result)
        assert value == 42

    def test_unwrap_or_raise_failure_default_exception(self) -> None:
        """Test unwrap_or_raise raises RuntimeError by default."""
        result = FlextResult[int].fail("operation failed")
        with pytest.raises(RuntimeError, match="operation failed"):
            FlextResult.unwrap_or_raise(result)

    def test_unwrap_or_raise_failure_custom_exception(self) -> None:
        """Test unwrap_or_raise raises custom exception type."""
        result = FlextResult[int].fail("validation error")
        with pytest.raises(ValueError, match="validation error"):
            FlextResult.unwrap_or_raise(result, ValueError)

    def test_unwrap_or_raise_failure_no_error_message(self) -> None:
        """Test unwrap_or_raise with empty error message."""
        result = FlextResult[int].fail("")
        with pytest.raises(RuntimeError, match="Unknown error occurred|Operation failed"):
            FlextResult.unwrap_or_raise(result)

    def test_collect_successes_all_successful(self) -> None:
        """Test collect_successes with all successful results."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ]
        successes = FlextResult.collect_successes(results)
        assert successes == [1, 2, 3]

    def test_collect_successes_mixed_results(self) -> None:
        """Test collect_successes with mixed success/failure results."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("error1"),
            FlextResult[int].ok(3),
            FlextResult[int].fail("error2"),
        ]
        successes = FlextResult.collect_successes(results)
        assert successes == [1, 3]

    def test_collect_successes_empty_list(self) -> None:
        """Test collect_successes with empty list."""
        results: list[FlextResult[int]] = []
        successes = FlextResult.collect_successes(results)
        assert successes == []

    def test_collect_failures_all_failures(self) -> None:
        """Test collect_failures with all failed results."""
        results = [
            FlextResult[int].fail("error1"),
            FlextResult[int].fail("error2"),
            FlextResult[int].fail("error3"),
        ]
        failures = FlextResult.collect_failures(results)
        assert failures == ["error1", "error2", "error3"]

    def test_collect_failures_mixed_results(self) -> None:
        """Test collect_failures with mixed success/failure results."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("error1"),
            FlextResult[int].ok(3),
            FlextResult[int].fail("error2"),
        ]
        failures = FlextResult.collect_failures(results)
        assert failures == ["error1", "error2"]

    def test_collect_failures_empty_list(self) -> None:
        """Test collect_failures with empty list."""
        results: list[FlextResult[int]] = []
        failures = FlextResult.collect_failures(results)
        assert failures == []

    def test_success_rate_all_successful(self) -> None:
        """Test success_rate with 100% success."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ]
        rate = FlextResult.success_rate(results)
        assert rate == 100.0

    def test_success_rate_all_failures(self) -> None:
        """Test success_rate with 0% success."""
        results = [
            FlextResult[int].fail("error1"),
            FlextResult[int].fail("error2"),
        ]
        rate = FlextResult.success_rate(results)
        assert rate == 0.0

    def test_success_rate_mixed_results(self) -> None:
        """Test success_rate with mixed results."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("error1"),
            FlextResult[int].ok(3),
            FlextResult[int].fail("error2"),
        ]
        rate = FlextResult.success_rate(results)
        assert rate == 50.0

    def test_success_rate_empty_list(self) -> None:
        """Test success_rate with empty list returns 0."""
        results: list[FlextResult[int]] = []
        rate = FlextResult.success_rate(results)
        assert rate == 0.0

    def test_batch_process_all_successful(self) -> None:
        """Test batch_process with all successful operations."""
        items = [1, 2, 3, 4]

        def double_value(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        successes, failures = FlextResult.batch_process(items, double_value)
        assert successes == [2, 4, 6, 8]
        assert failures == []

    def test_batch_process_with_failures(self) -> None:
        """Test batch_process with some failures."""
        items = [1, 2, 3, 4]

        def process_even_only(x: int) -> FlextResult[int]:
            if x % 2 == 0:
                return FlextResult[int].ok(x * 10)
            return FlextResult[int].fail(f"Odd number {x} not allowed")

        successes, failures = FlextResult.batch_process(items, process_even_only)
        assert successes == [20, 40]  # 2*10=20, 4*10=40
        assert failures == ["Odd number 1 not allowed", "Odd number 3 not allowed"]

    def test_batch_process_empty_list(self) -> None:
        """Test batch_process with empty input."""
        items: list[int] = []

        def identity(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x)

        successes, failures = FlextResult.batch_process(items, identity)
        assert successes == []
        assert failures == []


class TestFlextResultCollectionOperations:
    """Test FlextResult collection operations and advanced methods."""

    def test_all_success_all_successful(self) -> None:
        """Test all_success returns True when all results are successful."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ]
        assert FlextResult.all_success(*results) is True

    def test_all_success_with_failure(self) -> None:
        """Test all_success returns False when any result fails."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("error"),
            FlextResult[int].ok(3),
        ]
        assert FlextResult.all_success(*results) is False

    def test_all_success_empty(self) -> None:
        """Test all_success returns True for empty list."""
        assert FlextResult.all_success() is True

    def test_any_success_with_successes(self) -> None:
        """Test any_success returns True when any result is successful."""
        results = [
            FlextResult[int].fail("error1"),
            FlextResult[int].ok(2),
            FlextResult[int].fail("error2"),
        ]
        assert FlextResult.any_success(*results) is True

    def test_any_success_all_failures(self) -> None:
        """Test any_success returns False when all results fail."""
        results = [
            FlextResult[int].fail("error1"),
            FlextResult[int].fail("error2"),
        ]
        assert FlextResult.any_success(*results) is False

    def test_any_success_empty(self) -> None:
        """Test any_success returns False for empty list."""
        assert FlextResult.any_success() is False

    def test_first_success_with_successes(self) -> None:
        """Test first_success returns first successful result."""
        results = [
            FlextResult[int].fail("error1"),
            FlextResult[int].ok(42),
            FlextResult[int].ok(100),
        ]
        first = FlextResult.first_success(*results)
        assert first.is_success
        assert first.value == 42

    def test_first_success_all_failures(self) -> None:
        """Test first_success returns failure when no successes."""
        results = [
            FlextResult[int].fail("error1"),
            FlextResult[int].fail("error2"),
        ]
        first = FlextResult.first_success(*results)
        assert first.is_failure
        assert first.error is not None  # Just ensure error exists

    def test_first_success_empty(self) -> None:
        """Test first_success with empty list."""
        first = FlextResult[int].first_success()
        assert first.is_failure

    def test_try_all_first_succeeds(self) -> None:
        """Test try_all returns result from first successful function."""

        def failing_func1() -> int:
            raise ValueError("First function failed")

        def succeeding_func() -> int:
            return 42

        def failing_func2() -> int:
            raise RuntimeError("Should not reach here")

        result = FlextResult.try_all(failing_func1, succeeding_func, failing_func2)
        assert result.is_success
        assert result.value == 42

    def test_try_all_all_fail(self) -> None:
        """Test try_all returns failure when all functions fail."""

        def failing_func1() -> int:
            raise ValueError("First failed")

        def failing_func2() -> int:
            raise RuntimeError("Second failed")

        result = FlextResult.try_all(failing_func1, failing_func2)
        assert result.is_failure
        assert result.error is not None

    def test_try_all_empty(self) -> None:
        """Test try_all with no functions returns failure."""
        result = FlextResult[int].try_all()
        assert result.is_failure


class TestFlextResultSpecialMethods:
    """Test FlextResult special methods and protocols."""

    def test_bool_conversion_success(self) -> None:
        """Test __bool__ returns True for successful results."""
        result = FlextResult[int].ok(42)
        assert bool(result) is True

    def test_bool_conversion_failure(self) -> None:
        """Test __bool__ returns False for failed results."""
        result = FlextResult[int].fail("error")
        assert bool(result) is False

    def test_iterator_protocol_success(self) -> None:
        """Test __iter__ yields value and None for success."""
        result = FlextResult[str].ok("test")
        value, error = result
        assert value == "test"
        assert error is None

    def test_iterator_protocol_failure(self) -> None:
        """Test __iter__ yields None and error for failure."""
        result = FlextResult[str].fail("error message")
        value, error = result
        assert value is None
        assert error == "error message"

    def test_getitem_protocol_success(self) -> None:
        """Test __getitem__ provides subscript access for success."""
        result = FlextResult[str].ok("success")
        assert result[0] == "success"  # value
        assert result[1] is None  # error

    def test_getitem_protocol_failure(self) -> None:
        """Test __getitem__ provides subscript access for failure."""
        result = FlextResult[str].fail("error")
        assert result[0] is None  # value
        assert result[1] == "error"  # error

    def test_getitem_invalid_index(self) -> None:
        """Test __getitem__ raises IndexError for invalid indices."""
        result = FlextResult[str].ok("test")
        with pytest.raises(IndexError):
            _ = result[2]
        with pytest.raises(IndexError):
            _ = result[-1]

    def test_or_operator_success(self) -> None:
        """Test __or__ returns value for successful result."""
        result = FlextResult[str].ok("success")
        value = result | "default"
        assert value == "success"

    def test_or_operator_failure(self) -> None:
        """Test __or__ returns default for failed result."""
        result = FlextResult[str].fail("error")
        value = result | "default"
        assert value == "default"

    def test_equality_both_success(self) -> None:
        """Test __eq__ compares successful results by value."""
        result1 = FlextResult[int].ok(42)
        result2 = FlextResult[int].ok(42)
        result3 = FlextResult[int].ok(100)

        assert result1 == result2
        assert result1 != result3

    def test_equality_both_failure(self) -> None:
        """Test __eq__ compares failed results by error."""
        result1 = FlextResult[int].fail("error")
        result2 = FlextResult[int].fail("error")
        result3 = FlextResult[int].fail("different error")

        assert result1 == result2
        assert result1 != result3

    def test_equality_mixed_states(self) -> None:
        """Test __eq__ returns False for different states."""
        success = FlextResult[int].ok(42)
        failure = FlextResult[int].fail("error")

        assert success != failure
        assert failure != success

    def test_equality_different_types(self) -> None:
        """Test __eq__ returns False for different types."""
        result = FlextResult[int].ok(42)
        assert result != 42
        assert result != "not a result"
        assert result != None

    def test_hash_success(self) -> None:
        """Test __hash__ works for successful results."""
        result = FlextResult[str].ok("test")
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_hash_failure(self) -> None:
        """Test __hash__ works for failed results."""
        result = FlextResult[str].fail("error")
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_hash_consistency(self) -> None:
        """Test __hash__ is consistent for equal results."""
        result1 = FlextResult[int].ok(42)
        result2 = FlextResult[int].ok(42)
        assert hash(result1) == hash(result2)

    def test_hash_in_set(self) -> None:
        """Test FlextResult can be used in sets."""
        result1 = FlextResult[int].ok(42)
        result2 = FlextResult[int].ok(42)
        result3 = FlextResult[int].fail("error")

        result_set = {result1, result2, result3}
        assert len(result_set) == 2  # result1 and result2 are equal

    def test_repr_success(self) -> None:
        """Test __repr__ for successful result."""
        result = FlextResult[str].ok("test")
        repr_str = repr(result)
        assert "FlextResult" in repr_str
        assert "success" in repr_str.lower()

    def test_repr_failure(self) -> None:
        """Test __repr__ for failed result."""
        result = FlextResult[str].fail("error")
        repr_str = repr(result)
        assert "FlextResult" in repr_str
        assert "failure" in repr_str.lower() or "error" in repr_str


class TestFlextResultContextManager:
    """Test FlextResult context manager protocol."""

    def test_context_manager_success(self) -> None:
        """Test context manager with successful result."""
        result = FlextResult[str].ok("resource")

        with result as resource:
            assert resource == "resource"

    def test_context_manager_failure_raises(self) -> None:
        """Test context manager raises for failed result."""
        result = FlextResult[str].fail("resource not available")

        with pytest.raises(RuntimeError, match="resource not available"):
            with result as resource:
                # Should not reach here
                pass

    def test_context_manager_exception_handling(self) -> None:
        """Test context manager handles exceptions in block."""
        result = FlextResult[str].ok("resource")

        with pytest.raises(ValueError, match="test error"):
            with result as resource:
                assert resource == "resource"
                raise ValueError("test error")

    def test_context_manager_return_false_on_exception(self) -> None:
        """Test context manager __exit__ returns False to propagate exceptions."""
        result = FlextResult[str].ok("resource")

        try:
            with result as resource:
                assert resource == "resource"
                raise ValueError("test error")
        except ValueError:
            # Exception should propagate (not be suppressed)
            pass
        else:
            pytest.fail("Exception should have been raised")


class TestFlextResultAdditionalMethods:
    """Test additional FlextResult methods for complete coverage."""

    def test_expect_success(self) -> None:
        """Test expect returns value for successful result."""
        result = FlextResult[str].ok("success")
        value = result.expect("Should not fail")
        assert value == "success"

    def test_expect_failure_with_custom_message(self) -> None:
        """Test expect raises with custom message for failed result."""
        result = FlextResult[str].fail("original error")
        with pytest.raises(RuntimeError, match="Custom failure message"):
            result.expect("Custom failure message")

    def test_to_exception_failure_creates_exception(self) -> None:
        """Test to_exception creates Exception from failed result."""
        result = FlextResult[str].fail("operation failed")
        exception = result.to_exception()

        assert isinstance(exception, Exception)
        assert str(exception) == "operation failed"

    def test_then_alias_for_flat_map(self) -> None:
        """Test then as alias for flat_map."""

        def increment(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        result = FlextResult[int].ok(41)
        chained = result.then(increment)

        assert chained.is_success
        assert chained.value == 42


class TestFlextResultStandaloneFunctions:
    """Test standalone utility functions."""

    def test_ok_result_function(self) -> None:
        """Test ok_result standalone function."""
        from flext_core.result import ok_result

        result = ok_result("test data")
        assert result.is_success
        assert result.value == "test data"

    def test_fail_result_function_basic(self) -> None:
        """Test fail_result standalone function with basic error."""
        from flext_core.result import fail_result

        result = fail_result("operation failed")
        assert result.is_failure
        assert result.error == "operation failed"

    def test_fail_result_function_with_metadata(self) -> None:
        """Test fail_result standalone function with error code and data."""
        from flext_core.result import fail_result

        result = fail_result(
            "validation error",
            error_code="VALIDATION_ERROR",
            error_data={"field": "email", "reason": "invalid format"},
        )
        assert result.is_failure
        assert result.error == "validation error"
        assert result.error_code == "VALIDATION_ERROR"
        assert result.error_data["field"] == "email"
        assert result.error_data["reason"] == "invalid format"


class TestFlextResultEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for 100% coverage."""

    def test_map_with_exception_in_function(self) -> None:
        """Test map handles exceptions in mapping function."""
        result = FlextResult[int].ok(10)

        def failing_map(x: int) -> str:
            if x > 5:
                raise ValueError("Value too large")
            return str(x)

        mapped = result.map(failing_map)
        assert mapped.is_failure
        assert "Value too large" in mapped.error

    def test_flat_map_with_exception_in_function(self) -> None:
        """Test flat_map handles exceptions in mapping function."""
        result = FlextResult[int].ok(10)

        def failing_flat_map(x: int) -> FlextResult[str]:
            if x > 5:
                raise RuntimeError("Processing failed")
            return FlextResult[str].ok(str(x))

        mapped = result.flat_map(failing_flat_map)
        assert mapped.is_failure
        assert "Processing failed" in mapped.error

    def test_large_error_data_serialization(self) -> None:
        """Test FlextResult with large error data structures."""
        large_error_data = {
            f"field_{i}": f"error_message_{i}" * 100 for i in range(100)
        }
        result = FlextResult[str].fail(
            "validation failed", error_data=large_error_data
        )

        assert result.is_failure
        assert len(result.error_data) == 100
        assert result.error_data["field_50"] == "error_message_50" * 100

    def test_very_long_error_message(self) -> None:
        """Test FlextResult with extremely long error messages."""
        long_error = "A" * 10000  # 10KB error message
        result = FlextResult[str].fail(long_error)

        assert result.is_failure
        assert result.error == long_error
        assert len(result.error) == 10000

    def test_property_based_empty_error_message(self) -> None:
        """Test specific case of empty error message."""
        result = FlextResult[str].fail("")

        assert result.is_failure
        assert result.error == "Unknown error occurred"  # Default behavior

    def test_property_based_whitespace_error_message(self) -> None:
        """Test whitespace-only error message."""
        result = FlextResult[str].fail("   ")

        assert result.is_failure
        assert result.error == "Unknown error occurred"  # FlextResult normalizes whitespace

    @given(st.integers())
    def test_property_based_success_values(self, value: int) -> None:
        """Property-based test for success value handling."""
        result = FlextResult[int].ok(value)

        assert result.is_success
        assert result.value == value
