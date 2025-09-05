"""Comprehensive test coverage for FlextResult railway-oriented programming."""

from __future__ import annotations

import operator
from typing import cast

import pytest

from flext_core import FlextResult


class TestFlextResultComprehensiveCoverage:
    """Comprehensive tests for FlextResult covering all methods and patterns."""

    def test_ok_factory_method(self) -> None:
        """Test FlextResult.ok() factory method creates successful results."""
        result = FlextResult.ok("success")
        assert result.success is True
        assert result.is_success is True
        assert result.failure is False
        assert result.is_failure is False
        assert result.value == "success"
        assert result.data == "success"
        assert result.error is None

    def test_fail_factory_method(self) -> None:
        """Test FlextResult.fail() factory method creates failure results."""
        result: FlextResult[object] = FlextResult.fail("error message")
        assert result.success is False
        assert result.is_success is False
        assert result.failure is True
        assert result.is_failure is True
        assert result.error == "error message"

    def test_fail_with_error_code_and_data(self) -> None:
        """Test FlextResult.fail() with error code and metadata."""
        result: FlextResult[object] = FlextResult.fail(
            "validation error",
            error_code="VALIDATION_ERROR",
            error_data={"field": "email", "value": "invalid"},
        )
        assert result.failure is True
        assert result.error == "validation error"
        assert result.error_code == "VALIDATION_ERROR"
        assert result.error_data == {"field": "email", "value": "invalid"}
        assert result.metadata == {"field": "email", "value": "invalid"}

    def test_fail_empty_error_normalization(self) -> None:
        """Test that empty/whitespace errors are normalized."""
        result1: FlextResult[object] = FlextResult.fail("")
        result2: FlextResult[object] = FlextResult.fail("   ")
        assert result1.error == "Unknown error occurred"
        assert result2.error == "Unknown error occurred"

    def test_map_success_transformation(self) -> None:
        """Test map() transforms success values correctly."""
        result: FlextResult[object] = FlextResult.ok(10).map(lambda x: x * 2)
        assert result.success is True
        assert result.value == 20

    def test_map_failure_propagation(self) -> None:
        """Test map() propagates failures without transformation."""
        result = FlextResult.fail("error").map(lambda x: x * 2)
        assert result.failure is True
        assert result.error == "error"

    def test_map_transformation_exception_handling(self) -> None:
        """Test map() handles transformation exceptions."""
        result: FlextResult[object] = FlextResult.ok("text").map(
            int,
        )  # Will raise ValueError
        assert result.failure is True
        assert "Transformation error:" in (result.error or "")
        assert result.error_code == "EXCEPTION_ERROR"

    def test_flat_map_success_chaining(self) -> None:
        """Test flat_map() chains successful operations."""

        def double_and_validate(x: int) -> FlextResult[int]:
            doubled = x * 2
            return (
                FlextResult.ok(doubled) if doubled > 0 else FlextResult.fail("negative")
            )

        result = FlextResult.ok(5).flat_map(double_and_validate)
        assert result.success is True
        assert result.value == 10

    def test_flat_map_failure_propagation(self) -> None:
        """Test flat_map() propagates failures."""

        def process(_: object) -> FlextResult[int]:
            return FlextResult.ok(42)

        result = FlextResult.fail("error").flat_map(process)
        assert result.failure is True
        assert result.error == "error"

    def test_flat_map_chained_failure(self) -> None:
        """Test flat_map() handles failures from chained operations."""

        def failing_operation(_: object) -> FlextResult[int]:
            return FlextResult.fail("chained error")

        result = FlextResult.ok(10).flat_map(failing_operation)
        assert result.failure is True
        assert result.error == "chained error"

    def test_flat_map_exception_handling(self) -> None:
        """Test flat_map() handles exceptions in chained operations."""

        def raising_operation(_: object) -> FlextResult[int]:
            msg = "operation failed"
            raise ValueError(msg)

        result = FlextResult.ok(10).flat_map(raising_operation)
        assert result.failure is True
        assert "Chained operation failed:" in (result.error or "")
        assert result.error_code == "BIND_ERROR"

    def test_unwrap_success(self) -> None:
        """Test unwrap() returns value for successful results."""
        result = FlextResult.ok("success")
        assert result.unwrap() == "success"

    def test_unwrap_failure_raises_exception(self) -> None:
        """Test unwrap() raises RuntimeError for failures."""
        result: FlextResult[None] = FlextResult.fail("error")
        with pytest.raises(RuntimeError, match="error"):
            result.unwrap()

    def test_unwrap_or_success(self) -> None:
        """Test unwrap_or() returns value for successful results."""
        result = FlextResult.ok("success")
        assert result.unwrap_or("default") == "success"

    def test_unwrap_or_failure_returns_default(self) -> None:
        """Test unwrap_or() returns default for failures."""
        result: FlextResult[object] = FlextResult.fail("error")
        assert result.unwrap_or("default") == "default"

    def test_expect_success(self) -> None:
        """Test expect() returns value for successful results."""
        result = FlextResult.ok("success")
        assert result.expect("should work") == "success"

    def test_expect_failure_raises_with_message(self) -> None:
        """Test expect() raises RuntimeError with custom message for failures."""
        result: FlextResult[object] = FlextResult.fail("validation error")
        with pytest.raises(
            RuntimeError,
            match="Expected valid result: validation error",
        ):
            result.expect("Expected valid result")

    def test_expect_none_data_validation(self) -> None:
        """Test expect() validates None data for safety."""
        # Create result with None data directly (edge case)
        result = FlextResult(data=None)
        with pytest.raises(RuntimeError, match="Success result has None data"):
            result.expect("Should have data")

    def test_boolean_conversion(self) -> None:
        """Test __bool__() conversion for success/failure states."""
        success_result = FlextResult.ok("data")
        failure_result: FlextResult[str] = FlextResult.fail("error")
        assert bool(success_result) is True
        assert bool(failure_result) is False

    def test_iterator_destructuring_success(self) -> None:
        """Test __iter__() for destructuring successful results."""
        result = FlextResult.ok("data")
        value, error = result
        assert value == "data"
        assert error is None

    def test_iterator_destructuring_failure(self) -> None:
        """Test __iter__() for destructuring failed results."""
        result: FlextResult[object] = FlextResult.fail("error")
        value, error = result
        assert value is None
        assert error == "error"

    def test_subscript_access_success(self) -> None:
        """Test __getitem__() subscript access for successful results."""
        result = FlextResult.ok("data")
        assert result[0] == "data"
        assert result[1] is None

    def test_subscript_access_failure(self) -> None:
        """Test __getitem__() subscript access for failed results."""
        result: FlextResult[object] = FlextResult.fail("error")
        assert result[0] is None
        assert result[1] == "error"

    def test_subscript_access_invalid_index(self) -> None:
        """Test __getitem__() raises IndexError for invalid indices."""
        result: FlextResult[object] = FlextResult.ok("data")
        with pytest.raises(IndexError, match="FlextResult only supports indices 0"):
            _ = result[2]

    def test_or_operator_success(self) -> None:
        """Test __or__() operator returns value for successful results."""
        result = FlextResult.ok("success")
        assert (result | "default") == "success"

    def test_or_operator_failure(self) -> None:
        """Test __or__() operator returns default for failures."""
        result: FlextResult[object] = FlextResult.fail("error")
        assert (result | "default") == "default"

    def test_context_manager_success(self) -> None:
        """Test context manager __enter__/__exit__ for successful results."""
        result = FlextResult.ok("resource")
        with result as resource:
            assert resource == "resource"

    def test_context_manager_failure(self) -> None:
        """Test context manager raises RuntimeError for failures."""
        result: FlextResult[object] = FlextResult.fail("error")
        with pytest.raises(RuntimeError, match="error"), result:
            pass

    def test_value_or_none_success(self) -> None:
        """Test value_or_none property returns value for successful results."""
        result = FlextResult.ok("data")
        assert result.value_or_none == "data"

    def test_value_or_none_failure(self) -> None:
        """Test value_or_none property returns None for failures."""
        result: FlextResult[object] = FlextResult.fail("error")
        assert result.value_or_none is None

    def test_equality_comparison(self) -> None:
        """Test __eq__() equality comparison between results."""
        result1 = FlextResult.ok("data")
        result2 = FlextResult.ok("data")
        result3 = FlextResult.ok("different")
        result4: FlextResult[object] = FlextResult.fail("error")

        assert result1 == result2
        assert result1 != result3
        assert result1 != result4

    def test_equality_with_non_result(self) -> None:
        """Test __eq__() returns False when comparing with non-FlextResult objects."""
        result = FlextResult.ok("data")
        assert result != "data"
        assert result is not None
        assert result != 42

    def test_hash_success_results(self) -> None:
        """Test __hash__() for successful results."""
        result1 = FlextResult.ok("data")
        result2 = FlextResult.ok("data")
        result3 = FlextResult.ok("different")

        # Same data should have same hash
        assert hash(result1) == hash(result2)
        # Different data should have different hash (usually)
        assert hash(result1) != hash(result3)

    def test_hash_failure_results(self) -> None:
        """Test __hash__() for failed results."""
        result1: FlextResult[object] = FlextResult.fail("error", error_code="ERR001")
        result2: FlextResult[object] = FlextResult.fail("error", error_code="ERR001")
        result3: FlextResult[object] = FlextResult.fail("different error")

        assert hash(result1) == hash(result2)
        assert hash(result1) != hash(result3)

    def test_hash_non_hashable_data(self) -> None:
        """Test __hash__() handles non-hashable data gracefully."""
        # Create result with non-hashable data (dict)
        result = FlextResult.ok({"key": "value"})
        # Should not raise exception
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_repr_success(self) -> None:
        """Test __repr__() string representation for successful results."""
        result = FlextResult.ok("data")
        repr_str = repr(result)
        assert repr_str == "FlextResult(data='data', is_success=True, error=None)"

    def test_repr_failure(self) -> None:
        """Test __repr__() string representation for failed results."""
        result: FlextResult[object] = FlextResult.fail("error")
        repr_str = repr(result)
        assert repr_str == "FlextResult(data=None, is_success=False, error='error')"

    def test_then_alias_for_flat_map(self) -> None:
        """Test then() method is alias for flat_map()."""

        def double_operation(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        result = FlextResult.ok(5).then(double_operation)
        assert result.success is True
        assert result.value == 10

    def test_bind_alias_for_flat_map(self) -> None:
        """Test bind() method is alias for flat_map() (monadic bind)."""

        def increment_operation(x: int) -> FlextResult[int]:
            return FlextResult.ok(x + 1)

        result = FlextResult.ok(10).bind(increment_operation)
        assert result.success is True
        assert result.value == 11

    def test_or_else_success_returns_self(self) -> None:
        """Test or_else() returns self for successful results."""
        success_result = FlextResult.ok("success")
        alternative = FlextResult.ok("alternative")
        assert success_result.or_else(alternative) is success_result

    def test_or_else_failure_returns_alternative(self) -> None:
        """Test or_else() returns alternative for failed results."""
        failure_result: FlextResult[str] = FlextResult.fail("error")
        alternative: FlextResult[str] = FlextResult.ok("alternative")
        assert failure_result.or_else(alternative) is alternative

    def test_or_else_get_success_returns_self(self) -> None:
        """Test or_else_get() returns self for successful results."""
        success_result = FlextResult.ok("success")

        def alternative_func() -> FlextResult[str]:
            return FlextResult.ok("alternative")

        assert success_result.or_else_get(alternative_func) is success_result

    def test_or_else_get_failure_calls_function(self) -> None:
        """Test or_else_get() calls function for failed results."""
        failure_result: FlextResult[str] = FlextResult.fail("error")

        def alternative_func() -> FlextResult[str]:
            return FlextResult.ok("alternative")

        result = failure_result.or_else_get(alternative_func)
        assert result.success is True
        assert result.value == "alternative"

    def test_or_else_get_function_exception_handling(self) -> None:
        """Test or_else_get() handles exceptions in alternative function."""
        failure_result: FlextResult[str] = FlextResult.fail("error")

        def failing_func() -> FlextResult[str]:
            msg = "alternative failed"
            raise ValueError(msg)

        result = failure_result.or_else_get(failing_func)
        assert result.failure is True
        assert result.error == "alternative failed"

    def test_recover_success_returns_self(self) -> None:
        """Test recover() returns self for successful results."""
        success_result = FlextResult.ok("success")
        result = success_result.recover(lambda _: "recovered")
        assert result is success_result

    def test_recover_failure_applies_function(self) -> None:
        """Test recover() applies recovery function to error."""
        failure_result: FlextResult[object] = FlextResult.fail("original error")
        result = failure_result.recover(lambda err: f"recovered from: {err}")
        assert result.success is True
        assert result.value == "recovered from: original error"

    def test_recover_function_exception_handling(self) -> None:
        """Test recover() handles exceptions in recovery function."""
        failure_result: FlextResult[object] = FlextResult.fail("error")

        def failing_recovery(_: str) -> str:
            msg = "recovery failed"
            raise ValueError(msg)

        result = failure_result.recover(failing_recovery)
        assert result.failure is True
        assert result.error == "recovery failed"

    def test_recover_with_success_returns_self(self) -> None:
        """Test recover_with() returns self for successful results."""
        success_result = FlextResult.ok("success")
        result = success_result.recover_with(lambda _: FlextResult.ok("recovered"))
        assert result is success_result

    def test_recover_with_failure_applies_function(self) -> None:
        """Test recover_with() applies recovery function returning FlextResult."""
        failure_result: FlextResult[object] = FlextResult.fail("original error")
        result = failure_result.recover_with(
            lambda err: FlextResult.ok(f"recovered from: {err}"),
        )
        assert result.success is True
        assert result.value == "recovered from: original error"

    def test_recover_with_function_exception_handling(self) -> None:
        """Test recover_with() handles exceptions in recovery function."""
        failure_result = FlextResult.fail("error")

        def failing_recovery(_: str) -> FlextResult[str]:
            msg = "recovery failed"
            raise ValueError(msg)

        result = failure_result.recover_with(failing_recovery)
        assert result.failure is True
        assert result.error == "recovery failed"

    def test_tap_success_executes_side_effect(self) -> None:
        """Test tap() executes side effect on successful results."""
        side_effects: list[str] = []

        result = FlextResult.ok("data").tap(
            lambda x: side_effects.append(f"processed: {x}"),
        )

        assert result.success is True
        assert result.value == "data"
        assert side_effects == ["processed: data"]

    def test_tap_failure_skips_side_effect(self) -> None:
        """Test tap() skips side effect on failed results."""
        side_effects: list[str] = []

        result: FlextResult[object] = FlextResult.fail("error").tap(
            lambda x: side_effects.append(str(x))
        )

        assert result.failure is True
        assert result.error == "error"
        assert side_effects == []

    def test_tap_none_data_skips_side_effect(self) -> None:
        """Test tap() skips side effect when data is None."""
        side_effects: list[str] = []

        FlextResult(data=None).tap(lambda x: side_effects.append(str(x)))

        assert side_effects == []

    def test_tap_error_failure_executes_side_effect(self) -> None:
        """Test tap_error() executes side effect on failed results."""
        side_effects: list[str] = []

        result = FlextResult.fail("error").tap_error(
            lambda err: side_effects.append(f"error: {err}"),
        )

        assert result.failure is True
        assert result.error == "error"
        assert side_effects == ["error: error"]

    def test_tap_error_success_skips_side_effect(self) -> None:
        """Test tap_error() skips side effect on successful results."""
        side_effects: list[str] = []

        result = FlextResult.ok("data").tap_error(
            lambda err: side_effects.append(str(err)),
        )

        assert result.success is True
        assert result.value == "data"
        assert side_effects == []

    def test_filter_success_passes_predicate(self) -> None:
        """Test filter() passes when predicate is true for successful results."""
        result = FlextResult.ok(10).filter(lambda x: x > 5, "Value too small")
        assert result.success is True
        assert result.value == 10

    def test_filter_success_fails_predicate(self) -> None:
        """Test filter() fails when predicate is false for successful results."""
        result = FlextResult.ok(3).filter(lambda x: x > 5, "Value too small")
        assert result.failure is True
        assert result.error == "Value too small"

    def test_filter_failure_propagates(self) -> None:
        """Test filter() propagates failures without checking predicate."""
        result = FlextResult.fail("original error").filter(
            lambda x: x > 5,
            "Value too small",
        )
        assert result.failure is True
        assert result.error == "original error"

    def test_filter_predicate_exception_handling(self) -> None:
        """Test filter() handles exceptions in predicate function."""

        def bad_predicate(x: object) -> bool:
            return operator.gt(x, 5)

        result = FlextResult.ok("text").filter(
            bad_predicate,
            "Failed",
        )
        assert result.failure is True

    def test_zip_with_both_success(self) -> None:
        """Test zip_with() combines two successful results."""
        result1 = FlextResult.ok(10)
        result2 = FlextResult.ok(20)

        result = result1.zip_with(result2, operator.add)

        assert result.success is True
        assert result.value == 30

    def test_zip_with_first_failure(self) -> None:
        """Test zip_with() fails when first result is failure."""
        result1 = FlextResult.fail("first error")
        result2 = FlextResult.ok(20)

        result = result1.zip_with(result2, operator.add)

        assert result.failure is True
        assert result.error == "first error"

    def test_zip_with_second_failure(self) -> None:
        """Test zip_with() fails when second result is failure."""
        result1: FlextResult[int] = FlextResult.ok(10)
        result2: FlextResult[int] = FlextResult.fail("second error")

        result = result1.zip_with(result2, operator.add)

        assert result.failure is True
        assert result.error == "second error"

    def test_zip_with_none_data_handling(self) -> None:
        """Test zip_with() handles None data appropriately."""
        result1 = FlextResult(data=None)
        result2 = FlextResult.ok(20)

        result = result1.zip_with(result2, lambda x, y: (x, y))

        assert result.failure is True
        assert result.error == "Missing data for zip operation"

    def test_zip_with_function_exception_handling(self) -> None:
        """Test zip_with() handles exceptions in combining function."""
        result1 = FlextResult.ok(10)
        result2 = FlextResult.ok(0)

        result = result1.zip_with(result2, operator.truediv)  # Division by zero

        assert result.failure is True

    def test_to_either_success(self) -> None:
        """Test to_either() tuple conversion for successful results."""
        result = FlextResult.ok("data")
        data, error = result.to_either()
        assert data == "data"
        assert error is None

    def test_to_either_failure(self) -> None:
        """Test to_either() tuple conversion for failed results."""
        result = FlextResult.fail("error")
        data, error = result.to_either()
        assert data is None
        assert error == "error"

    def test_to_exception_success_returns_none(self) -> None:
        """Test to_exception() returns None for successful results."""
        result = FlextResult.ok("data")
        exception = result.to_exception()
        assert exception is None

    def test_to_exception_failure_returns_runtime_error(self) -> None:
        """Test to_exception() returns RuntimeError for failed results."""
        result = FlextResult.fail("error message")
        exception = result.to_exception()
        assert isinstance(exception, RuntimeError)
        assert str(exception) == "error message"

    def test_from_exception_success(self) -> None:
        """Test from_exception() class method wraps successful function call."""

        def successful_function() -> str:
            return "success"

        result = FlextResult.from_exception(successful_function)
        assert result.success is True
        assert result.value == "success"

    def test_from_exception_handles_exception(self) -> None:
        """Test from_exception() class method wraps exceptions as failures."""

        def failing_function() -> str:
            msg = "function failed"
            raise ValueError(msg)

        result = FlextResult.from_exception(failing_function)
        assert result.failure is True
        assert result.error == "function failed"

    def test_chain_results_all_success(self) -> None:
        """Test chain_results() static method with all successful results."""
        results = [FlextResult.ok(1), FlextResult.ok(2), FlextResult.ok(3)]
        chained = FlextResult.chain_results(*results)

        assert chained.success is True
        assert chained.value == [1, 2, 3]

    def test_chain_results_with_failure(self) -> None:
        """Test chain_results() static method fails on first failure."""
        results = [FlextResult.ok(1), FlextResult.fail("error"), FlextResult.ok(3)]
        chained = FlextResult.chain_results(*results)

        assert chained.failure is True
        assert chained.error == "error"

    def test_chain_results_empty_list(self) -> None:
        """Test chain_results() with empty input returns successful empty list."""
        chained = FlextResult.chain_results()

        assert chained.success is True
        assert chained.value == []

    def test_combine_all_success(self) -> None:
        """Test combine() static method with all successful results."""
        results = [FlextResult.ok(1), FlextResult.ok(2), FlextResult.ok(3)]
        combined = FlextResult.combine(*results)

        assert combined.success is True
        assert combined.value == [1, 2, 3]

    def test_combine_with_failure(self) -> None:
        """Test combine() static method fails on any failure."""
        results = [FlextResult.ok(1), FlextResult.fail("error"), FlextResult.ok(3)]
        combined = FlextResult.combine(*results)

        assert combined.failure is True
        assert combined.error == "error"

    def test_combine_filters_none_values(self) -> None:
        """Test combine() filters out None values from successful results."""
        results = [FlextResult.ok(1), FlextResult(data=None), FlextResult.ok(3)]
        combined = FlextResult.combine(*results)

        assert combined.success is True
        assert combined.value == [1, 3]

    def test_all_success_all_succeed(self) -> None:
        """Test all_success() static method returns True when all results succeed."""
        results = [FlextResult.ok(1), FlextResult.ok(2), FlextResult.ok(3)]
        assert FlextResult.all_success(*results) is True

    def test_all_success_some_fail(self) -> None:
        """Test all_success() static method returns False when any result fails."""
        results = [FlextResult.ok(1), FlextResult.fail("error"), FlextResult.ok(3)]
        assert FlextResult.all_success(*results) is False

    def test_any_success_some_succeed(self) -> None:
        """Test any_success() static method returns True when any result succeeds."""
        results: list[FlextResult[int]] = [
            FlextResult.fail("error1"),
            FlextResult.ok(2),
            FlextResult.fail("error2"),
        ]
        assert FlextResult.any_success(*results) is True

    def test_any_success_all_fail(self) -> None:
        """Test any_success() static method returns False when all results fail."""
        results: list[FlextResult[int]] = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("error3"),
        ]
        assert FlextResult.any_success(*results) is False

    def test_first_success_finds_first(self) -> None:
        """Test first_success() class method returns first successful result."""
        results = [
            FlextResult.fail("error1"),
            FlextResult.ok("success"),
            FlextResult.fail("error2"),
        ]
        first = FlextResult.first_success(*results)

        assert first.success is True
        assert first.value == "success"

    def test_first_success_all_fail(self) -> None:
        """Test first_success() class method fails when all results fail."""
        results = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("error3"),
        ]
        first = FlextResult.first_success(*results)

        assert first.failure is True
        assert first.error == "error3"  # Last error

    def test_try_all_first_succeeds(self) -> None:
        """Test try_all() class method returns first successful function result."""

        def func1() -> str:
            msg = "failed"
            raise ValueError(msg)

        def func2() -> str:
            return "success"

        def func3() -> str:
            return "also success"

        result = FlextResult.try_all(func1, func2, func3)
        assert result.success is True
        assert result.value == "success"

    def test_try_all_all_fail(self) -> None:
        """Test try_all() class method fails when all functions fail."""

        def func1() -> str:
            msg = "error1"
            raise ValueError(msg)

        def func2() -> str:
            msg = "error2"
            raise RuntimeError(msg)

        result = FlextResult.try_all(func1, func2)
        assert result.failure is True
        assert result.error == "error2"  # Last error

    def test_try_all_no_functions(self) -> None:
        """Test try_all() class method fails when no functions provided."""
        result: FlextResult[object] = FlextResult.try_all()
        assert result.failure is True
        assert result.error == "No functions provided"

    def test_safe_unwrap_or_none_success(self) -> None:
        """Test safe_unwrap_or_none() class method for successful results."""
        result = FlextResult.ok("data")
        value = FlextResult.safe_unwrap_or_none(result)
        assert value == "data"

    def test_safe_unwrap_or_none_failure(self) -> None:
        """Test safe_unwrap_or_none() class method for failed results."""
        result: FlextResult[object] = FlextResult.fail("error")
        value = FlextResult.safe_unwrap_or_none(result)
        assert value is None

    def test_unwrap_or_raise_success(self) -> None:
        """Test unwrap_or_raise() class method for successful results."""
        result = FlextResult.ok("data")
        value = FlextResult.unwrap_or_raise(result)
        assert value == "data"

    def test_unwrap_or_raise_failure(self) -> None:
        """Test unwrap_or_raise() class method raises exception for failures."""
        result: FlextResult[object] = FlextResult.fail("error")
        with pytest.raises(RuntimeError, match="error"):
            FlextResult.unwrap_or_raise(result)

    def test_unwrap_or_raise_custom_exception(self) -> None:
        """Test unwrap_or_raise() class method with custom exception type."""
        result: FlextResult[object] = FlextResult.fail("validation error")
        with pytest.raises(ValueError, match="validation error"):
            FlextResult.unwrap_or_raise(result, ValueError)

    def test_collect_successes_mixed_results(self) -> None:
        """Test collect_successes() class method filters successful values."""
        results = [
            FlextResult.ok(1),
            FlextResult.fail("error"),
            FlextResult.ok(3),
            FlextResult.fail("error2"),
        ]
        successes = FlextResult.collect_successes(results)
        assert successes == [1, 3]

    def test_collect_failures_mixed_results(self) -> None:
        """Test collect_failures() class method filters error messages."""
        results = [
            FlextResult.ok(1),
            FlextResult.fail("error1"),
            FlextResult.ok(3),
            FlextResult.fail("error2"),
        ]
        failures = FlextResult.collect_failures(results)
        assert failures == ["error1", "error2"]

    def test_success_rate_mixed_results(self) -> None:
        """Test success_rate() class method calculates percentage correctly."""
        results = [
            FlextResult.ok(1),
            FlextResult.fail("error"),
            FlextResult.ok(3),
            FlextResult.ok(4),
        ]
        rate = FlextResult.success_rate(results)
        assert rate == 75.0  # 3 out of 4 successful

    def test_success_rate_empty_list(self) -> None:
        """Test success_rate() class method returns 0.0 for empty list."""
        rate = FlextResult.success_rate([])
        assert rate == 0.0

    def test_batch_process_mixed_results(self) -> None:
        """Test batch_process() class method separates successes and failures."""

        def process_item(x: int) -> FlextResult[str]:
            if x > 0:
                return FlextResult.ok(f"processed_{x}")
            return FlextResult.fail(f"negative_{x}")

        items = [1, -1, 2, -2, 3]
        successes, failures = FlextResult.batch_process(items, process_item)

        assert successes == ["processed_1", "processed_2", "processed_3"]
        assert failures == ["negative_-1", "negative_-2"]

    def test_safe_call_success(self) -> None:
        """Test safe_call() class method wraps successful function execution."""

        def successful_func() -> str:
            return "success"

        result = FlextResult.safe_call(successful_func)
        assert result.success is True
        assert result.value == "success"

    def test_safe_call_exception(self) -> None:
        """Test safe_call() class method wraps exceptions as failures."""

        def failing_func() -> str:
            msg = "function failed"
            raise ValueError(msg)

        result = FlextResult.safe_call(failing_func)
        assert result.failure is True
        assert result.error == "function failed"

    def test_create_failure_alias(self) -> None:
        """Test create_failure() class method is alias for fail()."""
        result: FlextResult[object] = FlextResult.create_failure(
            "error",
            error_code="ERR001",
            error_data={"field": "test"},
        )
        assert result.failure is True
        assert result.error == "error"
        assert result.error_code == "ERR001"
        assert result.error_data == {"field": "test"}

    def test_property_aliases(self) -> None:
        """Test various property aliases work correctly."""
        success_result = FlextResult.ok("data")
        failure_result: FlextResult[object] = FlextResult.fail("error")

        # Test success aliases
        assert success_result.is_valid is True
        assert success_result.is_fail is False
        assert success_result.error_message is None

        # Test failure aliases
        assert failure_result.is_valid is False
        assert failure_result.is_fail is True
        assert failure_result.error_message == "error"

    def test_value_property_access_failure(self) -> None:
        """Test value property raises TypeError when accessed on failure."""
        result = FlextResult.fail("error")
        with pytest.raises(
            TypeError,
            match="Attempted to access value on failed result",
        ):
            _ = result.value

    def test_ensure_success_data_none_handling(self) -> None:
        """Test _ensure_success_data() handles None data case."""
        # Create a result with None data and test internal consistency
        result = FlextResult(data=None)
        with pytest.raises(RuntimeError, match="Success result has None data"):
            # This should trigger the None data validation in _ensure_success_data
            result._ensure_success_data()

    def test_is_success_state_type_guard(self) -> None:
        """Test _is_success_state() type guard functionality."""
        result = FlextResult.ok("data")
        # Test with actual data
        assert result._is_success_state("data") is True
        # Test with None (should return False)
        assert result._is_success_state(None) is False

        # Test with failed result
        failed_result: FlextResult[object] = FlextResult.fail("error")
        assert failed_result._is_success_state("data") is False

    def test_context_manager_none_data_edge_case(self) -> None:
        """Test context manager with None data edge case."""
        # Create result with None data but success state
        result = FlextResult(data=None)
        with pytest.raises(RuntimeError, match="Success result has None data"), result:
            pass

    def test_or_operator_none_data_handling(self) -> None:
        """Test __or__() operator handles None data correctly."""
        # Create result with None data but success state
        result = FlextResult(data=None)
        default_value = "default"
        # Ensure or-operator returns default when data is None
        assert (cast("FlextResult[object]", result) | default_value) == default_value

    def test_additional_edge_cases_for_complete_coverage(self) -> None:
        """Test additional edge cases to reach complete coverage."""
        # Test map with various exception types for complete error handling coverage
        result = FlextResult.ok("text")

        # Test with function that raises different exceptions
        def raise_attribute_error(_: str) -> str:
            msg = "attribute error"
            raise AttributeError(msg)

        mapped = result.map(raise_attribute_error)
        assert mapped.failure is True
        assert "Transformation error:" in (mapped.error or "")

        # Test flat_map with different exception types
        def raise_key_error(_: str) -> FlextResult[str]:
            msg = "key error"
            raise KeyError(msg)

        flat_mapped = result.flat_map(raise_key_error)
        assert flat_mapped.failure is True
        assert "Chained operation failed:" in (flat_mapped.error or "")

    def test_comprehensive_exception_coverage(self) -> None:
        """Test comprehensive exception handling coverage."""

        # Test try_all with ArithmeticError
        def divide_by_zero() -> float:
            return 1.0 / 0.0

        def successful_func() -> str:
            return "success"

        result = FlextResult.try_all(divide_by_zero, successful_func)
        assert result.success is True
        assert result.value == "success"

        # Test equality comparison exception handling
        class BadEqualityObject:
            def __eq__(self, other: object) -> bool:
                msg = "Equality failed"
                raise ValueError(msg)

            def __hash__(self) -> int:
                return hash("BadEqualityObject")

        result1 = FlextResult.ok(BadEqualityObject())
        result2 = FlextResult.ok("different")
        # Should return False when comparison raises exception
        assert (result1 == result2) is False

    def test_filter_none_data_edge_case(self) -> None:
        """Test filter() with None data edge case."""
        # Create result with None data but success state
        result = FlextResult(data=None)
        with pytest.raises(RuntimeError, match="Success result has None data"):
            result.filter(lambda x: x is not None, "Should not be None")

    def test_hash_object_with_dict_attributes(self) -> None:
        """Test __hash__() with object containing dict attributes."""

        class TestObject:
            def __init__(self) -> None:
                self.__dict__ = {"key": "value", "number": 42}

        obj = TestObject()
        result = FlextResult.ok(obj)
        # Should not raise exception and return valid hash
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_hash_complex_object_fallback(self) -> None:
        """Test __hash__() fallback for complex non-hashable objects."""

        class ComplexObject:
            def __init__(self) -> None:
                self.data = {"nested": [1, 2, 3]}

            def __getattribute__(self, name: str) -> object:
                if name == "__dict__":
                    msg = "No __dict__ attribute"
                    raise AttributeError(msg)
                return object.__getattribute__(self, name)

        # Create object without __dict__
        obj = ComplexObject()
        result = FlextResult.ok(obj)
        # Should fall back to type name + id hash
        hash_value = hash(result)
        assert isinstance(hash_value, int)
