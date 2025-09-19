"""Tests for FlextResult with high coverage target.

This test file focuses on covering all uncovered lines and edge cases
using real functional tests that validate the expected behavior.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from typing import cast
from unittest.mock import Mock

import pytest

from flext_core import FlextConstants, FlextResult


class TestFlextResultCompleteCoverage:
    """Complete coverage tests for FlextResult with real functionality testing."""

    def test_is_success_state_type_guard(self) -> None:
        """Test _is_success_state type guard functionality."""
        result_success = FlextResult[str].ok("test")
        result_failure = FlextResult[str].fail("error")

        # Test the type guard method exists and works
        assert hasattr(result_success, "_is_success_state")
        assert result_success._is_success_state("value") is True
        assert result_success._is_success_state(None) is False
        assert result_failure._is_success_state("value") is False

    def test_error_property_access(self) -> None:
        """Test error property access."""
        result = FlextResult[str].fail("test error")
        assert result.error == "test error"

        success_result = FlextResult[str].ok("value")
        assert success_result.error is None

    def test_error_data_property_access(self) -> None:
        """Test error_data property access."""
        error_data = {"key": "value", "code": 404}
        result = FlextResult[str].fail("error", error_data=error_data)

        assert result.error_data == error_data
        # Test the property is accessible
        assert result.error_data == error_data

    def test_fail_with_empty_error_normalization(self) -> None:
        """Test fail method with empty/whitespace error normalization."""
        # Empty string
        result1 = FlextResult[str].fail("")
        assert result1.error == "Unknown error occurred"

        # Whitespace only
        result2 = FlextResult[str].fail("   \n\t  ")
        assert result2.error == "Unknown error occurred"

        # None error (should not happen but test edge case)
        # This tests the fail classmethod normalization
        result3 = FlextResult[str].fail("normal error")
        assert result3.error == "normal error"

    def test_chain_results_edge_cases(self) -> None:
        """Test chain_results with edge cases."""
        # Empty results
        empty_chain = FlextResult.chain_results()
        assert empty_chain.is_success
        assert empty_chain.value == []

        # Single success
        single_result = FlextResult[str].ok("single")
        chain_single = FlextResult.chain_results(
            cast("FlextResult[object]", single_result),
        )
        assert chain_single.is_success
        assert chain_single.value == ["single"]

        # Mixed success and failure - should fail on first failure
        success1 = FlextResult[str].ok("first")
        failure = FlextResult[str].fail("error occurred")
        success2 = FlextResult[str].ok("second")  # This shouldn't be reached

        chain_mixed = FlextResult.chain_results(
            cast("FlextResult[object]", success1),
            cast("FlextResult[object]", failure),
            cast("FlextResult[object]", success2),
        )
        assert chain_mixed.is_failure
        assert chain_mixed.error is not None
        assert "error occurred" in chain_mixed.error

    def test_map_with_exception_handling(self) -> None:
        """Test map method with various exception types."""
        result = FlextResult[int].ok(42)

        # Test ValueError exception
        def raise__value__error(__: int) -> str:
            msg = "Invalid value"
            raise ValueError(msg)

        mapped_value_error = result.map(raise__value__error)
        assert mapped_value_error.is_failure
        assert mapped_value_error.error is not None
        assert "Transformation error: Invalid value" in mapped_value_error.error
        assert mapped_value_error.error_code == FlextConstants.Errors.EXCEPTION_ERROR

        # Test TypeError exception
        def raise__type__error(__: int) -> str:
            msg = "Type error"
            raise TypeError(msg)

        mapped_type_error = result.map(raise__type__error)
        assert mapped_type_error.is_failure
        assert mapped_type_error.error is not None
        assert "Transformation error: Type error" in mapped_type_error.error

        # Test AttributeError exception
        def raise__attribute__error(__: int) -> str:
            msg = "Attribute error"
            raise AttributeError(msg)

        mapped_attr_error = result.map(raise__attribute__error)
        assert mapped_attr_error.is_failure
        assert mapped_attr_error.error is not None
        assert "Transformation error: Attribute error" in mapped_attr_error.error

        # Test generic Exception (not in specific types)
        def raise__generic__error(__: int) -> str:
            msg = "Runtime error"
            raise RuntimeError(msg)

        mapped_generic_error = result.map(raise__generic__error)
        assert mapped_generic_error.is_failure
        assert mapped_generic_error.error is not None
        assert "Transformation failed: Runtime error" in mapped_generic_error.error
        assert mapped_generic_error.error_code == FlextConstants.Errors.MAP_ERROR

    def test_flat_map_with_exception_handling(self) -> None:
        """Test flat_map method with comprehensive exception handling."""
        result = FlextResult[int].ok(42)

        # Test specific exceptions
        def raise__type__error(__: int) -> FlextResult[str]:
            msg = "Type error in flat_map"
            raise TypeError(msg)

        flat_mapped = result.flat_map(raise__type__error)
        assert flat_mapped.is_failure
        assert flat_mapped.error is not None
        assert "Chained operation failed: Type error in flat_map" in flat_mapped.error
        assert flat_mapped.error_code == FlextConstants.Errors.BIND_ERROR

        # Test ValueError
        def raise__value__error(__: int) -> FlextResult[str]:
            msg = "Value error"
            raise ValueError(msg)

        flat_mapped_ve = result.flat_map(raise__value__error)
        assert flat_mapped_ve.is_failure
        assert flat_mapped_ve.error is not None
        assert "Chained operation failed: Value error" in flat_mapped_ve.error

        # Test AttributeError
        def raise__attr__error(__: int) -> FlextResult[str]:
            msg = "Attribute error"
            raise AttributeError(msg)

        flat_mapped_ae = result.flat_map(raise__attr__error)
        assert flat_mapped_ae.is_failure
        assert flat_mapped_ae.error is not None
        assert "Chained operation failed: Attribute error" in flat_mapped_ae.error

        # Test IndexError
        def raise__index__error(__: int) -> FlextResult[str]:
            msg = "Index error"
            raise IndexError(msg)

        flat_mapped_ie = result.flat_map(raise__index__error)
        assert flat_mapped_ie.is_failure
        assert flat_mapped_ie.error is not None
        assert "Chained operation failed: Index error" in flat_mapped_ie.error

        # Test KeyError
        def raise__key__error(__: int) -> FlextResult[str]:
            msg = "Key error"
            raise KeyError(msg)

        flat_mapped_ke = result.flat_map(raise__key__error)
        assert flat_mapped_ke.is_failure
        assert flat_mapped_ke.error is not None
        assert "Chained operation failed: 'Key error'" in flat_mapped_ke.error

        # Test generic Exception
        def raise__generic__error(__: int) -> FlextResult[str]:
            msg = "Runtime error in flat_map"
            raise RuntimeError(msg)

        flat_mapped_ge = result.flat_map(raise__generic__error)
        assert flat_mapped_ge.is_failure
        assert flat_mapped_ge.error is not None
        assert (
            "Unexpected chaining error: Runtime error in flat_map"
            in flat_mapped_ge.error
        )
        assert flat_mapped_ge.error_code == FlextConstants.Errors.CHAIN_ERROR

    def test_dunder_methods_comprehensive(self) -> None:
        """Test all dunder methods comprehensively."""
        success_result = FlextResult[str].ok("test")
        failure_result = FlextResult[str].fail("error")

        # Test __iter__ for both success and failure
        success_value, success_error = success_result
        assert success_value == "test"
        assert success_error is None

        failure_value, failure_error = failure_result
        assert failure_value is None
        assert failure_error == "error"

        # Test __getitem__
        assert success_result[0] == "test"
        assert success_result[1] is None
        assert failure_result[0] is None
        assert failure_result[1] == "error"

        # Test __getitem__ with invalid index
        with pytest.raises(IndexError, match="FlextResult only supports indices 0"):
            success_result[2]

        # Test __or__ operator (default values)
        assert success_result | "default" == "test"
        assert failure_result | "default" == "default"

        # Test __or__ with None data case
        none_result = FlextResult[str | None].ok(None)
        assert none_result | "default" == "default"

    def test_context_manager_functionality(self) -> None:
        """Test context manager (__enter__ and __exit__) functionality."""
        # Success case
        success_result = FlextResult[str].ok("context_value")
        with success_result as value:
            assert value == "context_value"

        # Failure case
        failure_result = FlextResult[str].fail("context_error")
        with (
            pytest.raises(RuntimeError, match="context_error"),
            failure_result as value,
        ):
            pass  # Should not reach here

    def test_hash_with_non_hashable_data(self) -> None:
        """Test __hash__ method with non-hashable data types."""
        # Test with dict (non-hashable) - use cast to satisfy MyPy variance
        dict_data: dict[str, object] = {"key": "value", "nested": {"inner": "data"}}
        result_with_dict = FlextResult[dict[str, object]].ok(dict_data)

        # Should not raise exception and return consistent hash
        hash1 = hash(result_with_dict)
        hash2 = hash(result_with_dict)
        assert hash1 == hash2

        # Test with object having __dict__
        class TestObject:
            def __init__(self) -> None:
                """Initialize the instance."""
                self.attr = "value"

        obj = TestObject()
        result_with_obj = FlextResult[TestObject].ok(obj)
        obj_hash = hash(result_with_obj)
        assert isinstance(obj_hash, int)

        # Test with complex object without __dict__
        mock_obj = Mock()
        del mock_obj.__dict__  # Remove __dict__ to test fallback
        result_with_mock = FlextResult[Mock].ok(mock_obj)
        mock_hash = hash(result_with_mock)
        assert isinstance(mock_hash, int)

    def test_expect_method_edge_cases(self) -> None:
        """Test expect method with edge cases."""
        # Success with non-None value
        success_result = FlextResult[str].ok("value")
        assert success_result.expect("Should work") == "value"

        # Failure case
        failure_result = FlextResult[str].fail("failed")
        with pytest.raises(RuntimeError, match="Custom message: failed"):
            failure_result.expect("Custom message")

        # Success with None value - expect should validate this
        none_success = FlextResult[str | None].ok(None)
        with pytest.raises(RuntimeError, match="Success result has None data"):
            none_success.expect("None validation")

    def test_or_else_methods(self) -> None:
        """Test or_else and or_else_get methods."""
        success = FlextResult[str].ok("original")
        failure = FlextResult[str].fail("error")
        alternative = FlextResult[str].ok("alternative")

        # or_else tests
        assert success.or_else(alternative) == success
        assert failure.or_else(alternative) == alternative

        # or_else_get tests
        def get_alternative() -> FlextResult[str]:
            return FlextResult[str].ok("function_alternative")

        assert success.or_else_get(get_alternative) == success
        result = failure.or_else_get(get_alternative)
        assert result.is_success
        assert result.value == "function_alternative"

        # or_else_get with exception
        def failing_function() -> FlextResult[str]:
            msg = "Function failed"
            raise ValueError(msg)

        error_result = failure.or_else_get(failing_function)
        assert error_result.is_failure
        assert error_result.error is not None
        assert "Function failed" in error_result.error

    def test_recover_methods_comprehensive(self) -> None:
        """Test recover and recover_with methods comprehensively."""
        success = FlextResult[str].ok("success")
        failure = FlextResult[str].fail("original_error")

        # recover method - success should pass through
        recovered_success = success.recover(lambda _: "recovered")
        assert recovered_success == success

        # recover method - failure should recover
        recovered_failure = failure.recover(lambda e: f"recovered_from_{e}")
        assert recovered_failure.is_success
        assert recovered_failure.value == "recovered_from_original_error"

        # recover method with exception
        def failing__recover(__: str) -> str:
            msg = "Recovery failed"
            raise ValueError(msg)

        failed_recovery = failure.recover(failing__recover)
        assert failed_recovery.is_failure
        assert failed_recovery.error is not None
        assert "Recovery failed" in failed_recovery.error

        # recover_with method - success should pass through
        recovered_with_success = success.recover_with(
            lambda _: FlextResult[str].ok("recovered_with"),
        )
        assert recovered_with_success == success

        # recover_with method - failure should recover
        recovered_with_failure = failure.recover_with(
            lambda e: FlextResult[str].ok(f"recovered_with_{e}"),
        )
        assert recovered_with_failure.is_success
        assert recovered_with_failure.value == "recovered_with_original_error"

        # recover_with method with exception
        def failing__recover__with(__: str) -> FlextResult[str]:
            msg = "Recovery with failed"
            raise TypeError(msg)

        failed_recovery_with = failure.recover_with(failing__recover__with)
        assert failed_recovery_with.is_failure
        assert failed_recovery_with.error is not None
        assert "Recovery with failed" in failed_recovery_with.error

    def test_tap_methods_with_side_effects(self) -> None:
        """Test tap and tap_error methods with real side effects."""
        side_effect_calls: list[str] = []

        def success_side_effect(value: str | None) -> None:
            side_effect_calls.append(f"success: {value}")

        def error_side_effect(error: str) -> None:
            side_effect_calls.append(f"error: {error}")

        # Test tap on success
        success_result = FlextResult[str].ok("tap_value")
        chained = success_result.tap(success_side_effect).tap_error(error_side_effect)
        assert chained == success_result  # Should return self
        assert "success: tap_value" in side_effect_calls
        assert len([c for c in side_effect_calls if "error:" in c]) == 0

        # Test tap_error on failure
        side_effect_calls.clear()
        failure_result = FlextResult[str].fail("tap_error")
        chained_failure = failure_result.tap(success_side_effect).tap_error(
            error_side_effect,
        )
        assert chained_failure == failure_result  # Should return self
        assert "error: tap_error" in side_effect_calls
        assert len([c for c in side_effect_calls if "success:" in c]) == 0

        # Test tap with None data - should not call function
        side_effect_calls.clear()
        none_success = FlextResult[str | None].ok(None)
        none_success.tap(success_side_effect)
        assert len(side_effect_calls) == 0  # Should not call function for None data

        # Test exception handling in side effects
        def failing__side__effect(__: str) -> None:
            msg = "Side effect failed"
            raise ValueError(msg)

        # Should not raise exception, just suppress it
        success_result.tap(failing__side__effect)

    def test_filter_method_comprehensive(self) -> None:
        """Test filter method with comprehensive scenarios."""
        success_result = FlextResult[int].ok(42)
        failure_result = FlextResult[int].fail("original_failure")

        # Filter success - predicate passes
        filtered_pass = success_result.filter(lambda x: x > 40, "Should pass")
        assert filtered_pass == success_result

        # Filter success - predicate fails
        filtered_fail = success_result.filter(lambda x: x > 50, "Should fail")
        assert filtered_fail.is_failure
        assert filtered_fail.error == "Should fail"

        # Filter failure - should pass through
        filtered_failure = failure_result.filter(lambda x: x > 40, "Won't be used")
        assert filtered_failure == failure_result

        # Filter with exception in predicate
        def failing__predicate(__: int) -> bool:
            msg = "Predicate failed"
            raise TypeError(msg)

        filtered_exception = success_result.filter(failing__predicate, "Error message")
        assert filtered_exception.is_failure
        assert filtered_exception.error is not None
        assert "Predicate failed" in filtered_exception.error

    def test_zip_with_method_comprehensive(self) -> None:
        """Test zip_with method with comprehensive scenarios."""
        success1 = FlextResult[int].ok(10)
        success2 = FlextResult[int].ok(20)
        failure1 = FlextResult[int].fail("first_error")
        failure2 = FlextResult[int].fail("second_error")

        # Both success
        zipped_success = success1.zip_with(success2, operator.add)
        assert zipped_success.is_success
        assert zipped_success.value == 30

        # First failure
        zipped_first_fail = failure1.zip_with(success2, operator.add)
        assert zipped_first_fail.is_failure
        assert zipped_first_fail.error is not None
        assert "first_error" in zipped_first_fail.error

        # Second failure
        zipped_second_fail = success1.zip_with(failure2, operator.add)
        assert zipped_second_fail.is_failure
        assert zipped_second_fail.error is not None
        assert "second_error" in zipped_second_fail.error

        # Missing data (None values) - should fail
        none_result1 = FlextResult[int | None].ok(None)
        none_result2 = FlextResult[int | None].ok(None)
        zipped_none = none_result1.zip_with(none_result2, lambda x, y: (x, y))
        assert zipped_none.is_failure
        assert zipped_none.error is not None
        assert "Missing data for zip operation" in zipped_none.error

        # Exception in zip function
        def failing_zip_func(_: int, ___: int) -> int:
            msg = "Division by zero"
            raise ZeroDivisionError(msg)

        zipped_exception = success1.zip_with(success2, failing_zip_func)
        assert zipped_exception.is_failure
        assert zipped_exception.error is not None
        assert "Division by zero" in zipped_exception.error

    def test_conversion_methods(self) -> None:
        """Test to_either and to_exception conversion methods."""
        success_result = FlextResult[str].ok("success_data")
        failure_result = FlextResult[str].fail("failure_message")

        # to_either method
        success_either = success_result.to_either()
        assert success_either == ("success_data", None)

        failure_either = failure_result.to_either()
        assert failure_either == (None, "failure_message")

        # to_exception method
        success_exception = success_result.to_exception()
        assert success_exception is None

        failure_exception = failure_result.to_exception()
        assert isinstance(failure_exception, RuntimeError)
        assert str(failure_exception) == "failure_message"

    def test_from_exception_class_method(self) -> None:
        """Test from_exception class method with various scenarios."""

        # Success case
        def successful_func() -> str:
            return "success"

        result_success = FlextResult.from_exception(successful_func)
        assert result_success.is_success
        assert result_success.value == "success"

        # Exception cases
        def failing_func_type_error() -> str:
            msg = "Type error occurred"
            raise TypeError(msg)

        result_type_error = FlextResult.from_exception(failing_func_type_error)
        assert result_type_error.is_failure
        assert result_type_error.error is not None
        assert "Type error occurred" in result_type_error.error

        def failing_func_value_error() -> str:
            msg = "Value error occurred"
            raise ValueError(msg)

        result_value_error = FlextResult.from_exception(failing_func_value_error)
        assert result_value_error.is_failure
        assert result_value_error.error is not None
        assert "Value error occurred" in result_value_error.error

        def failing_func_attr_error() -> str:
            msg = "Attribute error occurred"
            raise AttributeError(msg)

        result_attr_error = FlextResult.from_exception(failing_func_attr_error)
        assert result_attr_error.is_failure
        assert result_attr_error.error is not None
        assert "Attribute error occurred" in result_attr_error.error

        def failing_func_runtime_error() -> str:
            msg = "Runtime error occurred"
            raise RuntimeError(msg)

        result_runtime_error = FlextResult.from_exception(failing_func_runtime_error)
        assert result_runtime_error.is_failure
        assert result_runtime_error.error is not None
        assert "Runtime error occurred" in result_runtime_error.error

    def test_combine_static_method(self) -> None:
        """Test combine static method comprehensively."""
        success1 = FlextResult[str].ok("first")
        success2 = FlextResult[int].ok(42)
        success3 = FlextResult[bool].ok(True)
        failure = FlextResult[str].fail("failed")

        # All success

        combined_success = FlextResult.combine(
            cast("FlextResult[object]", success1),
            cast("FlextResult[object]", success2),
            cast("FlextResult[object]", success3),
        )
        assert combined_success.is_success
        expected_data = ["first", 42, True]
        assert combined_success.value == expected_data

        # With failure
        combined_failure = FlextResult.combine(
            cast("FlextResult[object]", success1),
            cast("FlextResult[object]", failure),
            cast("FlextResult[object]", success3),
        )
        assert combined_failure.is_failure
        assert combined_failure.error is not None
        assert "failed" in combined_failure.error

        # With None values - should include them
        none_success = FlextResult[None].ok(None)
        combined_with_none = FlextResult.combine(
            cast("FlextResult[object]", success1),
            cast("FlextResult[object]", none_success),
        )
        assert combined_with_none.is_success
        # None values should not be included based on the implementation
        assert combined_with_none.value == ["first"]

    def test_static_boolean_methods(self) -> None:
        """Test all_success and any_success static methods."""
        success1 = FlextResult[str].ok("first")
        success2 = FlextResult[str].ok("second")
        failure1 = FlextResult[str].fail("first_error")
        failure2 = FlextResult[str].fail("second_error")

        # all_success tests

        assert (
            FlextResult.all_success(
                cast("FlextResult[object]", success1),
                cast("FlextResult[object]", success2),
            )
            is True
        )
        assert (
            FlextResult.all_success(
                cast("FlextResult[object]", success1),
                cast("FlextResult[object]", failure1),
            )
            is False
        )
        assert (
            FlextResult.all_success(
                cast("FlextResult[object]", failure1),
                cast("FlextResult[object]", failure2),
            )
            is False
        )
        assert FlextResult.all_success() is True  # Empty case

        # any_success tests
        assert (
            FlextResult.any_success(
                cast("FlextResult[object]", success1),
                cast("FlextResult[object]", success2),
            )
            is True
        )
        assert (
            FlextResult.any_success(
                cast("FlextResult[object]", success1),
                cast("FlextResult[object]", failure1),
            )
            is True
        )
        assert (
            FlextResult.any_success(
                cast("FlextResult[object]", failure1),
                cast("FlextResult[object]", failure2),
            )
            is False
        )
        assert FlextResult.any_success() is False  # Empty case

    def test_first_success_class_method(self) -> None:
        """Test first_success class method comprehensively."""
        success1 = FlextResult[str].ok("first")
        success2 = FlextResult[str].ok("second")
        failure1 = FlextResult[str].fail("first_error")
        failure2 = FlextResult[str].fail("second_error")

        # First is success
        first_success_result: FlextResult[object] = FlextResult.first_success(
            cast("FlextResult[object]", success1),
            cast("FlextResult[object]", failure1),
            cast("FlextResult[object]", success2),
        )
        assert first_success_result == success1

        # Success in middle
        middle_success_result: FlextResult[object] = FlextResult.first_success(
            cast("FlextResult[object]", failure1),
            cast("FlextResult[object]", success2),
            cast("FlextResult[object]", failure2),
        )
        assert middle_success_result == success2

        # No success
        no_success_result: FlextResult[object] = FlextResult.first_success(
            cast("FlextResult[object]", failure1),
            cast("FlextResult[object]", failure2),
        )
        assert no_success_result.is_failure
        assert no_success_result.error is not None
        assert "second_error" in no_success_result.error

    def test_sequence_class_method(self) -> None:
        """Test sequence class method comprehensively."""
        success1 = FlextResult[str].ok("first")
        success2 = FlextResult[str].ok("second")
        success3 = FlextResult[str].ok("third")
        failure = FlextResult[str].fail("sequence_error")

        # All success
        sequenced_success = FlextResult.sequence([success1, success2, success3])
        assert sequenced_success.is_success
        assert sequenced_success.value == ["first", "second", "third"]

        # With failure
        sequenced_failure = FlextResult.sequence([success1, failure, success3])
        assert sequenced_failure.is_failure
        assert sequenced_failure.error is not None
        assert "sequence_error" in sequenced_failure.error
        assert sequenced_failure.error_code == failure.error_code

        # Empty list
        sequenced_empty: FlextResult[list[object]] = FlextResult.sequence([])
        assert sequenced_empty.is_success
        assert sequenced_empty.value == []

    def test_try_all_class_method(self) -> None:
        """Test try_all class method comprehensively."""

        def success_func() -> str:
            return "success"

        def failing_func1() -> str:
            msg = "Type error"
            raise TypeError(msg)

        def failing_func2() -> str:
            msg = "Value error"
            raise ValueError(msg)

        def failing_func3() -> str:
            msg = "Arithmetic error"
            raise ArithmeticError(msg)

        # First succeeds
        result_first_success = FlextResult.try_all(success_func, failing_func1)
        assert result_first_success.is_success
        assert result_first_success.value == "success"

        # Success after failures
        result_later_success = FlextResult.try_all(
            failing_func1,
            failing_func2,
            success_func,
        )
        assert result_later_success.is_success
        assert result_later_success.value == "success"

        # All fail
        result_all_fail = FlextResult.try_all(
            failing_func1,
            failing_func2,
            failing_func3,
        )
        assert result_all_fail.is_failure
        assert result_all_fail.error is not None
        assert "Arithmetic error" in result_all_fail.error

        # No functions
        result_no_funcs: FlextResult[object] = FlextResult.try_all()
        assert result_no_funcs.is_failure
        assert result_no_funcs.error is not None
        assert "No functions provided" in result_no_funcs.error

    def test_utility_methods_comprehensive(self) -> None:
        """Test all utility methods (formerly FlextResultUtils)."""
        success_result = FlextResult[str].ok("utility_value")
        failure_result = FlextResult[str].fail("utility_error")

        # safe_unwrap_or_none
        assert FlextResult.safe_unwrap_or_none(success_result) == "utility_value"
        assert FlextResult.safe_unwrap_or_none(failure_result) is None

        # unwrap_or_raise with default exception
        assert FlextResult.unwrap_or_raise(success_result) == "utility_value"
        with pytest.raises(RuntimeError, match="utility_error"):
            FlextResult.unwrap_or_raise(failure_result)

        # unwrap_or_raise with custom exception
        with pytest.raises(ValueError, match="utility_error"):
            FlextResult.unwrap_or_raise(failure_result, ValueError)

        # collect_successes and collect_failures
        results = [
            success_result,
            failure_result,
            FlextResult[str].ok("second_success"),
        ]
        successes = FlextResult.collect_successes(results)
        failures = FlextResult.collect_failures(results)

        assert successes == ["utility_value", "second_success"]
        assert failures == ["utility_error"]

        # success_rate
        success_rate = FlextResult.success_rate(results)
        assert success_rate == (2 / 3) * 100  # 2 successes out of 3

        # Empty list success_rate
        empty_rate = FlextResult.success_rate([])
        assert empty_rate == 0.0

        # batch_process
        items = ["item1", "item2", "item3"]

        def processor(item: str) -> FlextResult[str]:
            if item == "item2":
                return FlextResult[str].fail(f"failed_{item}")
            return FlextResult[str].ok(f"processed_{item}")

        batch_successes, batch_failures = FlextResult.batch_process(items, processor)
        assert batch_successes == ["processed_item1", "processed_item3"]
        assert batch_failures == ["failed_item2"]

        # safe_call
        def safe_success() -> str:
            return "safe_result"

        def safe_failure() -> str:
            msg = "safe_error"
            raise ValueError(msg)  # Use specific exception type

        safe_result_success = FlextResult.safe_call(safe_success)
        assert safe_result_success.is_success
        assert safe_result_success.value == "safe_result"

        safe_result_failure = FlextResult.safe_call(safe_failure)
        assert safe_result_failure.is_failure
        assert safe_result_failure.error is not None
        assert "safe_error" in safe_result_failure.error

    def test_advanced_operators_comprehensive(self) -> None:
        """Test all advanced monadic composition operators."""
        success_result = FlextResult[int].ok(42)
        failure_result = FlextResult[int].fail("operator_error")

        # Right shift operator (>>) - monadic bind
        def bind_func(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"bound_{x}")

        bound_result = success_result >> bind_func
        assert bound_result.is_success
        assert bound_result.value == "bound_42"

        bound_failure = failure_result >> bind_func
        assert bound_failure.is_failure

        # Left shift operator (<<) - functor map
        mapped_result = success_result << (lambda x: f"mapped_{x}")
        assert mapped_result.is_success
        assert mapped_result.value == "mapped_42"

        # Matrix multiplication operator (@) - applicative combination
        other_success = FlextResult[str].ok("other")
        combined_result = success_result @ other_success
        assert combined_result.is_success
        assert combined_result.value == (42, "other")

        combined_failure = failure_result @ other_success
        assert combined_failure.is_failure

        # Division operator (/) - alternative fallback
        alternative_result = success_result / failure_result
        assert alternative_result.is_success
        assert alternative_result.value == 42

        fallback_result = failure_result / success_result
        assert fallback_result.is_success
        assert fallback_result.value == 42

        # Modulo operator (%) - conditional filtering
        filtered_result = success_result % (lambda x: x > 40)
        assert filtered_result == success_result

        filtered_fail = success_result % (lambda x: x > 50)
        assert filtered_fail.is_failure

        # AND operator (&) - sequential composition
        and_result = success_result & other_success
        assert and_result.is_success
        assert and_result.value == (42, "other")

        # XOR operator (^) - error recovery
        def recovery_func(__error: str, /) -> int:
            return 999

        recovered_result = failure_result ^ recovery_func
        assert recovered_result.is_success
        assert recovered_result.value == 999

    def test_advanced_monadic_combinators(self) -> None:
        """Test advanced monadic combinators from Category Theory."""
        # traverse operation
        items = [1, 2, 3]

        def transform_func(x: int) -> FlextResult[str]:
            if x == 2:
                return FlextResult[str].fail(f"failed_at_{x}")
            return FlextResult[str].ok(f"item_{x}")

        # Success case
        success_items = [1, 3, 4]
        traversed_success = FlextResult.traverse(success_items, transform_func)
        assert traversed_success.is_success
        assert traversed_success.value == ["item_1", "item_3", "item_4"]

        # Failure case
        traversed_failure = FlextResult.traverse(items, transform_func)
        assert traversed_failure.is_failure
        assert traversed_failure.error is not None
        assert "failed_at_2" in traversed_failure.error

        # kleisli_compose
        def f(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"f_{x}")

        def g(x: str) -> FlextResult[bool]:
            return FlextResult[bool].ok(x.startswith("f"))

        result = FlextResult[int].ok(42)
        composed_func = result.kleisli_compose(f, g)
        composed_result = composed_func(42)
        assert composed_result.is_success
        assert composed_result.value is True

        # applicative_lift2
        def add_func(x: int, y: int) -> int:
            return x + y

        result1 = FlextResult[int].ok(10)
        result2 = FlextResult[int].ok(20)
        lifted_result = FlextResult.applicative_lift2(add_func, result1, result2)
        assert lifted_result.is_success
        assert lifted_result.value == 30

        # With failure
        failure_result = FlextResult[int].fail("lift_error")
        lifted_failure = FlextResult.applicative_lift2(
            add_func,
            result1,
            failure_result,
        )
        assert lifted_failure.is_failure

        # applicative_lift3
        def add_three_func(x: int, y: int, z: int) -> int:
            return x + y + z

        result3 = FlextResult[int].ok(30)
        lifted3_result = FlextResult.applicative_lift3(
            add_three_func,
            result1,
            result2,
            result3,
        )
        assert lifted3_result.is_success
        assert lifted3_result.value == 60

    def test_result_factory_methods(self) -> None:
        """Test Result factory methods and types."""
        # dict_result factory
        dict_result_type = FlextResult.Result.dict_result()
        assert dict_result_type == FlextResult[dict[str, object]]

        # Success type alias - just verify it exists and is a type
        success_type = FlextResult.Result.Success
        assert success_type is not None
        # In Python 3.12+ type aliases, Success should be equivalent to object
        assert str(success_type) == "Success" or str(success_type) == "object"

        # In Python 3.12+ type aliases, Success should be equivalent to object
        assert str(success_type) == "Success" or str(success_type) == "object"
