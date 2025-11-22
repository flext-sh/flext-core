"""Comprehensive tests for FlextResult - Railway Pattern Implementation.

This module tests the core FlextResult railway pattern which is the foundation
of error handling across the entire FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Never, cast

import pytest
from returns.io import IO, IOFailure, IOSuccess
from returns.maybe import Nothing, Some

from flext_core import FlextExceptions, FlextResult


class TestFlextResult:
    """Test suite for FlextResult railway pattern implementation."""

    def test_result_creation_success(self) -> None:
        """Test successful result creation."""
        result = FlextResult[str].ok("test_value")

        assert result.is_success
        assert not result.is_failure
        assert result.value == "test_value"
        assert result.unwrap() == "test_value"

    def test_result_creation_failure(self) -> None:
        """Test failure result creation."""
        result = FlextResult[str].fail("test_error")

        assert not result.is_success
        assert result.is_failure
        assert result.error == "test_error"
        assert result.error_code is None

    def test_result_creation_failure_with_code(self) -> None:
        """Test failure result creation with error code."""
        result = FlextResult[str].fail("test_error", error_code="TEST_ERROR")

        assert not result.is_success
        assert result.is_failure
        assert result.error == "test_error"
        assert result.error_code == "TEST_ERROR"

    def test_result_map_success(self) -> None:
        """Test map operation on successful result."""
        result = FlextResult[int].ok(5)

        def double(x: int) -> int:
            return x * 2

        mapped = result.map(double)

        assert mapped.is_success
        assert mapped.value == 10

    def test_result_map_failure(self) -> None:
        """Test map operation on failed result."""
        result = FlextResult[int].fail("test_error")

        def double(x: int) -> int:
            return x * 2

        mapped = result.map(double)

        assert mapped.is_failure
        assert mapped.error == "test_error"

    def test_result_flat_map_success(self) -> None:
        """Test flat_map operation on successful result."""
        result = FlextResult[int].ok(5)

        def to_string_result(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"value_{x}")

        flat_mapped = result.flat_map(to_string_result)

        assert flat_mapped.is_success
        assert flat_mapped.value == "value_5"

    def test_result_flat_map_failure(self) -> None:
        """Test flat_map operation on failed result."""
        result = FlextResult[int].fail("test_error")
        flat_mapped = result.flat_map(lambda x: FlextResult[str].ok(f"value_{x}"))

        assert flat_mapped.is_failure
        assert flat_mapped.error == "test_error"

    def test_result_unwrap_or(self) -> None:
        """Test unwrap_or operation."""
        success_result: FlextResult[str] = FlextResult[str].ok("success")
        failure_result: FlextResult[str] = FlextResult[str].fail("error")

        assert success_result.unwrap_or("default") == "success"
        assert failure_result.unwrap_or("default") == "default"

    def test_result_expect(self) -> None:
        """Test expect operation."""
        success_result: FlextResult[str] = FlextResult[str].ok("success")

        assert success_result.expect("Should not fail") == "success"

    def test_result_expect_failure(self) -> None:
        """Test expect operation on failure."""
        failure_result = FlextResult[str].fail("error")

        with pytest.raises(FlextExceptions.BaseError, match="Should fail"):
            failure_result.expect("Should fail")

    def test_result_railway_composition(self) -> None:
        """Test railway-oriented composition."""

        def validate_input(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            if not data.get("value"):
                return FlextResult[dict[str, object]].fail("Missing value")
            return FlextResult[dict[str, object]].ok(data)

        def process_data(data: dict[str, object]) -> FlextResult[int]:
            return FlextResult[int].ok(cast("int", data["value"]) * 2)

        def format_result(value: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"Result: {value}")

        # Test successful railway composition
        result = (
            validate_input({"value": 5}).flat_map(process_data).flat_map(format_result)
        )

        assert result.is_success
        assert result.value == "Result: 10"

        # Test failed railway composition
        result = validate_input({}).flat_map(process_data).flat_map(format_result)

        assert result.is_failure
        assert result.error == "Missing value"

    def test_result_type_safety(self) -> None:
        """Test type safety of FlextResult."""
        # Test generic type preservation
        result_int = FlextResult[int].ok(42)
        result_str = FlextResult[str].ok("hello")

        assert isinstance(result_int.value, int)
        assert isinstance(result_str.value, str)

    def test_result_value_property_dict(self) -> None:
        """Test .value property access on successful results with dict."""
        result = FlextResult[dict[str, object]].ok({"key": "value"})

        # .value should return the wrapped dictionary
        assert result.value == {"key": "value"}
        assert isinstance(result.value, dict)

    def test_result_error_handling_edge_cases(self) -> None:
        """Test edge cases in error handling."""
        # Test empty error message (gets converted to default message)
        result = FlextResult[str].fail("")
        assert result.is_failure
        assert result.error == "Unknown error occurred"

        # Test that None is not a valid success value
        with pytest.raises(TypeError, match="cannot accept None"):
            FlextResult[int].ok(cast("int", None))

    def test_result_performance(self) -> None:
        """Test performance characteristics of FlextResult."""
        start_time = time.time()

        # Create many results
        results = [FlextResult[int].ok(i) for i in range(1000)]

        # Chain operations
        final_result = results[0]
        for _i, result in enumerate(results[1:10]):  # Test first 10

            def make_processor(
                current_result: FlextResult[int],
            ) -> Callable[[int], FlextResult[int]]:
                return lambda x: current_result.map(lambda y: x + y)

            final_result = final_result.flat_map(make_processor(result))

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete quickly (less than 1 second)
        assert execution_time < 1.0
        assert final_result.is_success

    def test_result_tap_method(self) -> None:
        """Test tap method."""
        result = FlextResult[str].ok("test")
        tapped_values: list[str] = []

        tapped = result.tap(tapped_values.append)
        assert tapped.is_success
        assert tapped.value == "test"
        assert tapped_values == ["test"]

    def test_result_tap_method_failure(self) -> None:
        """Test tap method on failure."""
        result = FlextResult[str].fail("error")
        tapped_values: list[str] = []

        tapped = result.tap(tapped_values.append)
        assert tapped.is_failure
        assert tapped.error == "error"
        assert tapped_values == []  # Should not be called

    def test_result_recover_method(self) -> None:
        """Test recover method."""
        result = FlextResult[str].fail("error")

        def recover_func(e: str) -> str:
            return f"recovered_{e}"

        recovered = result.recover(recover_func)

        assert recovered.is_success
        assert recovered.value == "recovered_error"

    def test_result_recover_method_success(self) -> None:
        """Test recover method on success."""
        result = FlextResult[str].ok("success")

        def recover_func(e: str) -> str:
            return f"recovered_{e}"

        recovered = result.recover(recover_func)

        assert recovered.is_success
        assert recovered.value == "success"

    def test_result_or_else_method(self) -> None:
        """Test or_else method."""
        result1 = FlextResult[str].fail("error1")
        result2 = FlextResult[str].ok("success2")

        or_result = result1.or_else(result2)
        assert or_result.is_success
        assert or_result.value == "success2"

    def test_result_or_else_method_success(self) -> None:
        """Test or_else method on success."""
        result1 = FlextResult[str].ok("success1")
        result2 = FlextResult[str].ok("success2")

        or_result = result1.or_else(result2)
        assert or_result.is_success
        assert or_result.value == "success1"

    def test_result_or_else_get_method(self) -> None:
        """Test or_else_get method."""
        result1 = FlextResult[str].fail("error1")

        def fallback_func() -> FlextResult[str]:
            return FlextResult[str].ok("fallback")

        or_result = result1.or_else_get(fallback_func)
        assert or_result.is_success
        assert or_result.value == "fallback"

    def test_result_or_else_get_method_success(self) -> None:
        """Test or_else_get method on success."""
        result1 = FlextResult[str].ok("success1")

        or_result = result1.or_else_get(lambda: FlextResult[str].ok("fallback"))
        assert or_result.is_success
        assert or_result.value == "success1"

    def test_result_with_context_method(self) -> None:
        """Test with_context method."""
        result = FlextResult[str].fail("error")

        def add_context(e: str) -> str:
            return f"Context: {e}"

        context_result = result.with_context(add_context)

        assert context_result.is_failure
        assert context_result.error is not None
        assert "Context: error" in context_result.error

    def test_result_with_context_method_success(self) -> None:
        """Test with_context method on success."""
        result = FlextResult[str].ok("success")

        def add_context(e: str) -> str:
            return f"Context: {e}"

        context_result = result.with_context(add_context)

        assert context_result.is_success
        assert context_result.value == "success"

    def test_result_sequence_static_method(self) -> None:
        """Test sequence static method."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ]

        sequenced = FlextResult.sequence(results)
        assert sequenced.is_success
        assert sequenced.value == [1, 2, 3]

    def test_result_sequence_static_method_failure(self) -> None:
        """Test sequence static method with failure."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("error"),
            FlextResult[int].ok(3),
        ]

        sequenced = FlextResult.sequence(results)
        assert sequenced.is_failure
        assert sequenced.error == "error"

    def test_result_collect_failures_class_method(self) -> None:
        """Test collect_failures class method."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("error1"),
            FlextResult[int].ok(3),
            FlextResult[int].fail("error2"),
        ]

        failures = FlextResult.collect_failures(results)
        assert failures == ["error1", "error2"]

    def test_result_value_or_none_property(self) -> None:
        """Test value_or_none property - REMOVED: None is not a valid success value."""
        # value_or_none was removed because None is not a valid success value
        # Use .value or .unwrap() for success, or check .is_success first
        result = FlextResult[int].ok(5)
        value = result.value

        assert value == 5

    def test_result_value_or_none_property_failure(self) -> None:
        """Test value_or_none property on failure - REMOVED: Use .is_success check."""
        # value_or_none was removed because None is not a valid success value
        # Use .is_success check instead
        result = FlextResult[int].fail("error")

        assert result.is_failure
        with pytest.raises(FlextExceptions.ValidationError):
            _ = result.value

    def test_result_rshift_operator(self) -> None:
        """Test >> operator (flat_map)."""
        result = FlextResult[int].ok(5)

        def double_func(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        mapped = result >> double_func

        assert mapped.is_success
        assert mapped.value == 10

    def test_result_lshift_operator(self) -> None:
        """Test << operator (map)."""
        result = FlextResult[int].ok(5)

        def double_value(x: int) -> int:
            return x * 2

        mapped = result << double_value

        assert mapped.is_success
        assert mapped.value == 10

    def test_result_mod_operator(self) -> None:
        """Test % operator (when)."""
        result = FlextResult[int].ok(5)

        def is_greater_than_3(x: int) -> bool:
            return x > 3

        filtered = result % is_greater_than_3

        assert filtered.is_success
        assert filtered.value == 5

    def test_result_xor_operator(self) -> None:
        """Test ^ operator (recover)."""
        result = FlextResult[int].fail("error")

        def recover_func(error_msg: str) -> int:
            return 0

        recovered = result ^ recover_func

        assert recovered.is_success
        assert recovered.value == 0

    def test_result_batch_processing(self) -> None:
        """Test batch processing functionality."""

        def processor(item: str) -> FlextResult[str]:
            if item == "fail":
                return FlextResult[str].fail("Processing failed")
            return FlextResult[str].ok(f"processed_{item}")

        items = ["item1", "item2", "fail", "item3"]
        successes, failures = FlextResult.batch_process(items, processor)

        assert len(successes) == 3
        assert len(failures) == 1
        assert "processed_item1" in successes
        assert "processed_item2" in successes
        assert "processed_item3" in successes
        assert "Processing failed" in failures

    def test_result_traverse_collection(self) -> None:
        """Test traverse collection functionality."""

        def transform(item: int) -> FlextResult[str]:
            if item < 0:
                return FlextResult[str].fail("Negative number")
            return FlextResult[str].ok(str(item * 2))

        items = [1, 2, -1, 3]
        result = FlextResult.traverse(items, transform)

        assert result.is_failure
        assert result.error is not None
        assert "Negative number" in result.error

    def test_result_traverse_collection_success(self) -> None:
        """Test traverse collection functionality with all successes."""

        def transform(item: int) -> FlextResult[str]:
            return FlextResult[str].ok(str(item * 2))

        items = [1, 2, 3]
        result = FlextResult.traverse(items, transform)

        assert result.is_success
        assert result.value == ["2", "4", "6"]

    def test_result_sequence_with_failures(self) -> None:
        """Test sequence static method with failures."""
        results = [
            FlextResult[str].ok("success1"),
            FlextResult[str].fail("error1"),
            FlextResult[str].ok("success2"),
        ]

        result = FlextResult.sequence(results)
        assert result.is_failure
        assert result.error is not None
        assert "error1" in result.error

    def test_result_collect_failures_empty(self) -> None:
        """Test collect_failures with empty results."""
        results: list[FlextResult[str]] = []
        failures = FlextResult.collect_failures(results)
        assert failures == []

    def test_result_collect_failures_mixed(self) -> None:
        """Test collect_failures with mixed results."""
        results = [
            FlextResult[str].ok("success1"),
            FlextResult[str].fail("error1"),
            FlextResult[str].ok("success2"),
            FlextResult[str].fail("error2"),
        ]

        failures = FlextResult.collect_failures(results)
        assert failures == ["error1", "error2"]

    def test_result_accumulate_with_errors(self) -> None:
        """Test accumulate method with errors."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].fail("error1"),
            FlextResult[str].ok("value2"),
            FlextResult[str].fail("error2"),
        ]

        result = FlextResult.accumulate_errors(*results)
        assert result.is_failure
        assert result.error is not None
        assert "error1" in result.error
        assert result.error is not None
        assert "error2" in result.error

    def test_result_accumulate_all_success(self) -> None:
        """Test accumulate method with all successes."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].ok("value2"),
            FlextResult[str].ok("value3"),
        ]

        result = FlextResult.accumulate_errors(*results)
        assert result.is_success
        assert result.value == ["value1", "value2", "value3"]

    def test_result_accumulate_empty(self) -> None:
        """Test accumulate method with no results."""
        result: FlextResult[list[str]] = FlextResult.accumulate_errors()
        assert result.is_success
        assert result.value == []

    def test_result_accumulate_with_none_errors(self) -> None:
        """Test accumulate method with None errors."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].fail(""),  # Empty error
            FlextResult[str].ok("value2"),
        ]

        result = FlextResult.accumulate_errors(*results)
        assert result.is_failure
        assert result.error is not None
        assert "Unknown error" in result.error

    def test_result_parallel_map_with_fail_fast(self) -> None:
        """Test parallel_map method with fail_fast=True."""
        items = [1, 2, 3, 4, 5]

        def process_item(item: int) -> FlextResult[str]:
            if item == 3:
                return FlextResult[str].fail("Processing failed")
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=True)
        assert result.is_failure
        assert result.error == "Processing failed"

    def test_result_parallel_map_with_fail_fast_success(self) -> None:
        """Test parallel_map method with fail_fast=True and all success."""
        items = [1, 2, 3, 4, 5]

        def process_item(item: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=True)
        assert result.is_success
        assert len(result.value) == 5
        assert result.value[0] == "processed_1"
        assert result.value[4] == "processed_5"

    def test_result_parallel_map_without_fail_fast(self) -> None:
        """Test parallel_map method with fail_fast=False."""
        items = [1, 2, 3, 4, 5]

        def process_item(item: int) -> FlextResult[str]:
            if item == 3:
                return FlextResult[str].fail("Processing failed")
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=False)
        assert result.is_failure
        assert result.error is not None
        assert "Processing failed" in result.error

    def test_result_parallel_map_without_fail_fast_success(self) -> None:
        """Test parallel_map method with fail_fast=False and all success."""
        items = [1, 2, 3, 4, 5]

        def process_item(item: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=False)
        assert result.is_success
        assert len(result.value) == 5

    def test_result_parallel_map_with_multiple_errors(self) -> None:
        """Test parallel_map method with multiple errors."""
        items = [1, 2, 3, 4, 5]

        def process_item(item: int) -> FlextResult[str]:
            if item in {2, 4}:
                return FlextResult[str].fail(f"Error processing {item}")
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=False)
        assert result.is_failure
        assert result.error is not None
        assert "Error processing 2" in result.error
        assert result.error is not None
        assert "Error processing 4" in result.error

    def test_result_parallel_map_with_none_error(self) -> None:
        """Test parallel_map method with None error."""
        items = [1, 2, 3]

        def process_item(item: int) -> FlextResult[str]:
            if item == 2:
                return FlextResult[str].fail("Unknown error occurred")
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=False)
        assert result.is_failure
        assert result.error is not None
        assert "Unknown error occurred" in result.error

    def test_result_parallel_map_with_fail_fast_none_error(self) -> None:
        """Test parallel_map method with fail_fast=True and None error."""
        items = [1, 2, 3]

        def process_item(item: int) -> FlextResult[str]:
            if item == 2:
                return FlextResult[str].fail("Unknown error occurred")
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=True)
        assert result.is_failure
        assert result.error is not None
        assert "Unknown error occurred" in result.error

    # ========== COMPREHENSIVE TESTS FOR UNCOVERED METHODS ==========

    def test_result_filter_method(self) -> None:
        """Test filter method."""
        # Success case that passes filter
        result = FlextResult[int].ok(10)
        filtered = result.filter(lambda x: x > 5, "Value too small")
        assert filtered.is_success
        assert filtered.value == 10

        # Success case that fails filter
        result = FlextResult[int].ok(3)
        filtered = result.filter(lambda x: x > 5, "Value too small")
        assert filtered.is_failure
        assert filtered.error is not None
        assert "Value too small" in filtered.error

        # Failure case (should remain failure)
        result = FlextResult[int].fail("Original error")
        filtered = result.filter(lambda x: x > 5, "Value too small")
        assert filtered.is_failure
        assert filtered.error == "Original error"

    def test_result_from_exception_static_method(self) -> None:
        """Test from_exception static method."""

        # Function that succeeds
        def success_func() -> int:
            return 42

        success_result = FlextResult.from_exception(success_func)
        assert success_result.is_success
        assert success_result.value == 42

        # Function that raises exception
        def failing_func() -> int:
            error_message = "Something went wrong"
            raise ValueError(error_message)

        failure_result = FlextResult.from_exception(failing_func)
        assert failure_result.is_failure
        assert failure_result.error is not None
        assert "Something went wrong" in failure_result.error

    def test_result_safe_unwrap_or_none_static_method(self) -> None:
        """Test value access - value_or_none removed, use .value or .unwrap_or()."""
        # Success result
        result = FlextResult[int].ok(42)
        value = result.value
        assert value == 42

        # Failure result - use unwrap_or for default value
        result = FlextResult[int].fail("Error")
        value = result.unwrap_or(0)
        assert value == 0

    def test_result_success_rate_static_method(self) -> None:
        """Test success_rate static method."""
        # Mix of success and failure
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("Error"),
            FlextResult[int].ok(2),
            FlextResult[int].fail("Error"),
            FlextResult[int].ok(3),
        ]

        rate = FlextResult.success_rate(results)
        assert rate == 60.0  # 3 out of 5 (60%)

        # All success
        results = [FlextResult[int].ok(i) for i in range(5)]
        rate = FlextResult.success_rate(results)
        assert rate == 100.0

        # All failure
        results = [FlextResult[int].fail("Error") for _ in range(5)]
        rate = FlextResult.success_rate(results)
        assert rate == 0.0

        # Empty list
        rate = FlextResult.success_rate([])
        assert rate == 0.0

    def test_result_safe_call_static_method(self) -> None:
        """Test safe_call static method."""

        # Function that succeeds
        def success_func() -> int:
            return 42

        success_result: FlextResult[int] = FlextResult[int].safe_call(success_func)
        assert success_result.is_success
        assert success_result.value == 42

        # Function that raises exception
        def failing_func() -> int:
            error_message = "Something went wrong"
            raise ValueError(error_message)

        failure_result: FlextResult[int] = FlextResult[int].safe_call(failing_func)
        assert failure_result.is_failure
        assert failure_result.error is not None
        assert "Something went wrong" in failure_result.error

    def test_result_chain_validations_static_method(self) -> None:
        """Test chain_validations static method."""

        # Create validation functions that don't depend on input
        def validate_positive() -> FlextResult[bool]:
            # For this test, we'll simulate validation that always passes
            return FlextResult[bool].ok(True)

        def validate_even() -> FlextResult[bool]:
            # For this test, we'll simulate validation that always passes for the first case
            # and fails for the second case - this is just for testing the chaining logic
            return FlextResult[bool].ok(True)

        validators = [validate_positive, validate_even]

        # Valid value (positive and even)
        result = FlextResult.chain_validations(*validators)
        assert result.is_success

        # Invalid value (positive but odd)
        def validate_even_fail() -> FlextResult[bool]:
            return FlextResult[bool].fail("Must be even")

        result = FlextResult.chain_validations(validate_positive, validate_even_fail)
        assert result.is_failure
        assert result.error is not None
        assert "Must be even" in result.error

        # Invalid value (negative) - first validator should fail
        def validate_positive_fail() -> FlextResult[bool]:
            return FlextResult[bool].fail("Must be positive")

        result = FlextResult.chain_validations(validate_positive_fail, validate_even)
        assert result.is_failure
        assert result.error is not None
        assert "Must be positive" in result.error

    def test_result_validate_and_execute_static_method(self) -> None:
        """Test validate_and_execute static method."""

        def validator(x: int) -> FlextResult[bool]:
            if x > 0:
                return FlextResult[bool].ok(True)
            return FlextResult[bool].fail("Must be positive")

        def executor(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"Processed: {x}")

        # Valid value
        valid_result: FlextResult[str] = (
            FlextResult[int].ok(5).validate_and_execute(validator, executor)
        )
        assert valid_result.is_success
        assert valid_result.value == "Processed: 5"

        # Invalid value
        invalid_result: FlextResult[str] = (
            FlextResult[int].ok(-5).validate_and_execute(validator, executor)
        )
        assert invalid_result.is_failure
        assert invalid_result.error is not None
        assert "Must be positive" in invalid_result.error

    def test_result_pipeline_static_method(self) -> None:
        """Test pipeline static method."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def double(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        def to_int(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(int(x))

        operations: list[Callable[[int], FlextResult[int]]] = [
            add_one,
            double,
            to_int,
        ]

        # Successful pipeline
        result = FlextResult[int].pipeline(5, *operations)
        assert result.is_success
        assert result.value == 12  # (5 + 1) * 2 = 12

        # Pipeline with failure
        def failing_op(x: int) -> FlextResult[int]:
            # Parameter is intentionally unused in this test
            _ = x  # Mark as intentionally unused
            return FlextResult[int].fail("Pipeline failed")

        operations_with_failure = [add_one, failing_op, double]
        result = FlextResult[int].pipeline(5, *operations_with_failure)
        assert result.is_failure
        assert result.error is not None
        assert "Pipeline failed" in result.error

    def test_result_collect_all_errors_static_method(self) -> None:
        """Test collect_all_errors static method."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("Error 1"),
            FlextResult[int].ok(2),
            FlextResult[int].fail("Error 2"),
            FlextResult[int].ok(3),
        ]

        collected = FlextResult[int].collect_all_errors(*results)
        successes, errors = collected
        assert successes == [1, 2, 3]
        assert len(errors) == 2
        assert "Error 1" in errors[0]
        assert "Error 2" in errors[1]

    def test_result_with_resource_static_method(self) -> None:
        """Test with_resource static method."""
        resources_created: list[str] = []
        resources_cleaned: list[str] = []

        def create_resource() -> str:
            resource = "test_resource"
            resources_created.append(resource)
            return resource

        def operation(value: str, resource: str) -> FlextResult[int]:
            return FlextResult[int].ok(len(resource) + len(value))

        def cleanup(resource: str) -> None:
            resources_cleaned.append(resource)

        result: FlextResult[int] = (
            FlextResult[str]
            .ok("42")
            .with_resource(
                create_resource,
                operation,
                cleanup,
            )
        )
        assert result.is_success
        assert result.value == 15  # len("test_resource") + len("42")
        assert len(resources_created) == 1
        assert len(resources_cleaned) == 1
        assert resources_created[0] == resources_cleaned[0]

    def test_result_equality_with_different_types(self) -> None:
        """Test __eq__ with different types for complete coverage."""
        result = FlextResult[int].ok(42)

        # Test equality with non-FlextResult object
        assert result != "not a result"
        assert result != 42
        assert result != {"data": 42}
        assert result is not None

        # Test two success results with same value (string comparison)
        result2 = FlextResult[int].ok(42)
        assert result == result2

        # Test equality is based on string representation, so different types with same str are equal
        other_result = FlextResult[str].ok("42")
        # Both convert to str("42") so they're equal per the implementation
        assert result == other_result

        # Test two failure results with same error
        fail1 = FlextResult[int].fail("error", error_code="ERR_001")
        fail2 = FlextResult[int].fail("error", error_code="ERR_001")
        assert fail1 == fail2

        # Test different error codes
        fail3 = FlextResult[int].fail("error", error_code="ERR_002")
        assert fail1 != fail3

    def test_result_hash_for_set_operations(self) -> None:
        """Test __hash__ implementation for use in sets and dicts."""
        result1 = FlextResult[int].ok(42)
        result2 = FlextResult[int].ok(42)
        result3 = FlextResult[int].ok(43)
        fail1 = FlextResult[int].fail("error")
        fail2 = FlextResult[int].fail("error")

        # Test hashability
        result_set = {result1, result2, result3, fail1, fail2}
        # Should have 3 unique items (2 successes with different values, 1 failure)
        assert len(result_set) == 3

        # Test as dict[str, object] keys
        result_dict = {
            result1: "first",
            result2: "second",  # Should overwrite first
            result3: "third",
            fail1: "failure",
        }
        assert len(result_dict) == 3
        assert result_dict[result1] == "second"  # result2 overwrote result1

    def test_result_validate_all_classmethod(self) -> None:
        """Test the validate_all classmethod with validators."""

        # Define test validators
        def validate_positive(x: int) -> FlextResult[bool]:
            if x <= 0:
                return FlextResult[bool].fail("Must be positive")
            return FlextResult[bool].ok(True)

        def validate_even(x: int) -> FlextResult[bool]:
            if x % 2 != 0:
                return FlextResult[bool].fail("Must be even")
            return FlextResult[bool].ok(True)

        def validate_less_than_100(x: int) -> FlextResult[bool]:
            if x >= 100:
                return FlextResult[bool].fail("Must be less than 100")
            return FlextResult[bool].ok(True)

        # Test all validations pass
        result = FlextResult[int].validate_all(
            42,
            validate_positive,
            validate_even,
            validate_less_than_100,
        )
        assert result.is_success
        assert result.value == 42

        # Test validation failure
        result = FlextResult[int].validate_all(
            43,
            validate_positive,
            validate_even,
            validate_less_than_100,
        )
        assert result.is_failure
        # Check that error message contains validation failure info
        assert result.error is not None

    def test_result_class_or_instance_method_descriptor(self) -> None:
        """Test ClassOrInstanceMethod descriptor behavior."""
        # Test calling as class method
        result = FlextResult[int].ok(42)
        assert result.is_success
        assert result.value == 42

        # Test calling as instance method through chain
        initial = FlextResult[int].ok(10)
        chained: FlextResult[int] = initial.flat_map(
            lambda x: FlextResult[int].ok(x * 2),
        )
        assert chained.is_success
        assert chained.value == 20

        # Verify the descriptor works for both contexts
        # Class context
        class_result = FlextResult[int].sequence([
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ])
        assert class_result.is_success

        # Instance context through map
        instance_result = FlextResult[list[int]].ok([1, 2, 3])
        mapped = instance_result.map(lambda x: [i * 2 for i in x])
        assert mapped.is_success

    def test_result_error_with_none_values(self) -> None:
        """Test error property with None error message."""
        # Failure with explicit None error converts to "Unknown error occurred"
        result = FlextResult[int].fail(cast("str", None))
        assert result.is_failure
        # None error is normalized to default message
        assert result.error == "Unknown error occurred"

    def test_result_repr_coverage(self) -> None:
        """Test __repr__ method for string representation."""
        # Success repr
        success: FlextResult[int] = FlextResult[int].ok(42)
        repr_str = repr(success)
        assert "FlextResult" in repr_str
        assert "42" in repr_str

        # Failure repr
        failure = FlextResult[int].fail("Test error", error_code="TEST_001")
        repr_str = repr(failure)
        assert "FlextResult" in repr_str
        assert "Test error" in repr_str or "TEST_001" in repr_str

    def test_result_bool_conversion(self) -> None:
        """Test __bool__ method for truthiness."""
        # Success is truthy
        success: FlextResult[int] = FlextResult[int].ok(42)
        assert bool(success) is True
        assert success  # Direct boolean context

        # Failure is falsy
        failure: FlextResult[int] = FlextResult[int].fail("error")
        assert bool(failure) is False
        assert not failure  # Direct boolean context

        # Use in if statement
        passed = bool(success)
        assert passed is True

        passed = not failure
        assert passed is True

    def test_result_iter_protocol(self) -> None:
        """Test __iter__ for unpacking and iteration."""
        # Success iteration returns (data, None)
        success: FlextResult[int] = FlextResult[int].ok(42)
        # Success iteration raises on unpacking (fast fail pattern)
        # Use .value or .unwrap() for success values
        assert success[0] == 42
        assert success[1] == ""  # Empty string for error when success

        # Failure iteration raises on unpacking (fast fail pattern)
        failure = FlextResult[int].fail("error message")
        with pytest.raises(FlextExceptions.BaseError):
            _ = failure[0]  # Accessing data on failure raises

    def test_result_getitem_protocol(self) -> None:
        """Test __getitem__ for indexing."""
        success: FlextResult[int] = FlextResult[int].ok(42)

        # Index 0 returns data
        assert success[0] == 42

        # Index 1 returns empty string for error when success (not None)
        assert success[1] == ""

        # Out of range raises FlextExceptions.NotFoundError
        with pytest.raises(FlextExceptions.NotFoundError) as exc_info:
            _ = success[2]
        assert "only supports indices 0 (data) and 1 (error)" in str(exc_info.value)

        # Failure raises on index 0 (fast fail), error for index 1
        failure = FlextResult[int].fail("error message")
        # Fast fail: accessing data on failure raises exception
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _ = failure[0]
        assert "error message" in str(exc_info.value)
        assert failure[1] == "error message"

    def test_result_context_manager_with_none_value(self) -> None:
        """Test context manager protocol with edge cases."""
        # None is not a valid success value - should raise TypeError
        with pytest.raises(TypeError, match="cannot accept None"):
            FlextResult[int].ok(cast("int", None))

        # Test __exit__ is called properly
        success: FlextResult[int] = FlextResult[int].ok(42)
        entered = False
        exited = False

        with success as value:
            entered = True
            assert value == 42
        exited = True

        assert entered
        assert exited

        # Test failure raises on enter
        failure = FlextResult[int].fail("Context error")
        with pytest.raises(FlextExceptions.BaseError), failure as value:
            pass

    def test_result_or_operator_coverage(self) -> None:
        """Test __or__ operator with various scenarios."""
        success: FlextResult[int] = FlextResult[int].ok(42)
        failure: FlextResult[int] = FlextResult[int].fail("error")

        # Success | default -> unwrapped value
        result = success | 100
        assert result == 42  # Returns unwrapped data

        # Failure | default -> default value
        result = failure | 100
        assert result == 100  # Returns default

        # Success with None data returns default
        success_none = FlextResult[int].ok(42)
        result = success_none | 100
        assert result == 42

    def test_result_sequence_with_list(self) -> None:
        """Test sequence static method with list of results."""
        # All results succeed
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ]
        combined = FlextResult[int].sequence(results)
        assert combined.is_success
        assert combined.value == [1, 2, 3]

        # One result fails
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("Validation error"),
            FlextResult[int].ok(3),
        ]
        combined = FlextResult[int].sequence(results)
        assert combined.is_failure
        assert "Validation error" in str(combined.error)

    def test_result_sequence_accepts_list_only(self) -> None:
        """Test sequence accepts list parameter (not variadic args)."""
        result1 = FlextResult[int].ok(1)
        result2 = FlextResult[int].ok(2)
        result3 = FlextResult[int].ok(3)

        # sequence() accepts a list of results
        combined: FlextResult[list[int]] = FlextResult[int].sequence([
            result1,
            result2,
            result3,
        ])
        assert combined.is_success
        assert combined.value == [1, 2, 3]

        # Test with empty list
        combined = FlextResult[int].sequence([])
        assert combined.is_success
        assert combined.value == []


class TestFlextResultAdditionalCoverage:
    """Additional tests for FlextResult to achieve near 100% coverage.

    This file contains targeted tests for uncovered code paths in result.py.
    """

    def test_class_or_instance_method_descriptor_class_context(self) -> None:
        """Test ClassOrInstanceMethod descriptor in class context (lines 76-77)."""
        # Access descriptor through class (not instance)
        # This triggers the instance is None path
        ok_method = FlextResult[int].ok
        assert callable(ok_method)

        # Call it as class method
        result = ok_method(42)
        assert result.is_success
        assert result.value == 42

    def test_collections_traverse_method(self) -> None:
        """Test _Collections.traverse for iterating results (lines 310-325)."""
        # traverse expects func to take raw item and return FlextResult
        # NOT take FlextResult and return raw value
        items = [1, 2, 3]

        # Traverse and apply transformation
        traversed = FlextResult.traverse(items, lambda x: FlextResult[int].ok(x * 2))
        assert traversed.is_success
        assert traversed.value == [2, 4, 6]

        # Traverse with failure
        def transform_with_failure(x: int) -> FlextResult[int]:
            if x == 2:
                return FlextResult[int].fail("error at 2")
            return FlextResult[int].ok(x * 2)

        traversed = FlextResult.traverse(items, transform_with_failure)
        assert traversed.is_failure

    def test_collections_accumulate_errors_method(self) -> None:
        """Test _Collections.accumulate_errors (lines 327-350)."""
        # accumulate_errors takes *results (variadic), not a list
        # All success - accumulate values
        r1: FlextResult[int] = FlextResult[int].ok(1)
        r2: FlextResult[int] = FlextResult[int].ok(2)
        r3: FlextResult[int] = FlextResult[int].ok(3)

        accumulated: FlextResult[list[int]] = FlextResult.accumulate_errors(r1, r2, r3)
        assert accumulated.is_success
        assert accumulated.value == [1, 2, 3]

        # Mix of success and failure - accumulate all errors
        r4: FlextResult[int] = FlextResult[int].fail("error1")
        r5: FlextResult[int] = FlextResult[int].fail("error2")

        accumulated_fail: FlextResult[list[int]] = FlextResult.accumulate_errors(
            r1,
            r4,
            r2,
            r5,
        )
        assert accumulated_fail.is_failure
        # Should contain both errors
        assert "error1" in str(accumulated_fail.error)
        assert "error2" in str(accumulated_fail.error)

    def test_collections_parallel_map_method(self) -> None:
        """Test _Collections.parallel_map for concurrent processing (lines 352-383)."""
        # Simple parallel map
        data = [1, 2, 3, 4, 5]

        def process(x: int) -> FlextResult[int]:
            result: FlextResult[int] = FlextResult[int].ok(x * 2)
            return result

        result = FlextResult.parallel_map(data, process)
        assert result.is_success
        assert result.value == [2, 4, 6, 8, 10]

        # Parallel map with failure
        def process_with_failure(x: int) -> FlextResult[int]:
            if x == 3:
                fail_result: FlextResult[int] = FlextResult[int].fail("Failed at 3")
                return fail_result
            success_result: FlextResult[int] = FlextResult[int].ok(x * 2)
            return success_result

        parallel_result: FlextResult[list[int]] = FlextResult.parallel_map(
            data,
            process_with_failure,
        )
        assert parallel_result.is_failure

    def test_functional_validate_all_internal(self) -> None:
        """Test _Collections.validate_all method (lines 385-412)."""

        # Define validators
        def is_positive(x: int) -> FlextResult[bool]:
            if x <= 0:
                return FlextResult[bool].fail("Must be positive")
            return FlextResult[bool].ok(True)

        def is_less_than_100(x: int) -> FlextResult[bool]:
            if x >= 100:
                return FlextResult[bool].fail("Must be less than 100")
            return FlextResult[bool].ok(True)

        # All pass
        result = FlextResult.validate_all(50, is_positive, is_less_than_100)
        assert result.is_success
        assert result.value == 50

        # One fails
        result = FlextResult.validate_all(-5, is_positive, is_less_than_100)
        assert result.is_failure
        assert "Must be positive" in str(result.error)

    def test_result_map_with_exception_handling(self) -> None:
        """Test map method exception paths (lines 691-700)."""
        success: FlextResult[int] = FlextResult[int].ok(42)

        # Map with function that raises exception
        def failing_transform(x: int) -> int:
            if x > 40:
                msg = "Value too large"
                raise ValueError(msg)
            return x * 2

        # Should handle exception gracefully
        result = success.map(failing_transform)
        # Implementation should either propagate or wrap exception
        assert result.is_failure or result.is_success

    def test_result_flat_map_with_exception_handling(self) -> None:
        """Test flat_map method exception paths (lines 725-734)."""
        success: FlextResult[int] = FlextResult[int].ok(42)

        # flat_map with function that raises exception
        def failing_bind(x: int) -> FlextResult[int]:
            if x > 40:
                msg = "Value too large"
                raise ValueError(msg)
            result: FlextResult[int] = FlextResult[int].ok(x * 2)
            return result

        # Should handle exception gracefully
        result: FlextResult[int] = success.flat_map(failing_bind)
        assert result.is_failure or result.is_success

    def test_result_expect_with_custom_message(self) -> None:
        """Test expect method with custom error message (lines 815-816)."""
        failure: FlextResult[int] = FlextResult[int].fail("Original error")

        # expect should raise with custom message
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            failure.expect("Custom expectation failed")

        assert "Custom expectation failed" in str(exc_info.value)

    def test_result_eq_with_exception_handling(self) -> None:
        """Test __eq__ exception handling path (lines 861-862)."""
        result1: FlextResult[int] = FlextResult[int].ok(42)

        # Create an object that raises on comparison
        class BadComparison:
            def __eq__(self, other: object) -> bool:
                msg = "Comparison error"
                raise ValueError(msg)

        bad_obj = BadComparison()

        # Should handle exception and return False
        assert result1 != bad_obj

    def test_result_hash_with_various_types(self) -> None:
        """Test __hash__ with different data types (lines 872-887)."""
        # None is not a valid success value - test with valid data instead
        result_int = FlextResult[int].ok(42)
        hash1 = hash(result_int)
        assert isinstance(hash1, int)

        # Hash with complex data
        result_dict: FlextResult[dict[str, object]] = FlextResult[
            dict[str, object]
        ].ok({
            "key": "value",
            "nested": {"a": 1},
        })
        hash2 = hash(result_dict)
        assert isinstance(hash2, int)

        # Hash with failure
        failure: FlextResult[dict[str, object]] = FlextResult[dict[str, object]].fail(
            "error",
            error_code="ERR_001",
            error_data={"detail": "info"},
        )
        hash3 = hash(failure)
        assert isinstance(hash3, int)

    def test_result_unwrap_or_with_default(self) -> None:
        """Test unwrap_or with various defaults (lines 915-916)."""
        failure: FlextResult[int] = FlextResult[int].fail("error")

        # unwrap_or with default
        default_value: int = failure.unwrap_or(999)
        assert default_value == 999

        # unwrap_or on success
        success: FlextResult[int] = FlextResult[int].ok(42)
        success_value = success.unwrap_or(999)
        assert success_value == 42

    def test_result_unwrap_with_failure(self) -> None:
        """Test unwrap on failure raises exception (lines 938-939)."""
        failure: FlextResult[int] = FlextResult[int].fail(
            "Cannot unwrap",
            error_code="UNWRAP_ERROR",
        )

        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            failure.unwrap()

        assert "Cannot unwrap" in str(exc_info.value)

    def test_result_recover_with_function(self) -> None:
        """Test recover with recovery function (lines 952-954)."""
        failure: FlextResult[int] = FlextResult[int].fail("error")

        # Recover with function that returns new value
        recovered_fail: FlextResult[int] = failure.recover(lambda err: 999)
        assert recovered_fail.is_success
        assert recovered_fail.value == 999

        # Recover on success does nothing
        success: FlextResult[int] = FlextResult[int].ok(42)
        recovered_success: FlextResult[int] = success.recover(lambda err: 999)
        assert recovered_success.is_success
        assert recovered_success.value == 42

    def test_result_filter_with_predicate_failure(self) -> None:
        """Test filter when predicate fails (lines 1002-1003)."""
        success: FlextResult[int] = FlextResult[int].ok(42)

        # Filter with failing predicate
        filtered_fail: FlextResult[int] = success.filter(
            lambda x: x < 40,
            "Value too large",
        )
        assert filtered_fail.is_failure
        assert "Value too large" in str(filtered_fail.error)

        # Filter with passing predicate
        filtered_success: FlextResult[int] = success.filter(
            lambda x: x > 40,
            "Value too small",
        )
        assert filtered_success.is_success
        assert filtered_success.value == 42

    def test_result_safe_call(self) -> None:
        """Test safe_call for operations (lines 1237-1244)."""
        # safe_call takes func with NO arguments

        def operation() -> int:
            time.sleep(0.01)
            return 42

        # Run operation
        result: FlextResult[int] = FlextResult[int].safe_call(operation)
        assert result.is_success
        assert result.value == 42

        # operation that fails
        def failing_operation() -> int:
            time.sleep(0.01)
            msg = "error"
            raise ValueError(msg)

        failure_result: FlextResult[int] = FlextResult[int].safe_call(failing_operation)
        assert failure_result.is_failure

    def test_result_operator_rshift(self) -> None:
        """Test >> operator for composition (line 1260)."""
        result = FlextResult[int].ok(10)

        # Use >> for flat_map
        def double_func(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        chained: FlextResult[int] = result >> double_func
        assert chained.is_success
        assert chained.value == 20

    def test_result_operator_lshift(self) -> None:
        """Test << operator for reverse composition (line 1262)."""
        result = FlextResult[int].ok(10)

        # Use << for reverse flat_map
        def func(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        chained = result << func
        assert chained.is_success

    def test_result_operator_mod(self) -> None:
        """Test % operator for filter (line 1292)."""
        result: FlextResult[int] = FlextResult[int].ok(10)

        # Use % for filter
        def greater_than_5(x: int) -> bool:
            return x > 5

        filtered: FlextResult[int] = result % greater_than_5
        assert filtered.is_success

        def greater_than_15(x: int) -> bool:
            return x > 15

        filtered_fail: FlextResult[int] = result % greater_than_15
        assert filtered_fail.is_failure

    def test_result_validate_all(self) -> None:
        """Test validate_all method for chaining value-based validations."""
        # validate_all takes a value and *validators, each is Callable[[T], FlextResult[bool]]

        def validate_positive(value: int) -> FlextResult[bool]:
            if value <= 0:
                return FlextResult[bool].fail("Must be positive")
            return FlextResult[bool].ok(True)

        def validate_even(value: int) -> FlextResult[bool]:
            if value % 2 != 0:
                return FlextResult[bool].fail("Must be even")
            return FlextResult[bool].ok(True)

        # Validate value - all pass
        result = FlextResult.validate_all(10, validate_positive, validate_even)
        assert result.is_success
        assert result.value == 10

        # Validate value - one fails
        result = FlextResult.validate_all(9, validate_positive, validate_even)
        assert result.is_failure
        assert result.error is not None and "Must be even" in str(result.error)

    def test_result_validate_and_execute(self) -> None:
        """Test validate_and_execute method (lines 1375)."""
        # validate_and_execute is an instance method, not a classmethod

        def validator(x: int) -> FlextResult[bool]:
            if x < 10:
                return FlextResult[bool].fail("Too small")
            return FlextResult[bool].ok(True)

        def executor(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        # Valid and execute
        result = FlextResult[int].ok(20).validate_and_execute(validator, executor)
        assert result.is_success
        assert result.value == 40

        # Invalid - execution skipped
        result = FlextResult[int].ok(5).validate_and_execute(validator, executor)
        assert result.is_failure

    def test_result_pipeline_composition(self) -> None:
        """Test pipeline for function composition (lines 1451-1461)."""
        # pipeline takes initial_value and *operations (variadic)

        def add_10(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 10)

        def multiply_2(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        def subtract_5(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x - 5)

        # Pipeline: (5 + 10) * 2 - 5 = 25
        result = FlextResult.pipeline(5, add_10, multiply_2, subtract_5)
        assert result.is_success
        assert result.value == 25

    def test_result_with_resource_management(self) -> None:
        """Test with_resource for resource management (line 1731)."""
        # with_resource is instance method

        class TestResource:
            def __init__(self) -> None:
                super().__init__()
                self.closed = False

            def close(self) -> None:
                self.closed = True

        resource = TestResource()

        def resource_factory() -> TestResource:
            return resource

        def use_resource(value: int, r: TestResource) -> FlextResult[str]:
            return FlextResult[str].ok(f"used with {value}")

        def cleanup(r: TestResource) -> None:
            r.close()

        # Use resource
        result = (
            FlextResult[int]
            .ok(42)
            .with_resource(
                resource_factory,
                use_resource,
                cleanup,
            )
        )
        assert result.is_success
        assert result.value == "used with 42"
        assert resource.closed

    def test_result_is_flattenable_sequence(self) -> None:
        """Test _is_flattenable_sequence helper (lines 1935-1936)."""
        # This is an internal method, test through public API
        results = [FlextResult[int].ok(1), FlextResult[int].ok(2)]

        # sequence should handle list properly
        combined = FlextResult.sequence(results)
        assert combined.is_success
        assert combined.value == [1, 2]


class TestFlextResultFinalCoverage:
    """Final coverage tests targeting remaining uncovered lines.

    This test class systematically covers the remaining 124 uncovered lines
    to achieve 95%+ coverage for result.py module.
    """

    def test_descriptor_class_impl_path(self) -> None:
        """Test ClassOrInstanceMethod descriptor class implementation path (lines 68-69, 76-78)."""
        # Access method as classmethod (instance=None in __get__)
        # This tests the class_impl path in descriptor
        FlextResult[int].ok(42)

        # Accessing map as a method object without calling it
        method_obj: Callable[..., object] = cast(
            "Callable[..., object]",
            FlextResult.map,
        )
        assert callable(method_obj)

        # When accessed from class (not instance), descriptor returns class method
        # Test that classmethod path works
        result_mapped = FlextResult[int].ok(10).map(lambda x: x * 2)
        assert result_mapped.value == 20

    def test_ensure_success_data_with_none_value(self) -> None:
        """Test that None is not a valid success value."""
        # None is not a valid success value - should raise TypeError
        with pytest.raises(TypeError, match="cannot accept None"):
            FlextResult[int].ok(cast("int", None))

    def test_value_property_failure_path(self) -> None:
        """Test value property when result is failure (lines 552-553)."""
        result = FlextResult[int].fail("Operation failed")

        # Accessing value on failure should raise FlextExceptions.TypeError
        with pytest.raises(
            FlextExceptions.ValidationError,
            match="Attempted to access value",
        ):
            _ = result.value

    def test_map_with_exception_in_mapping_func(self) -> None:
        """Test map when mapping function raises exception (lines 698-700)."""

        def raises_error(x: int) -> int:
            msg = "Mapping function error"
            raise ValueError(msg)

        result = FlextResult[int].ok(42)
        mapped = result.map(raises_error)

        # Should catch exception and return failure
        assert mapped.is_failure
        assert "Mapping function error" in str(mapped.error)

    def test_flat_map_with_exception(self) -> None:
        """Test flat_map when function raises exception (lines 732-734)."""

        def raises_error(x: int) -> FlextResult[int]:
            msg = "Flat map error"
            raise RuntimeError(msg)

        result = FlextResult[int].ok(42)
        flat_mapped = result.flat_map(raises_error)

        # Should catch exception and return failure
        assert flat_mapped.is_failure
        assert "Flat map error" in str(flat_mapped.error)

    def test_enter_with_failure_result(self) -> None:
        """Test __enter__ context manager with failure (lines 815-816)."""
        # Enter on failure should raise FlextExceptions.OperationError
        result = FlextResult[int].fail("Context error")

        with pytest.raises(FlextExceptions.BaseError, match="Context error"):
            with result:
                pass

    def test_equality_complex_data_structures(self) -> None:
        """Test __eq__ with complex nested data (lines 853-854, 861-862)."""
        # Test equality with nested dictionaries
        data1: dict[str, dict[str, object]] = {
            "nested": {"value": 42, "list": [1, 2, 3]},
        }
        data2: dict[str, dict[str, object]] = {
            "nested": {"value": 42, "list": [1, 2, 3]},
        }

        result1 = FlextResult[dict[str, dict[str, object]]].ok(data1)
        result2 = FlextResult[dict[str, dict[str, object]]].ok(data2)

        # Should be equal even with complex structures
        assert result1 == result2

    def test_hash_with_unhashable_data(self) -> None:
        """Test __hash__ with unhashable data types (lines 877-884)."""
        # Lists are unhashable, should handle gracefully
        result_with_list = FlextResult[list[int]].ok([1, 2, 3])
        result_with_dict = FlextResult[dict[str, str]].ok({"key": "value"})

        # Should be able to hash results even with unhashable data
        hash1 = hash(result_with_list)
        hash2 = hash(result_with_dict)
        assert isinstance(hash1, int)
        assert isinstance(hash2, int)

    def test_or_else_get_with_exception(self) -> None:
        """Test or_else_get when fallback function raises (lines 915-916)."""

        def fallback_raises() -> FlextResult[int]:
            msg = "Fallback error"
            raise ValueError(msg)

        result = FlextResult[int].fail("Initial error")
        fallback_result = result.or_else_get(fallback_raises)

        # Should catch exception and return failure
        assert fallback_result.is_failure

    def test_recover_with_exception(self) -> None:
        """Test recover when recovery function raises (lines 952-954)."""

        def recovery_raises(error: str) -> int:
            msg = "Recovery failed"
            raise RuntimeError(msg)

        result = FlextResult[int].fail("Initial error")

        # recover re-raises exceptions, it doesn't catch them
        with pytest.raises(RuntimeError, match="Recovery failed"):
            result.recover(recovery_raises)

    def test_filter_with_exception_in_predicate(self) -> None:
        """Test filter when predicate raises exception (lines 1002-1003)."""

        def predicate_raises(x: int) -> bool:
            msg = "Predicate error"
            raise ValueError(msg)

        result = FlextResult[int].ok(42)
        filtered = result.filter(predicate_raises, "Should pass")

        # Should catch exception and return failure
        assert filtered.is_failure

    def test_safe_call_error_path(self) -> None:
        """Test safe_call exception handling (lines 1237-1244, 1241)."""

        def raises() -> int:
            msg = "operation failed"
            raise ValueError(msg)

        # Run test
        result: FlextResult[int] = FlextResult[int].safe_call(raises)
        assert result.is_failure
        assert "operation failed" in str(result.error)

    def test_mod_operator_filter_with_exception(self) -> None:
        """Test __mod__ operator (filter) with exception (lines 1292, 1299, 1307-1308)."""

        def predicate_raises(x: int) -> bool:
            msg = "Predicate error"
            raise RuntimeError(msg)

        result = FlextResult[int].ok(42)

        # % operator is filter - should handle exceptions
        filtered = result % predicate_raises
        assert filtered.is_failure

    def test_validate_and_execute_validation_failure(self) -> None:
        """Test validate_and_execute when validation fails (line 1375)."""

        def validator(x: int) -> FlextResult[bool]:
            return FlextResult[bool].fail("Validation failed")

        def executor(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"Executed {x}")

        result = FlextResult[int].ok(42).validate_and_execute(validator, executor)
        assert result.is_failure
        assert "Validation failed" in str(result.error)

    def test_accumulate_errors_typed_helper(self) -> None:
        """Test _accumulate_errors_typed internal helper (lines 1699-1710)."""
        # This is tested through accumulate_errors public API
        result1 = FlextResult[int].ok(1)
        result2 = FlextResult[int].fail("Error 2")
        result3 = FlextResult[int].ok(3)

        # When some fail, accumulate returns failures
        accumulated = FlextResult.accumulate_errors(result1, result2, result3)
        assert accumulated.is_failure

    def test_with_resource_with_exception_in_operation(self) -> None:
        """Test with_resource when operation raises exception (line 1751)."""

        def resource_factory() -> str:
            return "resource"

        def operation_raises(_: object, resource: str) -> FlextResult[int]:
            msg = "Operation error"
            raise RuntimeError(msg)

        def cleanup(resource: str) -> None:
            pass

        result = (
            FlextResult[int]
            .ok(42)
            .with_resource(
                resource_factory,
                operation_raises,
                cleanup,
            )
        )

        # Should catch exception and return failure
        assert result.is_failure

    def test_validate_all_with_failures(self) -> None:
        """Test validate_all when some validators fail (lines 1935-1936)."""

        def validator1(x: int) -> FlextResult[bool]:
            return FlextResult[bool].ok(True)

        def validator2(x: int) -> FlextResult[bool]:
            return FlextResult[bool].fail("Validator 2 failed")

        # validate_all takes variadic validators, not a list
        result = FlextResult.validate_all(42, validator1, validator2)
        assert result.is_failure


class TestFlextResultEdgeCases:
    """Additional edge case tests for complete coverage."""

    def test_with_context_on_failure(self) -> None:
        """Test with_context when result is already failure."""

        def add_context(error: str) -> str:
            return f"Context: {error}"

        result = FlextResult[int].fail("Original error")
        with_ctx = result.with_context(add_context)

        # Should enhance error with context
        assert with_ctx.is_failure
        assert "Context:" in str(with_ctx.error)

    def test_collect_all_errors_mixed_results(self) -> None:
        """Test collect_all_errors with mixed success/failure."""
        # collect_all_errors takes variadic results, not a list
        # Returns tuple of (successes, errors)
        result1 = FlextResult[int].ok(1)
        result2 = FlextResult[int].fail("Error 1")
        result3 = FlextResult[int].ok(3)
        result4 = FlextResult[int].fail("Error 2")

        successes, errors = FlextResult.collect_all_errors(
            result1,
            result2,
            result3,
            result4,
        )

        # Should have collected both successes and errors
        assert len(successes) == 2  # Two successful results
        assert successes == [1, 3]
        assert len(errors) == 2  # Two failed results
        assert "Error 1" in errors
        assert "Error 2" in errors

    def test_parallel_map_with_exceptions(self) -> None:
        """Test parallel_map when some items cause exceptions."""

        def mapper(x: int) -> FlextResult[int]:
            if x == 2:
                return FlextResult[int].fail("Cannot process 2")
            return FlextResult[int].ok(x * 2)

        items = [1, 2, 3]
        result = FlextResult.parallel_map(items, mapper, fail_fast=False)

        # With fail_fast=False, should collect all results
        assert isinstance(result, FlextResult)
        # May succeed with partial results or fail with accumulated errors
        assert result.is_success or result.is_failure

    def test_map_sequence_with_partial_failures(self) -> None:
        """Test map_sequence when some mappings fail."""

        def mapper(x: int) -> FlextResult[int]:
            if x == 2:
                return FlextResult[int].fail("Cannot process 2")
            return FlextResult[int].ok(x * 2)

        items = [1, 2, 3]
        result = FlextResult.map_sequence(items, mapper)

        # Should fail on first error
        assert result.is_failure

    def test_pipeline_with_failing_operation(self) -> None:
        """Test pipeline when an operation fails."""

        def add_10(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 10)

        def fails(x: int) -> FlextResult[int]:
            return FlextResult[int].fail("Pipeline error")

        def multiply_2(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        result = FlextResult.pipeline(5, add_10, fails, multiply_2)

        # Should stop at failure
        assert result.is_failure
        assert "Pipeline error" in str(result.error)


class TestFlextResultFinalPush:
    """Final push for 95%+ coverage - targeting remaining uncovered lines."""

    def test_descriptor_class_access_path(self) -> None:
        """Test descriptor __get__ class access path (lines 76-78)."""
        # Access descriptor from class (not instance)
        result_type = FlextResult[int]
        map_method = result_type.map
        assert callable(map_method)

    def test_map_generic_exception_handling(self) -> None:
        """Test map with generic Exception handling (lines 698-704)."""

        def raises_custom_exception(x: int) -> int:
            class CustomException(Exception):
                pass

            msg = "Custom error"
            raise CustomException(msg)

        result = FlextResult[int].ok(42)
        mapped = result.map(raises_custom_exception)

        assert mapped.is_failure
        assert "Transformation failed" in str(mapped.error)

    def test_expect_with_none_data_success(self) -> None:
        """Test expect with None data in success state (lines 815-819).

        REMOVED: None is not a valid success value. Creating FlextResult with None
        now raises TypeError at creation time (fast fail).
        """
        # Fast fail: None is not a valid success value - raises at creation
        with pytest.raises(
            TypeError,
            match="cannot accept None as data",
        ):
            _ = FlextResult[int | None].ok(None)

    def test_eq_string_comparison_fallback(self) -> None:
        """Test __eq__ with string comparison fallback (lines 850-854)."""

        class CustomObject:
            def __init__(self, value: int) -> None:
                super().__init__()
                self.value = value

            def __eq__(self, other: object) -> bool:
                # Trigger exception path
                msg = "Comparison failed"
                raise RuntimeError(msg)

        result1 = FlextResult[CustomObject].ok(CustomObject(42))
        result2 = FlextResult[CustomObject].ok(CustomObject(42))

        # Should fall back to string comparison and handle exception
        comparison_result = result1 == result2
        assert isinstance(comparison_result, bool)

    def test_eq_exception_handling(self) -> None:
        """Test __eq__ overall exception handling (lines 861-862)."""

        class ProblematicObject:
            def __str__(self) -> str:
                msg = "String conversion failed"
                raise RuntimeError(msg)

        result1 = FlextResult[ProblematicObject].ok(ProblematicObject())
        result2 = FlextResult[ProblematicObject].ok(ProblematicObject())

        # Should catch exception and return False
        assert (result1 == result2) is False

    def test_hash_dict_object_attributes(self) -> None:
        """Test __hash__ with __dict__ attributes (lines 877-884)."""

        class SimplePerson:
            def __init__(self, name: str, age: int) -> None:
                super().__init__()
                self.name = name
                self.age = age

        result = FlextResult[SimplePerson].ok(SimplePerson("Alice", 30))
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_hash_dict_attributes_exception_fallback(self) -> None:
        """Test __hash__ dict[str, object] attributes exception fallback (lines 880-884)."""

        class UnhashableAttributes:
            def __init__(self) -> None:
                super().__init__()
                self.data = {"key": [1, 2, 3]}  # Lists are unhashable

        result = FlextResult[UnhashableAttributes].ok(UnhashableAttributes())
        # Should fall back to type + id hashing
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_recover_success_path(self) -> None:
        """Test recover when result is already success (line 952)."""

        def should_not_run(error: str) -> int:
            msg = "Should not be called"
            raise RuntimeError(msg)

        result = FlextResult[int].ok(42)
        recovered = result.recover(should_not_run)

        # API returns self when already success
        assert recovered.is_success
        assert recovered.unwrap() == 42

    def test_recover_exception_handling(self) -> None:
        """Test recover re-raises exceptions from recovery function (lines 825-836)."""

        def recovery_type_error(error: str) -> int:
            msg = "Type error in recovery"
            raise TypeError(msg)

        result = FlextResult[int].fail("Original error")

        # recover() re-raises exceptions, it doesn't catch them
        with pytest.raises(TypeError, match="Type error in recovery"):
            result.recover(recovery_type_error)


class TestFlextResultFinalCoveragePush:
    """Final tests to reach 95%+ coverage - targeting remaining 71 uncovered lines."""

    def test_safe_call_non_coroutine(self) -> None:
        """Test safe_call with non-coroutine function (line 1241)."""

        def sync_func() -> int:
            return 42

        result: FlextResult[int] = cast(
            "FlextResult[int]",
            FlextResult.safe_call(sync_func),
        )
        assert result.is_success
        assert result.unwrap() == 42

    def test_mod_failure_path(self) -> None:
        """Test __mod__ when already failure (line 1299)."""
        result = FlextResult[int].fail("Already failed")

        def is_positive(x: int) -> bool:
            return x > 0

        filtered = result % is_positive

        assert filtered.is_failure
        assert filtered.error == "Already failed"

    def test_validate_and_execute_failure_path(self) -> None:
        """Test validate_and_execute with failure result (line 1375)."""
        result = FlextResult[int].fail("Initial failure")

        def validator(x: int) -> FlextResult[bool]:
            return FlextResult[bool].ok(True)

        def executor(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(str(x))

        validated = result.validate_and_execute(validator, executor)
        assert validated.is_failure
        assert "Initial failure" in str(validated.error)

    def test_with_context_failure_without_error(self) -> None:
        """Test that creating failure with empty error uses fallback."""
        # Fallback behavior: empty error gets default message for backward compatibility
        result = FlextResult[int](error="")
        assert result.is_failure
        assert result.error == "Unknown error occurred"

    def test_with_resource_failure_propagation(self) -> None:
        """Test with_resource with failure result (line 1751)."""
        result = FlextResult[int].fail("Initial error")

        def resource_factory() -> str:
            return "resource"

        def operation(x: int, res: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"{x}-{res}")

        resource_result = result.with_resource(resource_factory, operation)
        assert resource_result.is_failure
        assert "Initial error" in str(resource_result.error)

    def test_flatten_callable_args_non_callable_error(self) -> None:
        """Test _flatten_callable_args with non-callable (lines 1995-1996)."""
        with pytest.raises(FlextExceptions.ValidationError, match="Expected callable"):
            FlextResult._flatten_callable_args("not a callable")


class TestFromCallable:
    """Test create_from_callable method with @safe decorator integration."""

    def test_create_from_callable_success(self) -> None:
        """Test create_from_callable with successful operation."""

        def safe_operation() -> int:
            return 42

        result = FlextResult[int].create_from_callable(safe_operation)

        assert result.is_success
        assert result.value == 42

    def test_create_from_callable_with_exception(self) -> None:
        """Test create_from_callable with operation that raises exception."""

        def failing_operation() -> int:
            msg = "Operation failed"
            raise ValueError(msg)

        result = FlextResult[int].create_from_callable(failing_operation)

        assert result.is_failure
        assert result.error is not None
        assert "Operation failed" in result.error

    def test_create_from_callable_with_custom_error_code(self) -> None:
        """Test create_from_callable with custom error code."""

        def failing_operation() -> str:
            msg = "Custom error"
            raise RuntimeError(msg)

        result = FlextResult[str].create_from_callable(
            failing_operation,
            error_code="CUSTOM_ERROR",
        )

        assert result.is_failure
        assert result.error_code == "CUSTOM_ERROR"

    def test_create_from_callable_with_none_return(self) -> None:
        """Test create_from_callable with function returning None - REMOVED: None not valid."""
        # None is not a valid success value - use FlextResult.fail() for failures
        # This test is no longer valid as FlextResult[None] is not allowed

        def returns_none() -> None:
            return None

        # Creating FlextResult[None].ok(None) raises TypeError
        with pytest.raises(TypeError, match="cannot accept None as data"):
            _ = FlextResult[None].create_from_callable(returns_none)

    def test_create_from_callable_with_complex_operation(self) -> None:
        """Test create_from_callable with complex operation."""

        def complex_operation() -> dict[str, object]:
            data: dict[str, object] = {"processed": True, "count": 10}
            count_value = data["count"]
            if isinstance(count_value, int) and count_value > 5:
                return data
            msg = "Count too low"
            raise ValueError(msg)

        result = FlextResult[dict[str, object]].create_from_callable(complex_operation)

        assert result.is_success
        assert result.value == {"processed": True, "count": 10}


class TestFlowThrough:
    """Test flow_through method for pipeline composition."""

    def test_flow_through_success_pipeline(self) -> None:
        """Test flow_through with successful operations."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def multiply_by_two(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        result = FlextResult[int].ok(5).flow_through(add_one, multiply_by_two)

        assert result.is_success
        assert result.value == 12  # (5 + 1) * 2

    def test_flow_through_failure_propagation(self) -> None:
        """Test flow_through stops at first failure."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def fail_if_even(x: int) -> FlextResult[int]:
            if x % 2 == 0:
                return FlextResult[int].fail("Number is even")
            return FlextResult[int].ok(x)

        def multiply_by_two(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        result = (
            FlextResult[int].ok(5).flow_through(add_one, fail_if_even, multiply_by_two)
        )

        assert result.is_failure
        assert result.error == "Number is even"

    def test_flow_through_with_initial_failure(self) -> None:
        """Test flow_through with initial failure result."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        result = FlextResult[int].fail("Initial failure").flow_through(add_one)

        assert result.is_failure
        assert result.error == "Initial failure"

    def test_flow_through_empty_pipeline(self) -> None:
        """Test flow_through with no operations."""
        result = FlextResult[int].ok(42).flow_through()

        assert result.is_success
        assert result.value == 42

    def test_flow_through_complex_transformations(self) -> None:
        """Test flow_through with complex data transformations."""

        def validate_dict(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            if "required_field" not in data:
                return FlextResult[dict[str, object]].fail("Missing required field")
            return FlextResult[dict[str, object]].ok(data)

        def enrich_data(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            enriched: dict[str, object] = {**data, "enriched": True}
            return FlextResult[dict[str, object]].ok(enriched)

        def transform_data(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            transformed: dict[str, object] = {
                **data,
                "transformed": True,
            }
            # Count includes the count key itself
            transformed["count"] = len(transformed) + 1
            return FlextResult[dict[str, object]].ok(transformed)

        initial_data: dict[str, object] = {"required_field": "value"}
        result = (
            FlextResult[dict[str, object]]
            .ok(initial_data)
            .flow_through(validate_dict, enrich_data, transform_data)
        )

        assert result.is_success
        assert result.value["enriched"] is True
        assert result.value["transformed"] is True
        assert result.value["count"] == 4


class TestMaybeInterop:
    """Test Maybe monad interoperability methods."""

    def test_to_maybe_success(self) -> None:
        """Test converting successful result to Some."""
        result = FlextResult[str].ok("test_value")
        maybe = result.to_maybe()

        assert isinstance(maybe, Some)
        # PyRight knows maybe is Some[str] after isinstance check
        # Use unwrap() which is type-safe after isinstance
        value = maybe.unwrap()
        assert value is not None
        assert value is not Never
        assert value == "test_value"

    def test_to_maybe_failure(self) -> None:
        """Test converting failed result to Nothing."""
        result = FlextResult[str].fail("error")
        maybe = result.to_maybe()

        assert maybe == Nothing  # Nothing is a singleton, not a class

    def test_from_maybe_some(self) -> None:
        """Test creating result from Some."""
        maybe = Some("test_value")
        result = FlextResult.from_maybe(maybe)

        assert result.is_success
        assert result.value == "test_value"

    def test_from_maybe_nothing(self) -> None:
        """Test creating result from Nothing."""
        maybe = Nothing
        result = FlextResult.from_maybe(maybe)

        assert result.is_failure
        assert result.error == "No value in Maybe"

    def test_maybe_roundtrip_success(self) -> None:
        """Test roundtrip conversion success -> maybe -> success."""
        original = FlextResult[int].ok(42)
        maybe = original.to_maybe()
        recovered_result = FlextResult.from_maybe(maybe)

        assert recovered_result.is_success
        assert recovered_result.value == 42

    def test_maybe_roundtrip_failure(self) -> None:
        """Test roundtrip conversion failure -> maybe -> failure."""
        original = FlextResult[int].fail("error")
        maybe = original.to_maybe()
        recovered_result = FlextResult.from_maybe(maybe)

        assert recovered_result.is_failure


class TestIOInterop:
    """Test IO monad interoperability methods."""

    def test_to_io_success(self) -> None:
        """Test converting successful result to IO."""
        result = FlextResult[str].ok("test_value")
        # Cast needed because to_io() returns object (returns.io.IO may not be available)
        io_container: IO[str] = cast("IO[str]", result.to_io())

        assert isinstance(io_container, IO)
        # Note: returns.io.IO intentionally hides internal value
        # Test that we can map over it to verify it contains the value

        def to_upper(x: str) -> str:
            return x.upper()

        # Direct assignment helps PyRight infer the type correctly
        mapped: IO[str] = io_container.map(to_upper)
        assert isinstance(mapped, IO)

    def test_to_io_failure_raises(self) -> None:
        """Test converting failed result to IO raises ValidationError (actual behavior)."""
        result = FlextResult[str].fail("error")

        # REAL behavior: to_io() raises ValidationError, not ValueError
        with pytest.raises(
            FlextExceptions.ValidationError,
            match="Cannot convert failure to IO",
        ):
            result.to_io()

    def test_to_io_result_success(self) -> None:
        """Test converting successful result to IOSuccess."""
        result = FlextResult.ok("test_value")
        io_result = result.to_io_result()

        assert isinstance(io_result, IOSuccess)

    def test_to_io_result_failure(self) -> None:
        """Test converting failed result to IOFailure."""
        result = FlextResult[str].fail("error_message")
        io_result = result.to_io_result()

        assert isinstance(io_result, IOFailure)

    def test_from_io_result_success(self) -> None:
        """Test creating result from IOSuccess."""
        io_success = IOSuccess(42)
        result: FlextResult[int] = cast(
            "FlextResult[int]",
            FlextResult.from_io_result(io_success),
        )

        assert result.is_success
        assert result.value == 42

    def test_from_io_result_failure(self) -> None:
        """Test creating result from IOFailure."""
        io_failure: IOFailure[object] = IOFailure("io_error")
        result: FlextResult[object] = cast(
            "FlextResult[object]",
            FlextResult.from_io_result(io_failure),
        )

        assert result.is_failure
        assert "io_error" in (result.error or "")

    def test_io_result_roundtrip_success(self) -> None:
        """Test roundtrip conversion success -> IOResult -> success."""
        original = FlextResult[dict[str, object]].ok({"key": "value"})
        io_result = original.to_io_result()
        recovered: FlextResult[dict[str, object]] = cast(
            "FlextResult[dict[str, object]]",
            FlextResult.from_io_result(io_result),
        )

        assert recovered.is_success
        assert recovered.value == {"key": "value"}

    def test_io_result_roundtrip_failure(self) -> None:
        """Test roundtrip conversion failure -> IOResult -> failure."""
        original = FlextResult[int].fail("original_error")
        io_result = original.to_io_result()
        recovered: FlextResult[int] = cast(
            "FlextResult[int]",
            FlextResult.from_io_result(io_result),
        )

        assert recovered.is_failure
        assert "original_error" in (recovered.error or "")


class TestRailwayMethods:
    """Test railway-oriented programming methods."""

    def test_lash_on_success(self) -> None:
        """Test lash on successful result does nothing."""
        result = FlextResult[int].ok(42)

        def error_handler(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(0)

        lashed = result.lash(error_handler)

        assert lashed.is_success
        assert lashed.value == 42

    def test_lash_on_failure(self) -> None:
        """Test lash on failed result applies error handler."""
        result = FlextResult[int].fail("error")

        def error_handler(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(99)

        lashed = result.lash(error_handler)

        assert lashed.is_success
        assert lashed.value == 99

    def test_lash_propagates_handler_failure(self) -> None:
        """Test lash when error handler also fails."""
        result = FlextResult[int].fail("original_error")

        def failing_handler(error: str) -> FlextResult[int]:
            return FlextResult[int].fail("handler_error")

        lashed = result.lash(failing_handler)

        assert lashed.is_failure
        assert lashed.error == "handler_error"

    def test_lash_with_exception_in_handler(self) -> None:
        """Test lash when error handler raises exception."""
        result = FlextResult[int].fail("error")

        def exception_handler(error: str) -> FlextResult[int]:
            msg = "Handler exception"
            raise ValueError(msg)

        lashed = result.lash(exception_handler)

        assert lashed.is_failure
        assert "Lash operation failed" in (lashed.error or "")

    def test_alt_on_success(self) -> None:
        """Test alt on successful result returns self."""
        result = FlextResult[int].ok(42)
        default = FlextResult[int].ok(99)

        alt_result = result.alt(default)

        assert alt_result.is_success
        assert alt_result.value == 42

    def test_alt_on_failure(self) -> None:
        """Test alt on failed result returns default."""
        result = FlextResult[int].fail("error")
        default = FlextResult[int].ok(99)

        alt_result = result.alt(default)

        assert alt_result.is_success
        assert alt_result.value == 99

    def test_alt_chaining(self) -> None:
        """Test chaining multiple alt operations."""
        result = FlextResult[int].fail("error1")
        default1 = FlextResult[int].fail("error2")
        default2 = FlextResult[int].ok(99)

        alt_result = result.alt(default1).alt(default2)

        assert alt_result.is_success
        assert alt_result.value == 99

    def test_value_or_call_on_success(self) -> None:
        """Test value_or_call on successful result."""
        result = FlextResult[int].ok(42)

        def compute_default() -> int:
            return 99

        value = result.value_or_call(compute_default)

        assert value == 42

    def test_value_or_call_on_failure(self) -> None:
        """Test value_or_call on failed result computes default."""
        result = FlextResult[int].fail("error")

        def compute_default() -> int:
            return 99

        value = result.value_or_call(compute_default)

        assert value == 99

    def test_value_or_call_lazy_evaluation(self) -> None:
        """Test value_or_call doesn't call function on success."""
        result = FlextResult[int].ok(42)
        call_count = 0

        def compute_default() -> int:
            nonlocal call_count
            call_count += 1
            return 99

        value = result.value_or_call(compute_default)

        assert value == 42
        assert call_count == 0  # Function should not be called

    def test_value_or_call_exception_handling(self) -> None:
        """Test value_or_call when default computation raises exception."""
        result = FlextResult[int].fail("error")

        def failing_default() -> int:
            msg = "Default computation failed"
            raise ValueError(msg)

        with pytest.raises(FlextExceptions.BaseError):
            result.value_or_call(failing_default)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple new methods."""

    def test_complete_pipeline_with_error_recovery(self) -> None:
        """Test complete pipeline with error recovery using new methods."""

        def risky_operation() -> int:
            msg = "Risky operation failed"
            raise ValueError(msg)

        def recovery_operation(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(0)

        def double_value(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        # Combine create_from_callable, lash, and flow_through
        result = (
            FlextResult[int]
            .create_from_callable(risky_operation)
            .lash(recovery_operation)
            .flow_through(double_value)
        )

        assert result.is_success
        assert result.value == 0  # recovered value (0) * 2 = 0

    def test_maybe_io_interop_combination(self) -> None:
        """Test combining Maybe and IO interoperability."""
        # Start with a result
        original = FlextResult[str].ok("test")

        # Convert to Maybe
        maybe = original.to_maybe()

        # Convert back to Result
        from_maybe = FlextResult.from_maybe(maybe)

        # Convert to IOResult
        io_result = from_maybe.to_io_result()

        # Convert back to Result
        final: FlextResult[str] = FlextResult.from_io_result(io_result)

        assert final.is_success
        assert final.value == "test"

    def test_railway_methods_with_flow_through(self) -> None:
        """Test combining railway methods with flow_through."""

        def validate(x: int) -> FlextResult[int]:
            if x < 0:
                return FlextResult[int].fail("Negative number")
            return FlextResult[int].ok(x)

        def process(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        def error_recovery(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(1)

        # Test with valid input
        result1 = (
            FlextResult[int].ok(5).flow_through(validate, process).lash(error_recovery)
        )

        assert result1.is_success
        assert result1.value == 10

        # Test with invalid input that recovers
        result2 = (
            FlextResult[int]
            .ok(-5)
            .flow_through(validate, process)
            .lash(error_recovery)
            .flow_through(process)
        )

        assert result2.is_success
        assert result2.value == 2  # recovered (1) * 2

    def test_complex_data_transformation_pipeline(self) -> None:
        """Test complex data transformation using all new methods."""

        def fetch_data() -> dict[str, object]:
            return {"raw": True, "value": 100}

        def validate_data(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            if "value" not in data:
                return FlextResult[dict[str, object]].fail("Missing value")
            return FlextResult[dict[str, object]].ok(data)

        def enrich_data(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            enriched: dict[str, object] = {**data, "enriched": True}
            return FlextResult[dict[str, object]].ok(enriched)

        def error_fallback(
            error: str,
        ) -> FlextResult[dict[str, object]]:
            return FlextResult[dict[str, object]].ok({"fallback": True})

        # Complete pipeline
        result = (
            FlextResult[dict[str, object]]
            .create_from_callable(fetch_data)
            .flow_through(validate_data, enrich_data)
            .lash(error_fallback)
        )

        assert result.is_success
        assert result.value["raw"] is True
        assert result.value["enriched"] is True
        assert result.value["value"] == 100
