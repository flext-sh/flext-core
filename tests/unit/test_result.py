"""Comprehensive tests for FlextResult - Railway Pattern Implementation.

This module tests the core FlextResult railway pattern which is the foundation
of error handling across the entire FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable

import pytest

from flext_core import FlextResult


class TestFlextResult:
    """Test suite for FlextResult railway pattern implementation."""

    def test_result_creation_success(self) -> None:
        """Test successful result creation."""
        result = FlextResult[str].ok("test_value")

        assert result.is_success
        assert not result.is_failure
        assert result.value == "test_value"
        assert result.data == "test_value"  # Backward compatibility
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
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_success
        assert mapped.value == 10

    def test_result_map_failure(self) -> None:
        """Test map operation on failed result."""
        result = FlextResult[int].fail("test_error")
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_failure
        assert mapped.error == "test_error"

    def test_result_flat_map_success(self) -> None:
        """Test flat_map operation on successful result."""
        result = FlextResult[int].ok(5)
        flat_mapped = result.flat_map(lambda x: FlextResult[str].ok(f"value_{x}"))

        assert flat_mapped.is_success
        assert flat_mapped.value == "value_5"

    def test_result_flat_map_failure(self) -> None:
        """Test flat_map operation on failed result."""
        result = FlextResult[int].fail("test_error")
        flat_mapped = result.flat_map(lambda x: FlextResult[str].ok(f"value_{x}"))

        assert flat_mapped.is_failure
        assert flat_mapped.error == "test_error"

    def test_result_when_success(self) -> None:
        """Test when operation on successful result."""
        result = FlextResult[int].ok(5)
        when_result = result.when(lambda x: x > 3)

        assert when_result.is_success
        assert when_result.value == 5

    def test_result_when_failure(self) -> None:
        """Test when operation on failed result."""
        result = FlextResult[int].ok(2)
        when_result = result.when(lambda x: x > 3)

        assert when_result.is_failure
        assert (
            when_result.error is not None and "condition" in when_result.error.lower()
        )

    def test_result_tap_error(self) -> None:
        """Test tap_error operation."""
        result = FlextResult[int].fail("original_error")

        # Test side effect function that doesn't return anything
        side_effect_called = False

        def side_effect(e: str) -> None:
            nonlocal side_effect_called
            side_effect_called = True
            assert e == "original_error"

        tapped_error = result.tap_error(side_effect)

        assert tapped_error.is_failure
        assert tapped_error.error == "original_error"
        assert side_effect_called

    def test_result_unwrap_or(self) -> None:
        """Test unwrap_or operation."""
        success_result = FlextResult[str].ok("success")
        failure_result = FlextResult[str].fail("error")

        assert success_result.unwrap_or("default") == "success"
        assert failure_result.unwrap_or("default") == "default"

    def test_result_expect(self) -> None:
        """Test expect operation."""
        success_result = FlextResult[str].ok("success")

        assert success_result.expect("Should not fail") == "success"

    def test_result_expect_failure(self) -> None:
        """Test expect operation on failure."""
        failure_result = FlextResult[str].fail("error")

        with pytest.raises(RuntimeError, match="Should fail"):
            failure_result.expect("Should fail")

    def test_result_railway_composition(self) -> None:
        """Test railway-oriented composition."""

        def validate_input(data: dict) -> FlextResult[dict]:
            if not data.get("value"):
                return FlextResult[dict].fail("Missing value")
            return FlextResult[dict].ok(data)

        def process_data(data: dict) -> FlextResult[int]:
            return FlextResult[int].ok(data["value"] * 2)

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

    def test_result_backward_compatibility(self) -> None:
        """Test backward compatibility of .data and .value properties."""
        result = FlextResult[dict].ok({"key": "value"})

        # Both .data and .value should work for ecosystem compatibility
        assert result.data == {"key": "value"}
        assert result.value == {"key": "value"}
        assert result.data == result.value

    def test_result_error_handling_edge_cases(self) -> None:
        """Test edge cases in error handling."""
        # Test empty error message (gets converted to default message)
        result = FlextResult[str].fail("")
        assert result.is_failure
        assert result.error == "Unknown error occurred"

        # Test None value in success
        result = FlextResult[None].ok(None)
        assert result.is_success
        assert result.value is None

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

    def test_result_chain_results_static_method(self) -> None:
        """Test chain_results static method."""
        result1 = FlextResult[int].ok(1)
        result2 = FlextResult[int].ok(2)
        result3 = FlextResult[int].ok(3)

        chained = FlextResult.chain_results(result1, result2, result3)
        assert chained.is_success
        assert chained.value == [1, 2, 3]

    def test_result_chain_results_static_method_failure(self) -> None:
        """Test chain_results static method with failure."""
        result1 = FlextResult[int].ok(1)
        result2 = FlextResult[int].fail("error")
        result3 = FlextResult[int].ok(3)

        chained = FlextResult.chain_results(result1, result2, result3)
        assert chained.is_failure
        assert chained.error is not None and "Chain failed at result" in chained.error

    def test_result_combine_static_method(self) -> None:
        """Test combine static method."""
        result1 = FlextResult[str].ok("hello")
        result2 = FlextResult[str].ok("world")
        result3 = FlextResult[str].ok("test")

        combined = FlextResult.combine(result1, result2, result3)
        assert combined.is_success
        assert combined.value == ["hello", "world", "test"]

    def test_result_combine_static_method_failure(self) -> None:
        """Test combine static method with failure."""
        result1 = FlextResult[str].ok("hello")
        result2 = FlextResult[str].fail("error")
        result3 = FlextResult[str].ok("test")

        combined = FlextResult.combine(result1, result2, result3)
        assert combined.is_failure
        assert combined.error is not None and "error" in combined.error

    def test_result_tap_method(self) -> None:
        """Test tap method."""
        result = FlextResult[str].ok("test")
        tapped_values = []

        tapped = result.tap(tapped_values.append)
        assert tapped.is_success
        assert tapped.value == "test"
        assert tapped_values == ["test"]

    def test_result_tap_method_failure(self) -> None:
        """Test tap method on failure."""
        result = FlextResult[str].fail("error")
        tapped_values = []

        tapped = result.tap(tapped_values.append)
        assert tapped.is_failure
        assert tapped.error == "error"
        assert tapped_values == []  # Should not be called

    def test_result_recover_method(self) -> None:
        """Test recover method."""
        result = FlextResult[str].fail("error")
        recovered = result.recover(lambda e: f"recovered_{e}")

        assert recovered.is_success
        assert recovered.value == "recovered_error"

    def test_result_recover_method_success(self) -> None:
        """Test recover method on success."""
        result = FlextResult[str].ok("success")
        recovered = result.recover(lambda e: f"recovered_{e}")

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

        or_result = result1.or_else_get(lambda: FlextResult[str].ok("fallback"))
        assert or_result.is_success
        assert or_result.value == "fallback"

    def test_result_or_else_get_method_success(self) -> None:
        """Test or_else_get method on success."""
        result1 = FlextResult[str].ok("success1")

        or_result = result1.or_else_get(lambda: FlextResult[str].ok("fallback"))
        assert or_result.is_success
        assert or_result.value == "success1"

    def test_result_unless_method(self) -> None:
        """Test unless method."""
        result = FlextResult[int].ok(5)
        unless_result = result.unless(lambda x: x > 3)

        assert unless_result.is_failure
        assert (
            unless_result.error is not None
            and "condition" in unless_result.error.lower()
        )

    def test_result_unless_method_success(self) -> None:
        """Test unless method on success."""
        result = FlextResult[int].ok(2)
        unless_result = result.unless(lambda x: x > 3)

        assert unless_result.is_success
        assert unless_result.value == 2

    def test_result_with_context_method(self) -> None:
        """Test with_context method."""
        result = FlextResult[str].fail("error")
        context_result = result.with_context(lambda e: f"Context: {e}")

        assert context_result.is_failure
        assert (
            context_result.error is not None
            and "Context: error" in context_result.error
        )

    def test_result_with_context_method_success(self) -> None:
        """Test with_context method on success."""
        result = FlextResult[str].ok("success")
        context_result = result.with_context(lambda e: f"Context: {e}")

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

    def test_result_collect_successes_class_method(self) -> None:
        """Test collect_successes class method."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("error"),
            FlextResult[int].ok(3),
        ]

        successes = FlextResult.collect_successes(results)
        assert successes == [1, 3]

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

    def test_result_to_either_method(self) -> None:
        """Test to_either method."""
        result = FlextResult[int].ok(5)
        either = result.to_either()

        assert either == (5, None)

    def test_result_to_either_method_failure(self) -> None:
        """Test to_either method on failure."""
        result = FlextResult[int].fail("error")
        either = result.to_either()

        assert either == (None, "error")

    def test_result_to_exception_method(self) -> None:
        """Test to_exception method."""
        result = FlextResult[int].ok(5)
        exception = result.to_exception()

        assert exception is None

    def test_result_to_exception_method_failure(self) -> None:
        """Test to_exception method on failure."""
        result = FlextResult[int].fail("error")
        exception = result.to_exception()

        assert exception is not None
        assert isinstance(exception, Exception)

    def test_result_value_or_none_property(self) -> None:
        """Test value_or_none property."""
        result = FlextResult[int].ok(5)
        value = result.value_or_none

        assert value == 5

    def test_result_value_or_none_property_failure(self) -> None:
        """Test value_or_none property on failure."""
        result = FlextResult[int].fail("error")
        value = result.value_or_none

        assert value is None

    def test_result_rshift_operator(self) -> None:
        """Test >> operator (flat_map)."""
        result = FlextResult[int].ok(5)
        mapped = result >> (lambda x: FlextResult[int].ok(x * 2))

        assert mapped.is_success
        assert mapped.value == 10

    def test_result_lshift_operator(self) -> None:
        """Test << operator (map)."""
        result = FlextResult[int].ok(5)
        mapped = result << (lambda x: x * 2)

        assert mapped.is_success
        assert mapped.value == 10

    def test_result_matmul_operator(self) -> None:
        """Test @ operator (zip)."""
        result1 = FlextResult[int].ok(5)
        result2 = FlextResult[str].ok("hello")

        zipped = result1 @ result2
        assert zipped.is_success
        assert zipped.value == (5, "hello")

    def test_result_truediv_operator(self) -> None:
        """Test / operator (or_else)."""
        result1 = FlextResult[int].fail("error1")
        result2 = FlextResult[int].ok(10)

        or_result = result1 / result2
        assert or_result.is_success
        assert or_result.value == 10

    def test_result_mod_operator(self) -> None:
        """Test % operator (when)."""
        result = FlextResult[int].ok(5)
        filtered = result % (lambda x: x > 3)

        assert filtered.is_success
        assert filtered.value == 5

    def test_result_and_operator(self) -> None:
        """Test & operator (combine)."""
        result1 = FlextResult[int].ok(5)
        result2 = FlextResult[str].ok("hello")

        combined = result1 & result2
        assert combined.is_success
        assert combined.value == (5, "hello")

    def test_result_xor_operator(self) -> None:
        """Test ^ operator (recover)."""
        result = FlextResult[int].fail("error")
        recovered = result ^ (lambda _e: 0)

        assert recovered.is_success
        assert recovered.value == 0

    def test_result_all_success_static_method(self) -> None:
        """Test all_success static method."""
        result1 = FlextResult[str].ok("success1")
        result2 = FlextResult[str].ok("success2")
        result3 = FlextResult[str].ok("success3")

        all_success = FlextResult.all_success(result1, result2, result3)
        assert all_success is True

    def test_result_all_success_static_method_failure(self) -> None:
        """Test all_success static method with failure."""
        result1 = FlextResult[str].ok("success1")
        result2 = FlextResult[str].fail("error")
        result3 = FlextResult[str].ok("success3")

        all_success = FlextResult.all_success(result1, result2, result3)
        assert all_success is False

    def test_result_any_success_static_method(self) -> None:
        """Test any_success static method."""
        result1 = FlextResult[str].fail("error1")
        result2 = FlextResult[str].ok("success2")
        result3 = FlextResult[str].fail("error3")

        any_success = FlextResult.any_success(result1, result2, result3)
        assert any_success is True

    def test_result_any_success_static_method_all_fail(self) -> None:
        """Test any_success static method with all failures."""
        result1 = FlextResult[str].fail("error1")
        result2 = FlextResult[str].fail("error2")
        result3 = FlextResult[str].fail("error3")

        any_success = FlextResult.any_success(result1, result2, result3)
        assert any_success is False

    def test_result_all_success_empty_results(self) -> None:
        """Test all_success static method with no results."""
        all_success = FlextResult.all_success()
        assert all_success is True

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
        assert result.error is not None and "Negative number" in result.error

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
        assert result.error is not None and "error1" in result.error

    def test_result_combine_with_failures(self) -> None:
        """Test combine static method with failures."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].fail("error1"),
            FlextResult[str].ok("value2"),
        ]

        result = FlextResult.combine(*results)
        assert result.is_failure
        assert result.error is not None and "error1" in result.error

    def test_result_collect_successes_empty(self) -> None:
        """Test collect_successes with empty results."""
        results: list[FlextResult[str]] = []
        successes = FlextResult.collect_successes(results)
        assert successes == []

    def test_result_collect_failures_empty(self) -> None:
        """Test collect_failures with empty results."""
        results: list[FlextResult[str]] = []
        failures = FlextResult.collect_failures(results)
        assert failures == []

    def test_result_collect_successes_mixed(self) -> None:
        """Test collect_successes with mixed results."""
        results = [
            FlextResult[str].ok("success1"),
            FlextResult[str].fail("error1"),
            FlextResult[str].ok("success2"),
            FlextResult[str].fail("error2"),
        ]

        successes = FlextResult.collect_successes(results)
        assert successes == ["success1", "success2"]

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

    def test_result_chain_results_with_failures(self) -> None:
        """Test chain_results static method with failures."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].fail("Processing failed"),
            FlextResult[str].ok("value2"),
        ]

        result = FlextResult.chain_results(*results)
        assert result.is_failure
        assert result.error is not None and "Processing failed" in result.error

    def test_result_chain_results_all_success(self) -> None:
        """Test chain_results static method with all successes."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].ok("value2"),
            FlextResult[str].ok("value3"),
        ]

        result = FlextResult.chain_results(*results)
        assert result.is_success
        assert result.value == ["value1", "value2", "value3"]

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
        assert result.error is not None and "error1" in result.error
        assert result.error is not None and "error2" in result.error

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
        result = FlextResult.accumulate_errors()
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
        assert result.error is not None and "Unknown error" in result.error

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
        assert result.error is not None and "Processing failed" in result.error

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
        assert result.error is not None and "Error processing 2" in result.error
        assert result.error is not None and "Error processing 4" in result.error

    def test_result_parallel_map_with_none_error(self) -> None:
        """Test parallel_map method with None error."""
        items = [1, 2, 3]

        def process_item(item: int) -> FlextResult[str]:
            if item == 2:
                return FlextResult[str].fail("Unknown error occurred")
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=False)
        assert result.is_failure
        assert result.error is not None and "Unknown error occurred" in result.error

    def test_result_parallel_map_with_fail_fast_none_error(self) -> None:
        """Test parallel_map method with fail_fast=True and None error."""
        items = [1, 2, 3]

        def process_item(item: int) -> FlextResult[str]:
            if item == 2:
                return FlextResult[str].fail("Unknown error occurred")
            return FlextResult[str].ok(f"processed_{item}")

        result = FlextResult.parallel_map(items, process_item, fail_fast=True)
        assert result.is_failure
        assert result.error is not None and "Unknown error occurred" in result.error

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
        assert filtered.error is not None and "Value too small" in filtered.error

        # Failure case (should remain failure)
        result = FlextResult[int].fail("Original error")
        filtered = result.filter(lambda x: x > 5, "Value too small")
        assert filtered.is_failure
        assert filtered.error == "Original error"

    def test_result_zip_with_method(self) -> None:
        """Test zip_with method."""
        # Both success
        result1 = FlextResult[int].ok(5)
        result2 = FlextResult[int].ok(10)
        zipped = result1.zip_with(result2, operator.add)
        assert zipped.is_success
        assert zipped.value == 15

        # First success, second failure
        result1 = FlextResult[int].ok(5)
        result2 = FlextResult[int].fail("Second failed")
        zipped = result1.zip_with(result2, operator.add)
        assert zipped.is_failure
        assert zipped.error is not None and "Second failed" in zipped.error

        # First failure, second success
        result1 = FlextResult[int].fail("First failed")
        result2 = FlextResult[int].ok(10)
        zipped = result1.zip_with(result2, operator.add)
        assert zipped.is_failure
        assert zipped.error is not None and "First failed" in zipped.error

        # Both failure
        result1 = FlextResult[int].fail("First failed")
        result2 = FlextResult[int].fail("Second failed")
        zipped = result1.zip_with(result2, operator.add)
        assert zipped.is_failure

    def test_result_recover_with_method(self) -> None:
        """Test recover_with method."""
        # Success case (should not call recovery function)
        result = FlextResult[int].ok(42)
        recovered = result.recover_with(lambda _: FlextResult[int].ok(0))
        assert recovered.is_success
        assert recovered.value == 42

        # Failure case with successful recovery
        result = FlextResult[int].fail("Error occurred")
        recovered = result.recover_with(lambda _: FlextResult[int].ok(0))
        assert recovered.is_success
        assert recovered.value == 0

        # Failure case with failed recovery
        result = FlextResult[int].fail("Error occurred")
        recovered = result.recover_with(
            lambda _: FlextResult[int].fail("Recovery failed")
        )
        assert recovered.is_failure
        assert recovered.error is not None and "Recovery failed" in recovered.error

    def test_result_from_exception_static_method(self) -> None:
        """Test from_exception static method."""

        # Function that succeeds
        def success_func() -> int:
            return 42

        result = FlextResult.from_exception(success_func)
        assert result.is_success
        assert result.value == 42

        # Function that raises exception
        def failing_func() -> int:
            error_message = "Something went wrong"
            raise ValueError(error_message)

        result = FlextResult.from_exception(failing_func)
        assert result.is_failure
        assert result.error is not None and "Something went wrong" in result.error

    def test_result_first_success_static_method(self) -> None:
        """Test first_success static method."""
        # Mix of success and failure
        results = [
            FlextResult[int].fail("First error"),
            FlextResult[int].fail("Second error"),
            FlextResult[int].ok(42),
            FlextResult[int].ok(100),
        ]

        first = FlextResult.first_success(results)
        assert first.is_success
        assert first.value == 42

        # All failures
        results = [
            FlextResult[int].fail("First error"),
            FlextResult[int].fail("Second error"),
            FlextResult[int].fail("Third error"),
        ]

        first = FlextResult.first_success(results)
        assert first.is_failure
        assert (
            first.error is not None and "Third error" in first.error
        )  # Should be last error

        # Empty list
        first = FlextResult.first_success([])
        assert first.is_failure

    def test_result_try_all_static_method(self) -> None:
        """Test try_all static method."""
        call_count = 0

        def failing_func() -> FlextResult[int]:
            nonlocal call_count
            call_count += 1
            return FlextResult[int].fail(f"Attempt {call_count} failed")

        def success_func() -> FlextResult[int]:
            nonlocal call_count
            call_count += 1
            return FlextResult[int].ok(call_count)

        # Mix of failing and succeeding functions
        funcs = [failing_func, failing_func, success_func, failing_func]
        result = FlextResult.try_all(funcs)
        assert result.is_success
        assert result.value == 3  # Should succeed on third attempt

        # All failing functions
        call_count = 0
        funcs = [failing_func, failing_func, failing_func]
        result = FlextResult.try_all(funcs)
        assert result.is_failure

        # Empty list
        result = FlextResult.try_all([])
        assert result.is_failure

    def test_result_safe_unwrap_or_none_static_method(self) -> None:
        """Test safe_unwrap_or_none static method."""
        # Success result
        result = FlextResult[int].ok(42)
        value = FlextResult.safe_unwrap_or_none(result)
        assert value == 42

        # Failure result
        result = FlextResult[int].fail("Error")
        value = FlextResult.safe_unwrap_or_none(result)
        assert value is None

    def test_result_unwrap_or_raise_static_method(self) -> None:
        """Test unwrap_or_raise static method."""
        # Success result
        result = FlextResult[int].ok(42)
        value = FlextResult.unwrap_or_raise(result)
        assert value == 42

        # Failure result with default exception
        result = FlextResult[int].fail("Error occurred")
        with pytest.raises(RuntimeError) as exc_info:
            FlextResult.unwrap_or_raise(result)
        assert str(exc_info.value) == "Error occurred"

        # Failure result with custom exception
        result = FlextResult[int].fail("Error occurred")
        with pytest.raises(RuntimeError) as exc_info:
            FlextResult.unwrap_or_raise(result, RuntimeError)
        assert str(exc_info.value) == "Error occurred"

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

        result = FlextResult.safe_call(success_func)
        assert result.is_success
        assert result.value == 42

        # Function that raises exception
        def failing_func() -> int:
            error_message = "Something went wrong"
            raise ValueError(error_message)

        result = FlextResult.safe_call(failing_func)
        assert result.is_failure
        assert result.error is not None and "Something went wrong" in result.error

    def test_result_cast_fail_method(self) -> None:
        """Test cast_fail method."""
        # Success result should raise ValueError
        result = FlextResult[int].ok(42)
        with pytest.raises(ValueError, match="Cannot cast successful result to failed"):
            result.cast_fail()

        # Failure result should remain failure
        result = FlextResult[int].fail("Original error")
        cast_result = result.cast_fail()
        assert cast_result.is_failure
        assert cast_result.error == "Original error"

    def test_result_chain_validations_static_method(self) -> None:
        """Test chain_validations static method."""

        # Create validation functions that don't depend on input
        def validate_positive() -> FlextResult[None]:
            # For this test, we'll simulate validation that always passes
            return FlextResult[None].ok(None)

        def validate_even() -> FlextResult[None]:
            # For this test, we'll simulate validation that always passes for the first case
            # and fails for the second case - this is just for testing the chaining logic
            return FlextResult[None].ok(None)

        validators = [validate_positive, validate_even]

        # Valid value (positive and even)
        result = FlextResult.chain_validations(*validators)
        assert result.is_success

        # Invalid value (positive but odd)
        def validate_even_fail() -> FlextResult[None]:
            return FlextResult[None].fail("Must be even")

        result = FlextResult.chain_validations(validate_positive, validate_even_fail)
        assert result.is_failure
        assert result.error is not None and "Must be even" in result.error

        # Invalid value (negative) - first validator should fail
        def validate_positive_fail() -> FlextResult[None]:
            return FlextResult[None].fail("Must be positive")

        result = FlextResult.chain_validations(validate_positive_fail, validate_even)
        assert result.is_failure
        assert result.error is not None and "Must be positive" in result.error

    def test_result_validate_and_execute_static_method(self) -> None:
        """Test validate_and_execute static method."""

        def validator(x: int) -> FlextResult[None]:
            if x > 0:
                return FlextResult[None].ok(None)
            return FlextResult[None].fail("Must be positive")

        def executor(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"Processed: {x}")

        # Valid value
        result = FlextResult.ok(5).validate_and_execute(validator, executor)
        assert result.is_success
        assert result.value == "Processed: 5"

        # Invalid value
        result = FlextResult.ok(-5).validate_and_execute(validator, executor)
        assert result.is_failure
        assert result.error is not None and "Must be positive" in result.error

    def test_result_map_sequence_static_method(self) -> None:
        """Test map_sequence static method."""

        def double(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        # All successful mapping
        items = [1, 2, 3, 4, 5]
        result = FlextResult.map_sequence(items, double)
        assert result.is_success
        assert result.value == [2, 4, 6, 8, 10]

        # Mapping with failure
        def conditional_double(x: int) -> FlextResult[int]:
            if x == 3:
                return FlextResult[int].fail("Cannot process 3")
            return FlextResult[int].ok(x * 2)

        result = FlextResult.map_sequence(items, conditional_double)
        assert result.is_failure
        assert result.error is not None and "Cannot process 3" in result.error

    def test_result_pipeline_static_method(self) -> None:
        """Test pipeline static method."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def double(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        def to_int(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(int(x))

        operations: list[Callable[[int], FlextResult[int]]] = [add_one, double, to_int]

        # Successful pipeline
        result = FlextResult.pipeline(5, *operations)
        assert result.is_success
        assert result.value == 12  # (5 + 1) * 2 = 12

        # Pipeline with failure
        def failing_op(x: int) -> FlextResult[int]:
            # Parameter is intentionally unused in this test
            _ = x  # Mark as intentionally unused
            return FlextResult[int].fail("Pipeline failed")

        operations_with_failure = [add_one, failing_op, double]
        result = FlextResult.pipeline(5, *operations_with_failure)
        assert result.is_failure
        assert result.error is not None and "Pipeline failed" in result.error

    def test_result_or_try_method(self) -> None:
        """Test or_try method."""
        # Success result should not try alternatives
        result = FlextResult[int].ok(42)

        def alternative() -> FlextResult[int]:
            return FlextResult[int].ok(100)

        or_result = result.or_try(alternative)
        assert or_result.is_success
        assert or_result.value == 42

        # Failure result should try alternatives
        result = FlextResult[int].fail("Original error")

        def successful_alternative() -> FlextResult[int]:
            return FlextResult[int].ok(100)

        or_result = result.or_try(successful_alternative)
        assert or_result.is_success
        assert or_result.value == 100

        # All alternatives fail
        def failing_alternative() -> FlextResult[int]:
            return FlextResult[int].fail("Alternative failed")

        or_result = result.or_try(failing_alternative)
        assert or_result.is_failure

    def test_result_rescue_with_logging_method(self) -> None:
        """Test rescue_with_logging method."""
        logged_errors = []

        def logger(error: str) -> None:
            logged_errors.append(error)

        # Success result should not log
        result = FlextResult[int].ok(42)
        rescued = result.rescue_with_logging(logger)
        assert rescued.is_success
        assert rescued.value == 42
        assert len(logged_errors) == 0

        # Failure result should log error
        result = FlextResult[int].fail("Test error")
        rescued = result.rescue_with_logging(logger)
        assert rescued.is_failure
        assert len(logged_errors) == 1
        assert "Test error" in logged_errors[0]

    def test_result_kleisli_compose_static_method(self) -> None:
        """Test kleisli_compose static method."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def double(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        composed = FlextResult.ok(5).kleisli_compose(add_one, double)
        result = composed(5)
        assert result.is_success
        assert result.value == 12  # (5 + 1) * 2

    def test_result_applicative_lift2_static_method(self) -> None:
        """Test applicative_lift2 static method."""
        result1 = FlextResult[int].ok(5)
        result2 = FlextResult[int].ok(10)

        lifted = FlextResult.applicative_lift2(operator.add, result1, result2)
        assert lifted.is_success
        assert lifted.value == 15

        # With failure
        result2_fail = FlextResult[int].fail("Error")
        lifted = FlextResult.applicative_lift2(operator.add, result1, result2_fail)
        assert lifted.is_failure

    def test_result_applicative_lift3_static_method(self) -> None:
        """Test applicative_lift3 static method."""

        def add_three(x: int, y: int, z: int) -> int:
            return x + y + z

        result1 = FlextResult[int].ok(5)
        result2 = FlextResult[int].ok(10)
        result3 = FlextResult[int].ok(15)

        lifted = FlextResult.applicative_lift3(add_three, result1, result2, result3)
        assert lifted.is_success
        assert lifted.value == 30

    def test_result_if_then_else_static_method(self) -> None:
        """Test if_then_else static method."""

        def condition(x: int) -> bool:
            return x > 5

        def then_func(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"Large: {x}")

        def else_func(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"Small: {x}")

        # Condition is true
        result = FlextResult.ok(10).if_then_else(condition, then_func, else_func)
        assert result.is_success
        assert result.value == "Large: 10"

        # Condition is false
        result = FlextResult.ok(3).if_then_else(condition, then_func, else_func)
        assert result.is_success
        assert result.value == "Small: 3"

    def test_result_collect_all_errors_static_method(self) -> None:
        """Test collect_all_errors static method."""
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("Error 1"),
            FlextResult[int].ok(2),
            FlextResult[int].fail("Error 2"),
            FlextResult[int].ok(3),
        ]

        collected = FlextResult.collect_all_errors(*results)
        successes, errors = collected
        assert successes == [1, 2, 3]
        assert len(errors) == 2
        assert "Error 1" in errors[0]
        assert "Error 2" in errors[1]

    def test_result_concurrent_sequence_static_method(self) -> None:
        """Test concurrent_sequence static method."""
        # All success
        results = [FlextResult[int].ok(i) for i in range(5)]
        sequenced = FlextResult.concurrent_sequence(results, fail_fast=True)
        assert sequenced.is_success
        assert sequenced.value == [0, 1, 2, 3, 4]

        # With failure and fail_fast=True
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("Error"),
            FlextResult[int].ok(3),
        ]
        sequenced = FlextResult.concurrent_sequence(results, fail_fast=True)
        assert sequenced.is_failure

    def test_result_with_resource_static_method(self) -> None:
        """Test with_resource static method."""
        resources_created = []
        resources_cleaned = []

        def create_resource() -> str:
            resource = "test_resource"
            resources_created.append(resource)
            return resource

        def operation(value: str, resource: str) -> FlextResult[int]:
            return FlextResult[int].ok(len(resource) + len(value))

        def cleanup(resource: str) -> None:
            resources_cleaned.append(resource)

        result = FlextResult.ok("dummy").with_resource(
            create_resource, operation, cleanup
        )
        assert result.is_success
        assert result.value == 18  # len("test_resource") + len("dummy")
        assert len(resources_created) == 1
        assert len(resources_cleaned) == 1
        assert resources_created[0] == resources_cleaned[0]

    def test_result_bracket_static_method(self) -> None:
        """Test bracket static method."""
        finally_called = []

        def operation(value: str) -> FlextResult[int]:
            _ = value
            return FlextResult[int].ok(42)

        def finally_action(value: str) -> None:
            _ = value
            finally_called.append(True)

        result = FlextResult.ok("dummy").bracket(operation, finally_action)
        assert result.is_success
        assert result.value == 42
        assert len(finally_called) == 1

        # Test with failing operation
        finally_called.clear()

        def failing_operation(value: str) -> FlextResult[int]:
            _ = value
            return FlextResult[int].fail("Operation failed")

        result = FlextResult.ok("dummy").bracket(failing_operation, finally_action)
        assert result.is_failure
        assert len(finally_called) == 1  # Finally should still be called

    def test_result_with_timeout_static_method(self) -> None:
        """Test with_timeout static method."""

        def quick_operation(value: str) -> FlextResult[str]:
            _ = value
            return FlextResult[str].ok("Quick result")

        result = FlextResult.ok("dummy").with_timeout(1.0, quick_operation)
        assert result.is_success
        assert result.value == "Quick result"

        def slow_operation(value: str) -> FlextResult[str]:
            time.sleep(0.1)  # Short delay for testing
            return FlextResult[str].ok("Slow result in " + value)

        # This should still succeed as 0.1s is less than timeout
        result = FlextResult.ok("dummy").with_timeout(0.2, slow_operation)
        assert result.is_success

    def test_result_retry_until_success_static_method(self) -> None:
        """Test retry_until_success static method."""
        attempt_count = 0

        def unstable_operation(value: int) -> FlextResult[int]:
            nonlocal attempt_count
            _ = value
            attempt_count += 1
            if attempt_count < 3:
                return FlextResult[int].fail(f"Attempt {attempt_count} failed")
            return FlextResult[int].ok(attempt_count)

        result = FlextResult.ok(5).retry_until_success(
            unstable_operation, max_attempts=5, backoff_factor=1.1
        )
        assert result.is_success
        assert result.value == 3
        assert attempt_count == 3

        # Test max attempts exceeded
        attempt_count = 0

        def always_failing(value: int) -> FlextResult[int]:
            nonlocal attempt_count
            attempt_count += value
            return FlextResult[int].fail("Always fails")

        result = FlextResult.ok(5).retry_until_success(
            always_failing, max_attempts=2, backoff_factor=1.1
        )
        assert result.is_failure
        assert attempt_count == 10  # 5 + 5 from two attempts

    def test_result_transition_method(self) -> None:
        """Test transition method."""

        def state_machine(current_state: int) -> FlextResult[int]:
            if current_state < 10:
                return FlextResult[int].ok(current_state + 1)
            return FlextResult[int].fail("Max state reached")

        result = FlextResult.ok(5)
        transitioned = result.transition(state_machine)
        assert transitioned.is_success
        assert transitioned.value == 6

        result = FlextResult.ok(15)
        transitioned = result.transition(state_machine)
        assert transitioned.is_failure
        assert (
            transitioned.error is not None and "Max state reached" in transitioned.error
        )
