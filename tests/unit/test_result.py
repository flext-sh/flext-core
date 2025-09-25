"""Comprehensive tests for FlextResult - Railway Pattern Implementation.

This module tests the core FlextResult railway pattern which is the foundation
of error handling across the entire FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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
        assert "condition" in when_result.error.lower()

    def test_result_tap_error(self) -> None:
        """Test tap_error operation."""
        result = FlextResult[int].fail("original_error")
        tapped_error = result.tap_error(lambda e: f"tapped_{e}")

        assert tapped_error.is_failure
        assert tapped_error.error == "original_error"

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
        assert "Chain failed at result" in chained.error

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
        assert "error" in combined.error

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
        assert "condition" in unless_result.error.lower()

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
        assert "Context: error" in context_result.error

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
            FlextResult[str].ok("success2")
        ]
        
        result = FlextResult.sequence(results)
        assert result.is_failure
        assert "error1" in result.error

    def test_result_combine_with_failures(self) -> None:
        """Test combine static method with failures."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].fail("error1"),
            FlextResult[str].ok("value2")
        ]
        
        result = FlextResult.combine(*results)
        assert result.is_failure
        assert "error1" in result.error

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
            FlextResult[str].fail("error2")
        ]
        
        successes = FlextResult.collect_successes(results)
        assert successes == ["success1", "success2"]

    def test_result_collect_failures_mixed(self) -> None:
        """Test collect_failures with mixed results."""
        results = [
            FlextResult[str].ok("success1"),
            FlextResult[str].fail("error1"),
            FlextResult[str].ok("success2"),
            FlextResult[str].fail("error2")
        ]
        
        failures = FlextResult.collect_failures(results)
        assert failures == ["error1", "error2"]

    def test_result_chain_results_with_failures(self) -> None:
        """Test chain_results static method with failures."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].fail("Processing failed"),
            FlextResult[str].ok("value2")
        ]
        
        result = FlextResult.chain_results(*results)
        assert result.is_failure
        assert "Processing failed" in result.error

    def test_result_chain_results_all_success(self) -> None:
        """Test chain_results static method with all successes."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].ok("value2"),
            FlextResult[str].ok("value3")
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
            FlextResult[str].fail("error2")
        ]
        
        result = FlextResult.accumulate_errors(*results)
        assert result.is_failure
        assert "error1" in result.error
        assert "error2" in result.error

    def test_result_accumulate_all_success(self) -> None:
        """Test accumulate method with all successes."""
        results = [
            FlextResult[str].ok("value1"),
            FlextResult[str].ok("value2"),
            FlextResult[str].ok("value3")
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
            FlextResult[str].fail(None),  # None error
            FlextResult[str].ok("value2")
        ]
        
        result = FlextResult.accumulate_errors(*results)
        assert result.is_failure
        assert "Unknown error" in result.error
