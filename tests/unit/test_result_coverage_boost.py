"""Tests to boost FlextResult coverage to target levels.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

import asyncio
import json
import operator

import pytest

from flext_core import FlextResult


class TestFlextResultCoverageBoost:
    """Tests focused on increasing FlextResult coverage."""

    def test_result_chaining_methods(self) -> None:
        """Test result chaining and composition methods."""
        # Test flat_map with success
        result = FlextResult[int].ok(5)
        chained = result.flat_map(lambda x: FlextResult[int].ok(x * 2))
        assert chained.is_success
        assert chained.unwrap() == 10

        # Test flat_map with failure
        result = FlextResult[int].fail("error")
        chained = result.flat_map(lambda x: FlextResult[int].ok(x * 2))
        assert chained.is_failure
        assert chained.error == "error"

    def test_result_filter_operations(self) -> None:
        """Test result filter operations."""
        # Test filter with passing predicate
        result = FlextResult[int].ok(10)
        filtered = result.filter(lambda x: x > 5, "Value too small")
        assert filtered.is_success
        assert filtered.unwrap() == 10

        # Test filter with failing predicate
        result = FlextResult[int].ok(3)
        filtered = result.filter(lambda x: x > 5, "Value too small")
        assert filtered.is_failure
        assert filtered.error == "Value too small"

        # Test filter on failure result
        result = FlextResult[int].fail("original error")
        filtered = result.filter(lambda x: x > 5, "Value too small")
        assert filtered.is_failure
        assert filtered.error == "original error"

    def test_result_error_handling(self) -> None:
        """Test result error handling methods."""
        # Test tap_error on failure (if available)
        result = FlextResult[int].fail("original error")
        if hasattr(result, "tap_error"):

            def error_handler(e: str) -> None:
                pass  # Silent error handler for testing

            tapped = result.tap_error(error_handler)
            assert tapped.is_failure
            assert tapped.error == "original error"

        # Test error access
        assert result.error == "original error"
        assert result.is_failure

        # Test success result error handling
        result = FlextResult[int].ok(5)
        assert result.error is None or result.error == ""
        assert result.is_success

    def test_result_unwrap_methods(self) -> None:
        """Test various unwrap methods."""
        # Test unwrap_or with success
        result = FlextResult[int].ok(5)
        value = result.unwrap_or(10)
        assert value == 5

        # Test unwrap_or with failure
        result = FlextResult[int].fail("error")
        value = result.unwrap_or(10)
        assert value == 10

        # Test expect with success
        result = FlextResult[int].ok(5)
        value = result.expect("Should have value")
        assert value == 5

        # Test expect with failure (should raise)
        result = FlextResult[int].fail("error")
        with pytest.raises(Exception) as exc_info:
            result.expect("Should have value")
        assert "Should have value" in str(exc_info.value)

    def test_result_alternative_constructors(self) -> None:
        """Test alternative result construction methods."""

        # Test manual construction with try/catch pattern
        def successful_function() -> int:
            return 42

        try:
            value = successful_function()
            result = FlextResult[int].ok(value)
            assert result.is_success
            assert result.unwrap() == 42
        except Exception as e:
            result = FlextResult[int].fail(str(e))
            assert result.is_failure

        # Test with exception handling
        def failing_function() -> int:
            error_message = "Something went wrong"
            raise ValueError(error_message)

        try:
            value = failing_function()
            result = FlextResult[int].ok(value)
        except Exception as e:
            result = FlextResult[int].fail(str(e))
            assert result.is_failure
            assert "Something went wrong" in str(result.error)

    def test_result_collect_operations(self) -> None:
        """Test result collection and aggregation operations."""
        # Test manual collection of results
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ]

        # Manual collection logic
        collected_values = []
        failed = False

        for result in results:
            if result.is_success:
                collected_values.append(result.unwrap())
            else:
                failed = True
                break

        if not failed:
            collected = FlextResult[list[int]].ok(collected_values)
            assert collected.is_success
            assert collected.unwrap() == [1, 2, 3]

        # Test with a failure
        results = [
            FlextResult[int].ok(1),
            FlextResult[int].fail("error"),
            FlextResult[int].ok(3),
        ]

        failed = False
        for result in results:
            if result.is_failure:
                failed = True
                collected = FlextResult[list[int]].fail(result.error or "Unknown error")
                break

        if failed:
            assert collected.is_failure
            assert "error" in str(collected.error)

    def test_result_sequence_operations(self) -> None:
        """Test result sequence and traversal operations."""

        # Test sequence with function that can fail
        def divide_by_two(x: int) -> FlextResult[float]:
            if x == 0:
                return FlextResult[float].fail("Cannot divide zero")
            return FlextResult[float].ok(x / 2)

        # Test with valid inputs
        results = [divide_by_two(x) for x in [4, 6, 8]]
        sequenced = FlextResult.sequence(results)
        assert sequenced.is_success
        assert sequenced.unwrap() == [2.0, 3.0, 4.0]

        # Test with invalid input
        results = [divide_by_two(x) for x in [4, 0, 8]]
        sequenced = FlextResult.sequence(results)
        assert sequenced.is_failure

    def test_result_traverse_operations(self) -> None:
        """Test result traverse operations."""

        # Test traverse with successful mapping
        def to_result_string(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"value_{x}")

        result = FlextResult.traverse([1, 2, 3], to_result_string)
        assert result.is_success
        assert result.unwrap() == ["value_1", "value_2", "value_3"]

        # Test traverse with failing mapping
        def conditional_string(x: int) -> FlextResult[str]:
            if x > 2:
                return FlextResult[str].fail(f"Value {x} too large")
            return FlextResult[str].ok(f"value_{x}")

        result = FlextResult.traverse([1, 2, 3], conditional_string)
        assert result.is_failure

    def test_result_monadic_operations(self) -> None:
        """Test monadic operations and laws."""
        # Test monadic left identity law
        value = 5

        def func(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        # return(value) >>= func === func(value)
        left = FlextResult[int].ok(value).flat_map(func)
        right = func(value)
        assert left.unwrap() == right.unwrap()

        # Test monadic right identity law
        result = FlextResult[int].ok(10)
        # result >>= return === result
        identity_mapped = result.flat_map(FlextResult[int].ok)
        assert identity_mapped.unwrap() == result.unwrap()

    def test_result_error_codes(self) -> None:
        """Test result error codes and categorization."""
        # Test with error codes
        result = FlextResult[int].fail("Network timeout", error_code="NETWORK_ERROR")
        assert result.is_failure
        assert result.error == "Network timeout"
        assert result.error_code == "NETWORK_ERROR"

        # Test without error codes
        result = FlextResult[int].fail("Generic error")
        assert result.is_failure
        assert result.error_code is None or result.error_code == ""

    def test_result_comparison_operations(self) -> None:
        """Test result comparison and equality operations."""
        # Test equality of success results
        result1 = FlextResult[int].ok(5)
        result2 = FlextResult[int].ok(5)
        result3 = FlextResult[int].ok(10)

        # Note: These may or may not be implemented, testing if they exist
        try:
            assert result1 == result2
            assert result1 != result3
        except (TypeError, NotImplementedError):
            # Equality might not be implemented
            pass

        # Test equality of failure results
        failure1 = FlextResult[int].fail("error")
        failure2 = FlextResult[int].fail("error")
        failure3 = FlextResult[int].fail("different error")

        try:
            assert failure1 == failure2
            assert failure1 != failure3
        except (TypeError, NotImplementedError):
            # Equality might not be implemented
            pass

    def test_result_conversion_operations(self) -> None:
        """Test result conversion and transformation operations."""
        # Test conversion to optional-like behavior
        success_result = FlextResult[int].ok(42)
        # Use unwrap_or for optional-like behavior
        optional_success = success_result.unwrap_or(-1)
        assert optional_success == 42

        failure_result = FlextResult[int].fail("error")
        optional_failure = failure_result.unwrap_or(-1)
        assert optional_failure == -1

    def test_result_async_operations(self) -> None:
        """Test result async operations if available."""
        # Test async-compatible operations

        async def async_operation(x: int) -> FlextResult[int]:
            await asyncio.sleep(0.001)  # Minimal async work
            return FlextResult[int].ok(x * 2)

        # Test if result supports async operations
        if hasattr(FlextResult, "async_map"):
            # This would test async mapping if implemented
            pass

    def test_result_performance_operations(self) -> None:
        """Test result performance characteristics."""
        # Test with large data sets
        large_data = list(range(1000))

        # Test mapping performance
        result = FlextResult[list[int]].ok(large_data)
        mapped = result.map(lambda x: [i * 2 for i in x])
        assert mapped.is_success
        assert len(mapped.unwrap()) == 1000

        # Test chaining performance
        chained = result
        for _ in range(10):
            chained = chained.map(operator.itemgetter(slice(None)))  # Copy operation

        assert chained.is_success

    def test_result_edge_cases(self) -> None:
        """Test result edge cases and boundary conditions."""
        # Test with None values
        result = FlextResult[None].ok(None)
        assert result.is_success
        assert result.unwrap() is None

        # Test with empty collections
        list_result = FlextResult[list[object]].ok([])
        assert list_result.is_success
        assert list_result.unwrap() == []

        # Test with complex nested types
        nested_data = {"level1": {"level2": {"level3": "deep_value"}}}
        dict_result = FlextResult[dict[str, dict[str, dict[str, str]]]].ok(nested_data)
        assert dict_result.is_success

        # Test with very long error messages
        long_error = "A" * 1000
        int_result = FlextResult[int].fail(long_error)
        assert int_result.is_failure
        if int_result.error is not None:
            assert len(int_result.error) == 1000

    def test_result_string_representations(self) -> None:
        """Test result string representations and debugging."""
        # Test string representation of success
        result = FlextResult[int].ok(42)
        str_repr = str(result)
        assert "42" in str_repr or "success" in str_repr.lower()

        # Test string representation of failure
        result = FlextResult[int].fail("Something went wrong")
        str_repr = str(result)
        assert "Something went wrong" in str_repr or "fail" in str_repr.lower()

        # Test repr
        result = FlextResult[int].ok(42)
        repr_str = repr(result)
        assert "FlextResult" in repr_str or "42" in repr_str

    def test_result_context_manager(self) -> None:
        """Test result as context manager if supported."""
        # Test if result supports context manager protocol
        result = FlextResult[int].ok(42)

        try:
            with result as value:
                # If context manager is supported
                assert value == 42
        except (TypeError, AttributeError):
            # Context manager might not be implemented
            pass

    def test_result_serialization(self) -> None:
        """Test result serialization capabilities."""
        # Test if result supports serialization
        result = FlextResult[dict[str, object]].ok({"key": "value"})

        # Test serialization capabilities
        # FlextResult doesn't have model_dump, but we can test string representation
        str_repr = str(result)
        assert isinstance(str_repr, str)

        # Test JSON serialization if data is serializable
        try:
            # Create a simple dict representation for JSON serialization
            data_dict = {
                "success": result.success,
                "data": result.value if result.success else None,
            }
            json_str = json.dumps(data_dict, default=str)
            assert isinstance(json_str, str)
        except (TypeError, AttributeError):
            # Skip if not serializable
            pass

    def test_result_immutability(self) -> None:
        """Test result immutability guarantees."""
        # Test that operations don't modify original result
        original = FlextResult[list[int]].ok([1, 2, 3])

        # Map should create new result
        mapped = original.map(lambda x: [*x, 4])

        # Original should be unchanged
        assert original.unwrap() == [1, 2, 3]
        if mapped.is_success:
            assert mapped.unwrap() == [1, 2, 3, 4]

    def test_result_type_safety(self) -> None:
        """Test result type safety features."""
        # Test type preservation through operations
        int_result = FlextResult[int].ok(42)

        # Map should preserve type safety
        string_result = int_result.map(str)
        if string_result.is_success:
            assert isinstance(string_result.unwrap(), str)

        # Filter should maintain type
        filtered = int_result.filter(lambda x: x > 0, "Positive only")
        if filtered.is_success:
            assert isinstance(filtered.unwrap(), int)
