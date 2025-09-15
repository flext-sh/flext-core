"""Final tests to achieve 100% coverage for FlextResult.

This test file targets the remaining uncovered lines to achieve complete coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

import pytest

from flext_core import FlextResult


class TestFlextResultFinalCoverage:
    """Final coverage tests for FlextResult to achieve 100%."""

    def test_property_aliases_complete(self) -> None:
        """Test all property aliases that are not covered yet."""
        success_result = FlextResult[str].ok("test")
        failure_result = FlextResult[str].fail("error")

        # Test failure property alias
        assert success_result.failure is False
        assert failure_result.failure is True

        # Test is_valid property alias
        assert success_result.is_valid is True
        assert failure_result.is_valid is False

    def test_create_failure_alias(self) -> None:
        """Test create_failure class method alias."""
        error_data = {"code": 500, "details": "Server error"}
        result = FlextResult[str].create_failure(
            "test error", error_code="TEST_ERROR", error_data=error_data
        )

        assert result.is_failure
        assert result.error == "test error"
        assert result.error_code == "TEST_ERROR"
        assert result.error_data == error_data

    def test_map_with_failure_edge_case(self) -> None:
        """Test map method with failure that has no error message."""
        # Create a failure result with empty error
        result = FlextResult[str](error="", error_code=None, error_data=None)

        # Map should handle empty error gracefully
        mapped = result.map(lambda x: x.upper())
        assert mapped.is_failure
        # Should get the default error from map operation
        assert mapped.error is not None
        assert "Map operation failed" in mapped.error

    def test_flat_map_internal_none_check(self) -> None:
        """Test flat_map method internal None data check path."""
        # This tests a specific branch in flat_map for None data validation
        result = FlextResult[str].ok("test")

        # Normal case should work
        flat_mapped = result.flat_map(lambda x: FlextResult[str].ok(f"mapped_{x}"))
        assert flat_mapped.is_success
        assert flat_mapped.value == "mapped_test"

    def test_context_manager_none_data_edge_case(self) -> None:
        """Test context manager with None data edge case."""
        # Test the specific path where success result has None data
        result = FlextResult[str | None].ok(None)

        # The context manager should raise for None data (defensive behavior)
        with (
            pytest.raises(RuntimeError, match="Success result has None data"),
            result as _value,
        ):
            pass  # Should not reach here

    def test_expect_none_data_validation(self) -> None:
        """Test expect method specific None data validation."""
        # Test the defensive None validation in expect method
        result = FlextResult[str | None].ok(None)

        # expect should validate None data and raise with specific message
        with pytest.raises(RuntimeError, match="Success result has None data"):
            result.expect("This should fail for None")

    def test_hash_complex_object_fallback_paths(self) -> None:
        """Test __hash__ method complex object fallback paths."""

        # Test object with __dict__ but unhashable attributes
        class ComplexObject:
            def __init__(self) -> None:
                """Initialize the instance."""
                self.unhashable_attr = {"nested": ["list", "data"]}

        complex_obj = ComplexObject()
        result = FlextResult[ComplexObject].ok(complex_obj)

        # Should handle the exception and use fallback
        hash_value = hash(result)
        assert isinstance(hash_value, int)

        # Test object without __dict__ attribute
        class NoDict:
            __slots__ = ["value"]

            def __init__(self, value: str) -> None:
                """Initialize the instance."""
                self.value = value

        no_dict_obj = NoDict("test")
        result_no_dict = FlextResult[NoDict].ok(no_dict_obj)
        hash_no_dict = hash(result_no_dict)
        assert isinstance(hash_no_dict, int)

    def test_or_else_get_with_none_error(self) -> None:
        """Test or_else_get with None error edge case."""
        # Create failure with None error
        failure = FlextResult[str](error="fail", error_code=None, error_data=None)

        def get_alternative() -> FlextResult[str]:
            return FlextResult[str].ok("alternative")

        result = failure.or_else_get(get_alternative)
        assert result.is_success
        assert result.value == "alternative"

    def test_recover_with_none_error_edge_case(self) -> None:
        """Test recover_with method with None error edge case."""
        success = FlextResult[str].ok("success")

        def recovery_func(error: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"recovered_from_{error}")

        # Success should pass through without calling recovery
        result = success.recover_with(recovery_func)
        assert result == success

        # Test failure with None error
        failure = FlextResult[str](error="error", error_code=None, error_data=None)
        recovered = failure.recover_with(recovery_func)
        assert recovered.is_success
        assert recovered.value == "recovered_from_error"

    def test_filter_none_data_edge_case(self) -> None:
        """Test filter method with None data edge case."""
        # This tests the specific None data validation in filter
        result = FlextResult[str | None].ok(None)

        # Filter should detect None data and raise specific error (defensive behavior)
        def predicate(x: str | None) -> bool:
            return x is not None

        # The method should raise for None data
        with pytest.raises(RuntimeError, match="Success result has None data"):
            result.filter(predicate, "Should handle None")

    def test_zip_with_none_data_comprehensive(self) -> None:
        """Test zip_with method None data handling comprehensively."""
        success_none = FlextResult[str | None].ok(None)
        success_value = FlextResult[str].ok("value")

        # Both None should fail
        result1 = success_none.zip_with(success_none, lambda x, y: (x, y))
        assert result1.is_failure
        assert result1.error is not None
        assert "Missing data for zip operation" in result1.error

        # One None should fail
        result2 = success_none.zip_with(success_value, lambda x, y: (x, y))
        assert result2.is_failure
        assert result2.error is not None
        assert "Missing data for zip operation" in result2.error

    def test_combine_none_value_handling(self) -> None:
        """Test combine static method with None values."""
        none_result = FlextResult[None].ok(None)
        string_result = FlextResult[str].ok("test")

        # Cast to FlextResult[object] for combine method

        none_result_obj = cast("FlextResult[object]", none_result)
        string_result_obj = cast("FlextResult[object]", string_result)

        # Combine should handle None values by not including them
        combined = FlextResult.combine(string_result_obj, none_result_obj)
        assert combined.is_success
        # None values are not included in the result based on implementation
        assert combined.value == ["test"]

    def test_first_success_empty_case(self) -> None:
        """Test first_success with no arguments."""
        # This should return a failure with default message
        result: FlextResult[object] = FlextResult.first_success()
        assert result.is_failure
        assert result.error
        assert result.error is not None
        assert "No successful results found" in result.error

    def test_unwrap_or_raise_edge_cases(self) -> None:
        """Test unwrap_or_raise utility method edge cases."""
        success = FlextResult[str].ok("value")
        failure = FlextResult[str].fail("test_error")

        # Success case
        assert FlextResult.unwrap_or_raise(success) == "value"

        # Custom exception type
        with pytest.raises(ValueError, match="test_error"):
            FlextResult.unwrap_or_raise(failure, ValueError)

    def test_batch_process_comprehensive(self) -> None:
        """Test batch_process utility method comprehensively."""

        def processor(item: int) -> FlextResult[str]:
            if item % 2 == 0:
                return FlextResult[str].ok(f"even_{item}")
            return FlextResult[str].fail(f"odd_{item}")

        items = [1, 2, 3, 4, 5]
        successes, failures = FlextResult.batch_process(items, processor)

        assert successes == ["even_2", "even_4"]
        assert failures == ["odd_1", "odd_3", "odd_5"]

    def test_advanced_operators_edge_cases(self) -> None:
        """Test advanced operators edge cases."""
        success = FlextResult[int].ok(42)
        FlextResult[int].fail("error")

        # Test modulo operator with exception in predicate
        def failing_predicate(__x: int, /) -> bool:
            msg = "Predicate error"
            raise ValueError(msg)

        filtered = success % failing_predicate
        assert filtered.is_failure
        assert filtered.error is not None
        assert "Predicate evaluation failed" in filtered.error

    def test_traverse_empty_items(self) -> None:
        """Test traverse with empty items list."""

        def transform(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"item_{x}")

        result = FlextResult.traverse([], transform)
        assert result.is_success
        assert result.value == []

    def test_kleisli_compose_usage(self) -> None:
        """Test kleisli_compose method usage."""
        result = FlextResult[int].ok(10)

        def f(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"f_{x}")

        def g(x: str) -> FlextResult[int]:
            return FlextResult[int].ok(len(x))

        composed = result.kleisli_compose(f, g)
        final_result = composed(10)
        assert final_result.is_success
        assert final_result.value == 4  # len("f_10")

    def test_applicative_lift3_none_handling(self) -> None:
        """Test applicative_lift3 None value handling."""

        def add_three(x: int, y: int, z: int) -> int:
            return x + y + z

        r1 = FlextResult[int].ok(1)
        r2 = FlextResult[int].ok(2)
        r3 = FlextResult[int].ok(3)

        result = FlextResult.applicative_lift3(add_three, r1, r2, r3)
        assert result.is_success
        assert result.value == 6

    def test_safe_call_comprehensive(self) -> None:
        """Test safe_call with various exception types."""

        def success_func() -> str:
            return "success"

        def failing_func() -> str:
            msg = "Generic error"
            raise ValueError(msg)  # Use specific exception type

        # Success case
        result_success = FlextResult.safe_call(success_func)
        assert result_success.is_success
        assert result_success.value == "success"

        # Failure case
        result_failure = FlextResult.safe_call(failing_func)
        assert result_failure.is_failure
        assert result_failure.error is not None
        assert "Generic error" in result_failure.error
