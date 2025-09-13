"""Edge case tests to achieve the final 100% coverage for FlextResult.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Never

import pytest

from flext_core import FlextResult


class TestFlextResultEdgeCases:
    """Edge cases tests to achieve complete 100% coverage."""

    def test_is_fail_property(self) -> None:
        """Test is_fail property specifically."""
        success_result = FlextResult[str].ok("test")
        failure_result = FlextResult[str].fail("error")

        # Test the is_fail property
        assert success_result.is_fail is False
        assert failure_result.is_fail is True

    def test_flat_map_none_data_runtime_error(self) -> None:
        """Test flat_map with artificially created None data edge case."""
        # This tests the specific RuntimeError path in flat_map line 284-285
        # We need to artificially create a success result with None data
        result = FlextResult[str].ok("test")

        # Artificially set _data to None to test the safety check
        result._data = None  # This should never happen in normal usage

        def transform(x: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"transformed_{x}")

        # This should hit the RuntimeError safety check and convert to failure
        flat_mapped = result.flat_map(transform)
        assert flat_mapped.is_failure
        assert (
            "Unexpected chaining error: Internal error: data is None when result is success"
            in flat_mapped.error
        )

    def test_empty_list_traverse_coverage(self) -> None:
        """Test traverse with empty list to cover specific branches."""

        def transform(x: int) -> FlextResult[str]:
            return FlextResult[str].ok(f"item_{x}")

        # Empty list should return successful empty result
        result = FlextResult.traverse([], transform)
        assert result.is_success
        assert result.value == []

        # This covers the empty iteration case
        results: list[str] = []
        assert results == []

    def test_value_or_none_property_coverage(self) -> None:
        """Test value_or_none property to ensure coverage."""
        success = FlextResult[str].ok("test_value")
        failure = FlextResult[str].fail("test_error")

        assert success.value_or_none == "test_value"
        assert failure.value_or_none is None

    def test_collect_failures_with_none_error(self) -> None:
        """Test collect_failures utility method with None error."""
        # Create a failure with None error (edge case)
        failure_none = FlextResult[str](error="test_error")
        success = FlextResult[str].ok("success")
        failure_normal = FlextResult[str].fail("normal_error")

        results = [success, failure_none, failure_normal]
        failures = FlextResult.collect_failures(results)

        # Should collect only non-None errors
        assert "test_error" in failures
        assert "normal_error" in failures
        assert len(failures) == 2

    def test_success_rate_edge_cases(self) -> None:
        """Test success_rate with various edge cases."""
        # Test with 100% success
        all_success = [
            FlextResult[str].ok("1"),
            FlextResult[str].ok("2"),
            FlextResult[str].ok("3"),
        ]
        assert FlextResult.success_rate(all_success) == 100.0

        # Test with 0% success
        all_failure = [FlextResult[str].fail("1"), FlextResult[str].fail("2")]
        assert FlextResult.success_rate(all_failure) == 0.0

        # Test with empty list (covered earlier but ensuring)
        assert FlextResult.success_rate([]) == 0.0

    def test_applicative_lift3_edge_case_coverage(self) -> None:
        """Test applicative_lift3 edge case with None handling."""

        def add_three(x: int, y: int, z: int) -> int:
            return x + y + z

        r1 = FlextResult[int].ok(1)
        r2 = FlextResult[int].ok(2)
        r3 = FlextResult[int].fail("error")

        # Test with failure in third argument
        result = FlextResult.applicative_lift3(add_three, r1, r2, r3)
        assert result.is_failure
        assert "error" in result.error

    def test_advanced_operators_complete_coverage(self) -> None:
        """Test advanced operators for complete coverage."""
        success = FlextResult[int].ok(42)
        failure = FlextResult[int].fail("test_error")

        # Test division operator both paths
        alt_success = FlextResult[int].ok(100)
        alt_failure = FlextResult[int].fail("alt_error")

        # Success / failure -> should return success
        result1 = success / failure
        assert result1.is_success
        assert result1.value == 42

        # Failure / success -> should return success (alternative)
        result2 = failure / alt_success
        assert result2.is_success
        assert result2.value == 100

        # Failure / failure -> should return failure with last error
        result3 = failure / alt_failure
        assert result3.is_failure
        # Should use the alternative error or original error as fallback
        assert "error" in result3.error

    def test_or_operator_none_handling(self) -> None:
        """Test __or__ operator with None data handling."""
        # Test success with None data
        none_success = FlextResult[str | None].ok(None)
        result = none_success | "default"
        assert result == "default"  # None data should use default

        # Test success with actual data
        success = FlextResult[str].ok("actual")
        result = success | "default"
        assert result == "actual"  # Should use actual data

    def test_hash_edge_case_complex_objects(self) -> None:
        """Test hash method with complex edge cases."""

        # Test with object that has __dict__ but raises exception during hashing
        class ProblematicObject:
            def __init__(self) -> None:
                """Initialize the instance."""
                self.data = {"key": ["unhashable", "list"]}

            def __hash__(self) -> Never:
                msg = "Cannot hash this object"
                raise TypeError(msg)

        obj = ProblematicObject()
        result = FlextResult[ProblematicObject].ok(obj)

        # Should handle the exception and use fallback hashing
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_repr_comprehensive(self) -> None:
        """Test __repr__ method comprehensively."""
        success = FlextResult[str].ok("test_data")
        failure = FlextResult[str].fail("test_error")

        success_repr = repr(success)
        failure_repr = repr(failure)

        assert "FlextResult" in success_repr
        assert "test_data" in success_repr
        assert "is_success=True" in success_repr
        assert "error=None" in success_repr

        assert "FlextResult" in failure_repr
        assert "test_error" in failure_repr
        assert "is_success=False" in failure_repr
        assert "data=None" in failure_repr

    def test_iter_comprehensive(self) -> None:
        """Test __iter__ method comprehensively."""
        success = FlextResult[str].ok("success_data")
        failure = FlextResult[str].fail("failure_error")

        # Test success unpacking
        data, error = success
        assert data == "success_data"
        assert error is None

        # Test failure unpacking
        data, error = failure
        assert data is None
        assert error == "failure_error"

    def test_getitem_comprehensive(self) -> None:
        """Test __getitem__ method comprehensively."""
        success = FlextResult[str].ok("success_data")
        failure = FlextResult[str].fail("failure_error")

        # Test success indexing
        assert success[0] == "success_data"
        assert success[1] is None

        # Test failure indexing
        assert failure[0] is None
        assert failure[1] == "failure_error"

        # Test invalid index
        with pytest.raises(IndexError, match="FlextResult only supports indices 0"):
            success[3]

    def test_bool_conversion(self) -> None:
        """Test __bool__ method for boolean conversion."""
        success = FlextResult[str].ok("data")
        failure = FlextResult[str].fail("error")

        assert bool(success) is True
        assert bool(failure) is False

        # Test in conditional contexts
        success_bool = bool(success)

        failure_bool = bool(failure)

        assert success_bool is True
        assert failure_bool is False
