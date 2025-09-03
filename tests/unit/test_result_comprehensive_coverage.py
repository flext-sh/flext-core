"""Comprehensive coverage tests for result.py (currently 60% coverage).

Targeting specific uncovered methods and branches in FlextResult.
"""

from __future__ import annotations

import pytest

from flext_core import FlextResult


class TestFlextResultComprehensiveCoverage:
    """Tests targeting uncovered methods and branches in FlextResult."""

    def test_combine_method_comprehensive(self) -> None:
        """Test combine method with various scenarios."""
        # Test combine with all successful results
        success_results = [
            FlextResult.ok(1),
            FlextResult.ok("test"),
            FlextResult.ok(True),
            FlextResult.ok([1, 2, 3]),
        ]
        combined = FlextResult.combine(*success_results)
        assert combined.success
        assert len(combined.unwrap()) == 4

        # Test combine with one failure
        mixed_results = [
            FlextResult.ok(1),
            FlextResult[int].fail("error"),
            FlextResult.ok(3),
        ]
        combined = FlextResult.combine(*mixed_results)
        assert combined.failure

        # Test combine with all failures
        failure_results = [
            FlextResult[int].fail("error1"),
            FlextResult[str].fail("error2"),
            FlextResult[bool].fail("error3"),
        ]
        combined = FlextResult.combine(*failure_results)
        assert combined.failure

        # Test combine with empty list
        empty_combined = FlextResult.combine()
        assert empty_combined.success
        assert empty_combined.unwrap() == []

    def test_first_success_method_comprehensive(self) -> None:
        """Test first_success method with various scenarios."""
        # Test with first result successful
        results_first_success = [
            FlextResult.ok("first"),
            FlextResult.ok("second"),
            FlextResult[str].fail("error"),
        ]
        first = FlextResult.first_success(*results_first_success)
        assert first.success
        assert first.unwrap() == "first"

        # Test with middle result successful
        results_middle_success = [
            FlextResult[str].fail("error1"),
            FlextResult.ok("middle"),
            FlextResult[str].fail("error2"),
        ]
        first = FlextResult.first_success(*results_middle_success)
        assert first.success
        assert first.unwrap() == "middle"

        # Test with all failures
        all_failures = [
            FlextResult[str].fail("error1"),
            FlextResult[str].fail("error2"),
            FlextResult[str].fail("error3"),
        ]
        first = FlextResult.first_success(*all_failures)
        assert first.failure

        # Test with empty list
        empty_first = FlextResult.first_success()
        assert empty_first.failure

    def test_tap_error_method(self) -> None:
        """Test tap_error method for error side effects."""
        errors_seen = []

        # Test tap_error on failure
        failure = FlextResult[int].fail("original error")
        tapped = failure.tap_error(lambda e: errors_seen.append(e))
        assert tapped.failure
        assert len(errors_seen) == 1
        assert "original error" in errors_seen[0]

        # Test tap_error on success (should be no-op)
        success = FlextResult.ok(42)
        tapped_success = success.tap_error(
            lambda e: errors_seen.append(f"should not see: {e}")
        )
        assert tapped_success.success
        assert tapped_success.unwrap() == 42
        assert len(errors_seen) == 1  # Should not have added another error

    def test_bind_method_comprehensive(self) -> None:
        """Test bind method (alias for flat_map)."""
        # Test bind with successful operation
        success = FlextResult.ok(5)

        def multiply_by_two(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        bound = success.bind(multiply_by_two)
        assert bound.success
        assert bound.unwrap() == 10

        # Test bind with failing operation
        def divide_by_zero(x: int) -> FlextResult[int]:
            return FlextResult[int].fail("division by zero")

        bound_fail = success.bind(divide_by_zero)
        assert bound_fail.failure
        assert "division by zero" in (bound_fail.error or "")

        # Test bind on failure (should propagate)
        failure = FlextResult[int].fail("original error")
        bound_failure = failure.bind(multiply_by_two)
        assert bound_failure.failure
        assert "original error" in (bound_failure.error or "")

    def test_recover_method(self) -> None:
        """Test recover method for error recovery."""
        # Test recover on failure
        failure = FlextResult[int].fail("error")
        recovered = failure.recover(lambda e: 42)
        assert recovered.success
        assert recovered.unwrap() == 42

        # Test recover on success (should be no-op)
        success = FlextResult.ok(10)
        not_recovered = success.recover(lambda e: 99)
        assert not_recovered.success
        assert not_recovered.unwrap() == 10

    def test_or_else_method(self) -> None:
        """Test or_else method for alternative results."""
        # Test or_else on failure
        failure = FlextResult[int].fail("error")
        result = failure.or_else(42)
        assert result.success
        assert result.unwrap() == 42

        # Test or_else on success (should be no-op)
        success = FlextResult.ok(10)
        result = success.or_else(99)
        assert result.success
        assert result.unwrap() == 10

    def test_all_success_method_comprehensive(self) -> None:
        """Test all_success class method."""
        # Test with all successful results
        all_good = [
            FlextResult.ok(1),
            FlextResult.ok("test"),
            FlextResult.ok(True),
        ]
        assert FlextResult.all_success(*all_good) is True

        # Test with one failure
        with_failure = [
            FlextResult.ok(1),
            FlextResult[int].fail("error"),
            FlextResult.ok(3),
        ]
        assert FlextResult.all_success(*with_failure) is False

        # Test with all failures
        all_failures = [
            FlextResult[int].fail("error1"),
            FlextResult[str].fail("error2"),
        ]
        assert FlextResult.all_success(*all_failures) is False

        # Test with empty list
        assert FlextResult.all_success() is True

    def test_any_success_method_comprehensive(self) -> None:
        """Test any_success class method."""
        # Test with some successful results
        mixed = [
            FlextResult[int].fail("error1"),
            FlextResult.ok("success"),
            FlextResult[bool].fail("error2"),
        ]
        assert FlextResult.any_success(*mixed) is True

        # Test with all failures
        all_failures = [
            FlextResult[int].fail("error1"),
            FlextResult[str].fail("error2"),
            FlextResult[bool].fail("error3"),
        ]
        assert FlextResult.any_success(*all_failures) is False

        # Test with all successes
        all_successes = [
            FlextResult.ok(1),
            FlextResult.ok("test"),
            FlextResult.ok(True),
        ]
        assert FlextResult.any_success(*all_successes) is True

        # Test with empty list
        assert FlextResult.any_success() is False

    def test_unwrap_or_raise_method(self) -> None:
        """Test unwrap_or_raise method."""
        # Test unwrap_or_raise on success
        success = FlextResult.ok(42)
        assert success.unwrap_or_raise(success) == 42

        # Test unwrap_or_raise on failure
        failure = FlextResult[int].fail("test error")
        with pytest.raises(ValueError, match="test error"):
            failure.unwrap_or_raise(failure)

    def test_then_method(self) -> None:
        """Test then method for additional operations."""
        success = FlextResult.ok(42)

        # Test then on success
        result = success.then(lambda x: FlextResult.ok(x * 2))
        assert result.success
        assert result.unwrap() == 84

        # Test then on failure (should propagate)
        failure = FlextResult[int].fail("error")
        result = failure.then(lambda x: FlextResult.ok(x * 2))
        assert result.failure

    def test_expect_method(self) -> None:
        """Test expect method."""
        success = FlextResult.ok(42)

        # Test expect on success
        result = success.expect("Should be successful")
        assert result == 42

        # Test expect on failure
        failure = FlextResult[int].fail("error")
        with pytest.raises(Exception):
            failure.expect("Should not fail")

    def test_chain_operations(self) -> None:
        """Test chaining multiple operations."""
        result = (
            FlextResult.ok(10)
            .map(lambda x: x * 2)
            .flat_map(lambda x: FlextResult.ok(x + 5))
            .filter(lambda x: x > 15, "too small")
            .tap(lambda x: None)  # Side effect
            .map(lambda x: str(x))
        )

        assert result.success
        assert result.unwrap() == "25"

        # Test chain with failure
        failing_chain = (
            FlextResult.ok(10)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 50, "too small")  # This should fail
            .map(lambda x: str(x))
        )

        assert failing_chain.failure
        assert "too small" in (failing_chain.error or "")

    def test_error_code_handling(self) -> None:
        """Test error code functionality."""
        # Test failure with error code
        failure = FlextResult[int].fail("error message", error_code="ERR001")
        assert failure.failure
        assert failure.error_code == "ERR001"

        # Test success doesn't have error code
        success = FlextResult.ok(42)
        assert success.error_code is None

    def test_from_exception_class_method(self) -> None:
        """Test from_exception class method."""
        # Test with standard exception
        try:
            raise ValueError("test error")
        except ValueError:
            result = FlextResult.from_exception(lambda: e)
            assert result.failure
            assert "test error" in (result.error or "")

        # Test with custom exception
        class CustomError(Exception):
            pass

        try:
            raise CustomError("custom error")
        except CustomError:
            result = FlextResult.from_exception(lambda: e)
            assert result.failure
            assert "custom error" in (result.error or "")

    def test_edge_case_scenarios(self) -> None:
        """Test edge cases and boundary conditions."""
        # Test with None values
        none_success = FlextResult.ok(None)
        assert none_success.success
        assert none_success.unwrap() is None

        # Test with empty string error - FlextResult may convert empty errors to default
        empty_error = FlextResult[int].fail("")
        assert empty_error.failure
        # Empty string may be converted to a default error message
        assert empty_error.error is not None

        # Test with complex objects
        complex_obj = {"nested": {"deep": {"value": [1, 2, 3]}}}
        complex_result = FlextResult.ok(complex_obj)
        assert complex_result.success
        unwrapped = complex_result.unwrap()
        assert unwrapped["nested"]["deep"]["value"] == [1, 2, 3]

    def test_performance_with_large_data(self) -> None:
        """Test performance with large data structures."""
        # Test with large list
        large_list = list(range(10000))
        large_result = FlextResult.ok(large_list)
        assert large_result.success
        assert len(large_result.unwrap()) == 10000

        # Test mapping over large data
        mapped_result = large_result.map(lambda lst: [x * 2 for x in lst[:100]])
        assert mapped_result.success
        assert len(mapped_result.unwrap()) == 100
        assert mapped_result.unwrap()[0] == 0
        assert mapped_result.unwrap()[-1] == 198

    def test_type_safety_scenarios(self) -> None:
        """Test type safety in various scenarios."""
        # Test generic type preservation
        string_result: FlextResult[str] = FlextResult.ok("test")
        assert string_result.success

        int_result: FlextResult[int] = FlextResult.ok(42)
        assert int_result.success

        # Test type transformation through map
        transformed = string_result.map(len)
        assert transformed.success
        assert transformed.unwrap() == 4
