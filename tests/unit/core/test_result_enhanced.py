"""Enhanced tests for FlextResult improvements.

Tests all the new utility methods added to FlextResult for comprehensive
error handling and functional programming patterns.
"""

from __future__ import annotations

import operator
from typing import cast

import pytest

from flext_core import FlextResult
from flext_core.exceptions import FlextOperationError

# Constants
EXPECTED_TOTAL_PAGES = 8


class TestFlextResultEnhanced:
    """Test enhanced FlextResult functionality."""

    def test_then_alias_for_flat_map(self) -> None:
        """Test then method as alias for flat_map."""
        result = FlextResult.ok(5)

        def double(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        chained = result.then(double)
        assert chained.success
        if chained.data != 10:
            raise AssertionError(f"Expected {10}, got {chained.data}")

    def test_bind_alias_for_flat_map(self) -> None:
        """Test bind method as monadic operation."""
        result = FlextResult.ok("hello")

        def to_upper(s: str) -> FlextResult[str]:
            return FlextResult.ok(s.upper())

        bound = result.bind(to_upper)
        assert bound.success
        if bound.data != "HELLO":
            raise AssertionError(f"Expected {'HELLO'}, got {bound.data}")

    def test_or_else_success(self) -> None:
        """Test or_else with successful result."""
        success = FlextResult.ok("original")
        default = FlextResult.ok("default")

        result = success.or_else(default)
        assert result.success
        if result.data != "original":
            raise AssertionError(f"Expected {'original'}, got {result.data}")

    def test_or_else_failure(self) -> None:
        """Test or_else with failed result."""
        failure: FlextResult[str] = FlextResult.fail("error")
        default = FlextResult.ok("default")

        result = failure.or_else(default)
        assert result.success
        if result.data != "default":
            raise AssertionError(f"Expected {'default'}, got {result.data}")

    def test_or_else_get_success(self) -> None:
        """Test or_else_get with successful result."""
        success = FlextResult.ok("original")

        def get_default() -> FlextResult[str]:
            return FlextResult.ok("default")

        result = success.or_else_get(get_default)
        assert result.success
        if result.data != "original":
            raise AssertionError(f"Expected {'original'}, got {result.data}")

    def test_or_else_get_failure(self) -> None:
        """Test or_else_get with failed result."""
        failure: FlextResult[str] = FlextResult.fail("error")

        def get_default() -> FlextResult[str]:
            return FlextResult.ok("default")

        result = failure.or_else_get(get_default)
        assert result.success
        if result.data != "default":
            raise AssertionError(f"Expected {'default'}, got {result.data}")

    def test_recover_success(self) -> None:
        """Test recover with successful result."""
        success = FlextResult.ok("original")

        def recover_func(error: str) -> str:
            return f"recovered: {error}"

        result = success.recover(recover_func)
        assert result.success
        if result.data != "original":
            raise AssertionError(f"Expected {'original'}, got {result.data}")

    def test_recover_failure(self) -> None:
        """Test recover with failed result."""
        failure: FlextResult[str] = FlextResult.fail("error message")

        def recover_func(error: str) -> str:
            return f"recovered: {error}"

        result = failure.recover(recover_func)
        assert result.success
        if result.data != "recovered: error message":
            raise AssertionError(
                f"Expected {'recovered: error message'}, got {result.data}"
            )

    def test_recover_with_success(self) -> None:
        """Test recover_with with successful result."""
        success = FlextResult.ok("original")

        def recover_func(error: str) -> FlextResult[str]:
            return FlextResult.ok(f"recovered: {error}")

        result = success.recover_with(recover_func)
        assert result.success
        if result.data != "original":
            raise AssertionError(f"Expected {'original'}, got {result.data}")

    def test_recover_with_failure(self) -> None:
        """Test recover_with with failed result."""
        failure: FlextResult[str] = FlextResult.fail("error message")

        def recover_func(error: str) -> FlextResult[str]:
            return FlextResult.ok(f"recovered: {error}")

        result = failure.recover_with(recover_func)
        assert result.success
        if result.data != "recovered: error message":
            raise AssertionError(
                f"Expected {'recovered: error message'}, got {result.data}"
            )

    def test_tap_success(self) -> None:
        """Test tap with successful result."""
        side_effect_value = []

        def side_effect(value: str) -> None:
            side_effect_value.append(value)

        result = FlextResult.ok("test").tap(side_effect)

        assert result.success
        if result.data != "test":
            raise AssertionError(f"Expected {'test'}, got {result.data}")
        assert side_effect_value == ["test"]

    def test_tap_failure(self) -> None:
        """Test tap with failed result."""
        side_effect_value = []

        def side_effect(value: str) -> None:
            side_effect_value.append(value)

        result: FlextResult[str] = FlextResult.fail("error").tap(side_effect)

        assert result.is_failure
        if side_effect_value != []:
            raise AssertionError(f"Expected {[]}, got {side_effect_value}")

    def test_tap_error_success(self) -> None:
        """Test tap_error with successful result."""
        side_effect_value = []

        def side_effect(error: str) -> None:
            side_effect_value.append(error)

        result = FlextResult.ok("test").tap_error(side_effect)

        assert result.success
        if side_effect_value != []:
            raise AssertionError(f"Expected {[]}, got {side_effect_value}")

    def test_tap_error_failure(self) -> None:
        """Test tap_error with failed result."""
        side_effect_value = []

        def side_effect(error: str) -> None:
            side_effect_value.append(error)

        result: FlextResult[str] = FlextResult.fail("error message").tap_error(
            side_effect
        )

        assert result.is_failure
        if side_effect_value != ["error message"]:
            raise AssertionError(
                f"Expected {['error message']}, got {side_effect_value}"
            )

    def test_filter_success_predicate_true(self) -> None:
        """Test filter with successful result and true predicate."""
        result = FlextResult.ok(10)

        filtered = result.filter(lambda x: x > 5)
        assert filtered.success
        if filtered.data != 10:
            raise AssertionError(f"Expected {10}, got {filtered.data}")

    def test_filter_success_predicate_false(self) -> None:
        """Test filter with successful result and false predicate."""
        result = FlextResult.ok(3)

        filtered = result.filter(lambda x: x > 5)
        assert filtered.is_failure
        if "Filter predicate failed" not in (filtered.error or ""):
            raise AssertionError(
                f"Expected {'Filter predicate failed'} in {(filtered.error or '')}"
            )

    def test_filter_failure(self) -> None:
        """Test filter with failed result."""
        result: FlextResult[int] = FlextResult.fail("original error")

        filtered = result.filter(lambda _: True)
        assert filtered.is_failure
        if filtered.error != "original error":
            raise AssertionError(f"Expected {'original error'}, got {filtered.error}")

    def test_zip_with_both_success(self) -> None:
        """Test zip_with with both results successful."""
        result1 = FlextResult.ok(5)
        result2 = FlextResult.ok(3)

        combined = result1.zip_with(result2, operator.add)
        assert combined.success
        if combined.data != EXPECTED_TOTAL_PAGES:
            raise AssertionError(f"Expected {8}, got {combined.data}")

    def test_zip_with_first_failure(self) -> None:
        """Test zip_with with first result failed."""
        result1: FlextResult[int] = FlextResult.fail("first error")
        result2 = FlextResult.ok(3)

        combined = result1.zip_with(result2, operator.add)
        assert combined.is_failure
        if combined.error != "first error":
            raise AssertionError(f"Expected {'first error'}, got {combined.error}")

    def test_zip_with_second_failure(self) -> None:
        """Test zip_with with second result failed."""
        result1 = FlextResult.ok(5)
        result2: FlextResult[int] = FlextResult.fail("second error")

        combined = result1.zip_with(result2, operator.add)
        assert combined.is_failure
        if combined.error != "second error":
            raise AssertionError(f"Expected {'second error'}, got {combined.error}")

    def test_to_either_success(self) -> None:
        """Test to_either with successful result."""
        result = FlextResult.ok("data")
        data, error = result.to_either()

        if data != "data":
            raise AssertionError(f"Expected {'data'}, got {data}")
        assert error is None

    def test_to_either_failure(self) -> None:
        """Test to_either with failed result."""
        result: FlextResult[str] = FlextResult.fail("error message")
        data, error = result.to_either()

        assert data is None
        if error != "error message":
            raise AssertionError(f"Expected {'error message'}, got {error}")

    def test_to_exception_success(self) -> None:
        """Test to_exception with successful result."""
        result = FlextResult.ok("data")
        exception = result.to_exception()

        assert exception is None

    def test_to_exception_failure(self) -> None:
        """Test to_exception with failed result."""
        result: FlextResult[str] = FlextResult.fail("error message")
        exception = result.to_exception()

        assert isinstance(exception, FlextOperationError)
        if str(exception) != "[OPERATION_ERROR] error message":
            raise AssertionError(
                f"Expected {'[OPERATION_ERROR] error message'}, got {exception!s}"
            )

    def test_from_exception_success(self) -> None:
        """Test from_exception with successful function."""

        def successful_func() -> str:
            return "success"

        result = FlextResult.from_exception(successful_func)
        assert result.success
        if result.data != "success":
            raise AssertionError(f"Expected {'success'}, got {result.data}")

    def test_from_exception_failure(self) -> None:
        """Test from_exception with failing function."""

        def failing_func() -> str:
            msg = "test error"
            raise ValueError(msg)

        result = FlextResult.from_exception(failing_func)
        assert result.is_failure
        if "test error" not in (result.error or ""):
            raise AssertionError(f"Expected {'test error'} in {(result.error or '')}")

    def test_combine_all_success(self) -> None:
        """Test combine with all successful results."""
        results = [
            FlextResult.ok(1),
            FlextResult.ok(2),
            FlextResult.ok(3),
        ]

        # Cast to FlextResult[object] for combine method compatibility
        object_results = [cast("FlextResult[object]", r) for r in results]
        combined = FlextResult.combine(*object_results)
        assert combined.success
        if combined.data != [1, 2, 3]:
            raise AssertionError(f"Expected {[1, 2, 3]}, got {combined.data}")

    def test_combine_with_failure(self) -> None:
        """Test combine with one failed result."""
        results = [
            FlextResult.ok(1),
            FlextResult.fail("error"),
            FlextResult.ok(3),
        ]

        # Cast to FlextResult[object] for combine method compatibility
        object_results = [cast("FlextResult[object]", r) for r in results]
        combined = FlextResult.combine(*object_results)
        assert combined.is_failure
        if combined.error != "error":
            raise AssertionError(f"Expected {'error'}, got {combined.error}")

    def test_combine_empty(self) -> None:
        """Test combine with no results."""
        combined = FlextResult.combine()
        assert combined.success
        if combined.data != []:
            raise AssertionError(f"Expected {[]}, got {combined.data}")

    def test_all_success_true(self) -> None:
        """Test all_success with all successful results."""
        results = [
            FlextResult.ok(1),
            FlextResult.ok(2),
            FlextResult.ok(3),
        ]

        # Cast to FlextResult[object] for all_success method compatibility
        if not FlextResult.all_success(
            *[cast("FlextResult[object]", r) for r in results]
        ):
            raise AssertionError(
                f"Expected True, got {FlextResult.all_success(*[cast('FlextResult[object]', r) for r in results])}"
            )

    def test_all_success_false(self) -> None:
        """Test all_success with one failed result."""
        results = [
            FlextResult.ok(1),
            FlextResult.fail("error"),
            FlextResult.ok(3),
        ]

        # Cast to FlextResult[object] for all_success method compatibility
        if FlextResult.all_success(*[cast("FlextResult[object]", r) for r in results]):
            raise AssertionError(
                f"Expected False, got {FlextResult.all_success(*[cast('FlextResult[object]', r) for r in results])}"
            )

    def test_any_success_true(self) -> None:
        """Test any_success with at least one successful result."""
        results = [
            FlextResult.fail("error1"),
            FlextResult.ok(2),
            FlextResult.fail("error2"),
        ]

        # Cast to FlextResult[object] for any_success method compatibility
        if not FlextResult.any_success(
            *[cast("FlextResult[object]", r) for r in results]
        ):
            raise AssertionError(
                f"Expected True, got {FlextResult.any_success(*[cast('FlextResult[object]', r) for r in results])}"
            )

    def test_any_success_false(self) -> None:
        """Test any_success with all failed results."""
        results: list[FlextResult[str]] = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("error3"),
        ]

        # Cast to FlextResult[object] for any_success method compatibility
        if FlextResult.any_success(*[cast("FlextResult[object]", r) for r in results]):
            raise AssertionError(
                f"Expected False, got {FlextResult.any_success(*[cast('FlextResult[object]', r) for r in results])}"
            )

    def test_first_success_found(self) -> None:
        """Test first_success with successful result found."""
        results = [
            FlextResult.fail("error1"),
            FlextResult.ok("success"),
            FlextResult.ok("another"),
        ]

        first = FlextResult.first_success(*results)
        assert first.success
        if first.data != "success":
            raise AssertionError(f"Expected {'success'}, got {first.data}")

    def test_first_success_not_found(self) -> None:
        """Test first_success with no successful results."""
        results: list[FlextResult[str]] = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("error3"),
        ]

        first = FlextResult.first_success(*results)
        assert first.is_failure
        if first.error != "error3":  # Last error:
            raise AssertionError(f"Expected {'error3'} # Last error, got {first.error}")

    def test_try_all_success(self) -> None:
        """Test try_all with successful function."""

        def func1() -> str:
            msg = "error1"
            raise ValueError(msg)

        def func2() -> str:
            return "success"

        def func3() -> str:
            return "another"

        result = FlextResult.try_all(func1, func2, func3)
        assert result.success
        if result.data != "success":
            raise AssertionError(f"Expected {'success'}, got {result.data}")

    def test_try_all_failure(self) -> None:
        """Test try_all with all failing functions."""

        def func1() -> str:
            msg = "error1"
            raise ValueError(msg)

        def func2() -> str:
            msg = "error2"
            raise ValueError(msg)

        result = FlextResult.try_all(func1, func2)
        assert result.is_failure
        if "error2" not in (result.error or ""):
            raise AssertionError(f"Expected {'error2'} in {(result.error or '')}")

    def test_try_all_empty(self) -> None:
        """Test try_all with no functions."""
        result: FlextResult[object] = FlextResult.try_all()
        assert result.is_failure
        if "No functions provided" not in (result.error or ""):
            raise AssertionError(
                f"Expected {'No functions provided'} in {(result.error or '')}"
            )


@pytest.mark.integration
class TestFlextResultIntegration:
    """Integration tests for chaining multiple FlextResult operations."""

    def test_complex_chain(self) -> None:
        """Test complex chaining of multiple operations."""

        def validate_positive(x: int) -> FlextResult[int]:
            if x > 0:
                return FlextResult.ok(x)
            return FlextResult.fail("Number must be positive")

        def double(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        def to_string(x: int) -> FlextResult[str]:
            return FlextResult.ok(f"Result: {x}")

        # Test successful chain
        result = (
            FlextResult.ok(5)
            .then(validate_positive)
            .then(double)
            .map(lambda x: x + 1)  # 11
            .then(to_string)
            .tap(lambda _: None)  # Side effect removed
        )

        assert result.success
        if result.data != "Result: 11":
            raise AssertionError(f"Expected {'Result: 11'}, got {result.data}")

    def test_chain_with_recovery(self) -> None:
        """Test chain with error recovery."""

        def might_fail(x: int) -> FlextResult[int]:
            if x < 0:
                return FlextResult.fail("Negative number")
            return FlextResult.ok(x * 2)

        def recover_negative(error: str) -> FlextResult[int]:
            if "Negative" in error:
                return FlextResult.ok(0)
            return FlextResult.fail(error)

        # Test recovery
        result = (
            FlextResult.ok(-5)
            .then(might_fail)
            .recover_with(recover_negative)
            .map(lambda x: x + 10)
        )

        assert result.success
        if result.data != 10:
            raise AssertionError(f"Expected {10}, got {result.data}")
