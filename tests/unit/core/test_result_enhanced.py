"""Enhanced tests for FlextResult improvements.

Tests all the new utility methods added to FlextResult for comprehensive
error handling and functional programming patterns.
"""

from __future__ import annotations

import pytest

from flext_core import FlextResult


class TestFlextResultEnhanced:
    """Test enhanced FlextResult functionality."""

    def test_then_alias_for_flat_map(self) -> None:
        """Test then method as alias for flat_map."""
        result = FlextResult.ok(5)

        def double(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        chained = result.then(double)
        assert chained.is_success
        assert chained.data == 10

    def test_bind_alias_for_flat_map(self) -> None:
        """Test bind method as monadic operation."""
        result = FlextResult.ok("hello")

        def to_upper(s: str) -> FlextResult[str]:
            return FlextResult.ok(s.upper())

        bound = result.bind(to_upper)
        assert bound.is_success
        assert bound.data == "HELLO"

    def test_or_else_success(self) -> None:
        """Test or_else with successful result."""
        success = FlextResult.ok("original")
        default = FlextResult.ok("default")

        result = success.or_else(default)
        assert result.is_success
        assert result.data == "original"

    def test_or_else_failure(self) -> None:
        """Test or_else with failed result."""
        failure = FlextResult.fail("error")
        default = FlextResult.ok("default")

        result = failure.or_else(default)
        assert result.is_success
        assert result.data == "default"

    def test_or_else_get_success(self) -> None:
        """Test or_else_get with successful result."""
        success = FlextResult.ok("original")

        def get_default() -> FlextResult[str]:
            return FlextResult.ok("default")

        result = success.or_else_get(get_default)
        assert result.is_success
        assert result.data == "original"

    def test_or_else_get_failure(self) -> None:
        """Test or_else_get with failed result."""
        failure = FlextResult.fail("error")

        def get_default() -> FlextResult[str]:
            return FlextResult.ok("default")

        result = failure.or_else_get(get_default)
        assert result.is_success
        assert result.data == "default"

    def test_recover_success(self) -> None:
        """Test recover with successful result."""
        success = FlextResult.ok("original")

        def recover_func(error: str) -> str:
            return f"recovered: {error}"

        result = success.recover(recover_func)
        assert result.is_success
        assert result.data == "original"

    def test_recover_failure(self) -> None:
        """Test recover with failed result."""
        failure = FlextResult.fail("error message")

        def recover_func(error: str) -> str:
            return f"recovered: {error}"

        result = failure.recover(recover_func)
        assert result.is_success
        assert result.data == "recovered: error message"

    def test_recover_with_success(self) -> None:
        """Test recover_with with successful result."""
        success = FlextResult.ok("original")

        def recover_func(error: str) -> FlextResult[str]:
            return FlextResult.ok(f"recovered: {error}")

        result = success.recover_with(recover_func)
        assert result.is_success
        assert result.data == "original"

    def test_recover_with_failure(self) -> None:
        """Test recover_with with failed result."""
        failure = FlextResult.fail("error message")

        def recover_func(error: str) -> FlextResult[str]:
            return FlextResult.ok(f"recovered: {error}")

        result = failure.recover_with(recover_func)
        assert result.is_success
        assert result.data == "recovered: error message"

    def test_tap_success(self) -> None:
        """Test tap with successful result."""
        side_effect_value = []

        def side_effect(value: str) -> None:
            side_effect_value.append(value)

        result = FlextResult.ok("test").tap(side_effect)

        assert result.is_success
        assert result.data == "test"
        assert side_effect_value == ["test"]

    def test_tap_failure(self) -> None:
        """Test tap with failed result."""
        side_effect_value = []

        def side_effect(value: str) -> None:
            side_effect_value.append(value)

        result = FlextResult.fail("error").tap(side_effect)

        assert result.is_failure
        assert side_effect_value == []

    def test_tap_error_success(self) -> None:
        """Test tap_error with successful result."""
        side_effect_value = []

        def side_effect(error: str) -> None:
            side_effect_value.append(error)

        result = FlextResult.ok("test").tap_error(side_effect)

        assert result.is_success
        assert side_effect_value == []

    def test_tap_error_failure(self) -> None:
        """Test tap_error with failed result."""
        side_effect_value = []

        def side_effect(error: str) -> None:
            side_effect_value.append(error)

        result = FlextResult.fail("error message").tap_error(side_effect)

        assert result.is_failure
        assert side_effect_value == ["error message"]

    def test_filter_success_predicate_true(self) -> None:
        """Test filter with successful result and true predicate."""
        result = FlextResult.ok(10)

        filtered = result.filter(lambda x: x > 5)
        assert filtered.is_success
        assert filtered.data == 10

    def test_filter_success_predicate_false(self) -> None:
        """Test filter with successful result and false predicate."""
        result = FlextResult.ok(3)

        filtered = result.filter(lambda x: x > 5)
        assert filtered.is_failure
        assert "Filter predicate failed" in (filtered.error or "")

    def test_filter_failure(self) -> None:
        """Test filter with failed result."""
        result = FlextResult.fail("original error")

        filtered = result.filter(lambda _: True)
        assert filtered.is_failure
        assert filtered.error == "original error"

    def test_zip_with_both_success(self) -> None:
        """Test zip_with with both results successful."""
        result1 = FlextResult.ok(5)
        result2 = FlextResult.ok(3)

        combined = result1.zip_with(result2, lambda x, y: x + y)
        assert combined.is_success
        assert combined.data == 8

    def test_zip_with_first_failure(self) -> None:
        """Test zip_with with first result failed."""
        result1 = FlextResult.fail("first error")
        result2 = FlextResult.ok(3)

        combined = result1.zip_with(result2, lambda x, y: x + y)
        assert combined.is_failure
        assert combined.error == "first error"

    def test_zip_with_second_failure(self) -> None:
        """Test zip_with with second result failed."""
        result1 = FlextResult.ok(5)
        result2 = FlextResult.fail("second error")

        combined = result1.zip_with(result2, lambda x, y: x + y)
        assert combined.is_failure
        assert combined.error == "second error"

    def test_to_either_success(self) -> None:
        """Test to_either with successful result."""
        result = FlextResult.ok("data")
        data, error = result.to_either()

        assert data == "data"
        assert error is None

    def test_to_either_failure(self) -> None:
        """Test to_either with failed result."""
        result = FlextResult.fail("error message")
        data, error = result.to_either()

        assert data is None
        assert error == "error message"

    def test_to_exception_success(self) -> None:
        """Test to_exception with successful result."""
        result = FlextResult.ok("data")
        exception = result.to_exception()

        assert exception is None

    def test_to_exception_failure(self) -> None:
        """Test to_exception with failed result."""
        result = FlextResult.fail("error message")
        exception = result.to_exception()

        assert isinstance(exception, ValueError)
        assert str(exception) == "error message"

    def test_from_exception_success(self) -> None:
        """Test from_exception with successful function."""

        def successful_func() -> str:
            return "success"

        result = FlextResult.from_exception(successful_func)
        assert result.is_success
        assert result.data == "success"

    def test_from_exception_failure(self) -> None:
        """Test from_exception with failing function."""

        def failing_func() -> str:
            msg = "test error"
            raise ValueError(msg)

        result = FlextResult.from_exception(failing_func)
        assert result.is_failure
        assert "test error" in (result.error or "")

    def test_combine_all_success(self) -> None:
        """Test combine with all successful results."""
        results = [
            FlextResult.ok(1),
            FlextResult.ok(2),
            FlextResult.ok(3),
        ]

        combined = FlextResult.combine(*results)
        assert combined.is_success
        assert combined.data == [1, 2, 3]

    def test_combine_with_failure(self) -> None:
        """Test combine with one failed result."""
        results = [
            FlextResult.ok(1),
            FlextResult.fail("error"),
            FlextResult.ok(3),
        ]

        combined = FlextResult.combine(*results)
        assert combined.is_failure
        assert combined.error == "error"

    def test_combine_empty(self) -> None:
        """Test combine with no results."""
        combined = FlextResult.combine()
        assert combined.is_success
        assert combined.data == []

    def test_all_success_true(self) -> None:
        """Test all_success with all successful results."""
        results = [
            FlextResult.ok(1),
            FlextResult.ok(2),
            FlextResult.ok(3),
        ]

        assert FlextResult.all_success(*results) is True

    def test_all_success_false(self) -> None:
        """Test all_success with one failed result."""
        results = [
            FlextResult.ok(1),
            FlextResult.fail("error"),
            FlextResult.ok(3),
        ]

        assert FlextResult.all_success(*results) is False

    def test_any_success_true(self) -> None:
        """Test any_success with at least one successful result."""
        results = [
            FlextResult.fail("error1"),
            FlextResult.ok(2),
            FlextResult.fail("error2"),
        ]

        assert FlextResult.any_success(*results) is True

    def test_any_success_false(self) -> None:
        """Test any_success with all failed results."""
        results = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("error3"),
        ]

        assert FlextResult.any_success(*results) is False

    def test_first_success_found(self) -> None:
        """Test first_success with successful result found."""
        results = [
            FlextResult.fail("error1"),
            FlextResult.ok("success"),
            FlextResult.ok("another"),
        ]

        first = FlextResult.first_success(*results)
        assert first.is_success
        assert first.data == "success"

    def test_first_success_not_found(self) -> None:
        """Test first_success with no successful results."""
        results = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("error3"),
        ]

        first = FlextResult.first_success(*results)
        assert first.is_failure
        assert first.error == "error3"  # Last error

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
        assert result.is_success
        assert result.data == "success"

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
        assert "error2" in (result.error or "")

    def test_try_all_empty(self) -> None:
        """Test try_all with no functions."""
        result = FlextResult.try_all()
        assert result.is_failure
        assert "No functions provided" in (result.error or "")


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

        assert result.is_success
        assert result.data == "Result: 11"

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

        assert result.is_success
        assert result.data == 10
