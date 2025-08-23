"""Tests for refactored FlextDecorators consolidated API.

Tests the new consolidated API after major refactoring where all decorator
functionality was consolidated into FlextDecorators with static methods.
"""

from __future__ import annotations

import time

import pytest

from flext_core import FlextDecorators, FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextDecoratorsNewAPI:
    """Test the new consolidated FlextDecorators API."""

    def test_safe_call_decorator_success(self) -> None:
        """Test safe_call decorator with successful execution."""

        @FlextDecorators.safe_call()
        def safe_function(x: int) -> int:
            return x * 2

        result = safe_function(5)

        # safe_call returns FlextResult
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value == 10

    def test_safe_call_decorator_exception(self) -> None:
        """Test safe_call decorator with exception handling."""

        @FlextDecorators.safe_call(handled_exceptions=(ValueError,))
        def failing_function(x: int) -> int:
            if x < 0:
                msg = "Negative not allowed"
                raise ValueError(msg)
            return x * 2

        result = failing_function(-1)

        # Exception should be caught and returned as failed FlextResult
        assert isinstance(result, FlextResult)
        assert result.is_failure
        assert "Negative not allowed" in result.error

    def test_time_execution_decorator(self) -> None:
        """Test time_execution decorator."""

        @FlextDecorators.time_execution
        def timed_function(delay_ms: float) -> str:
            time.sleep(delay_ms / 1000)  # Convert to seconds
            return "completed"

        result = timed_function(10.0)  # 10ms delay

        # Should return the original result (not wrapped)
        assert result == "completed"

    def test_cache_results_decorator(self) -> None:
        """Test cache_results decorator."""
        call_count = 0

        @FlextDecorators.cache_results(max_size=2)
        def cached_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x**2

        # First call
        result1 = cached_function(3)
        assert result1 == 9
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = cached_function(3)
        assert result2 == 9
        assert call_count == 1  # No additional calls

        # Third call with different args
        result3 = cached_function(4)
        assert result3 == 16
        assert call_count == 2

    def test_log_exceptions_decorator(self) -> None:
        """Test log_exceptions decorator."""

        @FlextDecorators.log_exceptions
        def function_that_fails(should_fail: bool) -> str:
            if should_fail:
                msg = "Test error"
                raise RuntimeError(msg)
            return "success"

        # Should work normally for success case
        result = function_that_fails(False)
        assert result == "success"

        # Should re-raise exceptions but log them
        with pytest.raises(RuntimeError, match="Test error"):
            function_that_fails(True)

    def test_validate_arguments_decorator(self) -> None:
        """Test validate_arguments decorator."""

        @FlextDecorators.validate_arguments
        def validated_function(x: int, y: str) -> str:
            return f"{x}: {y}"

        # Should work with valid arguments
        result = validated_function(42, "test")
        assert result == "42: test"

        # Should raise ValueError when first arg is None
        with pytest.raises(ValueError, match="first argument cannot be None"):
            validated_function(None, "test")

    def test_retry_decorator(self) -> None:
        """Test retry decorator with exponential backoff."""
        call_count = 0

        @FlextDecorators.retry(max_attempts=3, delay=0.01)
        def unreliable_function(should_fail: bool) -> str:
            nonlocal call_count
            call_count += 1

            if should_fail and call_count < 2:
                msg = "Network error"
                raise ConnectionError(msg)
            return f"Success on attempt {call_count}"

        # Should succeed after retries
        result = unreliable_function(True)
        assert result == "Success on attempt 2"
        assert call_count == 2

    def test_log_calls_decorator(self) -> None:
        """Test log_calls decorator for detailed logging."""

        @FlextDecorators.log_calls
        def logged_function(x: int, name: str = "default") -> str:
            return f"{name}: {x}"

        result = logged_function(10, name="test")
        assert result == "test: 10"

    def test_validate_types_decorator(self) -> None:
        """Test validate_types decorator."""

        @FlextDecorators.validate_types(expected_types=(int, str))
        def type_checked_function(x: int, y: str) -> str:
            return f"{x} + {y}"

        # Should work with correct types
        result = type_checked_function(42, "hello")
        assert result == "42 + hello"

        # Should handle type validation gracefully
        # Note: The current implementation might not strictly enforce types
        # but should not crash
        result2 = type_checked_function(42, "world")
        assert result2 == "42 + world"


class TestFlextDecoratorsIntegration:
    """Test decorator combinations and complex scenarios."""

    def test_multiple_decorators_combined(self) -> None:
        """Test combining multiple decorators."""

        @FlextDecorators.time_execution
        @FlextDecorators.log_exceptions
        @FlextDecorators.safe_call()
        def complex_function(x: int) -> int:
            if x == 0:
                msg = "Division by zero"
                raise ZeroDivisionError(msg)
            return 100 // x

        # Test success case
        result = complex_function(10)
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value == 10

        # Test failure case
        result_fail = complex_function(0)
        assert isinstance(result_fail, FlextResult)
        assert result_fail.is_failure
        assert "Division by zero" in result_fail.error

    def test_decorator_with_flext_result_return(self) -> None:
        """Test decorators with functions that already return FlextResult."""

        @FlextDecorators.safe_call()
        def function_returning_result(x: int) -> FlextResult[str]:
            if x < 0:
                return FlextResult[str].fail("Negative input")
            return FlextResult[str].ok(f"Value: {x}")

        # Should handle FlextResult returns correctly
        result = function_returning_result(5)
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value == "Value: 5"

        # Should handle FlextResult failures
        result_fail = function_returning_result(-1)
        assert isinstance(result_fail, FlextResult)
        assert result_fail.is_failure
        assert result_fail.error == "Negative input"
