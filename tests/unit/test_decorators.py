"""Tests for FlextDecorators - simplified and focused on real API."""

from __future__ import annotations

import pytest

from flext_core import FlextDecorators


class TestFlextDecoratorsReliability:
    """Test FlextDecorators.Reliability functionality."""

    def test_safe_result_success(self) -> None:
        """Test safe_result decorator with successful function."""
        @FlextDecorators.Reliability.safe_result
        def successful_function(value: int) -> int:
            return value * 2

        result = successful_function(5)
        assert result.is_success
        assert result.value == 10

    def test_safe_result_failure(self) -> None:
        """Test safe_result decorator with failing function."""
        @FlextDecorators.Reliability.safe_result
        def failing_function() -> str:
            msg = "Test error"
            raise ValueError(msg)

        result = failing_function()
        assert result.is_failure
        assert "Test error" in (result.error or "")

    def test_safe_result_with_division_by_zero(self) -> None:
        """Test safe_result decorator with division by zero."""
        @FlextDecorators.Reliability.safe_result
        def divide(a: int, b: int) -> float:
            return a / b

        result = divide(10, 0)
        assert result.is_failure
        assert "division by zero" in (result.error or "")

    def test_retry_decorator_eventual_success(self) -> None:
        """Test retry decorator with eventual success."""
        attempt_count = 0

        @FlextDecorators.Reliability.retry(max_attempts=3)
        def flaky_function() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                msg = "Network error"
                raise ConnectionError(msg)
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3

    def test_retry_decorator_all_attempts_fail(self) -> None:
        """Test retry decorator when all attempts fail."""
        @FlextDecorators.Reliability.retry(max_attempts=2)
        def always_failing_function() -> str:
            msg = "Always fails"
            raise ValueError(msg)

        with pytest.raises(RuntimeError, match="All .* retry attempts failed"):
            always_failing_function()

    def test_retry_decorator_with_specific_exceptions(self) -> None:
        """Test retry decorator with specific exception types."""
        attempt_count = 0

        @FlextDecorators.Reliability.retry(max_attempts=2, exceptions=(ConnectionError,))
        def selective_retry_function() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                # This should be retried
                msg = "Connection failed"
                raise ConnectionError(msg)
            # This should NOT be retried
            msg = "Value error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Value error"):
            selective_retry_function()

    def test_timeout_decorator(self) -> None:
        """Test timeout decorator with fast function."""
        @FlextDecorators.Reliability.timeout(seconds=1)
        def fast_function() -> str:
            return "completed"

        result = fast_function()
        assert result == "completed"


class TestFlextDecoratorsValidation:
    """Test FlextDecorators.Validation functionality."""

    def test_validate_input_success(self) -> None:
        """Test validate_input decorator with valid input."""
        def positive_validator(value: object) -> bool:
            return isinstance(value, int) and value > 0

        @FlextDecorators.Validation.validate_input(validator=positive_validator)
        def process_positive_number(num: int) -> int:
            return num * 2

        result = process_positive_number(5)
        assert result == 10

    def test_validate_input_failure(self) -> None:
        """Test validate_input decorator with invalid input."""
        def positive_validator(value: object) -> bool:
            return isinstance(value, int) and value > 0

        @FlextDecorators.Validation.validate_input(validator=positive_validator)
        def process_positive_number(num: int) -> int:
            return num * 2

        with pytest.raises(Exception):  # Should raise validation error
            process_positive_number(-5)


class TestFlextDecoratorsPerformance:
    """Test FlextDecorators.Performance functionality."""

    def test_monitor_decorator(self) -> None:
        """Test monitor decorator functionality."""
        @FlextDecorators.Performance.monitor()
        def monitored_function(value: int) -> int:
            return value + 1

        result = monitored_function(10)
        assert result == 11

    def test_cache_decorator(self) -> None:
        """Test cache decorator functionality."""
        call_count = 0

        @FlextDecorators.Performance.cache(max_size=2)
        def expensive_function(value: int) -> int:
            nonlocal call_count
            call_count += 1
            return value * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same argument (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment


class TestFlextDecoratorsObservability:
    """Test FlextDecorators.Observability functionality."""

    def test_log_execution_decorator(self) -> None:
        """Test log_execution decorator functionality."""
        @FlextDecorators.Observability.log_execution()
        def logged_function(value: str) -> str:
            return value.upper()

        result = logged_function("hello")
        assert result == "HELLO"


class TestFlextDecoratorsIntegration:
    """Test combining multiple decorators."""

    def test_safe_result_with_retry(self) -> None:
        """Test combining safe_result and retry decorators."""
        attempt_count = 0

        @FlextDecorators.Reliability.safe_result
        @FlextDecorators.Reliability.retry(max_attempts=2)
        def flaky_operation(*, should_succeed: bool) -> str:
            nonlocal attempt_count
            attempt_count += 1
            if not should_succeed and attempt_count == 1:
                msg = "First attempt fails"
                raise ValueError(msg)
            return "success"

        # Test successful case
        result_success = flaky_operation(should_succeed=True)
        assert result_success.is_success
        assert result_success.value == "success"

        # Reset counter for failure test
        attempt_count = 0

        # Test case where retry eventually succeeds
        result_retry = flaky_operation(should_succeed=False)
        assert result_retry.is_success
        assert result_retry.value == "success"
        assert attempt_count == 2  # Should have retried once
