"""Behavioral tests for the railway, retry, and timeout decorators.

Every assertion targets the observable public contract of ``d.railway``,
``d.retry``, and ``d.timeout``: the returned ``r[T]`` outcome, the raw return
value, or the ``e.FlextTimeoutError`` raised to the caller. No private attribute,
internal collaborator, or logging side effect is inspected.
"""

from __future__ import annotations

import time

import pytest

from flext_tests import d, e, r


class TestsFlextCoreDecoratorsRailwayRetry:
    """Public-contract behavior of the railway/retry/timeout decorators."""

    # ------------------------------------------------------------------ railway
    def test_railway_wraps_return_value_in_success_result(self) -> None:
        # Arrange
        @d.railway()
        def successful_operation() -> str:
            return "success"

        # Act
        result = successful_operation()

        # Assert
        assert isinstance(result, r)
        assert result.success is True
        assert result.unwrap() == "success"

    def test_railway_converts_raised_exception_into_failure_result(self) -> None:
        # Arrange
        @d.railway()
        def failing_operation() -> str:
            error_msg = "Operation failed"
            raise ValueError(error_msg)

        # Act
        result = failing_operation()

        # Assert
        assert isinstance(result, r)
        assert result.failure is True
        assert result.error is not None
        assert "Operation failed" in result.error
        assert "failing_operation" in result.error
        assert "ValueError" in result.error

    @pytest.mark.parametrize(
        ("error_code", "expected_code"),
        [(None, "OPERATION_ERROR"), ("CUSTOM_ERROR", "CUSTOM_ERROR")],
    )
    def test_railway_failure_carries_expected_error_code(
        self, error_code: str | None, expected_code: str
    ) -> None:
        # Arrange
        @d.railway(error_code=error_code)
        def failing_operation() -> str:
            error_msg = "boom"
            raise RuntimeError(error_msg)

        # Act
        result = failing_operation()

        # Assert
        assert result.failure is True
        assert result.error_code == expected_code

    def test_railway_forwards_arguments_and_supports_result_chaining(self) -> None:
        # Arrange
        @d.railway()
        def add(left: int, right: int) -> int:
            return left + right

        # Act
        chained = add(2, 3).map(lambda total: total * 10)

        # Assert
        assert chained.success is True
        assert chained.unwrap() == 50

    # -------------------------------------------------------------------- retry
    def test_retry_returns_value_when_operation_succeeds_immediately(self) -> None:
        # Arrange
        calls = 0

        @d.retry(max_attempts=3)
        def successful_operation() -> str:
            nonlocal calls
            calls += 1
            return "success"

        # Act
        value = successful_operation()

        # Assert
        assert value == "success"
        assert calls == 1

    def test_retry_recovers_after_transient_failures(self) -> None:
        # Arrange
        attempts = 0

        @d.retry(max_attempts=3, delay_seconds=0.001)
        def flaky_operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                error_msg = f"Attempt {attempts} failed"
                raise RuntimeError(error_msg)
            return "success"

        # Act
        value = flaky_operation()

        # Assert
        assert value == "success"
        assert attempts == 3

    def test_retry_forwards_arguments_to_wrapped_callable(self) -> None:
        # Arrange
        @d.retry(max_attempts=2, delay_seconds=0.001)
        def concat(prefix: str, *, suffix: str) -> str:
            return f"{prefix}-{suffix}"

        # Act
        value = concat("a", suffix="b")

        # Assert
        assert value == "a-b"

    def test_retry_raises_timeout_error_when_attempts_are_exhausted(self) -> None:
        # Arrange
        @d.retry(max_attempts=2, delay_seconds=0.001)
        def always_fails() -> str:
            error_msg = "Always fails"
            raise ValueError(error_msg)

        # Act / Assert
        with pytest.raises(
            e.FlextTimeoutError, match="failed after 2 attempts"
        ) as info:
            always_fails()
        assert info.value.operation == "always_fails"

    # ------------------------------------------------------------------ timeout
    def test_timeout_returns_value_when_operation_is_fast_enough(self) -> None:
        # Arrange
        @d.timeout(timeout_seconds=1.0)
        def fast_operation() -> str:
            time.sleep(0.01)
            return "completed"

        # Act
        value = fast_operation()

        # Assert
        assert value == "completed"

    def test_timeout_raises_timeout_error_when_duration_exceeded(self) -> None:
        # Arrange
        @d.timeout(timeout_seconds=0.005)
        def slow_operation() -> str:
            time.sleep(0.05)
            return "should_not_reach"

        # Act / Assert
        with pytest.raises(e.FlextTimeoutError) as info:
            slow_operation()
        assert info.value.operation == "slow_operation"

    def test_timeout_uses_default_error_code_when_none_provided(self) -> None:
        # Arrange
        @d.timeout(timeout_seconds=0.005)
        def slow_operation() -> str:
            time.sleep(0.05)
            return "should_not_reach"

        # Act / Assert
        with pytest.raises(e.FlextTimeoutError) as info:
            slow_operation()
        assert info.value.error_code == "OPERATION_TIMEOUT"
