"""Behavioral tests for the railway, retry, and timeout decorators.

Every assertion targets the observable public contract of ``d.railway``,
``d.retry``, and ``d.timeout``: the returned ``r[T]`` outcome, the raw return
value, or the ``e.FlextTimeoutError`` raised to the caller. No private attribute,
internal collaborator, or logging side effect is inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

import pytest
from flext_tests import d, e, r

from flext_core import c


class TestsFlextCoreDecoratorsRailwayRetry:
    """Public-contract behavior of the railway/retry/timeout decorators."""

    # ------------------------------------------------------------------ railway
    @staticmethod
    def test_railway_wraps_return_value_in_success_result() -> None:
        # Arrange
        """Test railway wraps return value in success result."""

        @d.railway()
        def successful_operation() -> str:
            return "success"

        # Act
        result = successful_operation()

        # Assert
        assert isinstance(result, r)
        assert result.success is True
        assert result.unwrap() == "success"

    @staticmethod
    def test_railway_converts_raised_exception_into_failure_result() -> None:
        # Arrange
        """Test railway converts raised exception into failure result."""

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
        [(None, c.ErrorCode.OPERATION_ERROR.value), ("CUSTOM_ERROR", "CUSTOM_ERROR")],
    )
    def test_railway_failure_carries_expected_error_code(
        self,
        error_code: str | None,
        expected_code: str,
    ) -> None:
        # Arrange
        """Test railway failure carries expected error code."""

        @d.railway(error_code=error_code)
        def failing_operation() -> str:
            error_msg = "boom"
            raise RuntimeError(error_msg)

        # Act
        result = failing_operation()

        # Assert
        assert result.failure is True
        assert result.error_code == expected_code

    @staticmethod
    def test_railway_forwards_arguments_and_supports_result_chaining() -> None:
        # Arrange
        """Test railway forwards arguments and supports result chaining."""

        @d.railway()
        def add(left: int, right: int) -> int:
            return left + right

        # Act
        chained = add(2, 3).map(lambda total: total * 10)

        # Assert
        assert chained.success is True
        assert chained.unwrap() == 50

    # -------------------------------------------------------------------- retry
    @staticmethod
    def test_retry_returns_value_when_operation_succeeds_immediately() -> None:
        # Arrange
        """Test retry returns value when operation succeeds immediately."""
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

    @staticmethod
    def test_retry_recovers_after_transient_failures() -> None:
        # Arrange
        """Test retry recovers after transient failures."""
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

    @staticmethod
    def test_retry_forwards_arguments_to_wrapped_callable() -> None:
        # Arrange
        """Test retry forwards arguments to wrapped callable."""

        @d.retry(max_attempts=2, delay_seconds=0.001)
        def concat(prefix: str, *, suffix: str) -> str:
            return f"{prefix}-{suffix}"

        # Act
        value = concat("a", suffix="b")

        # Assert
        assert value == "a-b"

    @staticmethod
    def test_retry_raises_timeout_error_when_attempts_are_exhausted() -> None:
        # Arrange
        """Test retry raises timeout error when attempts are exhausted."""

        @d.retry(max_attempts=2, delay_seconds=0.001)
        def always_fails() -> str:
            error_msg = "Always fails"
            raise ValueError(error_msg)

        # Act / Assert
        with pytest.raises(
            e.FlextTimeoutError,
            match="failed after 2 attempts",
        ) as info:
            always_fails()
        assert info.value.operation == "always_fails"

    # ------------------------------------------------------------------ timeout
    @staticmethod
    def test_timeout_returns_value_when_operation_is_fast_enough() -> None:
        # Arrange
        """Test timeout returns value when operation is fast enough."""

        @d.timeout(timeout_seconds=1.0)
        def fast_operation() -> str:
            time.sleep(0.01)
            return "completed"

        # Act
        value = fast_operation()

        # Assert
        assert value == "completed"

    @staticmethod
    def test_timeout_raises_timeout_error_when_duration_exceeded() -> None:
        # Arrange
        """Test timeout raises timeout error when duration exceeded."""

        @d.timeout(timeout_seconds=0.005)
        def slow_operation() -> str:
            time.sleep(0.05)
            return "should_not_reach"

        # Act / Assert
        with pytest.raises(e.FlextTimeoutError) as info:
            slow_operation()
        assert info.value.operation == "slow_operation"

    @staticmethod
    def test_timeout_uses_default_error_code_when_none_provided() -> None:
        # Arrange
        """Test timeout uses default error code when none provided."""

        @d.timeout(timeout_seconds=0.005)
        def slow_operation() -> str:
            time.sleep(0.05)
            return "should_not_reach"

        # Act / Assert
        with pytest.raises(e.FlextTimeoutError) as info:
            slow_operation()
        assert info.value.error_code == c.ErrorCode.TIMEOUT_ERROR.value
