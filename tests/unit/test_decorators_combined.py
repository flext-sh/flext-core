"""Behavioral tests for the combined and stacked decorator surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings, strategies as st

from tests import d, e, r, tm

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreDecoratorsCombined:
    """Assert observable behavior of ``d.combined`` and decorator stacking."""

    @staticmethod
    def test_combined_without_railway_returns_raw_value() -> None:
        # Arrange
        """Test combined without railway returns raw value."""

        @d.combined(operation_name="test_op", track_perf=True)
        def simple_function() -> str:
            return "result"

        # Act
        outcome = simple_function()

        # Assert: non-railway composition is a transparent pass-through.
        tm.that(outcome, eq="result")

    @staticmethod
    def test_combined_with_railway_wraps_success_in_result() -> None:
        # Arrange
        """Test combined with railway wraps success in result."""

        @d.combined(operation_name="wrapped", railway_enabled=True)
        def operation() -> str:
            return "success"

        # Act
        outcome = operation()

        # Assert
        tm.ok(outcome, eq="success")

    @staticmethod
    def test_combined_with_railway_captures_exception_as_failure() -> None:
        # Arrange
        """Test combined with railway captures exception as failure."""

        @d.combined(
            operation_name="failing",
            railway_enabled=True,
            railway_error_code="COMBINED_FAIL",
        )
        def operation() -> str:
            error_msg = "boom"
            raise ValueError(error_msg)

        # Act
        outcome = operation()

        # Assert: failure is delivered as a value, not a raised exception.
        tm.that(outcome.failure, truth=True)
        tm.that(outcome.error_code, eq="COMBINED_FAIL")
        tm.that(outcome.error, none=False)
        tm.that("boom" in outcome.error, truth=True)

    @staticmethod
    def test_combined_preserves_wrapped_callable_name() -> None:
        # Arrange
        """Test combined preserves wrapped callable name."""

        @d.combined(operation_name="named_op")
        def business_operation() -> str:
            return "ok"

        # Assert: functools.wraps identity is part of the public contract.
        tm.that(business_operation.__name__, eq="business_operation")

    @staticmethod
    def test_railway_preserves_a_preexisting_result() -> None:
        """A preexisting result retains its identity and successful payload."""
        original = r[str].ok("already_wrapped")

        @d.railway()
        def returns_result() -> p.Result[str]:
            return original

        outcome = returns_result()

        tm.that(outcome is original, eq=True)
        tm.that(tm.ok(outcome), eq="already_wrapped")

    @staticmethod
    def test_retry_succeeds_after_transient_failure() -> None:
        # Arrange
        """Test retry succeeds after transient failure."""
        attempts = 0

        @d.retry(max_attempts=2, delay_seconds=0.001)
        def flaky_method() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                message = "First attempt fails"
                raise RuntimeError(message)
            return "success"

        # Act
        outcome = flaky_method()

        # Assert: the operation recovers and the retry count is observable.
        tm.that(outcome, eq="success")
        tm.that(attempts, eq=2)

    @staticmethod
    def test_retry_raises_flext_timeout_when_exhausted() -> None:
        # Arrange
        """Test retry raises flext timeout when exhausted."""

        @d.retry(max_attempts=2, delay_seconds=0.001)
        def always_failing() -> str:
            error_msg = "persistent failure"
            raise RuntimeError(error_msg)

        # Act / Assert: exhaustion surfaces as a typed FlextExceptions member.
        with pytest.raises(e.FlextTimeoutError) as exc_info:
            always_failing()
        tm.that(exc_info.value.operation, eq="always_failing")

    @staticmethod
    def test_stacking_log_operation_over_railway_returns_result() -> None:
        # Arrange
        """Test stacking log operation over railway returns result."""

        @d.log_operation("stacked")
        @d.log_operation("stacked")
        @d.railway()
        def stacked_operation() -> str:
            return "stacked_result"

        # Act
        outcome = stacked_operation()

        # Assert: the innermost railway shape survives the logging layers.
        tm.ok(outcome, eq="stacked_result")

    @staticmethod
    def test_stacking_railway_over_retry_recovers_then_wraps() -> None:
        # Arrange
        """Test stacking railway over retry recovers then wraps."""
        attempts = 0

        @d.railway()
        @d.retry(max_attempts=3, delay_seconds=0.001)
        def flaky_with_railway() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                error_msg = "Retry me"
                raise RuntimeError(error_msg)
            return "success"

        # Act
        outcome = flaky_with_railway()

        # Assert
        tm.ok(outcome, eq="success")
        tm.that(attempts, eq=2)

    @pytest.mark.parametrize(
        ("timeout_seconds", "should_raise"),
        [(5.0, False), (0.001, True)],
    )
    @staticmethod
    def test_timeout_enforces_duration_budget(
        timeout_seconds: float,
        *,
        should_raise: bool,
    ) -> None:
        # Arrange
        """Test timeout enforces duration budget."""

        @d.timeout(timeout_seconds=timeout_seconds)
        def measured() -> str:
            time.sleep(0.02)
            return "done"

        # Act / Assert
        if should_raise:
            with pytest.raises(e.FlextTimeoutError) as exc_info:
                measured()
            tm.that(exc_info.value.operation, eq="measured")
        else:
            tm.that(measured(), eq="done")

    @staticmethod
    @given(a=st.integers(), b=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50)
    def test_railway_division_always_returns_success_result(
        a: int,
        b: int,
    ) -> None:
        # Arrange
        """Test railway division always returns success result."""

        @d.railway(error_code="DIV")
        def divide(x: int, y: int) -> float:
            return x / y

        # Act
        outcome = divide(a, b)

        # Assert: with a non-zero divisor the railway always succeeds cleanly.
        tm.that(outcome.success, truth=True)
        tm.ok(outcome, eq=a / b)
