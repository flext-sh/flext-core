"""Behavioral tests for the combined and stacked decorator surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from flext_tests import d, e, r
from hypothesis import given, settings, strategies as st

from tests.utilities import TestsFlextUtilities, u

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextCoreDecoratorsCombined(u.TestsFlextDecoratorsLegacy):
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
        assert outcome == "result"

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
        assert isinstance(outcome, r)
        assert u.Tests.assert_success(outcome, expected_value="success") == "success"

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
        assert isinstance(outcome, r)
        assert outcome.failure
        assert outcome.error_code == "COMBINED_FAIL"
        assert outcome.error is not None
        assert "boom" in outcome.error

    @staticmethod
    def test_combined_preserves_wrapped_callable_name() -> None:
        # Arrange
        """Test combined preserves wrapped callable name."""

        @d.combined(operation_name="named_op")
        def business_operation() -> str:
            return "ok"

        # Assert: functools.wraps identity is part of the public contract.
        assert business_operation.__name__ == "business_operation"

    @staticmethod
    def test_railway_nests_a_preexisting_result() -> None:
        # Arrange
        """Test railway nests a preexisting result."""

        @d.railway()
        def returns_result() -> p.Result[str]:
            return r[str].ok("already_wrapped")

        # Act
        outcome = returns_result()

        # Assert: railway wraps the returned value verbatim, producing r[r[str]].
        inner = u.Tests.assert_success(outcome)
        assert isinstance(inner, r)
        assert u.Tests.assert_success(inner, expected_value="already_wrapped")

    def test_retry_succeeds_after_transient_failure(self) -> None:
        # Arrange
        """Test retry succeeds after transient failure."""
        service = self.ServiceWithLogger()

        @d.retry(max_attempts=2, delay_seconds=0.001)
        def flaky_method() -> str:
            return service.flaky_method()

        # Act
        outcome = flaky_method()

        # Assert: the operation recovers and the retry count is observable.
        assert outcome == "success"
        assert service.attempts == 2

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
        assert exc_info.value.operation == "always_failing"

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
        assert isinstance(outcome, r)
        assert u.Tests.assert_success(outcome, expected_value="stacked_result")

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
        assert isinstance(outcome, r)
        assert u.Tests.assert_success(outcome, expected_value="success")
        assert attempts == 2

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
            assert exc_info.value.operation == "measured"
        else:
            assert measured() == "done"

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
        assert isinstance(outcome, r)
        assert outcome.success
        assert u.Tests.assert_success(outcome) == a / b
