"""Behavioral tests for FlextUtilitiesReliability public contract.

Exercises the observable public surface of ``flext_core.FlextUtilitiesReliability``:
``retry``, ``try_``, ``guard_result`` and the ``m.RetryOptions`` model. Every
assertion targets return values, ``r[T]`` outcomes, raised exceptions and public
model state -- never private attributes or internal collaborators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from tests import m, r, tm, u

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests import p


def _counting_operation(
    fail_before: int,
    success_value: int,
) -> tuple[Callable[[], p.Result[int]], list[int]]:
    """Build an operation that fails ``fail_before`` times then succeeds.

    Returns the operation plus the shared attempts log so callers can assert how
    many times the public ``retry`` contract invoked the operation.

    Returns:
        The resulting ``tuple[Callable[[], p.Result[int]], list[int]]``.

    """
    attempts: list[int] = []

    def op() -> p.Result[int]:
        attempts.append(len(attempts))
        if len(attempts) >= fail_before:
            return r[int].ok(success_value)
        return r[int].fail("transient")

    return op, attempts


def _raising_result_operation(exc: Exception) -> Callable[[], p.Result[int]]:
    """Build a Result-returning operation that raises ``exc`` on every call.

    Returns:
        The resulting ``Callable[[], p.Result[int]]``.

    """

    def op() -> p.Result[int]:
        raise exc

    return op


def _raising_value_operation(exc: Exception) -> Callable[[], int]:
    """Build a plain-value operation that raises ``exc`` on every call.

    Returns:
        The resulting ``Callable[[], int]``.

    """

    def op() -> int:
        raise exc

    return op


class TestsFlextCoreUtilitiesReliability:
    """Behavioral coverage of the reliability utility public contract."""

    SUCCESS_VALUE: Final[int] = 42

    # -- retry ------------------------------------------------------------

    def test_retry_returns_first_attempt_result_when_operation_succeeds(self) -> None:
        """Test retry returns first attempt result when operation succeeds."""
        op, attempts = _counting_operation(
            fail_before=1,
            success_value=self.SUCCESS_VALUE,
        )

        result: p.Result[int] = u.retry(op, max_attempts=3, delay_seconds=0.0)

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=self.SUCCESS_VALUE)
        tm.that(len(attempts), eq=1)

    def test_retry_recovers_after_transient_failures(self) -> None:
        """Test retry recovers after transient failures."""
        op, attempts = _counting_operation(
            fail_before=3,
            success_value=self.SUCCESS_VALUE,
        )

        result: p.Result[int] = u.retry(op, max_attempts=5, delay_seconds=0.0)

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=self.SUCCESS_VALUE)
        tm.that(len(attempts), eq=3)

    def test_retry_reports_failure_after_exhausting_attempts(self) -> None:
        """Test retry reports failure after exhausting attempts."""
        op, attempts = _counting_operation(
            fail_before=99,
            success_value=self.SUCCESS_VALUE,
        )

        result: p.Result[int] = u.retry(op, max_attempts=3, delay_seconds=0.0)

        tm.that(result.failure, truth=True)
        error: str = result.error or ""
        tm.that("failed after 3 attempts" in error, truth=True)
        tm.that("transient" in error, truth=True)
        tm.that(len(attempts), eq=3)

    def test_retry_catches_retryable_exception_then_succeeds(self) -> None:
        """Test retry catches retryable exception then succeeds."""
        attempts: list[int] = []
        transient = ValueError("boom")
        expected_attempts = 2

        def op() -> p.Result[int]:
            attempts.append(len(attempts))
            if len(attempts) < expected_attempts:
                raise transient
            return r[int].ok(self.SUCCESS_VALUE)

        result: p.Result[int] = u.retry(op, max_attempts=4, delay_seconds=0.0)

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=self.SUCCESS_VALUE)
        tm.that(len(attempts), eq=2)

    @staticmethod
    def test_retry_surfaces_exception_message_when_all_attempts_raise() -> None:
        """Test retry surfaces exception message when all attempts raise."""
        result: p.Result[int] = u.retry(
            _raising_result_operation(RuntimeError("still broken")),
            max_attempts=2,
            delay_seconds=0.0,
        )

        tm.that(result.failure, truth=True)
        tm.that("still broken" in (result.error or ""), truth=True)

    @staticmethod
    @pytest.mark.parametrize("invalid_attempts", [0, -1])
    def test_retry_rejects_non_positive_max_attempts(
        invalid_attempts: int,
    ) -> None:
        """Test retry rejects non positive max attempts."""
        result: p.Result[int] = u.retry(
            lambda: r[int].fail("unused"),
            max_attempts=invalid_attempts,
            delay_seconds=0.0,
        )

        tm.that(result.failure, truth=True)
        tm.that("max_attempts" in (result.error or ""), truth=True)

    def test_retry_accepts_configuration_via_options_model(self) -> None:
        """Test retry accepts configuration via options model."""
        op, attempts = _counting_operation(
            fail_before=2,
            success_value=self.SUCCESS_VALUE,
        )
        options = m.RetryOptions(max_attempts=3, delay_seconds=0.0)

        result: p.Result[int] = u.retry(op, options)

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=self.SUCCESS_VALUE)
        tm.that(len(attempts), eq=2)

    # -- try_ -------------------------------------------------------------

    def test_try_wraps_return_value_into_success_result(self) -> None:
        """Test try wraps return value into success result."""
        result: p.Result[int] = u.try_(lambda: self.SUCCESS_VALUE)

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=self.SUCCESS_VALUE)

    @staticmethod
    def test_try_translates_configured_exception_into_labeled_failure() -> None:
        """Test try translates configured exception into labeled failure."""
        result: p.Result[int] = u.try_(
            _raising_value_operation(ValueError("boom")),
            catch=ValueError,
            op_name="parse",
        )

        tm.that(result.failure, truth=True)
        error: str = result.error or ""
        tm.that("parse" in error, truth=True)
        tm.that("boom" in error, truth=True)

    @staticmethod
    def test_try_propagates_exception_outside_configured_catch() -> None:
        """Test try propagates exception outside configured catch."""
        with pytest.raises(KeyError):
            u.try_(_raising_value_operation(KeyError("k")), catch=ValueError)

    # -- guard_result -----------------------------------------------------

    def test_guard_result_propagates_success_result_unchanged(self) -> None:
        """Test guard result propagates success result unchanged."""
        result: p.Result[int] = u.guard_result(lambda: r[int].ok(self.SUCCESS_VALUE))

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=self.SUCCESS_VALUE)

    @staticmethod
    def test_guard_result_propagates_failure_result_unchanged() -> None:
        """Test guard result propagates failure result unchanged."""
        result: p.Result[int] = u.guard_result(lambda: r[int].fail("domain error"))

        tm.that(result.failure, truth=True)
        tm.that(result.error, eq="domain error")

    @staticmethod
    def test_guard_result_translates_exception_into_labeled_failure() -> None:
        """Test guard result translates exception into labeled failure."""
        result: p.Result[int] = u.guard_result(
            _raising_result_operation(RuntimeError("io down")),
            catch=RuntimeError,
            op_name="io",
        )

        tm.that(result.failure, truth=True)
        error: str = result.error or ""
        tm.that("io" in error, truth=True)
        tm.that("io down" in error, truth=True)

    @staticmethod
    def test_guard_result_propagates_exception_outside_configured_catch() -> None:
        """Test guard result propagates exception outside configured catch."""
        with pytest.raises(KeyError):
            u.guard_result(_raising_result_operation(KeyError("k")), catch=ValueError)

    # -- RetryOptions model ----------------------------------------------

    @staticmethod
    def test_retry_options_defaults_are_none() -> None:
        """Test retry options defaults are none."""
        options = m.RetryOptions()

        tm.that(options.max_attempts, none=True)
        tm.that(options.delay_seconds, none=True)

    @staticmethod
    def test_retry_options_round_trips_through_model_dump() -> None:
        """Test retry options round trips through model dump."""
        options = m.RetryOptions(max_attempts=5, delay_seconds=0.5)

        tm.that(options.model_dump(), eq={"max_attempts": 5, "delay_seconds": 0.5})
