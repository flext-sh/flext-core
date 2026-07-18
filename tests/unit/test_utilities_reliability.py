"""Behavioral tests for FlextUtilitiesReliability public contract.

Exercises the observable public surface of ``flext_core.FlextUtilitiesReliability``:
``retry``, ``try_``, ``guard_result`` and the ``RetryOptions`` model. Every
assertion targets return values, ``r[T]`` outcomes, raised exceptions and public
model state -- never private attributes or internal collaborators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from flext_tests import r

from tests import u

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests import p


def _counting_operation(
    fail_before: int, success_value: int
) -> tuple[Callable[[], p.Result[int]], list[int]]:
    """Build an operation that fails ``fail_before`` times then succeeds.

    Returns the operation plus the shared attempts log so callers can assert how
    many times the public ``retry`` contract invoked the operation.
    """
    attempts: list[int] = []

    def op() -> p.Result[int]:
        attempts.append(len(attempts))
        if len(attempts) >= fail_before:
            return r[int].ok(success_value)
        return r[int].fail("transient")

    return op, attempts


def _raising_result_operation(exc: Exception) -> Callable[[], p.Result[int]]:
    """Build a Result-returning operation that raises ``exc`` on every call."""

    def op() -> p.Result[int]:
        raise exc

    return op


def _raising_value_operation(exc: Exception) -> Callable[[], int]:
    """Build a plain-value operation that raises ``exc`` on every call."""

    def op() -> int:
        raise exc

    return op


class TestsFlextCoreUtilitiesReliability:
    """Behavioral coverage of the reliability utility public contract."""

    SUCCESS_VALUE: Final[int] = 42

    # -- retry ------------------------------------------------------------

    def test_retry_returns_first_attempt_result_when_operation_succeeds(self) -> None:
        op, attempts = _counting_operation(
            fail_before=1, success_value=self.SUCCESS_VALUE
        )

        result: p.Result[int] = u.retry(op, max_attempts=3, delay_seconds=0.0)

        assert result.success
        assert result.value == self.SUCCESS_VALUE
        assert len(attempts) == 1

    def test_retry_recovers_after_transient_failures(self) -> None:
        op, attempts = _counting_operation(
            fail_before=3, success_value=self.SUCCESS_VALUE
        )

        result: p.Result[int] = u.retry(op, max_attempts=5, delay_seconds=0.0)

        assert result.success
        assert result.value == self.SUCCESS_VALUE
        assert len(attempts) == 3

    def test_retry_reports_failure_after_exhausting_attempts(self) -> None:
        op, attempts = _counting_operation(
            fail_before=99, success_value=self.SUCCESS_VALUE
        )

        result: p.Result[int] = u.retry(op, max_attempts=3, delay_seconds=0.0)

        assert result.failure
        error: str = result.error or ""
        assert "failed after 3 attempts" in error
        assert "transient" in error
        assert len(attempts) == 3

    def test_retry_catches_retryable_exception_then_succeeds(self) -> None:
        attempts: list[int] = []
        transient = ValueError("boom")

        def op() -> p.Result[int]:
            attempts.append(len(attempts))
            if len(attempts) < 2:
                raise transient
            return r[int].ok(self.SUCCESS_VALUE)

        result: p.Result[int] = u.retry(op, max_attempts=4, delay_seconds=0.0)

        assert result.success
        assert result.value == self.SUCCESS_VALUE
        assert len(attempts) == 2

    def test_retry_surfaces_exception_message_when_all_attempts_raise(self) -> None:
        result: p.Result[int] = u.retry(
            _raising_result_operation(RuntimeError("still broken")),
            max_attempts=2,
            delay_seconds=0.0,
        )

        assert result.failure
        assert "still broken" in (result.error or "")

    @pytest.mark.parametrize("invalid_attempts", [0, -1])
    def test_retry_rejects_non_positive_max_attempts(
        self, invalid_attempts: int
    ) -> None:
        result: p.Result[int] = u.retry(
            lambda: r[int].fail("unused"),
            max_attempts=invalid_attempts,
            delay_seconds=0.0,
        )

        assert result.failure
        assert "max_attempts" in (result.error or "")

    def test_retry_accepts_configuration_via_options_model(self) -> None:
        op, attempts = _counting_operation(
            fail_before=2, success_value=self.SUCCESS_VALUE
        )
        options = u.RetryOptions(max_attempts=3, delay_seconds=0.0)

        result: p.Result[int] = u.retry(op, options)

        assert result.success
        assert result.value == self.SUCCESS_VALUE
        assert len(attempts) == 2

    # -- try_ -------------------------------------------------------------

    def test_try_wraps_return_value_into_success_result(self) -> None:
        result: p.Result[int] = u.try_(lambda: self.SUCCESS_VALUE)

        assert result.success
        assert result.value == self.SUCCESS_VALUE

    def test_try_translates_configured_exception_into_labeled_failure(self) -> None:
        result: p.Result[int] = u.try_(
            _raising_value_operation(ValueError("boom")),
            catch=ValueError,
            op_name="parse",
        )

        assert result.failure
        error: str = result.error or ""
        assert "parse" in error
        assert "boom" in error

    def test_try_propagates_exception_outside_configured_catch(self) -> None:
        with pytest.raises(KeyError):
            u.try_(_raising_value_operation(KeyError("k")), catch=ValueError)

    # -- guard_result -----------------------------------------------------

    def test_guard_result_propagates_success_result_unchanged(self) -> None:
        result: p.Result[int] = u.guard_result(lambda: r[int].ok(self.SUCCESS_VALUE))

        assert result.success
        assert result.value == self.SUCCESS_VALUE

    def test_guard_result_propagates_failure_result_unchanged(self) -> None:
        result: p.Result[int] = u.guard_result(lambda: r[int].fail("domain error"))

        assert result.failure
        assert result.error == "domain error"

    def test_guard_result_translates_exception_into_labeled_failure(self) -> None:
        result: p.Result[int] = u.guard_result(
            _raising_result_operation(RuntimeError("io down")),
            catch=RuntimeError,
            op_name="io",
        )

        assert result.failure
        error: str = result.error or ""
        assert "io" in error
        assert "io down" in error

    def test_guard_result_propagates_exception_outside_configured_catch(self) -> None:
        with pytest.raises(KeyError):
            u.guard_result(_raising_result_operation(KeyError("k")), catch=ValueError)

    # -- RetryOptions model ----------------------------------------------

    def test_retry_options_defaults_are_none(self) -> None:
        options = u.RetryOptions()

        assert options.max_attempts is None
        assert options.delay_seconds is None

    def test_retry_options_round_trips_through_model_dump(self) -> None:
        options = u.RetryOptions(max_attempts=5, delay_seconds=0.5)

        assert options.model_dump() == {"max_attempts": 5, "delay_seconds": 0.5}
