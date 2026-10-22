"""Behavioral contract tests for the public ``FlextDecorators`` facade (``d``).

Every test asserts observable public behavior of a decorator: the value the
decorated callable returns, the ``r[T]`` railway outcome, the raised
``FlextExceptions`` member, or preserved callable metadata. No test inspects a
private attribute, patches an internal collaborator, or asserts an internal
call happened.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextContainer
from tests import c, d, e, tm


class TestsFlextCoreDecorators:
    """Public behavioral contract for FLEXT infrastructure decorators."""

    @staticmethod
    def test_railway_wraps_success_in_ok_result() -> None:
        """Railway returns an ``r`` carrying the callable's value on success."""

        @d.railway()
        def compute() -> str:
            return "value"

        outcome = compute()

        tm.that(outcome.success, truth=True)
        tm.that(outcome.value, eq="value")

    @staticmethod
    @pytest.mark.parametrize(
        ("raised", "message"),
        [
            (ValueError("bad value"), "bad value"),
            (KeyError("missing"), "missing"),
            (TypeError("wrong type"), "wrong type"),
            (RuntimeError("boom"), "boom"),
        ],
    )
    def test_railway_converts_exception_to_failure_result(
        raised: Exception,
        message: str,
    ) -> None:
        """Railway captures caught exceptions as a failure ``r`` with the code."""

        @d.railway(error_code="CUSTOM_CODE")
        def failing() -> str:
            raise raised

        outcome = failing()

        tm.that(outcome.failure, truth=True)
        tm.that(outcome.error_code, eq="CUSTOM_CODE")
        tm.that(outcome.error, none=False)
        tm.that(message in str(outcome.error), truth=True)

    @staticmethod
    def test_railway_defaults_error_code_when_omitted() -> None:
        """Railway supplies a default error code when none is provided."""

        @d.railway()
        def failing() -> int:
            msg = "nope"
            raise ValueError(msg)

        outcome = failing()

        tm.that(outcome.failure, truth=True)
        tm.that(outcome.error_code, eq=c.ErrorCode.OPERATION_ERROR.value)

    @staticmethod
    def test_retry_returns_value_on_first_success() -> None:
        """Retry returns the raw value without extra attempts on success."""
        calls = 0

        @d.retry(max_attempts=3, delay_seconds=0.001)
        def once() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        tm.that(once(), eq="ok")
        tm.that(calls, eq=1)

    @staticmethod
    def test_retry_recovers_after_transient_failures() -> None:
        """Retry re-invokes until the callable succeeds within the budget."""
        attempts = 0

        @d.retry(max_attempts=3, delay_seconds=0.001)
        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                msg = f"attempt {attempts}"
                raise RuntimeError(msg)
            return "recovered"

        tm.that(flaky(), eq="recovered")
        tm.that(attempts, eq=3)

    @staticmethod
    def test_retry_raises_timeout_error_when_exhausted() -> None:
        """Retry raises ``e.FlextTimeoutError`` carrying the operation name."""

        @d.retry(max_attempts=2, delay_seconds=0.001)
        def always_fails() -> str:
            msg = "permanent"
            raise ValueError(msg)

        with pytest.raises(e.FlextTimeoutError) as excinfo:
            always_fails()

        tm.that(excinfo.value.operation, eq="always_fails")
        tm.that("failed after 2 attempts" in str(excinfo.value), truth=True)

    @staticmethod
    def test_timeout_returns_value_when_within_budget() -> None:
        """Timeout passes the value through when the callable is fast enough."""

        @d.timeout(timeout_seconds=5.0)
        def quick() -> str:
            return "done"

        tm.that(quick(), eq="done")

    @staticmethod
    def test_timeout_raises_timeout_error_when_exceeded() -> None:
        """Timeout raises ``e.FlextTimeoutError`` naming the slow operation."""

        @d.timeout(timeout_seconds=0.0)
        def slow() -> int:
            return sum(range(100000))

        with pytest.raises(e.FlextTimeoutError) as excinfo:
            slow()

        tm.that(excinfo.value.operation, eq="slow")

    @staticmethod
    def test_inject_supplies_dependency_from_container() -> None:
        """Inject resolves the named service and passes it as the kwarg."""
        _ = FlextContainer.shared().bind("decorators_test_dep", 41)

        @d.inject(dependency="decorators_test_dep")
        def add_one(*, dependency: int = 0) -> int:
            return dependency + 1

        tm.that(add_one(), eq=42)

    @staticmethod
    def test_inject_does_not_override_explicit_argument() -> None:
        """Inject leaves a caller-supplied kwarg untouched."""
        _ = FlextContainer.shared().bind("decorators_test_dep", 41)

        @d.inject(dependency="decorators_test_dep")
        def echo(*, dependency: int = 0) -> int:
            return dependency

        tm.that(echo(dependency=99), eq=99)

    @staticmethod
    def test_deprecated_emits_warning_and_returns_value() -> None:
        """Deprecated warns with ``DeprecationWarning`` yet still runs the call."""

        @d.deprecated("use the replacement")
        def legacy() -> str:
            return "still works"

        with pytest.warns(DeprecationWarning, match="use the replacement"):
            tm.that(legacy(), eq="still works")

    @staticmethod
    def test_factory_returns_same_callable_that_still_runs() -> None:
        """Factory marks without wrapping: the callable identity and result hold."""

        def build() -> int:
            return 7

        marked = d.factory("builder")(build)

        tm.that(marked is build, truth=True)
        tm.that(marked(), eq=7)

    @staticmethod
    def test_combined_railway_wraps_success_in_result() -> None:
        """Combined with railway enabled returns an ``r`` value on success."""

        @d.combined(railway_enabled=True, railway_error_code="COMBINED_ERR")
        def add(a: int, b: int) -> int:
            return a + b

        outcome = add(2, 3)

        tm.that(outcome.success, truth=True)
        tm.that(outcome.value, eq=5)

    @staticmethod
    def test_combined_railway_reports_failure_with_error_code() -> None:
        """Combined with railway enabled maps exceptions to a failure ``r``."""

        @d.combined(railway_enabled=True, railway_error_code="COMBINED_ERR")
        def boom() -> int:
            msg = "broken"
            raise ValueError(msg)

        outcome = boom()

        tm.that(outcome.failure, truth=True)
        tm.that(outcome.error_code, eq="COMBINED_ERR")

    @staticmethod
    def test_combined_without_railway_returns_raw_value() -> None:
        """Combined without railway returns the callable's plain value."""

        @d.combined()
        def double(x: int) -> int:
            return x * 2

        tm.that(double(21), eq=42)

    @staticmethod
    def test_log_operation_returns_value_and_propagates_errors() -> None:
        """log_operation is transparent: passes values, re-raises exceptions."""

        @d.log_operation(operation_name="unit-op")
        def succeed(x: int) -> int:
            return x + 1

        tm.that(succeed(4), eq=5)

        @d.log_operation()
        def fail() -> int:
            msg = "surfaced"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="surfaced"):
            fail()

    @staticmethod
    def test_with_correlation_is_transparent_to_return_value() -> None:
        """with_correlation runs the callable and returns its value unchanged."""

        @d.with_correlation()
        def work() -> str:
            return "correlated"

        tm.that(work(), eq="correlated")

    @staticmethod
    def test_decorators_preserve_callable_name() -> None:
        """Wrapping decorators keep ``__name__`` via ``functools.wraps``."""

        @d.railway()
        def named_operation() -> int:
            return 1

        tm.that(named_operation.__name__, eq="named_operation")
