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
from flext_tests import d, e, r

from flext_core.container import FlextContainer


class TestsFlextCoreDecorators:
    """Public behavioral contract for FLEXT infrastructure decorators."""

    def test_railway_wraps_success_in_ok_result(self) -> None:
        """Railway returns an ``r`` carrying the callable's value on success."""

        @d.railway()
        def compute() -> str:
            return "value"

        outcome = compute()

        assert isinstance(outcome, r)
        assert outcome.success
        assert outcome.value == "value"

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
        self,
        raised: Exception,
        message: str,
    ) -> None:
        """Railway captures caught exceptions as a failure ``r`` with the code."""

        @d.railway(error_code="CUSTOM_CODE")
        def failing() -> str:
            raise raised

        outcome = failing()

        assert isinstance(outcome, r)
        assert outcome.failure
        assert outcome.error_code == "CUSTOM_CODE"
        assert outcome.error is not None
        assert message in outcome.error

    def test_railway_defaults_error_code_when_omitted(self) -> None:
        """Railway supplies a default error code when none is provided."""

        @d.railway()
        def failing() -> int:
            msg = "nope"
            raise ValueError(msg)

        outcome = failing()

        assert outcome.failure
        assert outcome.error_code == "OPERATION_ERROR"

    def test_retry_returns_value_on_first_success(self) -> None:
        """Retry returns the raw value without extra attempts on success."""
        calls = 0

        @d.retry(max_attempts=3, delay_seconds=0.001)
        def once() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        assert once() == "ok"
        assert calls == 1

    def test_retry_recovers_after_transient_failures(self) -> None:
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

        assert flaky() == "recovered"
        assert attempts == 3

    def test_retry_raises_timeout_error_when_exhausted(self) -> None:
        """Retry raises ``e.TimeoutError`` carrying the operation name."""

        @d.retry(max_attempts=2, delay_seconds=0.001)
        def always_fails() -> str:
            msg = "permanent"
            raise ValueError(msg)

        with pytest.raises(e.TimeoutError) as excinfo:
            always_fails()

        assert excinfo.value.operation == "always_fails"
        assert "failed after 2 attempts" in str(excinfo.value)

    def test_timeout_returns_value_when_within_budget(self) -> None:
        """Timeout passes the value through when the callable is fast enough."""

        @d.timeout(timeout_seconds=5.0)
        def quick() -> str:
            return "done"

        assert quick() == "done"

    def test_timeout_raises_timeout_error_when_exceeded(self) -> None:
        """Timeout raises ``e.TimeoutError`` naming the slow operation."""

        @d.timeout(timeout_seconds=0.0)
        def slow() -> int:
            return sum(range(100000))

        with pytest.raises(e.TimeoutError) as excinfo:
            slow()

        assert excinfo.value.operation == "slow"

    def test_inject_supplies_dependency_from_container(self) -> None:
        """Inject resolves the named service and passes it as the kwarg."""
        _ = FlextContainer.shared().bind("decorators_test_dep", 41)

        @d.inject(dependency="decorators_test_dep")
        def add_one(*, dependency: int = 0) -> int:
            return dependency + 1

        assert add_one() == 42

    def test_inject_does_not_override_explicit_argument(self) -> None:
        """Inject leaves a caller-supplied kwarg untouched."""
        _ = FlextContainer.shared().bind("decorators_test_dep", 41)

        @d.inject(dependency="decorators_test_dep")
        def echo(*, dependency: int = 0) -> int:
            return dependency

        assert echo(dependency=99) == 99

    def test_deprecated_emits_warning_and_returns_value(self) -> None:
        """Deprecated warns with ``DeprecationWarning`` yet still runs the call."""

        @d.deprecated("use the replacement")
        def legacy() -> str:
            return "still works"

        with pytest.warns(DeprecationWarning, match="use the replacement"):
            assert legacy() == "still works"

    def test_factory_returns_same_callable_that_still_runs(self) -> None:
        """Factory marks without wrapping: the callable identity and result hold."""

        def build() -> int:
            return 7

        marked = d.factory("builder")(build)

        assert marked is build
        assert marked() == 7

    def test_combined_railway_wraps_success_in_result(self) -> None:
        """Combined with railway enabled returns an ``r`` value on success."""

        @d.combined(railway_enabled=True, railway_error_code="COMBINED_ERR")
        def add(a: int, b: int) -> int:
            return a + b

        outcome = add(2, 3)

        assert isinstance(outcome, r)
        assert outcome.success
        assert outcome.value == 5

    def test_combined_railway_reports_failure_with_error_code(self) -> None:
        """Combined with railway enabled maps exceptions to a failure ``r``."""

        @d.combined(railway_enabled=True, railway_error_code="COMBINED_ERR")
        def boom() -> int:
            msg = "broken"
            raise ValueError(msg)

        outcome = boom()

        assert isinstance(outcome, r)
        assert outcome.failure
        assert outcome.error_code == "COMBINED_ERR"

    def test_combined_without_railway_returns_raw_value(self) -> None:
        """Combined without railway returns the callable's plain value."""

        @d.combined()
        def double(x: int) -> int:
            return x * 2

        assert double(21) == 42

    def test_log_operation_returns_value_and_propagates_errors(self) -> None:
        """log_operation is transparent: passes values, re-raises exceptions."""

        @d.log_operation(operation_name="unit-op")
        def succeed(x: int) -> int:
            return x + 1

        assert succeed(4) == 5

        @d.log_operation()
        def fail() -> int:
            msg = "surfaced"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="surfaced"):
            fail()

    def test_with_correlation_is_transparent_to_return_value(self) -> None:
        """with_correlation runs the callable and returns its value unchanged."""

        @d.with_correlation()
        def work() -> str:
            return "correlated"

        assert work() == "correlated"

    def test_decorators_preserve_callable_name(self) -> None:
        """Wrapping decorators keep ``__name__`` via ``functools.wraps``."""

        @d.railway()
        def named_operation() -> int:
            return 1

        assert named_operation.__name__ == "named_operation"
