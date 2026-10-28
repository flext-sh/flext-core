"""Behavioral tests for the combined and stacked decorator surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import pytest
from hypothesis import given, settings, strategies as st

from flext_core import FlextContainer
from tests import d, p, r, tm


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
        tm.that("boom" in str(outcome.error), truth=True)

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
    @pytest.mark.parametrize("succeeds", [True, False])
    @pytest.mark.parametrize(
        "decorator",
        [d.railway(), d.combined(railway_enabled=True)],
    )
    def test_railway_preserves_a_preexisting_result(
        decorator: p.RailwayDecorator,
        *,
        succeeds: bool,
    ) -> None:
        """Existing success and failure results retain their original identity."""
        original = (
            r[str].ok("already_wrapped") if succeeds else r[str].fail("already_failed")
        )

        @decorator
        def returns_result() -> p.Result[str]:
            return original

        outcome: p.Result[str] = returns_result()

        tm.ok(outcome is original)
        if succeeds:
            tm.ok(outcome, eq="already_wrapped")
        else:
            tm.fail(outcome)

    @staticmethod
    @pytest.mark.parametrize(
        "decorator",
        [d.railway(), d.combined(railway_enabled=True)],
    )
    def test_railway_normalizes_a_mixed_return_contract(
        decorator: p.RailwayDecorator,
    ) -> None:
        """A mixed raw/result callable always exposes one result layer."""
        original = r[str].ok("wrapped")

        @decorator
        def returns_mixed(*, prewrapped: bool) -> str | p.Result[str]:
            return original if prewrapped else "raw"

        wrapped: p.Result[str] = returns_mixed(prewrapped=True)
        raw: p.Result[str] = returns_mixed(prewrapped=False)

        tm.ok(wrapped is original)
        tm.ok(raw, eq="raw")

    @staticmethod
    def test_combined_injects_dependencies_from_container(
        clean_container: p.Container,
    ) -> None:
        """Test combined applies ``inject_deps`` on the standard path."""
        _ = clean_container
        di = FlextContainer.shared()
        _ = di.bind("answer.service", 42)

        @d.combined(inject_deps={"dep": "answer.service"}, operation_name="std")
        def fn(*, dep: int = 0) -> int:
            return dep + 1

        tm.that(fn(), eq=43)

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
        succeeding_attempt = 2

        @d.railway()
        @d.retry(max_attempts=3, delay_seconds=0.001)
        def flaky_with_railway() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < succeeding_attempt:
                error_msg = "Retry me"
                raise RuntimeError(error_msg)
            return "success"

        # Act
        outcome = flaky_with_railway()

        # Assert
        tm.ok(outcome, eq="success")
        tm.that(attempts, eq=succeeding_attempt)

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
