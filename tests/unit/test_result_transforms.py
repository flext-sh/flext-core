"""Behavioral tests for FlextResult transform combinators.

Every test asserts observable public contract only: the ``r[T]`` outcome
(``ok``/``fail``, ``.value``, ``.error``), combinator return values, side-effect
ordering, and result identity. No private attributes, no internal collaborators,
no patching of the unit under test.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import r, tm, u

if TYPE_CHECKING:
    from collections.abc import MutableSequence

from tests import p


def _floor_divide(dividend: int, divisor: int) -> int:
    """Divide two integers, raising ``ZeroDivisionError`` when the divisor is zero.

    Returns:
        The resulting ``int``.

    """
    return dividend // divisor


class TestsFlextResultTransforms:
    """Public-contract behavior of FlextResult transform combinators."""

    @staticmethod
    def test_safe_wraps_successful_call_as_success() -> None:
        """Safe returns a success carrying the wrapped function's return value."""
        result: p.Result[int] = r.safe(_floor_divide)(10, 2)

        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq=5)

    @staticmethod
    def test_safe_captures_raised_exception_as_failure() -> None:
        """Safe converts a raised exception into a failure preserving its message."""
        result: p.Result[int] = r.safe(_floor_divide)(10, 0)

        tm.fail(result)
        tm.that(result.error, eq="integer division or modulo by zero")

    @staticmethod
    def test_map_error_rewrites_failure_message() -> None:
        """map_error transforms the error of a failure and preserves its state."""
        cause = ValueError("root cause")
        result: p.Result[str] = r[str].fail(
            "original error",
            error_code="E_ORIGINAL",
            error_data={"key": "value"},
            exception=cause,
        )

        transformed = result.map_error(lambda e: f"PREFIX: {e}")

        tm.fail(transformed)
        tm.that(transformed.error, eq="PREFIX: original error")
        tm.that(transformed.error_code, eq=result.error_code)
        tm.that(transformed.error_data, eq=result.error_data)
        tm.that(transformed.exception is cause, truth=True)

    @staticmethod
    def test_map_error_leaves_success_untouched() -> None:
        """map_error is a no-op on a success result."""
        success: p.Result[str] = r[str].ok("value")

        unchanged = success.map_error(lambda e: f"PREFIX: {e}")

        tm.ok(unchanged, eq="value")
        tm.that(unchanged is success, truth=True)

    @staticmethod
    def test_recover_converts_failure_to_success() -> None:
        """Recover maps a failure's error into a success value."""
        failure: p.Result[int] = r[int].fail("missing")

        recovered = failure.recover(len)

        _ = u.Tests.assert_success(recovered)
        tm.that(recovered.value, eq=7)

    @staticmethod
    def test_recover_leaves_success_untouched() -> None:
        """Recover never fires on a success result."""
        success: p.Result[int] = r[int].ok(3)

        unchanged = success.recover(lambda _error: 99)

        _ = u.Tests.assert_success(unchanged)
        tm.that(unchanged.value, eq=3)

    @staticmethod
    def test_tap_runs_side_effect_on_success_only() -> None:
        """Tap fires its callback for a success and returns the same result."""
        events: MutableSequence[str] = []
        success: p.Result[str] = r[str].ok("ready")

        tapped = success.tap(events.append)

        tm.ok(tapped is success)
        tm.that(tuple(events), eq=("ready",))

    @staticmethod
    def test_tap_is_noop_on_failure() -> None:
        """Tap does not fire on a failure and returns the same result."""
        events: MutableSequence[str] = []
        failure: p.Result[str] = r[str].fail("broken")

        tapped = failure.tap(events.append)

        tm.ok(tapped is failure)
        tm.that(tuple(events), eq=())

    @staticmethod
    def test_tap_error_runs_side_effect_on_failure_only() -> None:
        """tap_error fires its callback for a failure and returns the same result."""
        errors: MutableSequence[str] = []
        failure: p.Result[str] = r[str].fail("broken")

        tapped = failure.tap_error(errors.append)

        tm.ok(tapped is failure)
        tm.that(tuple(errors), eq=("broken",))

    @staticmethod
    def test_tap_error_is_noop_on_success() -> None:
        """tap_error does not fire on a success and returns the same result."""
        errors: MutableSequence[str] = []
        success: p.Result[str] = r[str].ok("ready")

        tapped = success.tap_error(errors.append)

        tm.ok(tapped is success)
        tm.that(tuple(errors), eq=())

    @staticmethod
    def test_filter_leaves_failure_unchanged() -> None:
        """Filter never runs its predicate on a failure and preserves the error."""
        result: p.Result[int] = r[int].fail("error")

        threshold = 5
        filtered = result.filter(lambda x: x > threshold)

        tm.fail(filtered)
        tm.that(filtered.error, eq="error")

    @staticmethod
    def test_map_transforms_success_value() -> None:
        """Map applies the function to a success value."""
        result = r[int].ok(3).map(lambda x: x + 1)

        tm.ok(result, eq=4)

    @staticmethod
    def test_map_returning_none_reports_failure() -> None:
        """``map`` producing ``None`` fails closed instead of empty success."""
        result = r[str].ok("x").map(lambda _: None)

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, none=False)
        tm.that("cannot be None" in str(result.error), truth=True)

    @staticmethod
    def test_map_is_noop_on_failure() -> None:
        """Map skips the function and preserves the error on a failure."""
        result = r[int].fail("boom").map(lambda x: x + 1)

        tm.fail(result)
        tm.that(result.error, eq="boom")

    @staticmethod
    def test_flat_map_chains_fallible_success() -> None:
        """flat_map chains a result-returning function on a success."""
        result = r[int].ok(3).flat_map(lambda x: r[int].ok(x * 2))

        tm.ok(result, eq=6)

    @staticmethod
    def test_flat_map_short_circuits_on_failure() -> None:
        """flat_map does not run the continuation on a failure."""
        result = r[int].fail("stop").flat_map(lambda x: r[int].ok(x * 2))

        tm.fail(result)
        tm.that(result.error, eq="stop")

    @staticmethod
    @pytest.mark.parametrize(
        ("result", "fallback", "expected"),
        [(r[int].ok(7), 42, 7), (r[int].fail("missing"), 42, 42)],
    )
    def test_unwrap_or_returns_value_or_fallback(
        result: p.Result[int],
        fallback: int,
        expected: int,
    ) -> None:
        """unwrap_or yields the success value, or the fallback on failure."""
        tm.that(result.unwrap_or(fallback), eq=expected)


class TestsFlextResultSequenceTransforms:
    """Sequential flow and traversal contracts."""

    @staticmethod
    def test_traverse_collects_all_mapped_successes() -> None:
        """Traverse maps every item and gathers the values in order."""
        result = r.traverse([1, 2, 3], lambda x: r[int].ok(x * 2))

        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq=[2, 4, 6])

    @staticmethod
    def test_traverse_fails_fast_on_first_failure() -> None:
        """Traverse stops at the first item that maps to a failure."""
        rejected = 2
        result = r.traverse(
            [1, 2, 3],
            lambda x: r[int].fail("error") if x == rejected else r[int].ok(x),
        )

        _ = u.Tests.assert_failure(result)
        tm.that(result.error, eq="error")
