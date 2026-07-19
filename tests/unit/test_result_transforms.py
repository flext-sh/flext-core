"""Behavioral tests for FlextResult transform combinators.

Every test asserts observable public contract only: the ``r[T]`` outcome
(``ok``/``fail``, ``.value``, ``.error``), combinator return values, side-effect
ordering, and result identity. No private attributes, no internal collaborators,
no patching of the unit under test.
"""

from __future__ import annotations

from operator import floordiv
from typing import TYPE_CHECKING

import pytest
from flext_tests import r, tm

from tests import u

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from tests import p


class TestsFlextResultTransforms:
    """Public-contract behavior of FlextResult transform combinators."""

    def test_safe_wraps_successful_call_as_success(self) -> None:
        """Safe returns a success carrying the wrapped function's return value."""
        result: p.Result[int] = r.safe(floordiv)(10, 2)

        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq=5)

    def test_safe_captures_raised_exception_as_failure(self) -> None:
        """Safe converts a raised exception into a failure preserving its message."""
        result: p.Result[int] = r.safe(floordiv)(10, 0)

        tm.fail(result)
        tm.that(result.error, eq="integer division or modulo by zero")

    def test_map_error_rewrites_failure_message(self) -> None:
        """map_error transforms the error of a failure and preserves its state."""
        result: p.Result[str] = r[str].fail("original error")

        transformed = result.map_error(lambda e: f"PREFIX: {e}")

        tm.fail(transformed)
        tm.that(transformed.error, eq="PREFIX: original error")

    def test_map_error_leaves_success_untouched(self) -> None:
        """map_error is a no-op on a success result."""
        success: p.Result[str] = r[str].ok("value")

        unchanged = success.map_error(lambda e: f"PREFIX: {e}")

        tm.ok(unchanged)
        tm.that(unchanged.value, eq="value")

    def test_recover_converts_failure_to_success(self) -> None:
        """Recover maps a failure's error into a success value."""
        failure: p.Result[int] = r[int].fail("missing")

        recovered = failure.recover(len)

        _ = u.Tests.assert_success(recovered)
        tm.that(recovered.value, eq=7)

    def test_recover_leaves_success_untouched(self) -> None:
        """Recover never fires on a success result."""
        success: p.Result[int] = r[int].ok(3)

        unchanged = success.recover(lambda _error: 99)

        _ = u.Tests.assert_success(unchanged)
        tm.that(unchanged.value, eq=3)

    def test_tap_runs_side_effect_on_success_only(self) -> None:
        """Tap fires its callback for a success and returns the same result."""
        events: MutableSequence[str] = []
        success: p.Result[str] = r[str].ok("ready")

        tapped = success.tap(events.append)

        tm.that(tapped is success, eq=True)
        tm.that(tuple(events), eq=("ready",))

    def test_tap_is_noop_on_failure(self) -> None:
        """Tap does not fire on a failure and returns the same result."""
        events: MutableSequence[str] = []
        failure: p.Result[str] = r[str].fail("broken")

        tapped = failure.tap(events.append)

        tm.that(tapped is failure, eq=True)
        tm.that(tuple(events), eq=())

    def test_tap_error_runs_side_effect_on_failure_only(self) -> None:
        """tap_error fires its callback for a failure and returns the same result."""
        errors: MutableSequence[str] = []
        failure: p.Result[str] = r[str].fail("broken")

        tapped = failure.tap_error(errors.append)

        tm.that(tapped is failure, eq=True)
        tm.that(tuple(errors), eq=("broken",))

    def test_tap_error_is_noop_on_success(self) -> None:
        """tap_error does not fire on a success and returns the same result."""
        errors: MutableSequence[str] = []
        success: p.Result[str] = r[str].ok("ready")

        tapped = success.tap_error(errors.append)

        tm.that(tapped is success, eq=True)
        tm.that(tuple(errors), eq=())

    @pytest.mark.parametrize(
        ("value", "predicate_ceiling", "expect_success"),
        [(10, 5, True), (10, 20, False), (6, 5, True), (5, 5, False)],
    )
    def test_filter_keeps_or_drops_success_by_predicate(
        self, value: int, predicate_ceiling: int, expect_success: bool
    ) -> None:
        """Filter keeps a success when the predicate holds, else fails it."""
        result = r[int].ok(value)

        filtered = result.filter(lambda x: x > predicate_ceiling)

        if expect_success:
            tm.ok(filtered)
            tm.that(filtered.value, eq=value)
        else:
            tm.fail(filtered)

    def test_filter_leaves_failure_unchanged(self) -> None:
        """Filter never runs its predicate on a failure and preserves the error."""
        result: p.Result[int] = r[int].fail("error")

        filtered = result.filter(lambda x: x > 5)

        tm.fail(filtered)
        tm.that(filtered.error, eq="error")

    def test_map_transforms_success_value(self) -> None:
        """Map applies the function to a success value."""
        result = r[int].ok(3).map(lambda x: x + 1)

        tm.ok(result)
        tm.that(result.value, eq=4)

    def test_map_is_noop_on_failure(self) -> None:
        """Map skips the function and preserves the error on a failure."""
        result = r[int].fail("boom").map(lambda x: x + 1)

        tm.fail(result)
        tm.that(result.error, eq="boom")

    def test_flat_map_chains_fallible_success(self) -> None:
        """flat_map chains a result-returning function on a success."""
        result = r[int].ok(3).flat_map(lambda x: r[int].ok(x * 2))

        tm.ok(result)
        tm.that(result.value, eq=6)

    def test_flat_map_short_circuits_on_failure(self) -> None:
        """flat_map does not run the continuation on a failure."""
        result = r[int].fail("stop").flat_map(lambda x: r[int].ok(x * 2))

        tm.fail(result)
        tm.that(result.error, eq="stop")

    @pytest.mark.parametrize(
        ("result", "fallback", "expected"),
        [(r[int].ok(7), 42, 7), (r[int].fail("missing"), 42, 42)],
    )
    def test_unwrap_or_returns_value_or_fallback(
        self, result: p.Result[int], fallback: int, expected: int
    ) -> None:
        """unwrap_or yields the success value, or the fallback on failure."""
        tm.that(result.unwrap_or(fallback), eq=expected)

    def test_flow_through_chains_multiple_operations(self) -> None:
        """flow_through pipes a value through successive fallible steps."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def multiply_two(x: int) -> p.Result[int]:
            return r[int].ok(x * 2)

        final = r[int].ok(5).flow_through(add_one, multiply_two)

        tm.ok(final)
        tm.that(final.value, eq=12)

    def test_flow_through_stops_on_first_failure(self) -> None:
        """flow_through short-circuits at the first failing step."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> p.Result[int]:
            return r[int].fail("error")

        def multiply_two(x: int) -> p.Result[int]:
            return r[int].ok(x * 2)

        final = r[int].ok(5).flow_through(add_one, fail_op, multiply_two)

        tm.fail(final)
        tm.that(final.error, eq="error")

    def test_traverse_collects_all_mapped_successes(self) -> None:
        """Traverse maps every item and gathers the values in order."""
        result = r.traverse([1, 2, 3], lambda x: r[int].ok(x * 2))

        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq=[2, 4, 6])

    def test_traverse_fails_fast_on_first_failure(self) -> None:
        """Traverse stops at the first item that maps to a failure."""
        result = r.traverse(
            [1, 2, 3], lambda x: r[int].fail("error") if x == 2 else r[int].ok(x)
        )

        _ = u.Tests.assert_failure(result)
        tm.that(result.error, eq="error")
