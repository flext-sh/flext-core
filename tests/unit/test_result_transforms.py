"""Result decorator and transform tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import r, tm

from tests.unit._result_scenarios import (
    ResultOperationType,
)
from tests.utilities import u

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from tests.protocols import p


class TestsFlextResultTransforms:
    ResultOperationType = ResultOperationType

    def test_safe_decorator(self) -> None:
        """Test safe decorator wraps function in try/except."""

        def divide(a: int, b: int) -> int:
            return a // b

        divide_wrapped = r.safe(divide)
        result: p.Result[int] = divide_wrapped(10, 2)
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq=5)
        result_fail: p.Result[int] = divide_wrapped(10, 0)
        tm.fail(result_fail)

    def test_map_error(self) -> None:
        """Test map_error transforms error message."""
        result: p.Result[str] = r[str].fail("original error")
        transformed = result.map_error(lambda e: f"PREFIX: {e}")
        tm.fail(transformed)
        tm.that(transformed.error, eq="PREFIX: original error")
        success = r[str].ok("value")
        unchanged = success.map_error(lambda e: f"PREFIX: {e}")
        tm.ok(unchanged)
        tm.that(unchanged.value, eq="value")

    def test_recover_converts_failure_to_success(self) -> None:
        """Test recover maps failure errors into success values."""
        failure: p.Result[int] = r[int].fail("missing")
        recovered = failure.recover(len)
        _ = u.Tests.assert_success(recovered)
        tm.that(recovered.value, eq=7)

        success: p.Result[int] = r[int].ok(3)
        unchanged = success.recover(lambda _error: 99)
        _ = u.Tests.assert_success(unchanged)
        tm.that(unchanged.value, eq=3)

    def test_tap_runs_success_side_effect_only(self) -> None:
        """Test tap runs only on successful values and keeps the original result."""
        events: MutableSequence[str] = []
        success: p.Result[str] = r[str].ok("ready")
        tapped = success.tap(events.append)
        tm.that(tapped is success, eq=True)
        tm.that(tuple(events), eq=("ready",))

        failure: p.Result[str] = r[str].fail("broken")
        tapped_failure = failure.tap(events.append)
        tm.that(tapped_failure is failure, eq=True)
        tm.that(tuple(events), eq=("ready",))

    def test_tap_error_runs_failure_side_effect_only(self) -> None:
        """Test tap_error runs only on failures and keeps the original result."""
        errors: MutableSequence[str] = []
        failure: p.Result[str] = r[str].fail("broken")
        tapped_failure = failure.tap_error(errors.append)
        tm.that(tapped_failure is failure, eq=True)
        tm.that(tuple(errors), eq=("broken",))

        success: p.Result[str] = r[str].ok("ready")
        tapped_success = success.tap_error(errors.append)
        tm.that(tapped_success is success, eq=True)
        tm.that(tuple(errors), eq=("broken",))

    def test_filter_success(self) -> None:
        """Test filter with success result."""
        result = r[int].ok(10)
        filtered = result.filter(lambda x: x > 5)
        tm.ok(filtered)
        tm.that(filtered.value, eq=10)
        filtered_fail = result.filter(lambda x: x > 20)
        tm.fail(filtered_fail)

    def test_filter_failure(self) -> None:
        """Test filter with failure result returns unchanged."""
        result: p.Result[int] = r[int].fail("error")
        filtered = result.filter(lambda x: x > 5)
        tm.fail(filtered)
        tm.that(filtered.error, eq="error")

    def test_flow_through(self) -> None:
        """Test flow_through chains multiple operations."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def multiply_two(x: int) -> p.Result[int]:
            return r[int].ok(x * 2)

        result = r[int].ok(5)
        final = result.flow_through(add_one, multiply_two)
        tm.ok(final)
        tm.that(final.value, eq=12)

    def test_flow_through_failure(self) -> None:
        """Test flow_through stops on first failure."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> p.Result[int]:
            return r[int].fail("error")

        def multiply_two(x: int) -> p.Result[int]:
            return r[int].ok(x * 2)

        result = r[int].ok(5)
        final = result.flow_through(add_one, fail_op, multiply_two)
        tm.fail(final)
        tm.that(final.error, eq="error")

    def test_traverse_success(self) -> None:
        """Test traverse maps over sequence successfully."""
        items = [1, 2, 3]
        result = r.traverse(items, lambda x: r[int].ok(x * 2))
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq=[2, 4, 6])

    def test_traverse_failure(self) -> None:
        """Test traverse fails fast on first failure."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail("error") if x == 2 else r[int].ok(x),
        )
        _ = u.Tests.assert_failure(result)
        tm.that(result.error, eq="error")
