"""Behavioral tests for the r[T] public contract (creation + combinators)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_tests import r, tm

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.protocols import p


class TestsFlextResultOperations:
    """Assert the observable r[T] contract: success/failure state and combinators."""

    # --- creation + terminal state --------------------------------------

    @pytest.mark.parametrize("value", ["success", "", "multi word value"])
    def test_ok_reports_success_and_exposes_value(self, value: str) -> None:
        """r.ok is success, carries the value, and has no error."""
        result: p.Result[str] = r[str].ok(value)

        tm.that(result.success, eq=True)
        tm.that(result.failure, eq=False)
        tm.that(bool(result), eq=True)
        tm.that(result.value, eq=value)
        tm.that(result.unwrap(), eq=value)
        tm.that(result.error, none=True)

    @pytest.mark.parametrize("message", ["boom", "error message", "not found"])
    def test_fail_reports_failure_and_preserves_error(self, message: str) -> None:
        """r.fail is failure, preserves the error message, and has no value."""
        result: p.Result[str] = r[str].fail(message)

        tm.that(result.failure, eq=True)
        tm.that(result.success, eq=False)
        tm.that(bool(result), eq=False)
        tm.that(result.error, eq=message)

    @pytest.mark.parametrize("message", ["boom", "denied"])
    def test_value_and_unwrap_raise_on_failure(self, message: str) -> None:
        """Accessing value or unwrap on a failure raises with the error message."""
        result: p.Result[str] = r[str].fail(message)

        with pytest.raises(RuntimeError, match=message):
            _ = result.value
        with pytest.raises(RuntimeError, match=message):
            result.unwrap()

    # --- unwrap_or / | default ------------------------------------------

    @pytest.mark.parametrize(
        ("result", "default", "expected"),
        [
            (r[str].ok("value"), "default", "value"),
            (r[str].fail("error"), "default", "default"),
        ],
        ids=["success-keeps-value", "failure-yields-default"],
    )
    def test_unwrap_or_returns_value_on_success_default_on_failure(
        self, result: p.Result[str], default: str, expected: str
    ) -> None:
        """unwrap_or and the | operator both yield value on success, default on failure."""
        tm.that(result.unwrap_or(default), eq=expected)
        tm.that((result | default), eq=expected)

    # --- map -------------------------------------------------------------

    def test_map_transforms_the_success_value(self) -> None:
        """Map applies the function to a success payload."""
        result: p.Result[int] = r[int].ok(5)

        mapped = result.map(lambda x: x * 2)

        tm.that(mapped.success, eq=True)
        tm.that(mapped.unwrap(), eq=10)

    def test_map_is_a_no_op_on_failure_and_preserves_error(self) -> None:
        """Map does not run the function on a failure; the error passes through."""
        result: p.Result[int] = r[int].fail("boom")

        mapped = result.map(lambda x: x * 2)

        tm.that(mapped.failure, eq=True)
        tm.that(mapped.error, eq="boom")

    # --- flat_map --------------------------------------------------------

    def test_flat_map_chains_a_dependent_success(self) -> None:
        """flat_map threads the value into a further fallible step."""
        result: p.Result[int] = r[int].ok(5)

        chained = result.flat_map(lambda x: r[str].ok(f"value_{x}"))

        tm.that(chained.unwrap(), eq="value_5")

    def test_flat_map_short_circuits_on_failure(self) -> None:
        """flat_map skips the continuation when the source is a failure."""
        result: p.Result[int] = r[int].fail("boom")

        chained = result.flat_map(lambda x: r[str].ok(f"value_{x}"))

        tm.that(chained.failure, eq=True)
        tm.that(chained.error, eq="boom")

    def test_flat_map_propagates_a_failing_continuation(self) -> None:
        """flat_map surfaces the error produced by the continuation."""
        result: p.Result[int] = r[int].ok(5)

        chained = result.flat_map(lambda _: r[str].fail("downstream"))

        tm.that(chained.failure, eq=True)
        tm.that(chained.error, eq="downstream")

    # --- map_error -------------------------------------------------------

    def test_map_error_rewrites_only_the_failure_message(self) -> None:
        """map_error transforms the error text of a failure."""
        result: p.Result[str] = r[str].fail("original")

        remapped = result.map_error(lambda e: f"alt_{e}")

        tm.that(remapped.failure, eq=True)
        tm.that(remapped.error, eq="alt_original")

    def test_map_error_leaves_a_success_untouched(self) -> None:
        """map_error is a no-op on a success value."""
        result: p.Result[str] = r[str].ok("value")

        remapped = result.map_error(lambda e: f"alt_{e}")

        tm.that(remapped.unwrap(), eq="value")

    # --- lash (recover-with-result) -------------------------------------

    def test_lash_recovers_a_failure_into_a_success(self) -> None:
        """Lash replaces a failure with a recovery result."""
        result: p.Result[str] = r[str].fail("error")

        recovered = result.lash(lambda e: r[str].ok(f"recovered_{e}"))

        tm.that(recovered.unwrap(), eq="recovered_error")

    def test_lash_passes_a_success_through_unchanged(self) -> None:
        """Lash does not invoke the handler on a success."""
        result: p.Result[str] = r[str].ok("value")

        recovered = result.lash(lambda e: r[str].ok(f"recovered_{e}"))

        tm.that(recovered.unwrap(), eq="value")

    # --- filter ----------------------------------------------------------

    @pytest.mark.parametrize(
        ("value", "expected_success"),
        [(10, True), (3, False)],
        ids=["predicate-passes", "predicate-fails"],
    )
    def test_filter_keeps_value_when_predicate_holds(
        self, value: int, *, expected_success: bool
    ) -> None:
        """Filter keeps a success only when the predicate is satisfied."""
        result: p.Result[int] = r[int].ok(value)

        filtered = result.filter(lambda x: x > 5)

        assert filtered.success is expected_success
        if expected_success:
            tm.that(filtered.unwrap(), eq=value)

    # --- boolean protocol ------------------------------------------------

    @pytest.mark.parametrize(
        ("result", "expected"),
        [(r[str].ok("value"), True), (r[str].fail("error"), False)],
        ids=["success-truthy", "failure-falsy"],
    )
    def test_bool_reflects_success_state(
        self, result: p.Result[str], *, expected: bool
    ) -> None:
        """bool(result) is True for success and False for failure."""
        assert bool(result) is expected
        tm.that(bool(result), eq=expected)

    # --- railway composition --------------------------------------------

    def test_railway_composition_threads_successes_end_to_end(self) -> None:
        """A chain of map steps composes without breaking the success track."""
        composed: p.Result[str] = (
            r[int].ok(5).map(lambda v: v * 2).map(lambda v: f"result_{v}")
        )

        tm.that(composed.success, eq=True)
        tm.that(composed.unwrap(), eq="result_10")

    def test_railway_composition_short_circuits_at_first_failure(self) -> None:
        """A failure mid-chain halts every downstream step and keeps its error."""
        steps: list[Callable[[int], int]] = [lambda v: v + 1, lambda v: v * 10]
        result: p.Result[int] = r[int].fail("early")
        for step in steps:
            result = result.map(step)

        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="early")
