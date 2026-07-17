"""Behavioral contract tests for FlextResult (r[T]) public surface."""

from __future__ import annotations

import pytest
from flext_tests import r, tm

from tests import p


class TestsFlextCoreResult:
    """Assert the observable public contract of the r[T] railway type."""

    def test_ok_exposes_value_and_success_state(self) -> None:
        """A successful result carries its value and reports success."""
        result: p.Result[int] = r[int].ok(42)

        tm.that(result.success, eq=True)
        tm.that(result.failure, eq=False)
        tm.ok(result, eq=42)

    def test_fail_exposes_error_and_failure_state(self) -> None:
        """A failed result carries its error message and reports failure."""
        result: p.Result[int] = r[int].fail("boom")

        tm.that(result.success, eq=False)
        tm.that(result.failure, eq=True)
        tm.fail(result, has="boom")

    def test_value_access_on_failure_raises(self) -> None:
        """Accessing .value on a failure raises rather than inventing data."""
        result: p.Result[int] = r[int].fail("no value")

        with pytest.raises(RuntimeError):
            _ = result.value

    def test_fail_preserves_error_code_data_and_exception(self) -> None:
        """Failure metadata is preserved on the public surface."""
        cause = ValueError("root cause")
        result: p.Result[int] = r[int].fail(
            "bad",
            error_code="E_BAD",
            error_data={"field": "name"},
            exception=cause,
        )

        tm.that(result.error_code, eq="E_BAD")
        tm.that(result.error_data, eq={"field": "name"})
        assert result.exception is cause

    @pytest.mark.parametrize(
        ("success", "default", "expected"),
        [
            (True, 0, 7),
            (False, 0, 0),
            (False, 99, 99),
        ],
    )
    def test_unwrap_or_returns_value_or_default(
        self, *, success: bool, default: int, expected: int
    ) -> None:
        """unwrap_or yields the value on success and the default on failure."""
        result: p.Result[int] = r[int].ok(7) if success else r[int].fail("err")

        tm.that(result.unwrap_or(default), eq=expected)

    def test_unwrap_or_else_computes_default_on_failure(self) -> None:
        """unwrap_or_else lazily derives a fallback only for failures."""
        failure: p.Result[int] = r[int].fail("err")
        success: p.Result[int] = r[int].ok(3)

        tm.that(failure.unwrap_or_else(lambda: 55), eq=55)
        tm.that(success.unwrap_or_else(lambda: 55), eq=3)

    def test_or_operator_is_unwrap_or(self) -> None:
        """The | operator delivers value on success and default on failure."""
        tm.that(r[int].ok(10) | 0, eq=10)
        tm.that(r[int].fail("err") | -1, eq=-1)

    def test_bool_reflects_success(self) -> None:
        """Truthiness of a result mirrors its success state."""
        tm.that(bool(r[int].ok(1)), eq=True)
        tm.that(bool(r[int].fail("err")), eq=False)

    def test_map_transforms_success_value(self) -> None:
        """Mapping a success applies the function to its value."""
        mapped = r[int].ok(5).map(lambda x: x * 2)

        tm.ok(mapped, eq=10)

    def test_map_propagates_failure_unchanged(self) -> None:
        """Mapping a failure is a no-op and preserves the original error."""
        mapped = r[int].fail("orig").map(lambda x: x * 2)

        tm.fail(mapped, has="orig")

    def test_flat_map_chains_successful_results(self) -> None:
        """flat_map sequences a dependent fallible step."""
        chained = r[int].ok(4).flat_map(lambda x: r[str].ok(f"v{x}"))

        tm.ok(chained, eq="v4")

    def test_flat_map_short_circuits_on_failure(self) -> None:
        """flat_map does not run its function when the input failed."""
        chained = r[int].fail("stop").flat_map(lambda x: r[str].ok(f"v{x}"))

        tm.fail(chained, has="stop")

    def test_map_error_rewrites_error_message(self) -> None:
        """map_error transforms the error text on failure only."""
        rewritten = r[int].fail("raw").map_error(lambda e: f"wrapped:{e}")

        tm.fail(rewritten, has="wrapped:raw")

    def test_map_error_leaves_success_untouched(self) -> None:
        """map_error must not alter a successful result."""
        unchanged = r[int].ok(8).map_error(lambda e: f"wrapped:{e}")

        tm.ok(unchanged, eq=8)

    def test_lash_recovers_failure_with_new_result(self) -> None:
        """Lashing a failure swaps it for a recovery result."""
        recovered = r[int].fail("down").lash(lambda _: r[int].ok(0))

        tm.ok(recovered, eq=0)

    def test_recover_maps_failure_to_success_value(self) -> None:
        """Recovering a failure produces a success from its error."""
        recovered = r[int].fail("err").recover(lambda _: -1)

        tm.ok(recovered, eq=-1)

    @pytest.mark.parametrize(
        ("value", "keeps"),
        [(9, True), (2, False)],
    )
    def test_filter_keeps_or_rejects_by_predicate(
        self, *, value: int, keeps: bool
    ) -> None:
        """Filtering keeps a value passing the predicate, else fails."""
        filtered = r[int].ok(value).filter(lambda x: x > 5)

        tm.that(filtered.success, eq=keeps)

    def test_fold_dispatches_on_success_and_failure(self) -> None:
        """Folding routes to the matching branch for each outcome."""
        tm.that(r[int].ok(1).fold(lambda _: "F", lambda _: "S"), eq="S")
        tm.that(r[int].fail("e").fold(lambda _: "F", lambda _: "S"), eq="F")

    def test_tap_runs_only_on_success(self) -> None:
        """Tapping a success performs a side effect and returns it intact."""
        seen: list[int] = []
        result = r[int].ok(6).tap(seen.append)

        tm.ok(result, eq=6)
        tm.that(seen, eq=[6])

    def test_tap_error_runs_only_on_failure(self) -> None:
        """tap_error observes the error and returns the result intact."""
        seen: list[str] = []
        result = r[int].fail("bad").tap_error(seen.append)

        tm.fail(result, has="bad")
        tm.that(seen, eq=["bad"])

    def test_success_has_no_error_and_failure_complements_success(self) -> None:
        """Outcome flags are complementary and success carries no error text."""
        success: p.Result[int] = r[int].ok(1)
        failure: p.Result[int] = r[int].fail("e")

        tm.that(success.error, eq=None)
        tm.that(success.failure, eq=not success.success)
        tm.that(failure.failure, eq=not failure.success)

    def test_type_guards_classify_results(self) -> None:
        """successful_result / failed_result guard by observable outcome."""
        ok_result = r[int].ok(1)
        fail_result = r[int].fail("e")

        tm.that(r.successful_result(ok_result), eq=True)
        tm.that(r.failed_result(ok_result), eq=False)
        tm.that(r.failed_result(fail_result), eq=True)
        tm.that(r.successful_result(fail_result), eq=False)

    def test_results_satisfy_success_checkable_protocol(self) -> None:
        """Result instances honor the structural p.SuccessCheckable contract."""
        assert isinstance(r[int].ok(1), p.SuccessCheckable)
        assert isinstance(r[int].fail("e"), p.SuccessCheckable)
