"""Behavioral contract tests for the FlextResult railway API.

Every assertion targets observable public behavior: creation state, combinator
results, error propagation, and the functor/monad laws. No private attribute is
touched and no collaborator is mocked.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from flext_tests import r
from tests import p
from tests.unit._result_scenarios import ResultOperationType


class TestsFlextCoreResultLaws:
    """Verify algebraic result laws."""

    ResultOperationType = ResultOperationType

    # ------------------------------------------------------------------ #
    # Construction contract                                              #
    # ------------------------------------------------------------------ #

    def test_ok_reports_success_and_exposes_value(self) -> None:
        result = r[int].ok(42)
        assert result.success is True
        assert result.failure is False
        assert bool(result) is True
        assert result.value == 42
        assert result.unwrap() == 42

    def test_fail_reports_failure_and_exposes_error_message(self) -> None:
        result = r[int].fail("boom")
        assert result.failure is True
        assert result.success is False
        assert bool(result) is False
        assert result.error == "boom"

    # ------------------------------------------------------------------ #
    # unwrap / default recovery contract                                 #
    # ------------------------------------------------------------------ #

    def test_unwrap_on_failure_raises_with_error_in_message(self) -> None:
        result = r[int].fail("boom")
        with pytest.raises(RuntimeError, match="boom"):
            result.unwrap()

    @pytest.mark.parametrize(
        ("result", "default", "expected"),
        [(r[int].ok(7), 99, 7), (r[int].fail("nope"), 99, 99)],
    )
    def test_unwrap_or_returns_value_or_default(
        self, result: p.Result[int], default: int, expected: int
    ) -> None:
        assert result.unwrap_or(default) == expected

    def test_unwrap_or_else_invokes_supplier_only_on_failure(self) -> None:
        assert r[int].ok(5).unwrap_or_else(lambda: 100) == 5
        assert r[int].fail("x").unwrap_or_else(lambda: 100) == 100

    # ------------------------------------------------------------------ #
    # map / flat_map combinator contract                                 #
    # ------------------------------------------------------------------ #

    def test_map_transforms_success_value(self) -> None:
        assert r[int].ok(10).map(lambda v: v + 5).value == 15

    def test_map_leaves_failure_untouched(self) -> None:
        mapped = r[int].fail("boom").map(lambda v: v + 1)
        assert mapped.failure is True
        assert mapped.error == "boom"

    def test_flat_map_chains_successful_results(self) -> None:
        chained = r[int].ok(3).flat_map(lambda v: r[int].ok(v * 4))
        assert chained.success is True
        assert chained.value == 12

    def test_flat_map_short_circuits_on_failure(self) -> None:
        called: list[int] = []

        def spy(v: int) -> p.Result[int]:
            called.append(v)
            return r[int].ok(v)

        chained = r[int].fail("boom").flat_map(spy)
        assert chained.failure is True
        assert chained.error == "boom"
        assert called == []

    # ------------------------------------------------------------------ #
    # filter / recover / lash / fold / map_error contract                #
    # ------------------------------------------------------------------ #

    def test_filter_keeps_success_when_predicate_passes(self) -> None:
        assert r[int].ok(20).filter(lambda v: v > 10).success is True

    def test_filter_converts_success_to_failure_when_predicate_fails(self) -> None:
        filtered = r[int].ok(5).filter(lambda v: v > 10)
        assert filtered.failure is True

    def test_recover_replaces_failure_with_computed_value(self) -> None:
        assert r[int].fail("boom").recover(lambda _e: 0).value == 0

    def test_recover_leaves_success_untouched(self) -> None:
        assert r[int].ok(9).recover(lambda _e: 0).value == 9

    def test_lash_substitutes_a_new_result_on_failure(self) -> None:
        lashed = r[int].fail("boom").lash(lambda _e: r[int].ok(7))
        assert lashed.success is True
        assert lashed.value == 7

    def test_lash_passes_success_through(self) -> None:
        lashed = r[int].ok(1).lash(lambda _e: r[int].ok(7))
        assert lashed.value == 1

    @pytest.mark.parametrize(
        ("result", "expected"),
        [(r[int].ok(4), "ok:4"), (r[int].fail("boom"), "err:boom")],
    )
    def test_fold_dispatches_to_the_matching_branch(
        self, result: p.Result[int], expected: str
    ) -> None:
        folded = result.fold(lambda e: f"err:{e}", lambda v: f"ok:{v}")
        assert folded == expected

    def test_map_error_rewrites_failure_message_only(self) -> None:
        assert (
            r[int].fail("boom").map_error(lambda e: f"wrapped:{e}").error
            == "wrapped:boom"
        )
        assert r[int].ok(1).map_error(lambda e: f"wrapped:{e}").value == 1

    # ------------------------------------------------------------------ #
    # tap side-effect contract (returns the same result)                 #
    # ------------------------------------------------------------------ #

    def test_tap_runs_effect_on_success_and_returns_result(self) -> None:
        seen: list[int] = []
        result = r[int].ok(8).tap(seen.append)
        assert seen == [8]
        assert result.value == 8

    def test_tap_does_not_run_effect_on_failure(self) -> None:
        seen: list[int] = []
        result = r[int].fail("boom").tap(seen.append)
        assert seen == []
        assert result.failure is True

    def test_tap_error_runs_effect_only_on_failure(self) -> None:
        seen: list[str] = []
        r[int].ok(1).tap_error(seen.append)
        assert seen == []
        r[int].fail("boom").tap_error(seen.append)
        assert seen == ["boom"]

    # ------------------------------------------------------------------ #
    # Functor / Monad algebraic laws (property-based)                    #
    # ------------------------------------------------------------------ #

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_functor_identity_law(self, x: int) -> None:
        """map(id) preserves the value and success state."""
        mapped = r[int].ok(x).map(lambda v: v)
        assert mapped.success is True
        assert mapped.value == r[int].ok(x).value

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_functor_composition_law(self, x: int) -> None:
        """map(f).map(g) == map(g . f)."""

        def f(v: int) -> int:
            return v + 3

        def g(v: int) -> int:
            return v * 2

        sequential = r[int].ok(x).map(f).map(g)
        composed = r[int].ok(x).map(lambda v: g(f(v)))
        assert sequential.value == composed.value

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_monad_left_unit_law(self, x: int) -> None:
        """ok(x).flat_map(f) == f(x)."""

        def f(v: int) -> p.Result[int]:
            return r[int].ok(v * 4)

        assert r[int].ok(x).flat_map(f).value == f(x).value

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_monad_right_unit_law(self, x: int) -> None:
        """ok(x).flat_map(ok) == ok(x)."""
        chained = r[int].ok(x).flat_map(lambda v: r[int].ok(v))
        assert chained.success is True
        assert chained.value == x

    @given(err=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_error_propagates_unchanged_through_map(self, err: str) -> None:
        propagated = r[int].fail(err).map(lambda v: v + 1)
        assert propagated.failure is True
        assert propagated.error == err

    # ------------------------------------------------------------------ #
    # Structural protocol conformance                                    #
    # ------------------------------------------------------------------ #

    def test_results_satisfy_success_checkable_protocol_at_runtime(self) -> None:
        assert isinstance(r[str].ok("value"), p.SuccessCheckable)
        assert isinstance(r[str].fail("boom"), p.SuccessCheckable)
