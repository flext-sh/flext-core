"""Behavioral contract tests for the FlextResult railway API.

Every assertion targets observable public behavior: creation state, combinator
results, error propagation, and the functor/monad laws. No private attribute is
touched and no collaborator is mocked.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_tests import r
from hypothesis import given, settings, strategies as st

from tests import p


class TestsFlextCoreResultLaws:
    # ------------------------------------------------------------------ #
    # Construction contract                                              #
    # ------------------------------------------------------------------ #
    """Tests for ``FlextCoreResultLaws``."""

    @staticmethod
    def test_ok_reports_success_and_exposes_value() -> None:
        """Test ok reports success and exposes value."""
        result = r[int].ok(42)
        assert result.success is True
        assert result.failure is False
        assert bool(result) is True
        assert result.value == 42
        assert result.unwrap() == 42

    @staticmethod
    def test_fail_reports_failure_and_exposes_error_message() -> None:
        """Test fail reports failure and exposes error message."""
        result: p.Result[int] = r[int].fail("boom")
        assert result.failure is True
        assert result.success is False
        assert bool(result) is False
        assert result.error == "boom"

    # ------------------------------------------------------------------ #
    # unwrap / default recovery contract                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_unwrap_on_failure_raises_with_error_in_message() -> None:
        """Test unwrap on failure raises with error in message."""
        result: p.Result[int] = r[int].fail("boom")
        with pytest.raises(RuntimeError, match="boom"):
            result.unwrap()

    @staticmethod
    @pytest.mark.parametrize(
        ("result", "default", "expected"),
        [(r[int].ok(7), 99, 7), (r[int].fail("nope"), 99, 99)],
    )
    def test_unwrap_or_returns_value_or_default(
        result: p.Result[int],
        default: int,
        expected: int,
    ) -> None:
        """Test unwrap or returns value or default."""
        assert result.unwrap_or(default) == expected

    @staticmethod
    def test_unwrap_or_else_invokes_supplier_only_on_failure() -> None:
        """Test unwrap or else invokes supplier only on failure."""
        assert r[int].ok(5).unwrap_or_else(lambda: 100) == 5
        assert r[int].fail("x").unwrap_or_else(lambda: 100) == 100

    # ------------------------------------------------------------------ #
    # map / flat_map combinator contract                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_map_transforms_success_value() -> None:
        """Test map transforms success value."""
        assert r[int].ok(10).map(lambda v: v + 5).value == 15

    @staticmethod
    def test_map_leaves_failure_untouched() -> None:
        """Test map leaves failure untouched."""
        mapped = r[int].fail("boom").map(lambda v: v + 1)
        assert mapped.failure is True
        assert mapped.error == "boom"

    @staticmethod
    def test_flat_map_chains_successful_results() -> None:
        """Test flat map chains successful results."""
        chained = r[int].ok(3).flat_map(lambda v: r[int].ok(v * 4))
        assert chained.success is True
        assert chained.value == 12

    @staticmethod
    def test_flat_map_short_circuits_on_failure() -> None:
        """Test flat map short circuits on failure."""
        chained = (
            r[int].fail("boom").flat_map(lambda v: r[int].fail(f"step ran with {v}"))
        )
        assert chained.failure is True
        assert chained.error == "boom"

    # ------------------------------------------------------------------ #
    # filter / recover / lash / fold / map_error contract                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_filter_keeps_success_when_predicate_passes() -> None:
        """Test filter keeps success when predicate passes."""
        assert r[int].ok(20).filter(lambda v: v > 10).success is True

    @staticmethod
    def test_filter_converts_success_to_failure_when_predicate_fails() -> None:
        """Test filter converts success to failure when predicate fails."""
        filtered = r[int].ok(5).filter(lambda v: v > 10)
        assert filtered.failure is True

    @staticmethod
    def test_recover_replaces_failure_with_computed_value() -> None:
        """Test recover replaces failure with computed value."""
        assert r[int].fail("boom").recover(lambda _e: 0).value == 0

    @staticmethod
    def test_recover_leaves_success_untouched() -> None:
        """Test recover leaves success untouched."""
        assert r[int].ok(9).recover(lambda _e: 0).value == 9

    @staticmethod
    def test_lash_substitutes_a_new_result_on_failure() -> None:
        """Test lash substitutes a new result on failure."""
        lashed = r[int].fail("boom").lash(lambda _e: r[int].ok(7))
        assert lashed.success is True
        assert lashed.value == 7

    @staticmethod
    def test_lash_passes_success_through() -> None:
        """Test lash passes success through."""
        lashed = r[int].ok(1).lash(lambda _e: r[int].ok(7))
        assert lashed.value == 1

    @staticmethod
    @pytest.mark.parametrize(
        ("result", "expected"),
        [(r[int].ok(4), "ok:4"), (r[int].fail("boom"), "err:boom")],
    )
    def test_fold_dispatches_to_the_matching_branch(
        result: p.Result[int],
        expected: str,
    ) -> None:
        """Test fold dispatches to the matching branch."""
        folded = result.fold(lambda e: f"err:{e}", lambda v: f"ok:{v}")
        assert folded == expected

    @staticmethod
    def test_map_error_rewrites_failure_message_only() -> None:
        """Test map error rewrites failure message only."""
        assert (
            r[int].fail("boom").map_error(lambda e: f"wrapped:{e}").error
            == "wrapped:boom"
        )
        assert r[int].ok(1).map_error(lambda e: f"wrapped:{e}").value == 1

    # ------------------------------------------------------------------ #
    # tap side-effect contract (returns the same result)                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_tap_runs_effect_on_success_and_returns_result() -> None:
        """Test tap runs effect on success and returns result."""
        seen: list[int] = []
        result = r[int].ok(8).tap(seen.append)
        assert seen == [8]
        assert result.value == 8

    @staticmethod
    def test_tap_does_not_run_effect_on_failure() -> None:
        """Test tap does not run effect on failure."""
        seen: list[int] = []
        result: p.Result[int] = r[int].fail("boom").tap(seen.append)
        assert seen == []
        assert result.failure is True

    @staticmethod
    def test_tap_error_runs_effect_only_on_failure() -> None:
        """Test tap error runs effect only on failure."""
        seen: list[str] = []
        r[int].ok(1).tap_error(seen.append)
        assert seen == []
        r[int].fail("boom").tap_error(seen.append)
        assert seen == ["boom"]

    # ------------------------------------------------------------------ #
    # Functor / Monad algebraic laws (property-based)                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_functor_identity_law(x: int) -> None:
        """map(id) preserves the value and success state."""
        mapped = r[int].ok(x).map(lambda v: v)
        assert mapped.success is True
        assert mapped.value == r[int].ok(x).value

    @staticmethod
    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_functor_composition_law(x: int) -> None:
        """map(f).map(g) == map(g . f)."""

        def f(v: int) -> int:
            return v + 3

        def g(v: int) -> int:
            return v * 2

        sequential = r[int].ok(x).map(f).map(g)
        composed = r[int].ok(x).map(lambda v: g(f(v)))
        assert sequential.value == composed.value

    @staticmethod
    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_monad_left_unit_law(x: int) -> None:
        """ok(x).flat_map(f) == f(x)."""

        def f(v: int) -> p.Result[int]:
            return r[int].ok(v * 4)

        assert r[int].ok(x).flat_map(f).value == f(x).value

    @staticmethod
    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_monad_right_unit_law(x: int) -> None:
        """ok(x).flat_map(ok) == ok(x)."""
        chained = r[int].ok(x).flat_map(r[int].ok)
        assert chained.success is True
        assert chained.value == x

    @staticmethod
    @given(err=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_error_propagates_unchanged_through_map(err: str) -> None:
        """Test error propagates unchanged through map."""
        propagated = r[int].fail(err).map(lambda v: v + 1)
        assert propagated.failure is True
        assert propagated.error == err

    # ------------------------------------------------------------------ #
    # Structural protocol conformance                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_results_satisfy_success_checkable_protocol_at_runtime() -> None:
        """Test results satisfy success checkable protocol at runtime."""
        assert isinstance(r[str].ok("value"), p.SuccessCheckable)
        assert isinstance(r[str].fail("boom"), p.SuccessCheckable)

    @staticmethod
    def test_results_satisfy_result_protocol_at_runtime() -> None:
        """Test results satisfy result protocol at runtime."""
        assert isinstance(r[str].ok("value"), p.Result)
        assert isinstance(r[str].fail("boom"), p.Result)
