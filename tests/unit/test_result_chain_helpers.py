"""Behavioral tests for the FlextResult (``r[T]``) railway chaining contract.

Every test asserts observable public behavior of ``r[T]`` — success/failure
state, carried value/error, and the combinator surface (map, flat_map, filter,
recover, lash, fold, tap, map_error, unwrap family). No private attributes,
no internal collaborators, no implementation details are inspected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import r

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextCoreResultChainHelpers:
    """Public-contract behavior of the ``r[T]`` railway type."""

    def test_ok_carries_value_and_reports_success(self) -> None:
        """A success result exposes its value and reports success state."""
        result = r[int].ok(5)

        assert result.success is True
        assert result.failure is False
        assert bool(result) is True
        assert result.value == 5
        assert result.unwrap() == 5
        assert result.error is None

    def test_fail_carries_error_and_reports_failure(self) -> None:
        """A failure result reports failure state and carries its error message."""
        result = r[int].fail("boom")

        assert result.success is False
        assert result.failure is True
        assert bool(result) is False
        assert result.error == "boom"

    def test_value_access_on_failure_raises(self) -> None:
        """Reading ``.value`` on a failure is a contract violation that raises."""
        result = r[int].fail("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _ = result.value

    def test_unwrap_on_failure_raises(self) -> None:
        """``unwrap`` on a failure raises rather than inventing a value."""
        result = r[int].fail("boom")

        with pytest.raises(RuntimeError, match="Cannot unwrap failed result"):
            result.unwrap()

    def test_map_transforms_success_value(self) -> None:
        """``map`` applies the function to a success value, preserving success."""
        result = r[int].ok(5).map(lambda x: x * 2)

        assert result.success is True
        assert result.value == 10

    def test_map_short_circuits_on_failure(self) -> None:
        """``map`` leaves a failure untouched and never runs the function."""
        calls: list[int] = []

        def spy(value: int) -> int:
            calls.append(value)
            return value

        result = r[int].fail("prior").map(spy)

        assert result.failure is True
        assert result.error == "prior"
        assert calls == []

    def test_flat_map_chains_success_producing_results(self) -> None:
        """``flat_map`` composes fallible steps, flattening nested results."""
        result = r[int].ok(10).flat_map(lambda x: r[int].ok(x // 2))

        assert result.success is True
        assert result.value == 5

    def test_flat_map_propagates_downstream_failure(self) -> None:
        """A failure produced inside ``flat_map`` propagates as the chain result."""
        result = (
            r[int]
            .ok(0)
            .flat_map(
                lambda x: r[int].fail("division by zero") if x == 0 else r[int].ok(x),
            )
        )

        assert result.failure is True
        assert result.error == "division by zero"

    def test_flat_map_short_circuits_upstream_failure(self) -> None:
        """``flat_map`` never invokes its step when the upstream already failed."""
        calls: list[int] = []

        def spy(value: int) -> p.Result[int]:
            calls.append(value)
            return r[int].ok(value)

        result = r[int].fail("upstream").flat_map(spy)

        assert result.failure is True
        assert result.error == "upstream"
        assert calls == []

    def test_end_to_end_chain_composes_map_flat_map_filter(self) -> None:
        """A full railway chain of passing steps yields the final value."""
        result = (
            r[int]
            .ok(3)
            .map(lambda x: x + 1)
            .flat_map(lambda x: r[int].ok(x * 2))
            .filter(lambda x: x > 5)
        )

        assert result.unwrap() == 8

    @pytest.mark.parametrize(
        ("value", "keeps_success"),
        [(20, True), (5, False)],
    )
    def test_filter_keeps_or_rejects_by_predicate(
        self,
        value: int,
        *,
        keeps_success: bool,
    ) -> None:
        """``filter`` keeps success when the predicate holds, else fails it."""
        result = r[int].ok(value).filter(lambda x: x > 10)

        assert result.success is keeps_success

    @pytest.mark.parametrize(
        ("source", "expected"),
        [(r[int].ok(5), 5), (r[int].fail("boom"), 99)],
    )
    def test_unwrap_or_returns_value_or_default(
        self,
        source: r[int],
        expected: int,
    ) -> None:
        """``unwrap_or`` returns the value on success and the default on failure."""
        assert source.unwrap_or(99) == expected

    @pytest.mark.parametrize(
        ("source", "expected"),
        [(r[int].ok(5), 5), (r[int].fail("boom"), 7)],
    )
    def test_unwrap_or_else_computes_default_on_failure(
        self,
        source: r[int],
        expected: int,
    ) -> None:
        """``unwrap_or_else`` calls the supplier only when the result failed."""
        assert source.unwrap_or_else(lambda: 7) == expected

    def test_recover_replaces_failure_with_value(self) -> None:
        """``recover`` turns a failure into a success using the error."""
        recovered = r[int].fail("boom").recover(lambda _error: 0)

        assert recovered.success is True
        assert recovered.value == 0

    def test_recover_leaves_success_unchanged(self) -> None:
        """``recover`` is a no-op on a success result."""
        recovered = r[int].ok(5).recover(lambda _error: 99)

        assert recovered.value == 5

    def test_lash_chains_alternative_result_on_failure(self) -> None:
        """``lash`` swaps in an alternative result when the source failed."""
        result = r[int].fail("boom").lash(lambda _error: r[int].ok(-1))

        assert result.success is True
        assert result.value == -1

    def test_lash_leaves_success_unchanged(self) -> None:
        """``lash`` does not run its handler for a success result."""
        result = r[int].ok(5).lash(lambda _error: r[int].ok(0))

        assert result.value == 5

    def test_map_error_rewrites_failure_message(self) -> None:
        """``map_error`` transforms the error text of a failure."""
        result = r[int].fail("boom").map_error(lambda err: f"{err}!")

        assert result.failure is True
        assert result.error == "boom!"

    def test_map_error_leaves_success_unchanged(self) -> None:
        """``map_error`` does not touch a success value."""
        result = r[int].ok(5).map_error(lambda err: f"{err}!")

        assert result.value == 5

    @pytest.mark.parametrize(
        ("source", "expected"),
        [(r[int].ok(5), "ok:5"), (r[int].fail("boom"), "err:boom")],
    )
    def test_fold_collapses_to_single_value_per_branch(
        self,
        source: r[int],
        expected: str,
    ) -> None:
        """``fold`` applies the matching branch and returns a plain value."""
        folded = source.fold(
            lambda err: f"err:{err}",
            lambda value: f"ok:{value}",
        )

        assert folded == expected

    def test_tap_runs_side_effect_only_on_success(self) -> None:
        """``tap`` observes success values and skips failures."""
        seen_ok: list[int] = []
        seen_fail: list[int] = []

        r[int].ok(5).tap(seen_ok.append)
        r[int].fail("boom").tap(seen_fail.append)

        assert seen_ok == [5]
        assert seen_fail == []

    def test_tap_error_runs_side_effect_only_on_failure(self) -> None:
        """``tap_error`` observes failure errors and skips successes."""
        seen_fail: list[str] = []
        seen_ok: list[str] = []

        r[int].fail("boom").tap_error(seen_fail.append)
        r[int].ok(5).tap_error(seen_ok.append)

        assert seen_fail == ["boom"]
        assert seen_ok == []

    @pytest.mark.parametrize(
        ("source", "expected"),
        [(r[int].ok(5), 6), (r[int].fail("boom"), 0)],
    )
    def test_map_or_applies_func_or_returns_default(
        self,
        source: r[int],
        expected: int,
    ) -> None:
        """``map_or`` maps a success value, else returns the supplied default."""

        def mapper(value: int) -> int:
            return value + 1

        assert source.map_or(0, mapper) == expected
