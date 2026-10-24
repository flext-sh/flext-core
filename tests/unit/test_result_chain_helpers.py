"""Behavioral tests for the FlextResult (``r[T]``) railway chaining contract.

Every test asserts observable public behavior of ``r[T]`` — success/failure
state, carried value/error, and the combinator surface (map, flat_map, filter,
recover, lash, fold, tap, map_error, unwrap family). No private attributes,
no internal collaborators, no implementation details are inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import p, r, tm


class TestsFlextCoreResultChainHelpers:
    """Public-contract behavior of the ``r[T]`` railway type."""

    @staticmethod
    def test_ok_carries_value_and_reports_success() -> None:
        """A success result exposes its value and reports success state."""
        result = r[int].ok(5)

        tm.that(result.success is True, truth=True)
        tm.that(result.failure is False, truth=True)
        tm.that(bool(result) is True, truth=True)
        tm.that(result.value, eq=5)
        tm.that(result.unwrap(), eq=5)
        tm.that(result.error, none=True)

    @staticmethod
    def test_fail_carries_error_and_reports_failure() -> None:
        """A failure result reports failure state and carries its error message."""
        result: p.Result[int] = r[int].fail("boom")

        tm.that(result.success is False, truth=True)
        tm.that(result.failure is True, truth=True)
        tm.that(bool(result) is False, truth=True)
        tm.that(result.error, eq="boom")

    @staticmethod
    def test_value_access_on_failure_raises() -> None:
        """Reading ``.value`` on a failure is a contract violation that raises."""
        result: p.Result[int] = r[int].fail("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _ = result.value

    @staticmethod
    def test_unwrap_on_failure_raises() -> None:
        """``unwrap`` on a failure raises rather than inventing a value."""
        result: p.Result[int] = r[int].fail("boom")

        with pytest.raises(RuntimeError, match="Cannot unwrap failed result"):
            result.unwrap()

    @staticmethod
    def test_map_transforms_success_value() -> None:
        """``map`` applies the function to a success value, preserving success."""
        result = r[int].ok(5).map(lambda x: x * 2)

        tm.that(result.success is True, truth=True)
        tm.that(result.value, eq=10)

    @staticmethod
    def test_map_short_circuits_on_failure() -> None:
        """``map`` leaves a failure untouched and never runs the function."""
        result = r[int].fail("prior").map(lambda value: value * 2)

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, eq="prior")

    @staticmethod
    def test_flat_map_chains_success_producing_results() -> None:
        """``flat_map`` composes fallible steps, flattening nested results."""
        result = r[int].ok(10).flat_map(lambda x: r[int].ok(x // 2))

        tm.that(result.success is True, truth=True)
        tm.that(result.value, eq=5)

    @staticmethod
    def test_flat_map_propagates_downstream_failure() -> None:
        """A failure produced inside ``flat_map`` propagates as the chain result."""
        result = (
            r[int]
            .ok(0)
            .flat_map(
                lambda x: r[int].fail("division by zero") if x == 0 else r[int].ok(x),
            )
        )

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, eq="division by zero")

    @staticmethod
    def test_flat_map_short_circuits_upstream_failure() -> None:
        """``flat_map`` never invokes its step when the upstream already failed."""
        result = (
            r[int]
            .fail("upstream")
            .flat_map(lambda value: r[int].fail(f"step ran with {value}"))
        )

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, eq="upstream")

    @staticmethod
    def test_end_to_end_chain_composes_map_flat_map_filter() -> None:
        """A full railway chain of passing steps yields the final value."""
        threshold = 5
        result = (
            r[int]
            .ok(3)
            .map(lambda x: x + 1)
            .flat_map(lambda x: r[int].ok(x * 2))
            .filter(lambda x: x > threshold)
        )

        tm.that(result.unwrap(), eq=8)

    @staticmethod
    @pytest.mark.parametrize(("value", "keeps_success"), [(20, True), (5, False)])
    def test_filter_keeps_or_rejects_by_predicate(
        value: int,
        *,
        keeps_success: bool,
    ) -> None:
        """``filter`` keeps success when the predicate holds, else fails it."""
        threshold = 10
        result = r[int].ok(value).filter(lambda x: x > threshold)

        tm.that(result.success is keeps_success, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("source", "expected"),
        [(r[int].ok(5), 5), (r[int].fail("boom"), 99)],
    )
    def test_unwrap_or_returns_value_or_default(
        source: r[int],
        expected: int,
    ) -> None:
        """``unwrap_or`` returns the value on success and the default on failure."""
        tm.that(source.unwrap_or(99), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("source", "expected"),
        [(r[int].ok(5), 5), (r[int].fail("boom"), 7)],
    )
    def test_unwrap_or_else_computes_default_on_failure(
        source: r[int],
        expected: int,
    ) -> None:
        """``unwrap_or_else`` calls the supplier only when the result failed."""
        tm.that(source.unwrap_or_else(lambda: 7), eq=expected)


class TestsFlextCoreResultRecoveryAndObservation:
    """Recovery, folding, and observation preserve the railway contract."""

    @staticmethod
    def test_recover_replaces_failure_with_value() -> None:
        """``recover`` turns a failure into a success using the error."""
        recovered = r[int].fail("boom").recover(lambda _error: 0)

        tm.that(recovered.success is True, truth=True)
        tm.that(recovered.value, eq=0)

    @staticmethod
    def test_recover_leaves_success_unchanged() -> None:
        """``recover`` is a no-op on a success result."""
        recovered = r[int].ok(5).recover(lambda _error: 99)

        tm.that(recovered.value, eq=5)

    @staticmethod
    def test_lash_chains_alternative_result_on_failure() -> None:
        """``lash`` swaps in an alternative result when the source failed."""
        result = r[int].fail("boom").lash(lambda _error: r[int].ok(-1))

        tm.that(result.success is True, truth=True)
        tm.that(result.value, eq=-1)

    @staticmethod
    def test_lash_leaves_success_unchanged() -> None:
        """``lash`` does not run its handler for a success result."""
        result = r[int].ok(5).lash(lambda _error: r[int].ok(0))

        tm.that(result.value, eq=5)

    @staticmethod
    def test_map_error_rewrites_failure_message() -> None:
        """``map_error`` transforms the error text of a failure."""
        result: p.Result[int] = r[int].fail("boom").map_error(lambda err: f"{err}!")

        tm.that(result.failure is True, truth=True)
        tm.that(result.error, eq="boom!")

    @staticmethod
    def test_map_error_leaves_success_unchanged() -> None:
        """``map_error`` does not touch a success value."""
        result: p.Result[int] = r[int].ok(5).map_error(lambda err: f"{err}!")

        tm.that(result.value, eq=5)

    @staticmethod
    @pytest.mark.parametrize(
        ("source", "expected"),
        [(r[int].ok(5), "ok:5"), (r[int].fail("boom"), "err:boom")],
    )
    def test_fold_collapses_to_single_value_per_branch(
        source: r[int],
        expected: str,
    ) -> None:
        """``fold`` applies the matching branch and returns a plain value."""
        folded = source.fold(lambda err: f"err:{err}", lambda value: f"ok:{value}")

        tm.that(folded, eq=expected)

    @staticmethod
    def test_tap_runs_side_effect_only_on_success() -> None:
        """``tap`` observes success values and skips failures."""
        seen_ok: list[int] = []
        seen_fail: list[int] = []

        r[int].ok(5).tap(seen_ok.append)
        r[int].fail("boom").tap(seen_fail.append)

        tm.that(seen_ok, eq=[5])
        tm.that(seen_fail, eq=[])

    @staticmethod
    def test_tap_error_runs_side_effect_only_on_failure() -> None:
        """``tap_error`` observes failure errors and skips successes."""
        seen_fail: list[str] = []
        seen_ok: list[str] = []

        r[int].fail("boom").tap_error(seen_fail.append)
        r[int].ok(5).tap_error(seen_ok.append)

        tm.that(seen_fail, eq=["boom"])
        tm.that(seen_ok, eq=[])

    @staticmethod
    @pytest.mark.parametrize(
        ("source", "expected"),
        [(r[int].ok(5), 6), (r[int].fail("boom"), 0)],
    )
    def test_map_or_applies_func_or_returns_default(
        source: r[int],
        expected: int,
    ) -> None:
        """``map_or`` maps a success value, else returns the supplied default."""

        def mapper(value: int) -> int:
            return value + 1

        tm.that(source.map_or(0, mapper), eq=expected)
