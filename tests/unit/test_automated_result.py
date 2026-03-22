"""Tests for result module real API via flext_tests."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from flext_tests import tm
from hypothesis import given, settings, strategies as st

from flext_core import r, t
from tests import m

type SampleValue = t.Primitives | None


class TestAutomatedResult:
    """Real functionality tests for FlextResult (r[T])."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("hello", "hello"),
            (42, 42),
            (math.pi, math.pi),
            (True, True),
            (None, None),
        ],
    )
    def test_ok_preserves_value(
        self, value: SampleValue, expected: SampleValue
    ) -> None:
        result = r[SampleValue].ok(value)
        if expected is None:
            tm.that(result.is_success, eq=True)
            tm.that(result.value, eq=None)
            return
        tm.ok(result, eq=expected)

    def test_fail_with_error_code_and_data(self) -> None:
        config_model = m.Tests.Config()
        tm.that(hasattr(config_model, "model_dump"), eq=True)
        error_data = {"debug": True, "source": "tests"}
        result = r[str].fail("bad", error_code="E001", error_data=error_data)
        tm.fail(result, has="bad", code="E001", data={"debug": True})
        tm.that(result.is_failure, eq=True)

    def test_map_flat_map_filter_fold_flow_unwrap(self) -> None:
        mapped = r[int].ok(5).map(lambda x: x * 2)
        tm.ok(mapped, eq=10)
        chained = mapped.flat_map(lambda x: r[int].ok(x + 1)).filter(lambda x: x > 10)
        tm.ok(chained, eq=11)
        folded = chained.fold(lambda err: f"error:{err}", lambda val: f"ok:{val}")
        tm.that(folded, eq="ok:11")

        def add_two(x: int) -> r[int]:
            return r[int].ok(x + 2)

        def mul_three(x: int) -> r[int]:
            return r[int].ok(x * 3)

        flowed = r[int].ok(2).flow_through(add_two, mul_three)
        tm.ok(flowed, eq=12)
        tm.that(r[int].fail("x").unwrap_or(77), eq=77)

    def test_lash_tap_recover_and_properties(self) -> None:
        seen: list[int] = []
        recovered = (
            r[int]
            .fail("broken")
            .lash(lambda _: r[int].ok(9))
            .tap(lambda v: seen.append(v))
        )
        tm.ok(recovered, eq=9)
        tm.that(seen, eq=[9])
        fallback = r[int].fail("oops").recover(lambda err: len(err))
        tm.ok(fallback, eq=4)
        tm.that(recovered.is_success, eq=True)
        tm.that(recovered.is_failure, eq=False)
        tm.that(recovered.error, eq=None)

    def test_traverse_and_accumulate_errors(self) -> None:
        items = [1, 2, 3]
        traversed = r[int].traverse(items, lambda x: r[int].ok(x * 10))
        tm.ok(traversed, eq=[10, 20, 30])
        generated_ok = r[int].ok(1)
        generated_fail = r[str].fail("boom")
        tm.that(isinstance(generated_ok, list), eq=False)
        tm.that(isinstance(generated_fail, list), eq=False)
        accumulated = r[int].accumulate_errors(r[int].ok(1), r[int].fail("boom"))
        tm.fail(accumulated, has="boom")

    def test_safe_decorator_wraps_exceptions(self) -> None:
        @r.safe
        def parse_int(raw: str) -> int:
            return int(raw)

        tm.ok(parse_int("12"), eq=12)
        tm.fail(parse_int("NaN"), has="invalid literal")

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_identity_law(self, x: int) -> None:
        left = r[int].ok(x).map(lambda v: v)
        right = r[int].ok(x)
        tm.ok(left, eq=right.value)
        tm.that(left.is_success, eq=right.is_success)

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_composition_law(self, x: int) -> None:
        def f(v: int) -> int:
            return v + 3

        def g(v: int) -> int:
            return v * 2

        left = r[int].ok(x).map(f).map(g)
        right = r[int].ok(x).map(lambda v: g(f(v)))
        tm.ok(left, eq=right.value)

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_left_unit_law(self, x: int) -> None:
        def f(v: int) -> r[int]:
            return r[int].ok(v * 4)

        left = r[int].ok(x).flat_map(f)
        right = f(x)
        tm.ok(left, eq=right.value)

    @given(err=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_error_propagation(self, err: str) -> None:
        propagated = r[int].fail(err).map(lambda v: v + 1)
        tm.fail(propagated, has=err)
        tm.that(propagated.is_failure, eq=True)

    @pytest.mark.performance
    def test_result_chain_benchmark(
        self, benchmark: Callable[..., t.NormalizedValue]
    ) -> None:
        def chain() -> r[int]:
            return (
                r[int].ok(42).map(lambda x: x + 1).flat_map(lambda x: r[int].ok(x * 2))
            )

        _ = benchmark(chain)
        tm.ok(chain(), eq=86)
