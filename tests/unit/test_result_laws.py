"""Result law and protocol tests."""

from __future__ import annotations

from flext_tests import r, tm
from hypothesis import given, settings, strategies as st

from tests.protocols import p
from tests.unit._result_scenarios import (
    ResultOperationType,
)


class TestsFlextResultLaws:
    ResultOperationType = ResultOperationType

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_identity_law(self, x: int) -> None:
        """Functor identity: map(id) == id."""
        left = r[int].ok(x).map(lambda v: v)
        right = r[int].ok(x)
        tm.ok(left, eq=right.value)
        tm.that(left.success, eq=right.success)

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_composition_law(self, x: int) -> None:
        """Functor composition: map(f).map(g) == map(g . f)."""

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
        """Monad left unit: ok(x).flat_map(f) == f(x)."""

        def f(v: int) -> p.Result[int]:
            return r[int].ok(v * 4)

        left = r[int].ok(x).flat_map(f)
        right = f(x)
        tm.ok(left, eq=right.value)

    @given(err=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_error_propagation_property(self, err: str) -> None:
        """Errors propagate through map unchanged."""
        propagated = r[int].fail(err).map(lambda v: v + 1)
        tm.fail(propagated, has=err)
        tm.that(propagated.failure, eq=True)

    def test_instances_satisfy_success_checkable_runtime_protocol(self) -> None:
        """R instances conform to p.SuccessCheckable structural contract at runtime."""
        assert isinstance(r[str].ok("value"), p.SuccessCheckable)
        assert isinstance(r[str].fail("boom"), p.SuccessCheckable)
