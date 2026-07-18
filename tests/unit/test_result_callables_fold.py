"""Behavioral tests for r callable factories, flow_through, map_or and fold.

Every test asserts observable public contract only: the r[T] outcome
(success/failure, value, error) and the combinator return values. No private
attribute access, no patching of the unit under test, no spying on internal
collaborators.
"""

from __future__ import annotations


import pytest
from flext_tests import r, tm

from tests import p
from tests import t


class TestsFlextCoreResultCallablesFold:
    @pytest.mark.parametrize(
        "value",
        [True, False, 0, 1, "", "value"],
    )
    def test_ok_carries_value_as_success(
        self,
        value: bool | int | str,
    ) -> None:
        """ok() yields a success whose value is the wrapped payload."""
        result = r[bool | int | str].ok(value)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=value)

    def test_fail_yields_failure_with_error(self) -> None:
        """fail() yields a failure carrying the given error message."""
        result: p.Result[int] = r[int].fail("boom")
        tm.that(result.success, eq=False)
        tm.that(result.error, eq="boom")

    def test_flow_through_chains_all_functions_on_success(self) -> None:
        """flow_through threads the value through every function in order."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def double(x: int) -> p.Result[int]:
            return r[int].ok(x * 2)

        final = r[int].ok(5).flow_through(add_one, double)
        value = tm.ok(final)
        tm.that(value, eq=12)

    def test_flow_through_short_circuits_on_first_failure(self) -> None:
        """flow_through stops at the first failure; later functions never run."""
        calls: list[str] = []

        def add_one(x: int) -> p.Result[int]:
            calls.append("add_one")
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> p.Result[int]:
            calls.append("fail_op")
            return r[int].fail("stopped")

        def never_called(_x: int) -> p.Result[int]:
            calls.append("never_called")
            return r[int].ok(999)

        final = r[int].ok(5).flow_through(add_one, fail_op, never_called)
        tm.fail(final)
        tm.that(final.error, eq="stopped")
        tm.that(calls, eq=["add_one", "fail_op"])

    def test_create_from_callable_wraps_return_value_as_success(self) -> None:
        """create_from_callable succeeds with the callable's return value."""

        def produce() -> str:
            return "success"

        result = r.create_from_callable(produce)
        value = tm.ok(result)
        tm.that(value, eq="success")

    def test_create_from_callable_none_return_is_failure(self) -> None:
        """create_from_callable fails when the callable returns None."""

        def produce() -> str | None:
            return None

        result = r.create_from_callable(produce)
        error = tm.fail(result)
        tm.that(error, has="Callable returned None")

    def test_create_from_callable_captures_raised_exception(self) -> None:
        """create_from_callable converts a raised exception into a failure."""

        def produce() -> str:
            message = "Callable failed"
            raise ValueError(message)

        result = r.create_from_callable(produce)
        error = tm.fail(result)
        tm.that(error, has="Callable failed")

    def test_map_or_on_success_without_func_returns_value(self) -> None:
        """map_or returns the success value unchanged when no func is given."""
        result: p.Result[str] = r[str].ok("hello")
        tm.that(result.map_or("default"), eq="hello")

    def test_map_or_on_failure_without_func_returns_default(self) -> None:
        """map_or returns the default for a failure when no func is given."""
        result: p.Result[str] = r[str].fail("error")
        tm.that(result.map_or("default"), eq="default")

    def test_map_or_on_success_with_func_applies_func(self) -> None:
        """map_or applies func to the success value."""
        result: p.Result[str] = r[str].ok("hello")
        tm.that(result.map_or(0, len), eq=5)

    def test_map_or_on_failure_with_func_returns_default(self) -> None:
        """map_or ignores func and returns the default for a failure."""
        result: p.Result[str] = r[str].fail("error")
        tm.that(result.map_or(0, len), eq=0)

    def test_fold_applies_on_success_branch(self) -> None:
        """Fold runs the on_success branch for a success."""
        result: p.Result[str] = r[str].ok("hello")
        message = result.fold(
            on_success=lambda v: f"Got: {v}",
            on_failure=lambda e: f"Error: {e}",
        )
        tm.that(message, eq="Got: hello")

    def test_fold_applies_on_failure_branch(self) -> None:
        """Fold runs the on_failure branch for a failure."""
        result: p.Result[str] = r[str].fail("something broke")
        message = result.fold(
            on_success=lambda v: f"Got: {v}",
            on_failure=lambda e: f"Error: {e}",
        )
        tm.that(message, eq="Error: something broke")

    def test_fold_can_return_a_type_other_than_the_input(self) -> None:
        """Fold may project the result into an unrelated return type."""
        result: p.Result[str] = r[str].ok("hello")
        response: t.JsonMapping = result.fold(
            on_success=lambda v: {"status": 200, "data": v},
            on_failure=lambda e: {"status": 400, "error": e},
        )
        tm.that(response, eq={"status": 200, "data": "hello"})
