"""Result callable and fold tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import r, tm

from tests.unit._result_scenarios import (
    ResultOperationType,
)
from tests.utilities import u

if TYPE_CHECKING:
    from tests.protocols import p
    from tests.typings import t


class TestsFlextResultCallablesFold:
    ResultOperationType = ResultOperationType

    def test_ok_with_valid_value_succeeds(self) -> None:
        """Test ok(True) creates valid success result."""
        result = r[bool].ok(True)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=True)

    def test_flow_through_stops_on_failure(self) -> None:
        """Test flow_through stops when function returns failure."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> p.Result[int]:
            return r[int].fail("stopped")

        def never_called(_x: int) -> p.Result[int]:
            return r[int].ok(999)

        result = r[int].ok(5)
        final = result.flow_through(add_one, fail_op, never_called)
        tm.fail(final)
        tm.that(final.error, eq="stopped")

    def test_create_from_callable_success(self) -> None:
        """Test create_from_callable with successful callable."""

        def func() -> str:
            return "success"

        result = r.create_from_callable(func)
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq="success")

    def test_create_from_callable_none(self) -> None:
        """Test create_from_callable with callable returning None."""

        def func() -> str | None:
            return None

        result = r.create_from_callable(func)
        _ = u.Tests.assert_failure(result)
        error_msg = result.error
        tm.that(error_msg, none=False)
        tm.that(error_msg, has="Callable returned None")

    def test_create_from_callable_exception(self) -> None:
        """Test create_from_callable handles exceptions."""

        def func() -> str:
            error_msg = "Callable failed"
            raise ValueError(error_msg)

        result = r.create_from_callable(func)
        _ = u.Tests.assert_failure(result)
        error_msg = result.error
        tm.that(error_msg, none=False)
        tm.that(error_msg, has="Callable failed")

    def test_map_or_success_without_func(self) -> None:
        """Test map_or returns value on success when func is None."""
        result: p.Result[str] = r[str].ok("hello")
        value = result.map_or(None)
        tm.that(value, eq="hello")

    def test_map_or_failure_without_func(self) -> None:
        """Test map_or returns default on failure when func is None."""
        result: p.Result[str] = r[str].fail("error")
        value = result.map_or("default")
        tm.that(value, eq="default")

    def test_map_or_success_with_func(self) -> None:
        """Test map_or applies func on success."""
        result: p.Result[str] = r[str].ok("hello")
        length = result.map_or(0, len)
        tm.that(length, eq=5)

    def test_map_or_failure_with_func(self) -> None:
        """Test map_or returns default on failure even with func."""
        result: p.Result[str] = r[str].fail("error")
        length = result.map_or(0, len)
        tm.that(length, eq=0)

    def test_fold_success(self) -> None:
        """Test fold applies on_success function."""
        result: p.Result[str] = r[str].ok("hello")
        message = result.fold(
            on_success=lambda v: f"Got: {v}",
            on_failure=lambda e: f"Error: {e}",
        )
        tm.that(message, eq="Got: hello")

    def test_fold_failure(self) -> None:
        """Test fold applies on_failure function."""
        result: p.Result[str] = r[str].fail("something broke")
        message = result.fold(
            on_success=lambda v: f"Got: {v}",
            on_failure=lambda e: f"Error: {e}",
        )
        tm.that(message, eq="Error: something broke")

    def test_fold_different_return_types(self) -> None:
        """Test fold can return different types than input."""
        result: p.Result[str] = r[str].ok("hello")
        response: t.JsonMapping = result.fold(
            on_success=lambda v: {"status": 200, "data": v},
            on_failure=lambda e: {"status": 400, "error": e},
        )
        tm.that(
            response,
            eq={"status": 200, "data": "hello"},
        )
