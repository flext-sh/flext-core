"""Behavioral tests for r.safe / create_from_callable exception carrying."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import r, tm

from tests.unit._result_exception_support import TestsFlextResultExceptionCarrying

if TYPE_CHECKING:
    from collections.abc import Sized

    from tests import p


class TestsFlextCoreResultExceptionSafeCallable(TestsFlextResultExceptionCarrying):
    def test_safe_carries_exception(self) -> None:
        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        result: p.Result[float] = divide(10, 0)
        tm.that(result.failure, eq=True)
        tm.that(
            result.error is not None and "division by zero" in result.error.lower(),
            eq=True,
        )
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=ZeroDivisionError)

    def test_safe_no_exception_on_success(self) -> None:
        @r.safe
        def add(a: int, b: int) -> int:
            return a + b

        result: p.Result[int] = add(5, 3)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=8)
        tm.that(result.exception, none=True)

    def test_safe_success_supports_map_combinator(self) -> None:
        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        mapped: p.Result[float] = divide(10, 2).map(lambda value: value * 2)
        tm.that(mapped.success, eq=True)
        tm.that(mapped.value, eq=10.0)

    def test_safe_success_supports_flat_map_combinator(self) -> None:
        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        chained: p.Result[float] = divide(10, 2).flat_map(lambda value: r.ok(value + 1))
        tm.that(chained.success, eq=True)
        tm.that(chained.value, eq=6.0)

    def test_safe_failure_short_circuits_map_and_keeps_exception(self) -> None:
        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        mapped: p.Result[float] = divide(10, 0).map(lambda value: value * 2)
        tm.that(mapped.failure, eq=True)
        tm.that(mapped.exception, is_=ZeroDivisionError)

    def test_safe_failure_recovers_with_unwrap_or(self) -> None:
        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        tm.that(divide(10, 0).unwrap_or(-1.0), eq=-1.0)
        tm.that(divide(10, 2).unwrap_or(-1.0), eq=5.0)

    def test_safe_failure_unwrap_raises_with_error_message(self) -> None:
        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        with pytest.raises(RuntimeError, match="division by zero"):
            divide(10, 0).unwrap()

    def test_safe_captures_value_error(self) -> None:
        @r.safe
        def parse_int(value: str) -> int:
            if not value.isdigit():
                raise ValueError(f"'{value}' is not a valid integer")
            return int(value)

        result: p.Result[int] = parse_int("abc")
        tm.that(result.failure, eq=True)
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=ValueError)

    def test_safe_captures_type_error(self) -> None:
        @r.safe
        def get_length(obj: Sized) -> int:
            return len(obj)

        result: p.Result[int] = get_length(self.BrokenSized())
        tm.that(result.failure, eq=True)
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=TypeError)

    @pytest.mark.parametrize(
        "exception_type", [OSError, IndexError], ids=("os-error", "lookup-index-error")
    )
    def test_safe_captures_boundary_exceptions(
        self, exception_type: type[Exception]
    ) -> None:
        @r.safe
        def raise_boundary_error() -> int:
            msg = "boundary failure"
            raise exception_type(msg)

        result: p.Result[int] = raise_boundary_error()
        tm.that(result.failure, eq=True)
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=exception_type)

    def test_create_from_callable_carries_exception(self) -> None:
        def risky_operation() -> int:
            msg = "operation failed"
            raise RuntimeError(msg)

        result: p.Result[int] = r[int].create_from_callable(risky_operation)
        tm.that(result.failure, eq=True)
        tm.that(
            result.error is not None and "operation failed" in result.error, eq=True
        )
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=RuntimeError)

    def test_create_from_callable_success_no_exception(self) -> None:
        def safe_operation() -> str:
            return "success"

        result: p.Result[str] = r[str].create_from_callable(safe_operation)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq="success")
        tm.that(result.exception, none=True)

    def test_create_from_callable_none_return_is_failure(self) -> None:
        def empty_operation() -> str | None:
            return None

        result: p.Result[str] = r[str].create_from_callable(empty_operation)
        tm.that(result.failure, eq=True)
        tm.that(result.exception, none=True)

    def test_create_from_callable_with_error_code(self) -> None:
        def failing_operation() -> int:
            msg = "invalid value"
            raise ValueError(msg)

        result: p.Result[int] = r[int].create_from_callable(
            failing_operation, error_code="INVALID_VALUE"
        )
        tm.that(result.failure, eq=True)
        tm.that(result.error_code, eq="INVALID_VALUE")
        tm.that(result.exception, none=False)
