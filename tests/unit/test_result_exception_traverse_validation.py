"""Result exception traversal and validation tests."""

from __future__ import annotations

from flext_tests import r, tm

from tests.constants import c
from tests.protocols import p
from tests.unit._result_exception_support import TestsFlextResultExceptionCarrying


class TestsFlextResultExceptionTraverseValidation(TestsFlextResultExceptionCarrying):
    def test_traverse_propagates_exception(self) -> None:
        exc = ValueError("item error")
        items = [1, 2, 3]

        def process_with_failure(value: int) -> p.Result[int]:
            if value == 2:
                return r[int].fail("error", exception=exc)
            return r[int].ok(value * 2)

        result = r[int].traverse(items, process_with_failure, fail_fast=True)
        tm.that(result.failure, eq=True)
        tm.that(result.exception is exc, eq=True)

    def test_traverse_callback_exception_returns_failure(self) -> None:
        exc = RuntimeError("traverse callback failed")

        def process_with_exception(value: int) -> p.Result[int]:
            if value == 2:
                raise exc
            return r[int].ok(value * 2)

        result = r[int].traverse([1, 2, 3], process_with_exception, fail_fast=True)

        tm.that(result.failure, eq=True)
        tm.that(result.exception is exc, eq=True)
        tm.that(result.error, eq=str(exc))

    def test_traverse_accumulate_preserves_exceptions(self) -> None:
        exc1 = ValueError("error 1")
        exc2 = TypeError("error 2")
        items = [1, 2, 3]

        def process_with_failures(value: int) -> p.Result[int]:
            if value == 1:
                return r[int].fail("error 1", exception=exc1)
            if value == 3:
                return r[int].fail("error 2", exception=exc2)
            return r[int].ok(value * 2)

        result = r[int].traverse(items, process_with_failures, fail_fast=False)
        tm.that(result.failure, eq=True)
        tm.that(result.error, none=False)
        if result.error is not None:
            tm.that("error 1" in result.error and "error 2" in result.error, eq=True)

    def test_traverse_accumulates_callback_exceptions(self) -> None:
        def process_with_exception(value: int) -> p.Result[int]:
            if value in {1, 3}:
                raise RuntimeError(f"callback error {value}")
            return r[int].ok(value * 2)

        result = r[int].traverse([1, 2, 3], process_with_exception, fail_fast=False)

        tm.that(result.failure, eq=True)
        tm.that(result.error, none=False)
        if result.error is not None:
            tm.that("callback error 1" in result.error, eq=True)
            tm.that("callback error 3" in result.error, eq=True)

    def test_from_validation_carries_exception(self) -> None:
        invalid_data = {"name": "Alice", "age": "not_an_int"}
        result = r[TestsFlextResultExceptionCarrying.UserModel].from_validation(
            invalid_data,
            TestsFlextResultExceptionCarrying.UserModel,
        )
        tm.that(result.failure, eq=True)
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=c.ValidationError)

    def test_error_or_pattern_unchanged(self) -> None:
        result_success = r[int].ok(42)
        result_failure: p.Result[int] = r[int].fail("error message")
        error_success = result_success.error or "fallback"
        error_failure = result_failure.error or "fallback"
        tm.that(error_success, eq="fallback")
        tm.that(error_failure, eq="error message")

    def test_error_or_pattern_with_exception(self) -> None:
        exc = RuntimeError("runtime error")
        result: p.Result[int] = r[int].fail("error", exception=exc)
        error_msg = result.error or "fallback"
        tm.that(error_msg, eq="error")
        tm.that(result.exception is exc, eq=True)

    def test_ok_bool_succeeds(self) -> None:
        result = r[bool].ok(True)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=True)

    def test_ok_with_valid_value_succeeds(self) -> None:
        value = 42
        result: p.Result[int] = r[int].ok(value)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=42)
        tm.that(result.exception, none=True)
