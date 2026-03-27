from __future__ import annotations

from collections.abc import Sequence, Sized
from typing import cast

from flext_tests import tm
from pydantic import BaseModel, ValidationError

from flext_core import r
from tests import t


class TestResultExceptionCarrying:
    class BrokenSized:
        """Sized t.NormalizedValue that raises on __len__."""

        def __len__(self) -> int:
            """Raise TypeError on length call."""
            msg = "no length"
            raise TypeError(msg)

    class UserModel(BaseModel):
        """User model for testing."""

        name: str
        age: int

    def test_fail_no_exception_backward_compat(self) -> None:
        error_msg = "Operation failed"
        result: r[int] = r[int].fail(error_msg)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error, eq=error_msg)
        tm.that(result.exception, none=True)
        result_as_obj = result
        tm.that(hasattr(result_as_obj, "is_success"), eq=True)
        tm.that(hasattr(result_as_obj, "is_failure"), eq=True)

    def test_fail_with_exception(self) -> None:
        error_msg = "Division by zero"
        exc = ZeroDivisionError("cannot divide by zero")
        result: r[float] = r[float].fail(error_msg, exception=exc)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error, eq=error_msg)
        tm.that(result.exception is exc, eq=True)
        tm.that(result.exception, is_=ZeroDivisionError)

    def test_fail_with_exception_and_error_code(self) -> None:
        error_msg = "Invalid input"
        error_code = "INVALID_INPUT"
        exc = ValueError("expected integer")
        result: r[str] = r[str].fail(error_msg, error_code=error_code, exception=exc)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error, eq=error_msg)
        tm.that(result.error_code, eq=error_code)
        tm.that(result.exception is exc, eq=True)

    def test_fail_with_exception_and_error_data(self) -> None:
        error_msg = "Validation failed"
        error_data: t.StrMapping = {
            "field": "email",
            "reason": "invalid format",
        }
        exc = ValueError("invalid email")
        result: r[t.StrMapping] = r[t.StrMapping].fail(
            error_msg,
            error_data=error_data,
            exception=exc,
        )
        tm.that(result.is_failure, eq=True)
        tm.that(result.error, eq=error_msg)
        tm.that(result.error_data, none=False)
        if result.error_data is not None:
            normalized_data = cast("BaseModel", result.error_data).model_dump()
            tm.that(normalized_data.get("field"), eq="email")
            tm.that(normalized_data.get("reason"), eq="invalid format")
        tm.that(result.exception is exc, eq=True)

    def test_fail_with_none_error_and_exception(self) -> None:
        exc = RuntimeError("something went wrong")
        result: r[int] = r[int].fail(None, exception=exc)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error, eq="")
        tm.that(result.exception is exc, eq=True)

    def test_safe_carries_exception(self) -> None:
        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        result: r[float] = divide(10, 0)
        tm.that(result.is_failure, eq=True)
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

        result: r[int] = add(5, 3)
        tm.that(result.is_success, eq=True)
        tm.that(result.value, eq=8)
        tm.that(result.exception, none=True)

    def test_safe_captures_value_error(self) -> None:
        @r.safe
        def parse_int(value: str) -> int:
            if not value.isdigit():
                raise ValueError(f"'{value}' is not a valid integer")
            return int(value)

        result: r[int] = parse_int("abc")
        tm.that(result.is_failure, eq=True)
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=ValueError)

    def test_safe_captures_type_error(self) -> None:
        @r.safe
        def get_length(obj: Sized) -> int:
            return len(obj)

        result: r[int] = get_length(self.BrokenSized())
        tm.that(result.is_failure, eq=True)
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=TypeError)

    def test_create_from_callable_carries_exception(self) -> None:
        def risky_operation() -> int:
            msg = "operation failed"
            raise RuntimeError(msg)

        result: r[int] = r[int].create_from_callable(risky_operation)
        tm.that(result.is_failure, eq=True)
        tm.that(
            result.error is not None and "operation failed" in result.error,
            eq=True,
        )
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=RuntimeError)

    def test_create_from_callable_success_no_exception(self) -> None:
        def safe_operation() -> str:
            return "success"

        result: r[str] = r[str].create_from_callable(safe_operation)
        tm.that(result.is_success, eq=True)
        tm.that(result.value, eq="success")
        tm.that(result.exception, none=True)

    def test_create_from_callable_with_error_code(self) -> None:
        def failing_operation() -> int:
            msg = "invalid value"
            raise ValueError(msg)

        result: r[int] = r[int].create_from_callable(
            failing_operation,
            error_code="INVALID_VALUE",
        )
        tm.that(result.is_failure, eq=True)
        tm.that(result.error_code, eq="INVALID_VALUE")
        tm.that(result.exception, none=False)

    def test_map_propagates_exception_on_failure(self) -> None:
        exc = ValueError("original error")
        result: r[int] = r[int].fail("error", exception=exc)
        mapped: r[int] = result.map(lambda value: value * 2)
        tm.that(mapped.is_failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    def test_map_success_no_exception(self) -> None:
        result: r[int] = r[int].ok(5)
        mapped: r[int] = result.map(lambda value: value * 2)
        tm.that(mapped.is_success, eq=True)
        tm.that(mapped.value, eq=10)
        tm.that(mapped.exception, none=True)

    def test_map_chain_preserves_exception(self) -> None:
        exc = RuntimeError("chain error")
        result: r[int] = r[int].fail("error", exception=exc)
        mapped: r[int] = result.map(lambda value: value + 1).map(
            lambda value: value * 2,
        )
        tm.that(mapped.is_failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    def test_flat_map_propagates_exception_on_failure(self) -> None:
        exc = TypeError("type error")
        result: r[int] = r[int].fail("error", exception=exc)
        flat_mapped: r[str] = result.flat_map(lambda value: r[str].ok(str(value)))
        tm.that(flat_mapped.is_failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    def test_flat_map_success_no_exception(self) -> None:
        result: r[int] = r[int].ok(5)
        flat_mapped: r[str] = result.flat_map(lambda value: r[str].ok(str(value)))
        tm.that(flat_mapped.is_success, eq=True)
        tm.that(flat_mapped.value, eq="5")
        tm.that(flat_mapped.exception, none=True)

    def test_flat_map_chain_preserves_exception(self) -> None:
        exc = KeyError("missing key")
        result: r[int] = r[int].fail("error", exception=exc)
        flat_mapped: r[str] = result.flat_map(
            lambda value: r[int].ok(value + 1),
        ).flat_map(
            lambda value: r[str].ok(str(value)),
        )
        tm.that(flat_mapped.is_failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    def test_alt_propagates_exception(self) -> None:
        exc = ValueError("original")
        result: r[int] = r[int].fail("error", exception=exc)
        altered: r[int] = result.map_error(lambda error: f"transformed: {error}")
        tm.that(altered.is_failure, eq=True)
        tm.that(altered.exception is exc, eq=True)
        tm.that(altered.error is not None and "transformed" in altered.error, eq=True)

    def test_alt_success_no_exception(self) -> None:
        result: r[int] = r[int].ok(42)
        altered: r[int] = result.map_error(lambda error: f"error: {error}")
        tm.that(altered.is_success, eq=True)
        tm.that(altered.value, eq=42)
        tm.that(altered.exception, none=True)

    def test_lash_propagates_exception(self) -> None:
        exc = RuntimeError("recovery needed")
        result: r[int] = r[int].fail("error", exception=exc)
        recovered = result.lash(lambda _: r[int].ok(0))
        tm.that(recovered.is_success, eq=True)
        tm.that(recovered.value, eq=0)

    def test_lash_preserves_exception_on_recovery_failure(self) -> None:
        exc = ValueError("original error")
        result: r[int] = r[int].fail("error", exception=exc)
        recovery_exc = RuntimeError("recovery failed")
        recovered: r[int] = result.lash(
            lambda error: r[int].fail(
                f"recovery failed: {error}",
                exception=recovery_exc,
            ),
        )
        tm.that(recovered.is_failure, eq=True)
        tm.that(recovered.exception is recovery_exc, eq=True)

    def test_traverse_propagates_exception(self) -> None:
        exc = ValueError("item error")
        items = [1, 2, 3]

        def process_with_failure(value: int) -> r[int]:
            if value == 2:
                return r[int].fail("error", exception=exc)
            return r[int].ok(value * 2)

        result = r[int].traverse(items, process_with_failure, fail_fast=True)
        tm.that(result.is_failure, eq=True)
        tm.that(result.exception is exc, eq=True)

    def test_traverse_accumulate_preserves_exceptions(self) -> None:
        exc1 = ValueError("error 1")
        exc2 = TypeError("error 2")
        items = [1, 2, 3]

        def process_with_failures(value: int) -> r[int]:
            if value == 1:
                return r[int].fail("error 1", exception=exc1)
            if value == 3:
                return r[int].fail("error 2", exception=exc2)
            return r[int].ok(value * 2)

        result = r[int].traverse(items, process_with_failures, fail_fast=False)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error, none=False)
        if result.error is not None:
            tm.that("error 1" in result.error and "error 2" in result.error, eq=True)

    def test_from_validation_carries_exception(self) -> None:
        invalid_data = {"name": "Alice", "age": "not_an_int"}
        result = r[TestResultExceptionCarrying.UserModel].from_validation(
            invalid_data,
            TestResultExceptionCarrying.UserModel,
        )
        tm.that(result.is_failure, eq=True)
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=ValidationError)

    def test_error_or_pattern_unchanged(self) -> None:
        result_success = r[int].ok(42)
        result_failure: r[int] = r[int].fail("error message")
        error_success = result_success.error or "fallback"
        error_failure = result_failure.error or "fallback"
        tm.that(error_success, eq="fallback")
        tm.that(error_failure, eq="error message")

    def test_error_or_pattern_with_exception(self) -> None:
        exc = RuntimeError("runtime error")
        result: r[int] = r[int].fail("error", exception=exc)
        error_msg = result.error or "fallback"
        tm.that(error_msg, eq="error")
        tm.that(result.exception is exc, eq=True)

    def test_ok_none_succeeds(self) -> None:
        result = r[int | None].ok(None)
        tm.that(result.is_success, eq=True)
        tm.that(result.value, none=True)

    def test_ok_with_valid_value_succeeds(self) -> None:
        value = 42
        result: r[int] = r[int].ok(value)
        tm.that(result.is_success, eq=True)
        tm.that(result.value, eq=42)
        tm.that(result.exception, none=True)

    def test_monadic_operations_unchanged(self) -> None:
        result: r[int] = r[int].ok(5)
        final: r[int] = (
            result
            .map(lambda value: value * 2)
            .flat_map(lambda value: r[int].ok(value + 1))
            .filter(lambda value: value > 5)
        )
        tm.that(final.is_success, eq=True)
        tm.that(final.value, eq=11)
        tm.that(final.exception, none=True)

    def test_monadic_chain_with_exception_in_middle(self) -> None:
        exc = ValueError("chain error")
        result = r[int].ok(5)
        final: r[int] = (
            result
            .map(lambda value: value * 2)
            .flat_map(lambda _: r[int].fail("error", exception=exc))
            .map(lambda value: value + 1)
        )
        tm.that(final.is_failure, eq=True)
        tm.that(final.exception is exc, eq=True)

    def test_recover_with_exception(self) -> None:
        exc = RuntimeError("recovery needed")
        result: r[int] = r[int].fail("error", exception=exc)
        recovered: r[int] = result.recover(lambda _: 0)
        tm.that(recovered.is_success, eq=True)
        tm.that(recovered.value, eq=0)

    def test_fold_with_exception(self) -> None:
        exc = ValueError("fold error")
        result: r[int] = r[int].fail("error", exception=exc)
        folded: str = result.fold(
            on_failure=lambda error: f"failed: {error}",
            on_success=lambda value: f"success: {value}",
        )
        tm.that(folded, eq="failed: error")

    def test_tap_with_exception(self) -> None:
        exc = RuntimeError("tap error")
        result: r[int] = r[int].fail("error", exception=exc)
        side_effect_called = False

        def side_effect(_: int) -> None:
            nonlocal side_effect_called
            side_effect_called = True

        tapped: r[int] = result.tap(side_effect)
        tm.that(tapped.is_failure, eq=True)
        tm.that(not side_effect_called, eq=True)
        tm.that(tapped.exception is exc, eq=True)

    def test_tap_error_with_exception(self) -> None:
        exc = ValueError("tap_error test")
        result: r[int] = r[int].fail("error", exception=exc)
        side_effect_called = False

        def side_effect(_: str) -> None:
            nonlocal side_effect_called
            side_effect_called = True

        tapped: r[int] = result.tap_error(side_effect)
        tm.that(tapped.is_failure, eq=True)
        tm.that(side_effect_called, eq=True)
        tm.that(tapped.exception is exc, eq=True)

    def test_exception_property_none_on_success(self) -> None:
        result: r[int] = r[int].ok(42)
        exc = result.exception
        tm.that(exc, none=True)

    def test_exception_property_set_on_failure(self) -> None:
        exc = RuntimeError("test error")
        result: r[int] = r[int].fail("error", exception=exc)
        retrieved_exc = result.exception
        tm.that(retrieved_exc is exc, eq=True)
        tm.that(retrieved_exc, is_=RuntimeError)

    def test_exception_property_type_check(self) -> None:
        exc = ValueError("value error")
        result: r[int] = r[int].fail("error", exception=exc)
        retrieved_exc = result.exception
        tm.that(retrieved_exc, is_=BaseException)
        tm.that(retrieved_exc, is_=ValueError)

    def test_exception_property_multiple_exception_types(self) -> None:
        exceptions: Sequence[BaseException] = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            KeyError("key error"),
            ZeroDivisionError("zero division"),
        ]
        for exc in exceptions:
            result: r[int] = r[int].fail("error", exception=exc)
            tm.that(result.exception is exc, eq=True)
            tm.that(result.exception, is_=type(exc))


__all__ = ["TestResultExceptionCarrying"]
