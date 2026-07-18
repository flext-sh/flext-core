"""Behavioral tests for r traverse, validation, and exception carrying.

All assertions target the public FlextResult contract (success/failure,
value, error, exception, combinators) and the FlextExceptions family via
from_validation. No private attributes or internal collaborators are touched.
"""

from __future__ import annotations


import pytest
from flext_tests import r, tm

from tests import c
from tests import t
from tests.unit._result_exception_support import TestsFlextResultExceptionCarrying

from tests import p


class TestsFlextCoreResultExceptionTraverseValidation(
    TestsFlextResultExceptionCarrying,
):
    @pytest.mark.parametrize(
        "raised",
        [ValueError("item error"), TypeError("bad type"), KeyError("missing")],
    )
    def test_traverse_fail_fast_carries_originating_failure_exception(
        self,
        raised: Exception,
    ) -> None:
        # Arrange
        def process(value: int) -> p.Result[int]:
            if value == 2:
                return r[int].fail("boom", exception=raised)
            return r[int].ok(value * 2)

        # Act
        result = r[int].traverse([1, 2, 3], process, fail_fast=True)

        # Assert
        tm.that(result.failure, eq=True)
        tm.that(result.exception is raised, eq=True)

    def test_traverse_fail_fast_wraps_callback_exception(self) -> None:
        # Arrange
        exc = RuntimeError("traverse callback failed")

        def process(value: int) -> p.Result[int]:
            if value == 2:
                raise exc
            return r[int].ok(value * 2)

        # Act
        result = r[int].traverse([1, 2, 3], process, fail_fast=True)

        # Assert
        tm.that(result.failure, eq=True)
        tm.that(result.exception is exc, eq=True)
        tm.that(result.error, eq=str(exc))

    def test_traverse_fail_fast_all_success_returns_mapped_sequence(self) -> None:
        # Arrange
        def double(value: int) -> p.Result[int]:
            return r[int].ok(value * 2)

        # Act
        result = r[int].traverse([1, 2, 3], double, fail_fast=True)

        # Assert
        tm.that(result.success, eq=True)
        tm.that(list(result.value), eq=[2, 4, 6])
        tm.that(result.exception, none=True)

    def test_traverse_accumulate_reports_every_failure(self) -> None:
        # Arrange
        def process(value: int) -> p.Result[int]:
            if value == 1:
                return r[int].fail("error 1", exception=ValueError("error 1"))
            if value == 3:
                return r[int].fail("error 2", exception=TypeError("error 2"))
            return r[int].ok(value * 2)

        # Act
        result = r[int].traverse([1, 2, 3], process, fail_fast=False)

        # Assert
        tm.that(result.failure, eq=True)
        tm.that(result.error, none=False)
        tm.that(result.error, has=["error 1", "error 2"])

    def test_traverse_accumulate_reports_every_callback_exception(self) -> None:
        # Arrange
        def process(value: int) -> p.Result[int]:
            if value in {1, 3}:
                msg = f"callback error {value}"
                raise RuntimeError(msg)
            return r[int].ok(value * 2)

        # Act
        result = r[int].traverse([1, 2, 3], process, fail_fast=False)

        # Assert
        tm.that(result.failure, eq=True)
        tm.that(result.error, none=False)
        tm.that(result.error, has=["callback error 1", "callback error 3"])

    def test_traverse_accumulate_all_success_returns_mapped_sequence(self) -> None:
        # Arrange
        def double(value: int) -> p.Result[int]:
            return r[int].ok(value * 2)

        # Act
        result = r[int].traverse([1, 2, 3], double, fail_fast=False)

        # Assert
        tm.that(result.success, eq=True)
        tm.that(list(result.value), eq=[2, 4, 6])

    def test_from_validation_failure_carries_validation_error(self) -> None:
        # Arrange
        invalid = {"name": "Alice", "age": "not_an_int"}

        # Act
        result = r[TestsFlextResultExceptionCarrying.UserModel].from_validation(
            invalid,
            TestsFlextResultExceptionCarrying.UserModel,
        )

        # Assert
        tm.that(result.failure, eq=True)
        tm.that(result.exception, none=False)
        tm.that(result.exception, is_=c.ValidationError)

    def test_from_validation_success_returns_populated_model(self) -> None:
        # Arrange
        valid: dict[str, t.JsonPayload] = {"name": "Alice", "age": 30}

        # Act
        result = r[TestsFlextResultExceptionCarrying.UserModel].from_validation(
            valid,
            TestsFlextResultExceptionCarrying.UserModel,
        )

        # Assert
        tm.that(result.success, eq=True)
        user = result.value
        tm.that(user.name, eq="Alice")
        tm.that(user.age, eq=30)
        tm.that(result.exception, none=True)

    def test_success_error_is_none_and_yields_fallback(self) -> None:
        # Arrange / Act
        result = r[int].ok(42)

        # Assert
        tm.that(result.error, none=True)
        tm.that(result.error or "fallback", eq="fallback")

    def test_failure_error_message_is_preserved(self) -> None:
        # Arrange / Act
        result: p.Result[int] = r[int].fail("error message")

        # Assert
        tm.that(result.error or "fallback", eq="error message")

    def test_failure_preserves_both_error_message_and_exception(self) -> None:
        # Arrange
        exc = RuntimeError("runtime error")

        # Act
        result: p.Result[int] = r[int].fail("error", exception=exc)

        # Assert
        tm.that(result.error, eq="error")
        tm.that(result.exception is exc, eq=True)

    def test_failure_short_circuits_map_and_unwrap_or_uses_default(self) -> None:
        # Arrange
        exc = RuntimeError("runtime error")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        # Act
        mapped = result.map(lambda value: value + 1)

        # Assert
        tm.that(mapped.failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)
        tm.that(result.unwrap_or(-1), eq=-1)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, True), (False, False), (42, 42)],
    )
    def test_ok_reports_success_and_wraps_value(
        self,
        value: bool | int,
        expected: bool | int,
    ) -> None:
        # Arrange / Act
        result = r[bool | int].ok(value)

        # Assert
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=expected)
        tm.that(result.exception, none=True)
