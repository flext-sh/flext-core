"""Behavioral tests for r[T] failure results that carry exception context.

Every assertion targets the public FlextResult contract a caller depends on:
failure state, error/error_code/error_data payload, the carried exception, and
how all of that propagates through the result combinators. No private
attributes, no internal patching, no mock spying on the unit under test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import pytest

from flext_tests import r, tm
from tests.constants import c
from tests.models import m
from tests.typings import t

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextCoreResultExceptionFailures:
    @pytest.mark.parametrize(
        ("error_msg", "expected_error"),
        [
            ("Operation failed", "Operation failed"),
            ("Division by zero", "Division by zero"),
            (None, ""),
        ],
    )
    def test_fail_without_exception_exposes_error_and_no_exception(
        self, error_msg: str | None, expected_error: str
    ) -> None:
        result: p.Result[int] = r[int].fail(error_msg)

        tm.that(result.failure, eq=True)
        tm.that(result.success, eq=False)
        tm.that(result.error, eq=expected_error)
        tm.that(result.exception, none=True)

    def test_fail_preserves_carried_exception_identity_and_type(self) -> None:
        exc = ZeroDivisionError("cannot divide by zero")

        result: p.Result[float] = r[float].fail("Division by zero", exception=exc)

        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="Division by zero")
        tm.that(result.exception is exc, eq=True)
        tm.that(result.exception, is_=ZeroDivisionError)

    def test_fail_with_none_error_and_exception_normalizes_error_to_empty(self) -> None:
        exc = RuntimeError("something went wrong")

        result: p.Result[int] = r[int].fail(None, exception=exc)

        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="")
        tm.that(result.exception is exc, eq=True)

    def test_fail_exposes_error_code_alongside_exception(self) -> None:
        exc = ValueError("expected integer")

        result: p.Result[str] = r[str].fail(
            "Invalid input", error_code="INVALID_INPUT", exception=exc
        )

        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="Invalid input")
        tm.that(result.error_code, eq="INVALID_INPUT")
        tm.that(result.exception is exc, eq=True)

    def test_fail_exposes_error_data_mapping_alongside_exception(self) -> None:
        error_data: t.StrMapping = {"field": "email", "reason": "invalid format"}
        exc = ValueError("invalid email")

        result: p.Result[t.StrMapping] = r[t.StrMapping].fail(
            "Validation failed", error_data=error_data, exception=exc
        )

        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="Validation failed")
        tm.that(result.error_data, none=False)
        if result.error_data is not None:
            tm.that(result.error_data.get("field"), eq="email")
            tm.that(result.error_data.get("reason"), eq="invalid format")
        tm.that(result.exception is exc, eq=True)

    def test_fail_enriches_error_data_from_exception_metadata(self) -> None:
        class MetadataError(ValueError):
            metadata: m.Metadata
            correlation_id: str

            def __init__(self) -> None:
                super().__init__("invalid email")
                self.metadata = m.Metadata(
                    attributes={"field": "email", "details": {"retryable": False}}
                )
                self.correlation_id = "corr-123"

        result: p.Result[str] = r[str].fail(
            "Validation failed", exception=MetadataError()
        )

        tm.that(result.failure, eq=True)
        tm.that(result.error_data, none=False)
        if result.error_data is not None:
            tm.that(result.error_data.get("field"), eq="email")
            tm.that(result.error_data.get("details"), eq={"retryable": False})
            tm.that(result.error_data.get(c.ContextKey.CORRELATION_ID), eq="corr-123")

    def test_map_on_failure_short_circuits_and_keeps_exception(self) -> None:
        exc = ValueError("boom")
        failure: p.Result[int] = r[int].fail("bad", exception=exc)

        mapped: p.Result[int] = failure.map(lambda value: value + 1)

        tm.that(mapped.failure, eq=True)
        tm.that(mapped.error, eq="bad")
        tm.that(mapped.exception is exc, eq=True)

    def test_flat_map_on_failure_short_circuits_and_keeps_exception(self) -> None:
        exc = ValueError("boom")
        failure: p.Result[int] = r[int].fail("bad", exception=exc)

        chained: p.Result[int] = failure.flat_map(lambda value: r[int].ok(value))

        tm.that(chained.failure, eq=True)
        tm.that(chained.exception is exc, eq=True)

    def test_map_error_transforms_message_and_preserves_exception(self) -> None:
        exc = ValueError("boom")
        failure: p.Result[int] = r[int].fail("bad", exception=exc)

        remapped: p.Result[int] = failure.map_error(lambda msg: msg.upper())

        tm.that(remapped.failure, eq=True)
        tm.that(remapped.error, eq="BAD")
        tm.that(remapped.exception is exc, eq=True)

    def test_unwrap_or_returns_default_for_carried_failure(self) -> None:
        failure: p.Result[int] = r[int].fail("bad", exception=ValueError("boom"))

        tm.that(failure.unwrap_or(99), eq=99)

    def test_unwrap_raises_on_carried_failure(self) -> None:
        failure: p.Result[int] = r[int].fail("bad", exception=ValueError("boom"))

        with pytest.raises(RuntimeError, match="bad"):
            failure.unwrap()

    def test_recover_produces_success_from_carried_failure(self) -> None:
        failure: p.Result[int] = r[int].fail("bad", exception=ValueError("boom"))

        recovered: p.Result[int] = failure.recover(lambda _error: 7)

        tm.that(recovered.success, eq=True)
        tm.that(recovered.unwrap(), eq=7)

    def test_typed_value_model_survives_recover_from_failure(self) -> None:
        class UserModel(m.Value):
            name: Annotated[str, m.Field(description="User name")]
            age: Annotated[int, m.Field(description="User age")]

        fallback = UserModel(name="anon", age=0)
        failure: p.Result[UserModel] = r[UserModel].fail(
            "lookup failed", exception=KeyError("missing")
        )

        recovered: p.Result[UserModel] = failure.recover(lambda _error: fallback)

        tm.that(recovered.success, eq=True)
        tm.that(recovered.unwrap().name, eq="anon")
        tm.that(recovered.unwrap().age, eq=0)
