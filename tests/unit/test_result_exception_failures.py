"""Result exception failure carrying tests."""

from __future__ import annotations

from flext_tests import tm

from tests import c, m, p, r, t
from tests.unit._result_exception_support import TestsFlextResultExceptionCarrying


class TestsFlextResultExceptionFailures(TestsFlextResultExceptionCarrying):
    def test_fail_no_exception_backward_compat(self) -> None:
        error_msg = "Operation failed"
        result: p.Result[int] = r[int].fail(error_msg)
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq=error_msg)
        tm.that(result.exception, none=True)

    def test_fail_with_exception(self) -> None:
        error_msg = "Division by zero"
        exc = ZeroDivisionError("cannot divide by zero")
        result: p.Result[float] = r[float].fail(error_msg, exception=exc)
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq=error_msg)
        tm.that(result.exception is exc, eq=True)
        tm.that(result.exception, is_=ZeroDivisionError)

    def test_fail_with_exception_and_error_code(self) -> None:
        error_msg = "Invalid input"
        error_code = "INVALID_INPUT"
        exc = ValueError("expected integer")
        result: p.Result[str] = r[str].fail(
            error_msg, error_code=error_code, exception=exc
        )
        tm.that(result.failure, eq=True)
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
        result: p.Result[t.StrMapping] = r[t.StrMapping].fail(
            error_msg,
            error_data=error_data,
            exception=exc,
        )
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq=error_msg)
        tm.that(result.error_data, none=False)
        if result.error_data is not None:
            tm.that(result.error_data.get("field"), eq="email")
            tm.that(result.error_data.get("reason"), eq="invalid format")
        tm.that(result.exception is exc, eq=True)

    def test_fail_with_exception_extracts_exception_metadata(self) -> None:
        class MetadataError(ValueError):
            metadata: m.Metadata
            correlation_id: str

            def __init__(self) -> None:
                super().__init__("invalid email")
                self.metadata = m.Metadata(
                    attributes={
                        "field": "email",
                        "details": {"retryable": False},
                    }
                )
                self.correlation_id = "corr-123"

        result: p.Result[str] = r[str].fail(
            "Validation failed",
            exception=MetadataError(),
        )

        tm.that(result.error_data, none=False)
        if result.error_data is not None:
            tm.that(result.error_data.get("field"), eq="email")
            tm.that(result.error_data.get("details"), eq={"retryable": False})
            tm.that(
                result.error_data.get(c.ContextKey.CORRELATION_ID),
                eq="corr-123",
            )

    def test_fail_with_none_error_and_exception(self) -> None:
        exc = RuntimeError("something went wrong")
        result: p.Result[int] = r[int].fail(None, exception=exc)
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="")
        tm.that(result.exception is exc, eq=True)
