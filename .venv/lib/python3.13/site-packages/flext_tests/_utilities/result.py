"""Extracted mixin for flext_tests."""

from __future__ import annotations

from types import EllipsisType

from flext_core import p
from flext_tests import c


class FlextTestsResultUtilitiesMixin:
    """Result helpers for test assertions."""

    @staticmethod
    def assert_failure[TResult](
        result: p.ResultView[TResult], expected_error: str | None = None
    ) -> str:
        """Assert result is failure and return error message."""
        if result.success:
            msg = f"Expected failure but got success: {result.value}"
            raise AssertionError(msg)
        error = result.error
        if error is None:
            msg = "Expected error but got None"
            raise AssertionError(msg)
        if expected_error and expected_error not in error:
            msg = f"Expected error containing '{expected_error}' but got: {error}"
            raise AssertionError(msg)
        return f"{error}"

    @staticmethod
    def assert_success[TResult](
        result: p.ResultView[TResult],
        error_msg: str | None = None,
        *,
        expected_value: TResult | EllipsisType = ...,
    ) -> TResult:
        """Assert result is success, optionally validate the value, and return it."""
        # mro-p68a.17.3.2.1.7: observation stays covariant across result payloads.
        if not result.success:
            raise AssertionError(
                error_msg or c.Tests.ERR_OK_FAILED.format(error=result.error)
            )
        value: TResult = result.value
        if expected_value is not ... and value != expected_value:
            msg = f"Expected success value {expected_value!r} but got {value!r}"
            raise AssertionError(msg)
        return value
