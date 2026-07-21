"""Focused public logging behaviors not covered by the core logger contract suite."""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import pytest

from flext_tests import tm
from tests import u

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests import p, t


class TestsFlextCoverageLoggings:
    """Verify logging coverage contracts."""

    @staticmethod
    def make_result_logger(name: str) -> p.Logger:
        """Build a real public logger for behavior tests."""
        return u.create_module_logger(name)

    @staticmethod
    def assert_log_result_success(
        result: p.ResultLike[bool] | None,
    ) -> p.ResultLike[bool]:
        if result is None:
            msg = "Expected result to not be None"
            raise AssertionError(msg)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=True)
        return result

    def assert_captured_log_success(
        self,
        emit: Callable[[], p.ResultLike[bool] | None],
        *,
        contains: str,
        expected_tokens: t.StrSequence = (),
    ) -> p.ResultLike[bool]:
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = emit()
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline:
                output = stream.getvalue()
                if contains in output:
                    break
                time.sleep(0.01)

        output = stream.getvalue()
        tm.that(contains in output, eq=True)
        for token in expected_tokens:
            tm.that(token in output, eq=True)
        return self.assert_log_result_success(result)

    @pytest.mark.parametrize(
        ("scope", "context"),
        [
            ("application", {"app_name": "test-app"}),
            ("request", {"correlation_id": "flext-123"}),
            ("operation", {"operation": "sync-users"}),
        ],
    )
    def test_clear_scope_returns_success_for_bound_scopes(
        self, scope: str, context: dict[str, str]
    ) -> None:
        bind_result = u.bind_context(scope=scope, **context)
        _ = self.assert_log_result_success(bind_result)

        result = u.clear_scope(scope)

        _ = self.assert_log_result_success(result)

    def test_clear_scope_ignores_unknown_scope(self) -> None:
        result = u.clear_scope("nonexistent")

        _ = self.assert_log_result_success(result)

    def test_fetch_logger_returns_usable_service_logger(self) -> None:
        logger = u.fetch_logger("user-service")

        tm.that(logger, none=False)
        result = self.assert_captured_log_success(
            lambda: logger.info("service ready"), contains="service ready"
        )
        tm.that(result.value, eq=True)

    def test_logging_preserves_message_template_with_format_arguments(self) -> None:
        logger = self.make_result_logger("tests.loggings.format")

        result = self.assert_captured_log_success(
            lambda: logger.info("User %s logged in", "john"),
            contains="User %s logged in",
        )
        tm.that(result.value, eq=True)

    def test_exception_logging_accepts_explicit_exception_and_context(self) -> None:
        logger = self.make_result_logger("tests.loggings.exception")
        error_message = "disk failure"

        try:
            raise OSError(error_message)
        except OSError as exc:
            result = self.assert_captured_log_success(
                lambda exc=exc: logger.exception(
                    "io operation failed", exception=exc, operation="file_read"
                ),
                contains="io operation failed",
                expected_tokens=("file_read",),
            )

        tm.that(result.value, eq=True)

    def test_logging_with_large_context_emits_context_tokens(self) -> None:
        logger = self.make_result_logger("tests.loggings.context")
        large_context = {f"key_{index}": f"value_{index}" for index in range(10)}

        result = self.assert_captured_log_success(
            lambda: logger.info("Message with large context", **large_context),
            contains="Message with large context",
            expected_tokens=("key_9", "value_9"),
        )

        tm.that(result.value, eq=True)


__all__: list[str] = ["TestsFlextCoverageLoggings"]
