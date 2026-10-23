"""Focused public logging behaviors not covered by the core logger contract suite.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr
from functools import partial
from typing import TYPE_CHECKING, NoReturn

import pytest

from tests import c, m, tm, u

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from tests import p, t


class TestsFlextCoverageLoggings:
    """Tests for ``FlextCoverageLoggings``."""

    pytestmark = pytest.mark.usefixtures("restore_logging")

    @staticmethod
    @pytest.fixture
    def restore_logging() -> Generator[None]:
        """Select an observable severity without assuming the configured default."""
        original_settings, original_options = u.logging_settings(), u.logging_options()
        try:
            selected = u.resolve_logging_settings(log_level=c.LogLevel.DEBUG)
            u.configure_structlog(
                settings=m.StructlogOptions(async_logging=False),
                runtime_settings=selected,
            )
            yield
        finally:
            u.configure_structlog(
                settings=original_options, runtime_settings=original_settings
            )

    @staticmethod
    def make_result_logger(name: str) -> p.Logger:
        """Build a real public logger for behavior tests.

        Returns:
            The resulting ``p.Logger``.

        """
        return u.create_module_logger(name)

    @staticmethod
    def assert_log_result_success(result: p.Result[bool] | None) -> p.Result[bool]:
        """Provide ``assert_log_result_success``.

        Returns:
            The resulting ``p.Result[bool]``.

        Raises:
            AssertionError: If Expected result to not be None.

        """
        if result is None:
            msg = "Expected result to not be None"
            raise AssertionError(msg)
        tm.ok(result.success)
        tm.ok(result.value)
        return result

    def assert_captured_log_success(
        self,
        emit: Callable[[], p.Result[bool] | None],
        *,
        contains: str,
        expected_tokens: t.StrSequence = (),
    ) -> p.Result[bool]:
        """Provide ``assert_captured_log_success``.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        stream = io.StringIO()
        with redirect_stderr(stream):
            result = emit()

        output = stream.getvalue()
        tm.ok(contains in output)
        for token in expected_tokens:
            tm.ok(token in output)
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
        self,
        scope: str,
        context: t.MappingKV[str, str],
    ) -> None:
        """Test clear scope returns success for bound scopes."""
        bind_result = u.bind_context(scope=scope, **context)
        _ = self.assert_log_result_success(bind_result)

        result = u.clear_scope(scope)

        _ = self.assert_log_result_success(result)

    def test_clear_scope_ignores_unknown_scope(self) -> None:
        """Test clear scope ignores unknown scope."""
        result = u.clear_scope("nonexistent")

        _ = self.assert_log_result_success(result)

    def test_fetch_logger_returns_usable_service_logger(self) -> None:
        """Test fetch logger returns usable service logger."""
        logger = u.fetch_logger("user-service")

        tm.that(logger, none=False)
        result = self.assert_captured_log_success(
            lambda: logger.info("service ready"),
            contains="service ready",
        )
        tm.ok(result.value)

    def test_logging_interpolates_printf_arguments_once(self) -> None:
        """The native adapter retains the public printf-style call contract."""
        logger = self.make_result_logger("tests.loggings.format")

        result = self.assert_captured_log_success(
            lambda: logger.info("User %s logged in", "john"),
            contains="User john logged in",
        )
        tm.ok(result.value)

    def test_exception_logging_accepts_explicit_exception_and_context(self) -> None:
        """Test exception logging accepts explicit exception and context."""
        logger = self.make_result_logger("tests.loggings.exception")
        error_message = "disk failure"

        def _fail() -> NoReturn:
            raise OSError(error_message)

        try:
            _fail()
        except OSError as exc:
            result = self.assert_captured_log_success(
                partial(
                    logger.exception,
                    "io operation failed",
                    exception=exc,
                    operation="file_read",
                ),
                contains="io operation failed",
                expected_tokens=("file_read",),
            )

        tm.ok(result.value)

    def test_logging_with_large_context_emits_context_tokens(self) -> None:
        """Test logging with large context emits context tokens."""
        logger = self.make_result_logger("tests.loggings.context")
        large_context = {f"key_{index}": f"value_{index}" for index in range(10)}

        result = self.assert_captured_log_success(
            lambda: logger.info("Message with large context", **large_context),
            contains="Message with large context",
            expected_tokens=("key_9", "value_9"),
        )

        tm.ok(result.value)
