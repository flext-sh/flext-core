"""Behavior contract for flext_core.loggings.FlextUtilitiesLogging — public API only.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr
from functools import partial
from typing import TYPE_CHECKING, ClassVar, NoReturn

import pytest

from tests import c, p, tm, u

if TYPE_CHECKING:
    from collections.abc import Callable

from tests import t

LOG_LEVELS: tuple[tuple[str, bool], ...] = (
    ("debug", True),
    ("info", True),
    ("warning", True),
    ("error", True),
    ("critical", True),
    ("trace", True),
)


class TestsFlextLoggings:
    """Behavior contract for the ``FlextUtilitiesLogging`` public API.

    Covers create, bind, log, track, and strict returns.
    """

    # Exercise all standard severities, including trace, then restore the policy.
    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [
        pytest.mark.usefixtures("restore_logging"),
        pytest.mark.parametrize("logging_trace", [True]),
    ]

    @classmethod
    def _assert_log_output[TResult: p.Result[bool] | None](
        cls,
        emit: Callable[[], TResult],
        *,
        contains: str,
        expect_output: bool = True,
        expected_tokens: t.StrSequence = (),
    ) -> TResult:
        stream = io.StringIO()
        with redirect_stderr(stream):
            result = emit()
        output = stream.getvalue()

        if expect_output:
            tm.ok(contains in output)
            for token in expected_tokens:
                tm.ok(token in output)
        else:
            tm.that(output, eq="")
        return result

    @pytest.fixture
    @staticmethod
    def logger() -> p.Logger:
        """Provide ``logger``.

        Returns:
            The resulting ``p.Logger``.

        """
        return u.create_module_logger("tests.flext_core.loggings")

    @staticmethod
    def test_create_module_logger_returns_usable_logger_instance(
        logger: p.Logger,
    ) -> None:
        """Test create module logger returns usable logger instance."""
        tm.that(logger, none=False)

    def test_bind_returns_logger_accepting_subsequent_log_calls(
        self,
        logger: p.Logger,
    ) -> None:
        """Test bind returns logger accepting subsequent log calls."""
        bound = logger.bind(service_name="svc", correlation_id="cid")
        tm.that(bound, none=False)
        result = self._assert_log_output(
            lambda: bound.info("bound ok"),
            contains="bound ok",
        )
        tm.ok(result)

    def test_new_returns_fresh_bound_logger_without_prior_context(
        self,
        logger: p.Logger,
    ) -> None:
        """Test new returns fresh bound logger without prior context."""
        refreshed = logger.bind(initial="x").new(fresh="y")
        tm.that(refreshed, none=False)
        result = self._assert_log_output(
            lambda: refreshed.info("new ok"),
            contains="new ok",
        )
        tm.ok(result)

    @staticmethod
    def test_unbind_missing_key_fails_loud(logger: p.Logger) -> None:
        """Test unbind missing key fails loud."""
        bound = logger.bind(a=c.ParserBooleanToken.ONE)
        with pytest.raises(KeyError):
            bound.unbind("missing")

    @staticmethod
    def test_unbind_without_safe_raises_on_missing_key(logger: p.Logger) -> None:
        """Test unbind without safe raises on missing key."""
        with pytest.raises(KeyError):
            logger.unbind("missing")

    def test_unbind_with_safe_ignores_missing_key(self, logger: p.Logger) -> None:
        """Test unbind with safe ignores missing key."""
        result = self._assert_log_output(
            lambda: logger.unbind("missing", safe=True).info("safe unbind ok"),
            contains="safe unbind ok",
        )
        tm.ok(result)

    def test_try_unbind_ignores_missing_key(self, logger: p.Logger) -> None:
        """Test try unbind ignores missing key."""
        result = self._assert_log_output(
            lambda: logger.try_unbind("missing").info("try unbind ok"),
            contains="try unbind ok",
        )
        tm.ok(result)

    @staticmethod
    def test_build_exception_context_captures_exception_metadata(
        logger: p.Logger,
    ) -> None:
        """Test build exception context captures exception metadata."""
        ctx = logger.build_exception_context(
            exception=ValueError("boom"),
            exc_info=False,
            context={"op": "test"},
        )
        tm.that(ctx, is_=dict, has="exception_type")

    @staticmethod
    def test_build_exception_context_without_exception_returns_context_dict(
        logger: p.Logger,
    ) -> None:
        """Test build exception context without exception returns context dict."""
        ctx = logger.build_exception_context(
            exception=None,
            exc_info=False,
            context={"op": "test"},
        )
        tm.that(ctx, is_=dict)

    def test_performance_tracker_context_manager_completes_without_error(
        self,
        logger: p.Logger,
    ) -> None:
        """Test performance tracker context manager completes without error."""
        result = 0

        def emit() -> p.Result[bool] | None:
            with u.PerformanceTracker(logger, "operation_under_test"):
                nonlocal result
                result = 1 + 1
            return None

        _ = self._assert_log_output(emit, contains="operation_under_test success")
        tm.that(result, eq=2)

    @pytest.mark.parametrize(("level", "expect_output"), LOG_LEVELS)
    def test_every_log_level_returns_success_result_with_value_true(
        self,
        logger: p.Logger,
        level: str,
        *,
        expect_output: bool,
    ) -> None:
        """Test every log level returns success result with value true."""
        result = self._assert_log_output(
            lambda: getattr(logger, level)("test %s message", level),
            contains=f"test {level} message",
            expect_output=expect_output,
        )
        tm.ok(result)
        tm.ok(result.value)

    @pytest.mark.parametrize(("level", "expect_output"), LOG_LEVELS)
    def test_every_log_level_accepts_structured_kwargs_and_returns_success(
        self,
        logger: p.Logger,
        level: str,
        *,
        expect_output: bool,
    ) -> None:
        """Test every log level accepts structured kwargs and returns success."""
        result = self._assert_log_output(
            lambda: getattr(logger, level)(
                "test message",
                request_id="r1",
                actor="tester",
            ),
            contains="test message",
            expect_output=expect_output,
        )
        tm.ok(result)
        tm.ok(result.value)

    def test_log_method_accepts_level_as_first_argument(self, logger: p.Logger) -> None:
        """Test log method accepts level as first argument."""
        for level, expect_output in LOG_LEVELS[:-1]:

            def emit(level: str = level) -> p.Result[bool]:
                return logger.log(level, "test %s message", level)

            result = self._assert_log_output(
                emit,
                contains=f"test {level} message",
                expect_output=expect_output,
            )
            tm.ok(result)
            tm.ok(result.value)

    def test_exception_log_captures_inside_except_block(self, logger: p.Logger) -> None:
        """Test exception log captures inside except block."""
        message = "captured"

        def _fail() -> NoReturn:
            raise ValueError(message)

        try:
            _fail()
        except ValueError:
            result = self._assert_log_output(
                lambda: logger.exception(message),
                contains=message,
            )
            tm.ok(result)
            tm.ok(result.value)

    def test_fetch_logger_returns_usable_service_logger(self) -> None:
        """Test fetch logger returns usable service logger."""
        service_logger = u.fetch_logger("user-service")

        tm.that(service_logger, none=False)
        result = self._assert_log_output(
            lambda: service_logger.info("service ready"),
            contains="service ready",
        )
        tm.ok(result)
        tm.ok(result.value)

    def test_exception_logging_accepts_explicit_exception_and_context(
        self,
        logger: p.Logger,
    ) -> None:
        """Test exception logging accepts explicit exception and context."""
        error_message = "disk failure"

        def _fail() -> NoReturn:
            raise OSError(error_message)

        try:
            _fail()
        except OSError as exc:
            result = self._assert_log_output(
                partial(
                    logger.exception,
                    "io operation failed",
                    exception=exc,
                    operation="file_read",
                ),
                contains="io operation failed",
                expected_tokens=("file_read",),
            )

        tm.ok(result)
        tm.ok(result.value)

    def test_logging_with_large_context_emits_context_tokens(
        self,
        logger: p.Logger,
    ) -> None:
        """Test logging with large context emits context tokens."""
        large_context = {f"key_{index}": f"value_{index}" for index in range(10)}

        result = self._assert_log_output(
            lambda: logger.info("Message with large context", **large_context),
            contains="Message with large context",
            expected_tokens=("key_9", "value_9"),
        )

        tm.ok(result)
        tm.ok(result.value)

    @staticmethod
    def test_log_source_points_to_call_site_not_logging_internals(
        logger: p.Logger,
    ) -> None:
        """Test log source points to call site not logging internals."""
        marker = "source probe"
        stream = io.StringIO()
        with redirect_stderr(stream):
            result = logger.info(marker)

        output = stream.getvalue()
        tm.ok(result)
        tm.ok(marker in output)
        tm.ok("tests/unit/test_loggings_full_coverage.py" in output)
        for internal_path in c.LOGGING_INTERNAL_PATH_FRAGMENTS:
            tm.fail(internal_path in output)
