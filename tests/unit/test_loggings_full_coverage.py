"""Behavior contract for flext_core.loggings.FlextUtilitiesLogging — public API only."""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from tests.protocols import p
from tests.utilities import u

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.typings import t

LOG_LEVELS: tuple[tuple[str, bool], ...] = (
    ("debug", False),
    ("info", True),
    ("warning", True),
    ("error", True),
    ("critical", True),
    ("trace", False),
)


class TestsFlextLoggings:
    """Behavior contract for FlextUtilitiesLogging public API: create, bind, log, track, strict returns."""

    @classmethod
    def _assert_log_output[TResult: p.ResultLike[bool] | None](
        cls,
        emit: Callable[[], TResult],
        *,
        contains: str,
        expect_output: bool = True,
        expected_tokens: t.StrSequence = (),
    ) -> TResult:
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = emit()
            deadline = time.monotonic() + 0.25
            while (
                expect_output
                and time.monotonic() < deadline
                and contains not in stream.getvalue()
            ):
                time.sleep(0.01)
        output = stream.getvalue()

        if expect_output:
            tm.that(contains in output, eq=True)
            for token in expected_tokens:
                tm.that(token in output, eq=True)
        else:
            tm.that(output, eq="")
        return result

    @pytest.fixture
    def logger(self) -> p.Logger:
        return u.create_module_logger("tests.flext_core.loggings")

    def test_create_module_logger_returns_usable_logger_instance(
        self,
        logger: p.Logger,
    ) -> None:
        tm.that(logger, none=False)

    def test_bind_returns_logger_accepting_subsequent_log_calls(
        self,
        logger: p.Logger,
    ) -> None:
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
        refreshed = logger.bind(initial="x").new(fresh="y")
        tm.that(refreshed, none=False)
        result = self._assert_log_output(
            lambda: refreshed.info("new ok"),
            contains="new ok",
        )
        tm.ok(result)

    def test_unbind_missing_key_fails_loud(
        self,
        logger: p.Logger,
    ) -> None:
        bound = logger.bind(a="1")
        with pytest.raises(KeyError):
            bound.unbind("missing")

    def test_unbind_without_safe_raises_on_missing_key(
        self,
        logger: p.Logger,
    ) -> None:
        with pytest.raises(KeyError):
            logger.unbind("missing")

    def test_unbind_with_safe_ignores_missing_key(
        self,
        logger: p.Logger,
    ) -> None:
        result = self._assert_log_output(
            lambda: logger.unbind("missing", safe=True).info("safe unbind ok"),
            contains="safe unbind ok",
        )
        tm.ok(result)

    def test_try_unbind_ignores_missing_key(
        self,
        logger: p.Logger,
    ) -> None:
        result = self._assert_log_output(
            lambda: logger.try_unbind("missing").info("try unbind ok"),
            contains="try unbind ok",
        )
        tm.ok(result)

    def test_build_exception_context_captures_exception_metadata(
        self,
        logger: p.Logger,
    ) -> None:
        ctx = logger.build_exception_context(
            exception=ValueError("boom"),
            exc_info=False,
            context={"op": "test"},
        )
        tm.that(ctx, is_=dict, has="exception_type")

    def test_build_exception_context_without_exception_returns_context_dict(
        self,
        logger: p.Logger,
    ) -> None:
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
        def emit() -> p.ResultLike[bool] | None:
            with u.PerformanceTracker(logger, "operation_under_test"):
                nonlocal result
                result = 1 + 1
            return None

        result = 0
        _ = self._assert_log_output(
            emit,
            contains="operation_under_test success",
        )
        tm.that(result, eq=2)

    @pytest.mark.parametrize(("level", "expect_output"), LOG_LEVELS)
    def test_every_log_level_returns_success_result_with_value_true(
        self,
        logger: p.Logger,
        level: str,
        expect_output: bool,
    ) -> None:
        result = self._assert_log_output(
            lambda: getattr(logger, level)("test %s message", level),
            contains="test %s message",
            expect_output=expect_output,
        )
        tm.ok(result)
        tm.that(result.value, eq=True)

    @pytest.mark.parametrize(("level", "expect_output"), LOG_LEVELS)
    def test_every_log_level_accepts_structured_kwargs_and_returns_success(
        self,
        logger: p.Logger,
        level: str,
        expect_output: bool,
    ) -> None:
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
        tm.that(result.value, eq=True)

    def test_log_method_accepts_level_as_first_argument(
        self,
        logger: p.Logger,
    ) -> None:
        for level, expect_output in LOG_LEVELS[:-1]:
            result = self._assert_log_output(
                lambda level=level: logger.log(level, "test %s message", level),
                contains="test %s message",
                expect_output=expect_output,
            )
            tm.ok(result)
            tm.that(result.value, eq=True)

    def test_exception_log_captures_inside_except_block(
        self,
        logger: p.Logger,
    ) -> None:
        message = "captured"
        try:
            raise ValueError(message)
        except ValueError:
            result = self._assert_log_output(
                lambda: logger.exception(message),
                contains=message,
            )
            tm.ok(result)
            tm.that(result.value, eq=True)

    def test_log_source_points_to_call_site_not_logging_internals(
        self,
        logger: p.Logger,
    ) -> None:
        marker = "source probe"
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = logger.info(marker)
            deadline = time.monotonic() + 0.25
            while marker not in stream.getvalue() and time.monotonic() < deadline:
                time.sleep(0.01)

        output = stream.getvalue()
        tm.ok(result)
        tm.that(marker in output, eq=True)
        tm.that("tests/unit/test_loggings_full_coverage.py" in output, eq=True)
        tm.that("_logging_context_parts" in output, eq=False)
        tm.that("_loggings_parts" in output, eq=False)
