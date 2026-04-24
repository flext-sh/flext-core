"""Behavior contract for flext_core.loggings.FlextLogger — public API only."""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests import p, u

LOG_LEVELS: tuple[str, ...] = (
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "trace",
)


class TestsFlextCoreLoggings:
    """Behavior contract for FlextLogger public API: create, bind, log, track, strict returns."""

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
        bound.info("bound ok")

    def test_new_returns_fresh_bound_logger_without_prior_context(
        self,
        logger: p.Logger,
    ) -> None:
        refreshed = logger.bind(initial="x").new(fresh="y")
        tm.that(refreshed, none=False)
        refreshed.info("new ok")

    def test_unbind_with_safe_flag_ignores_missing_keys(
        self,
        logger: p.Logger,
    ) -> None:
        bound = logger.bind(a="1")
        tm.that(bound.unbind("missing", safe=True), none=False)

    def test_unbind_without_safe_raises_on_missing_key(
        self,
        logger: p.Logger,
    ) -> None:
        with pytest.raises(KeyError):
            logger.unbind("missing")

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
        with u.PerformanceTracker(logger, "operation_under_test"):
            result = 1 + 1
        tm.that(result, eq=2)

    @pytest.mark.parametrize("level", LOG_LEVELS)
    def test_every_log_level_returns_success_result_with_value_true(
        self,
        logger: p.Logger,
        level: str,
    ) -> None:
        result = getattr(logger, level)("test %s message", level)
        tm.ok(result)
        tm.that(result.value, eq=True)

    @pytest.mark.parametrize("level", LOG_LEVELS)
    def test_every_log_level_accepts_structured_kwargs_and_returns_success(
        self,
        logger: p.Logger,
        level: str,
    ) -> None:
        result = getattr(logger, level)(
            "test message",
            request_id="r1",
            actor="tester",
        )
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_log_method_accepts_level_as_first_argument(
        self,
        logger: p.Logger,
    ) -> None:
        for level in ("debug", "info", "warning", "error", "critical"):
            result = logger.log(level, "test %s message", level)
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
            result = logger.exception(message)
            tm.ok(result)
            tm.that(result.value, eq=True)
