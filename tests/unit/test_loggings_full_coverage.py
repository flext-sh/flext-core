"""Behavior contract for flext_core.loggings.FlextLogger — public API only."""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests import u


class TestsFlextCoreLoggings:
    """Behavior contract for FlextLogger public API: create, bind, unbind, log, track."""

    def test_create_module_logger_returns_usable_logger_instance(self) -> None:
        logger = u.create_module_logger("test_create")
        tm.that(logger, none=False)
        logger.info("ready")

    def test_bind_returns_logger_accepting_subsequent_log_calls(self) -> None:
        bound = u.create_module_logger("test_bind").bind(
            service_name="svc",
            correlation_id="cid",
        )
        tm.that(bound, none=False)
        bound.info("bound ok")
        bound.debug("debug")
        bound.warning("warn")
        bound.error("err")

    def test_new_returns_fresh_bound_logger_without_prior_context(self) -> None:
        logger = u.create_module_logger("test_new").bind(initial="x")
        refreshed = logger.new(fresh="y")
        tm.that(refreshed, none=False)
        refreshed.info("new ok")

    def test_unbind_with_safe_flag_ignores_missing_keys(self) -> None:
        logger = u.create_module_logger("test_unbind_safe").bind(a="1")
        result = logger.unbind("missing", safe=True)
        tm.that(result, none=False)

    def test_unbind_without_safe_raises_on_missing_key(self) -> None:
        logger = u.create_module_logger("test_unbind_strict")
        with pytest.raises(KeyError):
            logger.unbind("missing")

    def test_build_exception_context_captures_exception_metadata(self) -> None:
        logger = u.create_module_logger("test_exc")
        context = logger.build_exception_context(
            exception=ValueError("boom"),
            exc_info=False,
            context={"op": "test"},
        )
        tm.that(context, is_=dict, has="exception_type")

    def test_build_exception_context_without_exception_returns_context_dict(
        self,
    ) -> None:
        logger = u.create_module_logger("test_exc_none")
        context = logger.build_exception_context(
            exception=None,
            exc_info=False,
            context={"op": "test"},
        )
        tm.that(context, is_=dict)

    def test_performance_tracker_context_manager_completes_without_error(self) -> None:
        logger = u.create_module_logger("test_perf")
        with u.PerformanceTracker(logger, "operation_under_test"):
            result = 1 + 1
        tm.that(result, eq=2)
