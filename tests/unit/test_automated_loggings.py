"""Real API tests for FlextLogger."""

from __future__ import annotations

from time import perf_counter

from flext_tests import tm, tt, u
from hypothesis import given, strategies as st

from flext_core import FlextLogger, FlextRuntime


class TestAutomatedFlextLogger:
    def setup_method(self) -> None:
        FlextRuntime.reset_structlog_state_for_testing()
        _ = FlextLogger.clear_global_context()

    def test_create_module_logger(self) -> None:
        logger = FlextLogger.create_module_logger("tests.logger")
        tm.that(logger.name, eq="tests.logger")

    def test_standard_log_methods_return_success(self) -> None:
        logger = FlextLogger.create_module_logger("tests.logger.methods")
        tm.ok(logger.info("info-message"), eq=True)
        tm.ok(logger.warning("warning-message"), eq=True)
        tm.ok(logger.error("error-message"), eq=True)
        tm.ok(logger.debug("debug-message"), eq=True)

    def test_bind_unbind_and_new(self) -> None:
        logger = FlextLogger.create_module_logger("tests.logger.bind")
        request_id = u.generate("ulid", 8)
        bound = logger.bind(request_id=request_id)
        unbound = bound.unbind("request_id", safe=True)
        fresh = logger.new(flow_id=request_id)
        tm.that(bound.name, eq=logger.name)
        tm.that(unbound.name, eq=logger.name)
        tm.that(fresh.name, eq=logger.name)

    def test_global_context_bind_and_clear(self) -> None:
        correlation_id = u.generate()
        bind_result = FlextLogger.bind_global_context(correlation_id=correlation_id)
        clear_result = FlextLogger.clear_global_context()
        tm.ok(bind_result, eq=True)
        tm.ok(clear_result, eq=True)

    @given(st.text(min_size=1, max_size=100))
    def test_hypothesis_info_returns_result(self, message: str) -> None:
        logger = FlextLogger.create_module_logger("tests.logger.hypothesis")
        tm.ok(logger.info(message), eq=True)

    def test_benchmark_info_throughput(self) -> None:
        logger = FlextLogger.create_module_logger("tests.logger.benchmark")
        users = tt.batch("user", count=1)
        tm.that(len(users), gt=0)
        start = perf_counter()
        for _ in range(1200):
            tm.ok(logger.info("benchmark"), eq=True)
        elapsed = perf_counter() - start
        tm.that(elapsed, gt=0.0)
