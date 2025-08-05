"""Performance benchmarks and advanced testing for FLEXT Core.

This module contains comprehensive performance tests, benchmarks, and stress tests
for core FLEXT functionality, focusing on production performance characteristics.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import pytest

from flext_core.container import get_flext_container
from flext_core.core import FlextCore
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger, FlextLoggerFactory
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable


class TestPerformanceBenchmarks:
    """Performance benchmarks for critical FLEXT operations."""

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_flext_result_performance(self, benchmark: object) -> None:
        """Benchmark FlextResult creation and operations."""

        def create_and_chain_results() -> FlextResult[str]:
            """Create and chain multiple FlextResult operations."""
            return (
                FlextResult.ok("initial")
                .map(lambda x: f"{x}_step1")
                .map(lambda x: f"{x}_step2")
                .flat_map(lambda x: FlextResult.ok(f"{x}_final"))
            )

        benchmark_func = cast("Callable[[object], object]", benchmark)
        result = cast("FlextResult[str]", benchmark_func(create_and_chain_results))
        assert result.success
        assert result.data == "initial_step1_step2_final"

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_container_performance(self, benchmark: object) -> None:
        """Benchmark container registration and retrieval performance."""

        def container_operations() -> FlextResult[str]:
            """Perform multiple container operations."""
            container = get_flext_container()

            # Register services
            container.register("service1", "value1")
            container.register("service2", "value2")
            container.register("service3", "value3")

            # Retrieve services
            result1 = container.get("service1")
            result2 = container.get("service2")
            container.get("service3")

            if result1.success and result2.success:
                return FlextResult.ok("all_retrieved")
            return FlextResult.fail("retrieval_failed")

        benchmark_func = cast("Callable[[object], object]", benchmark)
        result = cast("FlextResult[bool]", benchmark_func(container_operations))
        assert result.success

    @pytest.mark.performance
    @pytest.mark.slow
    def test_logging_performance_bulk(self) -> None:
        """Test logging performance with bulk operations."""
        logger = FlextLogger("performance_test", "INFO")

        start_time = time.time()

        # Bulk logging operations
        for log_id in range(1000):
            logger.info(
                "Bulk log message %d",
                log_id,
                operation="bulk_test",
                iteration=log_id,
                timestamp=time.time(),
            )

        duration = time.time() - start_time

        # Should complete 1000 log operations in under 1 second
        assert duration < 1.0, f"Bulk logging took too long: {duration:.3f}s"

    @pytest.mark.performance
    @pytest.mark.architecture
    def test_handler_chain_performance(self) -> None:
        """Test handler chain performance with multiple handlers."""

        class FastHandler(FlextHandlers.Handler[str, str]):
            """Fast processing handler."""

            def handle(self, message: object) -> FlextResult[object]:
                return FlextResult.ok(f"processed_{message}")

        # Create handler chain
        chain = FlextHandlers.flext_create_chain()

        # Add multiple handlers
        for handler_id in range(10):
            chain.add_handler(FastHandler(f"handler_{handler_id}"))

        start_time = time.time()

        # Process messages through chain
        for msg_id in range(100):
            result = chain.process(f"message_{msg_id}")
            assert result.success

        duration = time.time() - start_time

        # Should complete 100 messages through 10 handlers in under 0.5 seconds
        assert duration < 0.5, f"Handler chain took too long: {duration:.3f}s"


class TestStressTests:
    """Stress tests for FLEXT Core components."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_container_stress_test(self) -> None:
        """Stress test container with many services."""
        container = get_flext_container()

        # Register many services
        service_count = 1000
        for service_id in range(service_count):
            container.register(f"service_{service_id}", f"value_{service_id}")

        # Verify all services can be retrieved
        start_time = time.time()
        for service_id in range(service_count):
            result = container.get(f"service_{service_id}")
            assert result.success
            assert result.data == f"value_{service_id}"

        duration = time.time() - start_time

        # Should retrieve 1000 services in under 1 second
        assert duration < 1.0, f"Service retrieval took too long: {duration:.3f}s"

    @pytest.mark.performance
    @pytest.mark.slow
    def test_result_chain_stress_test(self) -> None:
        """Stress test FlextResult with long operation chains."""

        def build_long_chain(length: int) -> FlextResult[int]:
            """Build a long chain of FlextResult operations."""
            result = FlextResult.ok(0)

            for _ in range(length):
                result = result.map(lambda x: x + 1)

            return result

        start_time = time.time()
        result = build_long_chain(10000)
        duration = time.time() - start_time

        assert result.success
        assert result.data == 10000

        # Should complete 10,000 chained operations in under 1 second
        assert duration < 1.0, f"Long chain took too long: {duration:.3f}s"


class TestMemoryPerformance:
    """Memory and resource performance tests."""

    @pytest.mark.performance
    def test_logger_factory_memory_efficiency(self) -> None:
        """Test logger factory memory efficiency with caching."""
        factory = FlextLoggerFactory

        # Clear any existing loggers
        factory.clear_loggers()

        # Create many loggers with same name (should use cache)
        loggers = []
        for _ in range(100):
            logger = factory.get_logger("cached_logger", "INFO")
            loggers.append(logger)

        # All should be the same instance due to caching
        first_logger = loggers[0]
        for logger in loggers[1:]:
            assert logger is first_logger, "Logger caching not working properly"

    @pytest.mark.performance
    @pytest.mark.architecture
    def test_core_singleton_performance(self) -> None:
        """Test FlextCore singleton performance."""

        # Multiple calls should return same instance
        cores = []
        start_time = time.time()

        for _ in range(1000):
            core = FlextCore.get_instance()
            cores.append(core)

        duration = time.time() - start_time

        # All should be the same instance
        first_core = cores[0]
        for core in cores[1:]:
            assert core is first_core, "Singleton pattern not working"

        # Should complete 1000 singleton calls very quickly
        assert duration < 0.1, f"Singleton calls took too long: {duration:.3f}s"


class TestConcurrencyPerformance:
    """Concurrency and thread safety performance tests."""

    @pytest.mark.performance
    def test_result_thread_safety(self) -> None:
        """Test FlextResult thread safety (immutability)."""
        import threading

        results = []
        errors = []

        def create_and_modify_result(thread_id: int) -> None:
            """Create and work with results in parallel."""
            try:
                base_result = FlextResult.ok(f"thread_{thread_id}")

                # Chain operations
                final_result = base_result.map(lambda x: f"{x}_step1").map(
                    lambda x: f"{x}_step2"
                )

                results.append((thread_id, final_result.data))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Create multiple threads
        threads = []
        for thread_id in range(10):
            thread = threading.Thread(
                target=create_and_modify_result, args=(thread_id,)
            )
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        duration = time.time() - start_time

        # No errors should occur
        assert not errors, f"Thread safety errors: {errors}"

        # All results should be correct
        assert len(results) == 10
        for thread_id, result_data in results:
            expected = f"thread_{thread_id}_step1_step2"
            assert result_data == expected

        # Should complete quickly
        assert duration < 1.0, f"Concurrent operations took too long: {duration:.3f}s"
