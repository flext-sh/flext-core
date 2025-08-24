"""Advanced performance testing utilities using pytest-benchmark.

Provides comprehensive performance testing, profiling, and benchmarking
capabilities with memory tracking, complexity analysis, and regression detection.
"""

# ruff: noqa: S101, ARG001, ARG002, ANN401, PLC0415
from __future__ import annotations

import concurrent.futures
import gc
import time
import tracemalloc
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

T = TypeVar("T")


class ComplexityAnalyzer:
    """Analyze algorithmic complexity and performance characteristics."""

    def __init__(self) -> None:
        self.measurements: list[dict[str, Any]] = []

    def measure_complexity(
        self,
        function: Callable[[int], Any],
        input_sizes: list[int],
        operation_name: str = "operation",
    ) -> dict[str, Any]:
        """Measure function performance across different input sizes."""
        results = []

        for size in input_sizes:
            start_time = time.perf_counter()
            gc.collect()

            # Measure execution
            function(size)

            end_time = time.perf_counter()
            duration = end_time - start_time

            results.append(
                {
                    "input_size": size,
                    "duration_seconds": duration,
                    "duration_ms": duration * 1000,
                },
            )

        # Analyze complexity pattern
        complexity_analysis = self._analyze_complexity_pattern(results)

        measurement = {
            "operation": operation_name,
            "results": results,
            "complexity_analysis": complexity_analysis,
            "timestamp": time.time(),
        }

        self.measurements.append(measurement)
        return measurement

    def _analyze_complexity_pattern(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze if the performance follows common complexity patterns."""
        if len(results) < 2:
            return {"pattern": "insufficient_data"}

        # Simple pattern detection
        sizes = [r["input_size"] for r in results]
        times = [r["duration_seconds"] for r in results]

        # Check if it's roughly linear
        if len(times) >= 3:
            ratio_1 = times[1] / times[0] if times[0] > 0 else 0
            ratio_2 = times[2] / times[1] if times[1] > 0 else 0
            size_ratio_1 = sizes[1] / sizes[0] if sizes[0] > 0 else 0
            size_ratio_2 = sizes[2] / sizes[1] if sizes[1] > 0 else 0

            if abs(ratio_1 - size_ratio_1) < 0.5 and abs(ratio_2 - size_ratio_2) < 0.5:
                return {"pattern": "linear", "confidence": "medium"}

            # Check for quadratic (simplified)
            if ratio_1 > size_ratio_1 * 1.5 and ratio_2 > size_ratio_2 * 1.5:
                return {"pattern": "quadratic_or_worse", "confidence": "low"}

        return {"pattern": "unknown", "confidence": "low"}


class StressTestRunner:
    """Run stress tests with configurable load patterns."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def run_load_test(
        self,
        function: Callable[[], Any],
        iterations: int = 1000,
        concurrent: bool = False,
        operation_name: str = "load_test",
    ) -> dict[str, Any]:
        """Run load test with specified iterations."""
        start_time = time.perf_counter()
        failures = 0
        successes = 0

        for _i in range(iterations):
            try:
                function()
                successes += 1
            except Exception:
                failures += 1

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        result = {
            "operation": operation_name,
            "iterations": iterations,
            "successes": successes,
            "failures": failures,
            "failure_rate": failures / iterations if iterations > 0 else 0,
            "total_duration_seconds": total_duration,
            "avg_duration_ms": (total_duration / iterations * 1000)
            if iterations > 0
            else 0,
            "operations_per_second": iterations / total_duration
            if total_duration > 0
            else 0,
        }

        self.results.append(result)
        return result

    def run_endurance_test(
        self,
        function: Callable[[], Any],
        duration_seconds: float = 60.0,
        operation_name: str = "endurance_test",
    ) -> dict[str, Any]:
        """Run endurance test for specified duration."""
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds

        iterations = 0
        failures = 0

        while time.perf_counter() < end_time:
            try:
                function()
                iterations += 1
            except Exception:
                failures += 1
                iterations += 1

        actual_duration = time.perf_counter() - start_time

        result = {
            "operation": operation_name,
            "planned_duration_seconds": duration_seconds,
            "actual_duration_seconds": actual_duration,
            "iterations": iterations,
            "failures": failures,
            "failure_rate": failures / iterations if iterations > 0 else 0,
            "operations_per_second": iterations / actual_duration
            if actual_duration > 0
            else 0,
        }

        self.results.append(result)
        return result


class PerformanceProfiler:
    """Advanced performance profiling with memory and time tracking."""

    def __init__(self) -> None:
        self.measurements: list[dict[str, Any]] = []

    @contextmanager
    def profile_memory(self, operation_name: str = "operation"):
        """Profile memory usage during operation."""
        tracemalloc.start()
        gc.collect()  # Clean up before measurement

        snapshot_before = tracemalloc.take_snapshot()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            snapshot_after = tracemalloc.take_snapshot()
            tracemalloc.stop()

            # Calculate memory difference
            top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
            memory_usage = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB

            self.measurements.append(
                {
                    "operation": operation_name,
                    "duration_seconds": end_time - start_time,
                    "memory_mb": memory_usage,
                    "peak_memory_stats": top_stats[:5],  # Top 5 allocations
                },
            )

    def assert_memory_efficient(
        self,
        max_memory_mb: float = 10.0,
        operation_name: str | None = None,
    ) -> None:
        """Assert last operation was memory efficient."""
        if not self.measurements:
            msg = "No measurements recorded"
            raise AssertionError(msg)

        last_measurement = self.measurements[-1]
        if operation_name and last_measurement["operation"] != operation_name:
            msg = f"Expected operation {operation_name}, got {last_measurement['operation']}"
            raise AssertionError(msg)

        memory_used = last_measurement["memory_mb"]
        assert memory_used <= max_memory_mb, (
            f"Memory usage {memory_used:.2f}MB exceeds limit {max_memory_mb:.2f}MB\n"
            f"Top allocations: {last_measurement['peak_memory_stats']}"
        )


class BenchmarkUtils:
    """Advanced benchmarking utilities with statistical analysis."""

    @staticmethod
    def benchmark_with_warmup(
        benchmark: BenchmarkFixture,
        func: Callable[..., T],
        warmup_rounds: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Benchmark function with warmup rounds for consistent results."""
        # Warmup rounds
        for _ in range(warmup_rounds):
            func(*args, **kwargs)

        # Actual benchmark
        return benchmark(func, *args, **kwargs)

    @staticmethod
    def benchmark_complexity(
        benchmark: BenchmarkFixture,
        func: Callable[[int], T],
        sizes: list[int],
        expected_complexity: str = "O(n)",
    ) -> dict[str, Any]:
        """Benchmark function complexity across different input sizes."""
        results = {}

        for size in sizes:
            # Benchmark for this size
            result = benchmark(func, size)
            stats = benchmark.stats

            results[size] = {
                "result": result,
                "mean_time": stats.mean,
                "stddev": stats.stddev,
                "min_time": stats.min,
                "max_time": stats.max,
            }

        # Analyze complexity
        complexity_analysis = BenchmarkUtils._analyze_complexity(
            results,
            expected_complexity,
        )
        results["complexity_analysis"] = complexity_analysis

        return results

    @staticmethod
    def _analyze_complexity(
        results: dict[int, dict[str, Any]],
        expected: str,
    ) -> dict[str, Any]:
        """Analyze time complexity from benchmark results."""
        sizes = sorted(results.keys())
        times = [results[size]["mean_time"] for size in sizes]

        if len(sizes) < 2:
            return {"error": "Need at least 2 data points"}

        # Calculate growth ratios
        growth_ratios = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1]
            growth_ratios.append(time_ratio / size_ratio)

        avg_growth = sum(growth_ratios) / len(growth_ratios)

        # Determine actual complexity
        if avg_growth < 1.5:
            actual_complexity = "O(1) or O(log n)"
        elif avg_growth < 2.5:
            actual_complexity = "O(n)"
        elif avg_growth < 4.0:
            actual_complexity = "O(n log n)"
        else:
            actual_complexity = "O(n²) or worse"

        return {
            "expected": expected,
            "actual": actual_complexity,
            "average_growth_ratio": avg_growth,
            "growth_ratios": growth_ratios,
            "matches_expected": expected.lower() in actual_complexity.lower(),
        }

    @staticmethod
    def benchmark_regression(
        benchmark: BenchmarkFixture,
        func: Callable[..., T],
        baseline_time: float,
        tolerance_percent: float = 10.0,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Benchmark with regression detection."""
        result = benchmark(func, *args, **kwargs)
        current_time = benchmark.stats.mean

        # Check for regression
        max_allowed_time = baseline_time * (1 + tolerance_percent / 100)

        assert current_time <= max_allowed_time, (
            f"Performance regression detected: "
            f"current={current_time:.4f}s, "
            f"baseline={baseline_time:.4f}s, "
            f"max_allowed={max_allowed_time:.4f}s "
            f"(+{tolerance_percent}%)"
        )

        return result

    @staticmethod
    def parallel_benchmark(
        benchmark: BenchmarkFixture,
        func: Callable[..., T],
        thread_counts: list[int],
        *args: Any,
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        """Benchmark function with different thread counts."""
        results = {}

        for thread_count in thread_counts:

            def parallel_execution(workers=thread_count):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers,
                ) as executor:
                    futures = [
                        executor.submit(func, *args, **kwargs)
                        for _ in range(workers)
                    ]
                    return [future.result() for future in futures]

            benchmark_result = benchmark(parallel_execution)
            stats = benchmark.stats

            results[thread_count] = {
                "result": benchmark_result,
                "mean_time": stats.mean,
                "throughput": thread_count / stats.mean,  # operations per second
                "efficiency": (thread_count / stats.mean)
                / thread_count,  # relative to single thread
            }

        return results


class MemoryProfiler:
    """Memory profiling utilities for memory leak detection."""

    @staticmethod
    @contextmanager
    def track_memory_leaks(max_increase_mb: float = 5.0) -> Generator[None]:
        """Context manager to detect memory leaks."""
        gc.collect()
        tracemalloc.start()

        snapshot_before = tracemalloc.take_snapshot()

        try:
            yield
        finally:
            gc.collect()
            snapshot_after = tracemalloc.take_snapshot()

            top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
            memory_increase = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

            tracemalloc.stop()

            assert memory_increase <= max_increase_mb, (
                f"Potential memory leak detected: "
                f"memory increased by {memory_increase:.2f}MB "
                f"(limit: {max_increase_mb:.2f}MB)\n"
                f"Top allocations:\n" + "\n".join(str(stat) for stat in top_stats[:10])
            )

    @staticmethod
    def memory_stress_test(
        func: Callable[..., T],
        iterations: int = 1000,
        max_memory_growth_mb: float = 10.0,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Stress test function for memory stability."""
        with MemoryProfiler.track_memory_leaks(max_memory_growth_mb):
            last_result = None
            for _ in range(iterations):
                last_result = func(*args, **kwargs)
                if _ % 100 == 0:  # Periodic cleanup
                    gc.collect()

            return last_result


class AsyncBenchmark:
    """Benchmarking utilities for async functions."""

    @staticmethod
    async def benchmark_async(
        func: Callable[..., Any],
        iterations: int = 100,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Benchmark async function."""
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            await func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "stddev": (
                sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)
            )
            ** 0.5,
        }

    @staticmethod
    async def benchmark_concurrency(
        func: Callable[..., Any],
        concurrency_levels: list[int],
        *args: Any,
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        """Benchmark async function with different concurrency levels."""
        import asyncio

        results = {}

        for concurrency in concurrency_levels:

            async def concurrent_execution(workers=concurrency):
                tasks = [func(*args, **kwargs) for _ in range(workers)]
                return await asyncio.gather(*tasks)

            start_time = time.perf_counter()
            result = await concurrent_execution()
            end_time = time.perf_counter()

            total_time = end_time - start_time
            throughput = concurrency / total_time

            results[concurrency] = {
                "result": result,
                "total_time": total_time,
                "throughput": throughput,
                "avg_time_per_operation": total_time / concurrency,
            }

        return results


# Performance testing markers and fixtures
class PerformanceMarkers:
    """Custom pytest markers for performance tests."""

    slow = pytest.mark.slow
    performance = pytest.mark.performance
    memory = pytest.mark.memory
    benchmark = pytest.mark.benchmark
    regression = pytest.mark.regression

    @staticmethod
    def complexity(expected: str) -> pytest.MarkDecorator:
        """Mark test with expected complexity."""
        return pytest.mark.complexity(expected=expected)

    @staticmethod
    def memory_limit(mb: float) -> pytest.MarkDecorator:
        """Mark test with memory limit."""
        return pytest.mark.memory_limit(mb=mb)

    @staticmethod
    def timeout(seconds: float) -> pytest.MarkDecorator:
        """Mark test with timeout."""
        return pytest.mark.timeout(seconds)


# Export utilities
__all__ = [
    "AsyncBenchmark",
    "BenchmarkUtils",
    "MemoryProfiler",
    "PerformanceMarkers",
    "PerformanceProfiler",
]
