"""Performance testing utilities using pytest-benchmark.

Provides performance testing, profiling, and benchmarking
capabilities with memory tracking, complexity analysis, and regression detection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import time
import tracemalloc
from collections.abc import Awaitable, Callable, Generator
from contextlib import _GeneratorContextManager, contextmanager
from typing import Protocol

import pytest

from flext_core import FlextTypes, P, T


class FlextTestsPerformance:
    """Unified performance testing utilities for FLEXT ecosystem.

    Consolidates all performance testing patterns, profiling, benchmarking,
    memory tracking, complexity analysis, and regression detection into
    a single class interface. Provides performance analysis
    capabilities for synchronous and asynchronous code.
    """

    # === Complexity Analysis ===

    class ComplexityAnalyzer:
        """Analyze algorithmic complexity and performance characteristics."""

        def __init__(self) -> None:
            """Initialize complexityanalyzer:."""
            self.measurements: list[FlextTypes.Core.Dict] = []

        def measure_complexity(
            self,
            function: Callable[[int], object],
            input_sizes: list[int],
            operation_name: str = "operation",
        ) -> FlextTypes.Core.Dict:
            """Measure function performance across different input sizes."""
            results: list[FlextTypes.Core.Dict] = []

            for size in input_sizes:
                start_time = time.perf_counter()
                gc.collect()

                # Measure execution
                function(size)

                end_time = time.perf_counter()
                duration = end_time - start_time

                result_dict: FlextTypes.Core.Dict = {
                    "input_size": size,
                    "duration_seconds": duration,
                    "duration_ms": duration * 1000,
                }
                results.append(result_dict)

            # Analyze complexity pattern
            complexity_analysis = self._analyze_complexity_pattern(results)

            measurement: FlextTypes.Core.Dict = {
                "operation": operation_name,
                "results": results,
                "complexity_analysis": complexity_analysis,
                "timestamp": time.time(),
            }

            self.measurements.append(measurement)
            return measurement

        def _analyze_complexity_pattern(
            self,
            results: list[FlextTypes.Core.Dict],
        ) -> FlextTypes.Core.Dict:
            """Analyze if the performance follows common complexity patterns."""
            if len(results) < 2:
                return {"pattern": "insufficient_data"}

            sizes = [
                float(r["input_size"])
                for r in results
                if isinstance(r["input_size"], (int, float))
            ]
            times = [
                float(r["duration_seconds"])
                for r in results
                if isinstance(r["duration_seconds"], (int, float))
            ]

            # Check if it's roughly linear
            if len(times) >= 3:
                ratio_1 = times[1] / times[0] if times[0] > 0 else 0
                ratio_2 = times[2] / times[1] if times[1] > 0 else 0
                size_ratio_1 = sizes[1] / sizes[0] if sizes[0] > 0 else 0
                size_ratio_2 = sizes[2] / sizes[1] if sizes[1] > 0 else 0

                if (
                    abs(ratio_1 - size_ratio_1) < 0.5
                    and abs(ratio_2 - size_ratio_2) < 0.5
                ):
                    return {"pattern": "linear", "confidence": "medium"}

                # Check for quadratic (simplified)
                if ratio_1 > size_ratio_1 * 1.5 and ratio_2 > size_ratio_2 * 1.5:
                    return {"pattern": "quadratic_or_worse", "confidence": "low"}

            return {"pattern": "unknown", "confidence": "low"}

    # === Stress Testing ===

    class StressTestRunner:
        """Run stress tests with configurable load patterns."""

        def __init__(self) -> None:
            """Initialize stresstestrunner:."""
            self.results: list[FlextTypes.Core.Dict] = []

        def run_load_test(
            self,
            function: Callable[[], object],
            iterations: int = 1000,
            *,  # concurrent is keyword-only to avoid boolean trap
            _concurrent: bool = False,  # Currently unused but reserved for future use
            operation_name: str = "load_test",
        ) -> FlextTypes.Core.Dict:
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

            result: FlextTypes.Core.Dict = {
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
            function: Callable[[], object],
            duration_seconds: float = 60.0,
            operation_name: str = "endurance_test",
        ) -> FlextTypes.Core.Dict:
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

            result: FlextTypes.Core.Dict = {
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

    # === Performance Profiling ===

    class PerformanceProfiler:
        """Advanced performance profiling with memory and time tracking."""

        def __init__(self) -> None:
            """Initialize performanceprofiler:."""
            self.measurements: list[FlextTypes.Core.Dict] = []

        @contextmanager
        def profile_memory(self, operation_name: str = "operation") -> Generator[None]:
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
                memory_usage = (
                    sum(stat.size_diff for stat in top_stats) / 1024 / 1024
                )  # MB

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

            memory_used = (
                float(last_measurement["memory_mb"])
                if isinstance(last_measurement["memory_mb"], (int, float, str))
                else 0.0
            )
            assert memory_used <= max_memory_mb, (
                f"Memory usage {memory_used:.2f}MB exceeds limit {max_memory_mb:.2f}MB\n"
                f"Top allocations: {last_measurement['peak_memory_stats']}"
            )

    # === Benchmark Utils ===

    class BenchmarkUtils:
        """Advanced benchmarking utilities with statistical analysis."""

        @staticmethod
        def benchmark_with_warmup(
            benchmark: FlextTestsPerformance.BenchmarkProtocol,
            func: Callable[P, T],
            warmup_rounds: int = 5,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> T:
            """Benchmark function with warmup rounds for consistent results."""
            # Warmup rounds
            for _ in range(warmup_rounds):
                func(*args, **kwargs)

            # Actual benchmark
            return benchmark(func, *args, **kwargs)

        @staticmethod
        def benchmark_complexity(
            benchmark: FlextTestsPerformance.BenchmarkProtocol,
            func: Callable[[int], T],
            sizes: list[int],
            expected_complexity: str = "O(n)",
        ) -> FlextTypes.Core.Dict:
            """Benchmark function complexity across different input sizes."""
            results: dict[int, FlextTypes.Core.Dict] = {}

            for size in sizes:
                # Benchmark for this size
                result = benchmark(func, size)
                stats = getattr(benchmark, "stats", None)

                results[size] = {
                    "result": result,
                    "mean_time": getattr(stats, "mean", 0.0) if stats else 0.0,
                    "stddev": getattr(stats, "stddev", 0.0) if stats else 0.0,
                    "min_time": getattr(stats, "min", 0.0) if stats else 0.0,
                    "max_time": getattr(stats, "max", 0.0) if stats else 0.0,
                }

            # Analyze complexity
            complexity_analysis = (
                FlextTestsPerformance.BenchmarkUtils._analyze_complexity(
                    results,
                    expected_complexity,
                )
            )

            # Convert to FlextTypes.Core.Dict for return type
            final_results: FlextTypes.Core.Dict = {
                str(k): v for k, v in results.items()
            }
            final_results["complexity_analysis"] = complexity_analysis

            return final_results

        @staticmethod
        def _analyze_complexity(
            results: dict[int, FlextTypes.Core.Dict],
            expected: str,
        ) -> FlextTypes.Core.Dict:
            """Analyze time complexity from benchmark results."""
            sizes = sorted(results.keys())
            times: list[float] = []
            for size in sizes:
                mean_time = results[size]["mean_time"]
                if isinstance(mean_time, (int, float)):
                    times.append(float(mean_time))
                elif isinstance(mean_time, str):
                    try:
                        times.append(float(mean_time))
                    except ValueError:
                        times.append(0.0)
                else:
                    times.append(0.0)

            if len(sizes) < 2:
                return {"error": "Need at least 2 data points"}

            # Calculate growth ratios
            growth_ratios: list[float] = []
            for i in range(1, len(sizes)):
                size_ratio = sizes[i] / sizes[i - 1]
                time_ratio = times[i] / times[i - 1] if times[i - 1] != 0 else 1.0
                growth_ratio = time_ratio / size_ratio if size_ratio != 0 else 1.0
                growth_ratios.append(growth_ratio)

            # Calculate average growth ratio
            growth_sum = sum(growth_ratios)  # All elements are already floats
            avg_growth = growth_sum / len(growth_ratios) if growth_ratios else 1.0

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
            benchmark: FlextTestsPerformance.BenchmarkProtocol,
            func: Callable[P, T],
            baseline_time: float,
            tolerance_percent: float = 10.0,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> T:
            """Benchmark with regression detection."""
            result = benchmark(func, *args, **kwargs)
            stats = getattr(benchmark, "stats", None)
            current_time = getattr(stats, "mean", 0.0) if stats else 0.0

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
            benchmark: FlextTestsPerformance.BenchmarkProtocol,
            func: Callable[P, T],
            thread_counts: list[int],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> dict[int, FlextTypes.Core.Dict]:
            """Benchmark function with different thread counts."""
            results: dict[int, FlextTypes.Core.Dict] = {}

            for thread_count in thread_counts:

                def parallel_execution(workers: int = thread_count) -> list[T]:
                    """parallel_execution method."""
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=workers,
                    ) as executor:
                        futures = [
                            executor.submit(func, *args, **kwargs)
                            for _ in range(workers)
                        ]
                        return [future.result() for future in futures]

                benchmark_result = benchmark(parallel_execution)
                stats = getattr(benchmark, "stats", None)
                mean_time = getattr(stats, "mean", 0.0) if stats else 0.0

                results[thread_count] = {
                    "result": benchmark_result,
                    "mean_time": mean_time,
                    "throughput": thread_count / mean_time
                    if mean_time > 0
                    else 0.0,  # operations per second
                    "efficiency": (thread_count / mean_time / thread_count)
                    if mean_time > 0
                    else 0.0,  # relative to single thread
                }

            return results

    # === Memory Profiling ===

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
                memory_increase = (
                    sum(stat.size_diff for stat in top_stats) / 1024 / 1024
                )

                tracemalloc.stop()

                assert memory_increase <= max_increase_mb, (
                    f"Potential memory leak detected: "
                    f"memory increased by {memory_increase:.2f}MB "
                    f"(limit: {max_increase_mb:.2f}MB)\n"
                    f"Top allocations:\n"
                    + "\n".join(str(stat) for stat in top_stats[:10])
                )

        @staticmethod
        def memory_stress_test(
            func: Callable[P, T],
            iterations: int = 1000,
            max_memory_growth_mb: float = 10.0,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> T:
            """Stress test function for memory stability."""
            with FlextTestsPerformance.MemoryProfiler.track_memory_leaks(
                max_memory_growth_mb,
            ):
                last_result: T | None = None
                for _ in range(iterations):
                    last_result = func(*args, **kwargs)
                    if _ % 100 == 0:  # Periodic cleanup
                        gc.collect()

                if last_result is None:
                    msg = "Function never returned a value"
                    raise RuntimeError(msg)
                return last_result

    # === Async Benchmarking ===

    class AsyncBenchmark:
        """Benchmarking utilities for async functions."""

        @staticmethod
        async def benchmark_async(
            func: Callable[P, Awaitable[T]],
            iterations: int = 100,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> dict[str, float]:
            """Benchmark async function."""
            times: list[float] = []

            for _ in range(iterations):
                start_time = time.perf_counter()
                await func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            if not times:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0}

            mean_time = sum(times) / len(times)
            return {
                "mean": mean_time,
                "min": min(times),
                "max": max(times),
                "stddev": (sum((t - mean_time) ** 2 for t in times) / len(times))
                ** 0.5,
            }

        @staticmethod
        async def benchmark_concurrency(
            func: Callable[P, Awaitable[T]],
            concurrency_levels: list[int],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> dict[int, FlextTypes.Core.Dict]:
            """Benchmark async function with different concurrency levels."""
            results: dict[int, FlextTypes.Core.Dict] = {}

            for concurrency in concurrency_levels:

                async def concurrent_execution(
                    workers: int = concurrency,
                ) -> FlextTypes.Core.List:
                    """concurrent_execution method."""
                    tasks = [func(*args, **kwargs) for _ in range(workers)]
                    return await asyncio.gather(*tasks)

                start_time = time.perf_counter()
                result = await concurrent_execution()
                end_time = time.perf_counter()

                total_time = end_time - start_time
                throughput = concurrency / total_time if total_time > 0 else 0.0

                results[concurrency] = {
                    "result": result,
                    "total_time": total_time,
                    "throughput": throughput,
                    "avg_time_per_operation": total_time / concurrency
                    if concurrency > 0
                    else 0.0,
                }

            return results

    # === Performance Markers ===

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

    # === Convenience Factory Methods ===

    @staticmethod
    def create_complexity_analyzer() -> FlextTestsPerformance.ComplexityAnalyzer:
        """Create ComplexityAnalyzer instance."""
        return FlextTestsPerformance.ComplexityAnalyzer()

    @staticmethod
    def create_stress_runner() -> FlextTestsPerformance.StressTestRunner:
        """Create StressTestRunner instance."""
        return FlextTestsPerformance.StressTestRunner()

    @staticmethod
    def create_profiler() -> FlextTestsPerformance.PerformanceProfiler:
        """Create PerformanceProfiler instance."""
        return FlextTestsPerformance.PerformanceProfiler()

    @staticmethod
    def create_benchmark_utils() -> FlextTestsPerformance.BenchmarkUtils:
        """Create BenchmarkUtils instance."""
        return FlextTestsPerformance.BenchmarkUtils()

    @staticmethod
    def create_memory_profiler() -> FlextTestsPerformance.MemoryProfiler:
        """Create MemoryProfiler instance."""
        return FlextTestsPerformance.MemoryProfiler()

    @staticmethod
    def create_async_benchmark() -> FlextTestsPerformance.AsyncBenchmark:
        """Create AsyncBenchmark instance."""
        return FlextTestsPerformance.AsyncBenchmark()

    @staticmethod
    def create_markers() -> FlextTestsPerformance.PerformanceMarkers:
        """Create PerformanceMarkers instance."""
        return FlextTestsPerformance.PerformanceMarkers()

    # === Quick Access Methods ===

    @staticmethod
    def quick_memory_profile(
        operation_name: str = "operation",
    ) -> _GeneratorContextManager[None]:
        """Quick access to memory profiling."""
        profiler = FlextTestsPerformance.PerformanceProfiler()
        return profiler.profile_memory(operation_name)

    @staticmethod
    def quick_stress_test(
        func: Callable[[], object],
        iterations: int = 1000,
        operation_name: str = "load_test",
    ) -> FlextTypes.Core.Dict:
        """Quick access to stress testing."""
        runner = FlextTestsPerformance.StressTestRunner()
        return runner.run_load_test(func, iterations, operation_name=operation_name)

    @staticmethod
    def quick_complexity_check(
        function: Callable[[int], object],
        input_sizes: list[int],
        operation_name: str = "operation",
    ) -> FlextTypes.Core.Dict:
        """Quick access to complexity analysis."""
        analyzer = FlextTestsPerformance.ComplexityAnalyzer()
        return analyzer.measure_complexity(function, input_sizes, operation_name)

    class BenchmarkProtocol(Protocol):
        """Protocol for pytest-benchmark fixture."""

        def __call__(
            self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs,
        ) -> T:
            """Call the benchmark function."""
            ...

        @property
        def stats(self) -> FlextTypes.Core.Dict:
            """Benchmark statistics."""
            ...


# === REMOVED COMPATIBILITY ALIASES AND FACADES ===
# Legacy compatibility removed as per user request
# All compatibility facades, aliases and protocol facades have been commented out
# Only FlextTestsPerformance class is now exported

# Main class alias for backward compatibility - REMOVED
# FlextTestsPerformances = FlextTestsPerformance

# Legacy ComplexityAnalyzer class - REMOVED (commented out)
# class ComplexityAnalyzer:
#     """Compatibility facade for ComplexityAnalyzer - use FlextTestsPerformance instead."""
#     ... all methods commented out

# Legacy StressTestRunner class - REMOVED (commented out)
# class StressTestRunner:
#     """Compatibility facade for StressTestRunner - use FlextTestsPerformance instead."""
#     ... all methods commented out

# Legacy PerformanceProfiler class - REMOVED (commented out)
# class PerformanceProfiler:
#     """Compatibility facade for PerformanceProfiler - use FlextTestsPerformance instead."""
#     ... all methods commented out

# Legacy BenchmarkUtils class - REMOVED (commented out)
# class BenchmarkUtils:
#     """Compatibility facade for BenchmarkUtils - use FlextTestsPerformance instead."""
#     ... all methods commented out

# Legacy MemoryProfiler class - REMOVED (commented out)
# class MemoryProfiler:
#     """Compatibility facade for MemoryProfiler - use FlextTestsPerformance instead."""
#     ... all methods commented out

# Legacy AsyncBenchmark class - REMOVED (commented out)
# class AsyncBenchmark:
#     """Compatibility facade for AsyncBenchmark - use FlextTestsPerformance instead."""
#     ... all methods commented out

# Legacy PerformanceMarkers class - REMOVED (commented out)
# class PerformanceMarkers:
#     """Compatibility facade for PerformanceMarkers - use FlextTestsPerformance instead."""
#     ... all methods commented out

# Export only the unified class
__all__ = [
    "FlextTestsPerformance",
]
