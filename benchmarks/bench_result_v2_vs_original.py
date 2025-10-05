"""Performance benchmark: FlextResultV2 (returns backend) vs FlextResult (pure Python).

This benchmark measures the performance overhead of wrapping returns.Result
compared to the pure Python implementation.

Metrics:
- Simple operations (ok, fail, is_success)
- Map chains (1, 10, 100 operations)
- Flat_map chains
- Metadata preservation overhead
- Memory usage

Run with:
    python benchmarks/bench_result_v2_vs_original.py
"""

import sys
import time
from collections.abc import Callable

# Add src to path
sys.path.insert(0, "src")

# Original implementation
from flext_core.result import FlextResult

# POC implementation (requires returns)
try:
    from flext_core.result_v2_returns_backend import FlextResultV2

    HAS_RETURNS = True
except ImportError:
    HAS_RETURNS = False


class BenchmarkRunner:
    """Run and compare benchmarks."""

    def __init__(self, iterations: int = 100000) -> None:
        self.iterations = iterations

    def time_operation(self, name: str, operation: Callable) -> float:
        """Time an operation over multiple iterations."""
        start = time.perf_counter()
        for _ in range(self.iterations):
            operation()
        end = time.perf_counter()
        duration = end - start
        self.iterations / duration if duration > 0 else 0
        return duration

    def run_benchmark_group(self, name: str, benchmarks: dict):
        """Run a group of related benchmarks."""
        results = {}
        for bench_name, operation in benchmarks.items():
            results[bench_name] = self.time_operation(bench_name, operation)

        return results

    def run_all(self) -> None:
        """Run all benchmark groups."""
        # ====================================================================
        # Benchmark 1: Construction
        # ====================================================================
        self.run_benchmark_group(
            "Construction Operations",
            {
                "FlextResult.ok(42)": lambda: FlextResult.ok(42),
                "FlextResultV2.ok(42)": lambda: FlextResultV2.ok(42)
                if HAS_RETURNS
                else None,
                "FlextResult.fail('error')": lambda: FlextResult.fail("error"),
                "FlextResultV2.fail('error')": lambda: FlextResultV2.fail("error")
                if HAS_RETURNS
                else None,
                "FlextResult.fail(code+data)": lambda: FlextResult.fail(
                    "error",
                    error_code="ERR",
                    error_data={"field": "test"},
                ),
                "FlextResultV2.fail(code+data)": lambda: FlextResultV2.fail(
                    "error",
                    error_code="ERR",
                    error_data={"field": "test"},
                )
                if HAS_RETURNS
                else None,
            },
        )

        # ====================================================================
        # Benchmark 2: Property Access
        # ====================================================================
        r1 = FlextResult.ok(42)
        r2 = FlextResultV2.ok(42) if HAS_RETURNS else None

        self.run_benchmark_group(
            "Property Access",
            {
                "FlextResult.is_success": lambda: r1.is_success,
                "FlextResultV2.is_success": lambda: r2.is_success
                if HAS_RETURNS
                else None,
                "FlextResult.value": lambda: r1.value,
                "FlextResultV2.value": lambda: r2.value if HAS_RETURNS else None,
                "FlextResult.data (legacy)": lambda: r1.data,
                "FlextResultV2.data (legacy)": lambda: r2.data if HAS_RETURNS else None,
            },
        )

        # ====================================================================
        # Benchmark 3: Single Map Operation
        # ====================================================================
        self.run_benchmark_group(
            "Single Map Operation",
            {
                "FlextResult.ok().map()": lambda: FlextResult.ok(5).map(
                    lambda x: x * 2
                ),
                "FlextResultV2.ok().map()": lambda: FlextResultV2.ok(5).map(
                    lambda x: x * 2
                )
                if HAS_RETURNS
                else None,
            },
        )

        # ====================================================================
        # Benchmark 4: Map Chain (10 operations)
        # ====================================================================
        def chain_10_maps_original():
            return (
                FlextResult.ok(1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
            )

        def chain_10_maps_v2():
            return (
                FlextResultV2.ok(1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
                .map(lambda x: x + 1)
            )

        self.run_benchmark_group(
            "Map Chain (10 operations)",
            {
                "FlextResult chain": chain_10_maps_original,
                "FlextResultV2 chain": chain_10_maps_v2 if HAS_RETURNS else None,
            },
        )

        # ====================================================================
        # Benchmark 5: Flat Map Operations
        # ====================================================================
        def validate_original(x: int) -> FlextResult[int]:
            if x > 0:
                return FlextResult.ok(x)
            return FlextResult.fail("invalid")

        def validate_v2(x: int) -> FlextResultV2[int]:
            if x > 0:
                return FlextResultV2.ok(x)
            return FlextResultV2.fail("invalid")

        self.run_benchmark_group(
            "Flat Map (monadic bind)",
            {
                "FlextResult.flat_map()": lambda: FlextResult.ok(5).flat_map(
                    validate_original
                ),
                "FlextResultV2.flat_map()": lambda: FlextResultV2.ok(5).flat_map(
                    validate_v2
                )
                if HAS_RETURNS
                else None,
            },
        )

        # ====================================================================
        # Benchmark 6: Metadata Preservation
        # ====================================================================
        def metadata_chain_original():
            return (
                FlextResult.fail("error", error_code="ERR_001", error_data={"x": 1})
                .map(str.upper)
                .map(str.strip)
                .map(str.lower)
            )

        def metadata_chain_v2():
            return (
                FlextResultV2.fail("error", error_code="ERR_001", error_data={"x": 1})
                .map(str.upper)
                .map(str.strip)
                .map(str.lower)
            )

        self.run_benchmark_group(
            "Metadata Preservation (3 maps on failure)",
            {
                "FlextResult metadata": metadata_chain_original,
                "FlextResultV2 metadata": metadata_chain_v2 if HAS_RETURNS else None,
            },
        )

        # ====================================================================
        # Benchmark 7: Unwrap Operations
        # ====================================================================
        success1 = FlextResult.ok(100)
        success2 = FlextResultV2.ok(100) if HAS_RETURNS else None

        self.run_benchmark_group(
            "Value Extraction",
            {
                "FlextResult.unwrap()": success1.unwrap,
                "FlextResultV2.unwrap()": lambda: success2.unwrap()
                if HAS_RETURNS
                else None,
                "FlextResult.unwrap_or()": lambda: success1.unwrap_or(0),
                "FlextResultV2.unwrap_or()": lambda: success2.unwrap_or(0)
                if HAS_RETURNS
                else None,
            },
        )

        # ====================================================================
        # Summary
        # ====================================================================


if __name__ == "__main__":
    if not HAS_RETURNS:
        sys.exit(1)

    runner = BenchmarkRunner(iterations=100000)
    runner.run_all()
