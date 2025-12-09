"""Performance benchmarks for FlextContainer operations.

This module provides benchmarks for critical container operations to establish
baseline performance metrics before optimizations. Benchmarks measure:
- register() - Service registration timing
- get() - Service resolution timing
- register_factory() / register_resource() - Factory/resource registration
- has_service() / list_services() - Lookup operations
- Volume tests: 10, 100, 1000, 10000 services

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from flext_core.container import FlextContainer
from flext_core.typings import t


class PerformanceBenchmark:
    """Container for benchmark results and utilities."""

    @staticmethod
    def measure_time(
        func: Callable[[], t.GeneralValueType],
    ) -> tuple[t.GeneralValueType, float]:
        """Measure execution time of a function.

        Args:
            func: Function to measure.

        Returns:
            Tuple of (result, elapsed_time_in_seconds).

        """
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        return result, elapsed

    @staticmethod
    def benchmark_register(
        container: FlextContainer,
        count: int,
    ) -> float:
        """Benchmark service registration.

        Args:
            container: Container instance.
            count: Number of services to register.

        Returns:
            Elapsed time in seconds.

        """
        _, elapsed = PerformanceBenchmark.measure_time(
            lambda: [
                container.register(f"service_{i}", f"value_{i}") for i in range(count)
            ],
        )
        return elapsed

    @staticmethod
    def benchmark_get(
        container: FlextContainer,
        count: int,
    ) -> float:
        """Benchmark service resolution.

        Args:
            container: Container instance with services registered.
            count: Number of services to resolve.

        Returns:
            Elapsed time in seconds.

        """
        _, elapsed = PerformanceBenchmark.measure_time(
            lambda: [container.get(f"service_{i}") for i in range(count)],
        )
        return elapsed

    @staticmethod
    def benchmark_register_factory(
        container: FlextContainer,
        count: int,
    ) -> float:
        """Benchmark factory registration.

        Args:
            container: Container instance.
            count: Number of factories to register.

        Returns:
            Elapsed time in seconds.

        """

        def make_factory(captured_i: int) -> Callable[[], str]:
            """Create factory function that captures i in closure."""
            return lambda: f"value_{captured_i}"

        _, elapsed = PerformanceBenchmark.measure_time(
            lambda: [
                container.register_factory(
                    f"factory_{i}",
                    make_factory(i),
                )
                for i in range(count)
            ],
        )
        return elapsed

    @staticmethod
    def benchmark_register_resource(
        container: FlextContainer,
        count: int,
    ) -> float:
        """Benchmark resource registration.

        Args:
            container: Container instance.
            count: Number of resources to register.

        Returns:
            Elapsed time in seconds.

        """

        def make_resource(captured_i: int) -> Callable[[], str]:
            """Create resource function that captures i in closure."""
            return lambda: f"value_{captured_i}"

        _, elapsed = PerformanceBenchmark.measure_time(
            lambda: [
                container.register_resource(
                    f"resource_{i}",
                    make_resource(i),
                )
                for i in range(count)
            ],
        )
        return elapsed

    @staticmethod
    def benchmark_has_service(
        container: FlextContainer,
        count: int,
    ) -> float:
        """Benchmark has_service() lookups.

        Args:
            container: Container instance with services registered.
            count: Number of lookups to perform.

        Returns:
            Elapsed time in seconds.

        """
        _, elapsed = PerformanceBenchmark.measure_time(
            lambda: [container.has_service(f"service_{i}") for i in range(count)],
        )
        return elapsed

    @staticmethod
    def benchmark_list_services(
        container: FlextContainer,
        iterations: int = 100,
    ) -> float:
        """Benchmark list_services() calls.

        Args:
            container: Container instance with services registered.
            iterations: Number of list_services() calls.

        Returns:
            Elapsed time in seconds.

        """
        _, elapsed = PerformanceBenchmark.measure_time(
            lambda: [container.list_services() for _ in range(iterations)],
        )
        return elapsed


class TestContainerPerformance:
    """Performance benchmarks for container operations."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize(
        "count", [10, 100, 1000, 10000], ids=["10", "100", "1000", "10000"]
    )
    def test_register_performance(
        self,
        count: int,
    ) -> None:
        """Benchmark register() with different volumes."""
        container = FlextContainer.create()
        PerformanceBenchmark.benchmark_register(container, count)
        # Log results (pytest will display)

    @pytest.mark.benchmark
    @pytest.mark.parametrize(
        "count", [10, 100, 1000, 10000], ids=["10", "100", "1000", "10000"]
    )
    def test_get_performance(
        self,
        count: int,
    ) -> None:
        """Benchmark get() with different volumes."""
        container = FlextContainer.create()
        # Pre-register services
        for i in range(count):
            container.register(f"service_{i}", f"value_{i}")
        PerformanceBenchmark.benchmark_get(container, count)
        # Log results

    @pytest.mark.benchmark
    @pytest.mark.parametrize("count", [10, 100, 1000], ids=["10", "100", "1000"])
    def test_register_factory_performance(
        self,
        count: int,
    ) -> None:
        """Benchmark register_factory() with different volumes."""
        container = FlextContainer.create()
        PerformanceBenchmark.benchmark_register_factory(container, count)
        # Log results

    @pytest.mark.benchmark
    @pytest.mark.parametrize("count", [10, 100, 1000], ids=["10", "100", "1000"])
    def test_register_resource_performance(
        self,
        count: int,
    ) -> None:
        """Benchmark register_resource() with different volumes."""
        container = FlextContainer.create()
        PerformanceBenchmark.benchmark_register_resource(container, count)
        # Log results

    @pytest.mark.benchmark
    @pytest.mark.parametrize(
        "count", [10, 100, 1000, 10000], ids=["10", "100", "1000", "10000"]
    )
    def test_has_service_performance(
        self,
        count: int,
    ) -> None:
        """Benchmark has_service() with different volumes."""
        container = FlextContainer.create()
        # Pre-register services
        for i in range(count):
            container.register(f"service_{i}", f"value_{i}")
        PerformanceBenchmark.benchmark_has_service(container, count)
        # Log results

    @pytest.mark.benchmark
    def test_list_services_performance(self) -> None:
        """Benchmark list_services() with different service counts."""
        for count in [10, 100, 1000, 10000]:
            container = FlextContainer.create()
            # Pre-register services
            for i in range(count):
                container.register(f"service_{i}", f"value_{i}")
            PerformanceBenchmark.benchmark_list_services(container, iterations=100)
            # Log results
