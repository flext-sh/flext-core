"""Memory usage benchmarks for FlextContainer.

This module provides benchmarks for memory usage of container operations to
establish baseline memory metrics before optimizations. Benchmarks measure:
- Memory usage of container singleton
- Memory leaks in resources
- Lifecycle of factories and resources
- Memory optimization opportunities

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable

import pytest

from tests import FlextContainer


def get_memory_usage() -> int:
    """Get current memory usage in bytes.

    Returns:
        Memory usage in bytes (approximate).

    """
    gc.collect()
    return sys.getsizeof(gc.get_objects())


class TestContainerMemory:
    """Memory usage benchmarks for container operations."""

    @pytest.mark.benchmark
    def test_container_singleton_memory(self) -> None:
        """Benchmark memory usage of container singleton."""
        gc.collect()
        initial_memory = get_memory_usage()
        FlextContainer.shared()
        gc.collect()
        after_creation = get_memory_usage()
        _ = after_creation - initial_memory

    @pytest.mark.benchmark
    def test_memory_with_services(self) -> None:
        """Benchmark memory usage with registered services."""
        container = FlextContainer.shared()
        gc.collect()
        initial_memory = get_memory_usage()
        for i in range(100):
            _ = container.bind(f"service_{i}", f"value_{i}")
        gc.collect()
        after_registration = get_memory_usage()
        _ = after_registration - initial_memory

    @pytest.mark.benchmark
    def test_memory_with_factories(self) -> None:
        """Benchmark memory usage with registered factories."""
        container = FlextContainer.shared()
        gc.collect()
        initial_memory = get_memory_usage()

        def make_factory(captured_i: int) -> Callable[[], str]:
            """Create factory function that captures i in closure."""
            return lambda: f"value_{captured_i}"

        for i in range(100):
            _ = container.factory(f"factory_{i}", make_factory(i))
        gc.collect()
        after_registration = get_memory_usage()
        _ = after_registration - initial_memory

    @pytest.mark.benchmark
    def test_memory_after_clear_all(self) -> None:
        """Benchmark memory usage after clear_all()."""
        container = FlextContainer.shared()
        for i in range(100):
            _ = container.bind(f"service_{i}", f"value_{i}")
        gc.collect()
        before_clear = get_memory_usage()
        container.clear()
        gc.collect()
        after_clear = get_memory_usage()
        _ = before_clear - after_clear

    @pytest.mark.benchmark
    def test_memory_leak_detection(self) -> None:
        """Detect potential memory leaks in resource lifecycle."""
        gc.collect()
        initial_memory = get_memory_usage()
        for _ in range(10):
            container = FlextContainer.shared()
            for i in range(10):
                _ = container.bind(f"service_{i}", f"value_{i}")
            container.clear()
            del container
        gc.collect()
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        assert memory_increase < 1000000, (
            f"Potential memory leak: {memory_increase} bytes increase"
        )
