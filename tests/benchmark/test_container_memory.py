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
from flext_core import FlextContainer


def get_memory_usage() -> int:
    """Get current memory usage in bytes.

    Returns:
        Memory usage in bytes (approximate).

    """
    # Force garbage collection
    gc.collect()
    # Return approximate memory usage
    # This is a simplified measurement - real memory profiling would use tracemalloc
    return sys.getsizeof(gc.get_objects())


class TestContainerMemory:
    """Memory usage benchmarks for container operations."""

    @pytest.mark.benchmark
    def test_container_singleton_memory(self) -> None:
        """Benchmark memory usage of container singleton."""
        # Force GC before measurement
        gc.collect()
        initial_memory = get_memory_usage()

        # Create container
        FlextContainer.get_global()

        # Force GC after creation
        gc.collect()
        after_creation = get_memory_usage()

        _ = after_creation - initial_memory

    @pytest.mark.benchmark
    def test_memory_with_services(self) -> None:
        """Benchmark memory usage with registered services."""
        container = FlextContainer.create()

        # Force GC before measurement
        gc.collect()
        initial_memory = get_memory_usage()

        # Register services (reduced from 1000 to 100 for memory efficiency)
        for i in range(100):
            container.register(f"service_{i}", f"value_{i}")

        # Force GC after registration
        gc.collect()
        after_registration = get_memory_usage()

        _ = after_registration - initial_memory

    @pytest.mark.benchmark
    def test_memory_with_factories(self) -> None:
        """Benchmark memory usage with registered factories."""
        container = FlextContainer.create()

        # Force GC before measurement
        gc.collect()
        initial_memory = get_memory_usage()

        # Register factories (reduced from 1000 to 100 for memory efficiency)
        def make_factory(captured_i: int) -> Callable[[], str]:
            """Create factory function that captures i in closure."""
            return lambda: f"value_{captured_i}"

        for i in range(100):
            container.register_factory(
                f"factory_{i}",
                make_factory(i),
            )

        # Force GC after registration
        gc.collect()
        after_registration = get_memory_usage()

        _ = after_registration - initial_memory

    @pytest.mark.benchmark
    def test_memory_after_clear_all(self) -> None:
        """Benchmark memory usage after clear_all()."""
        container = FlextContainer.create()

        # Register services (reduced from 1000 to 100 for memory efficiency)
        for i in range(100):
            container.register(f"service_{i}", f"value_{i}")

        # Force GC before clear
        gc.collect()
        before_clear = get_memory_usage()

        # Clear all
        container.clear_all()

        # Force GC after clear
        gc.collect()
        after_clear = get_memory_usage()

        _ = before_clear - after_clear

    @pytest.mark.benchmark
    def test_memory_leak_detection(self) -> None:
        """Detect potential memory leaks in resource lifecycle."""
        # Force GC before measurement
        gc.collect()
        initial_memory = get_memory_usage()

        # Create and destroy containers multiple times (reduced iterations for memory efficiency)
        for _ in range(10):
            container = FlextContainer.create()
            for i in range(10):
                container.register(f"service_{i}", f"value_{i}")
            container.clear_all()
            del container

        # Force GC after operations
        gc.collect()
        final_memory = get_memory_usage()

        memory_increase = final_memory - initial_memory

        # If memory increase is significant, it might indicate a leak
        # This is a basic check - real leak detection would use tracemalloc
        assert memory_increase < 1_000_000, (
            f"Potential memory leak: {memory_increase} bytes increase"
        )
