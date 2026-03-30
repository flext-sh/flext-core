# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Benchmark tests for flext container performance and memory usage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests.benchmark import (
        test_container_memory as test_container_memory,
        test_container_performance as test_container_performance,
        test_refactor_nesting_performance as test_refactor_nesting_performance,
    )
    from tests.benchmark.test_container_memory import (
        TestContainerMemory as TestContainerMemory,
        get_memory_usage as get_memory_usage,
    )
    from tests.benchmark.test_container_performance import (
        TestContainerPerformance as TestContainerPerformance,
    )
    from tests.benchmark.test_refactor_nesting_performance import (
        TestPerformanceBenchmarks as TestPerformanceBenchmarks,
    )

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "TestContainerMemory": [
        "tests.benchmark.test_container_memory",
        "TestContainerMemory",
    ],
    "TestContainerPerformance": [
        "tests.benchmark.test_container_performance",
        "TestContainerPerformance",
    ],
    "TestPerformanceBenchmarks": [
        "tests.benchmark.test_refactor_nesting_performance",
        "TestPerformanceBenchmarks",
    ],
    "get_memory_usage": ["tests.benchmark.test_container_memory", "get_memory_usage"],
    "test_container_memory": ["tests.benchmark.test_container_memory", ""],
    "test_container_performance": ["tests.benchmark.test_container_performance", ""],
    "test_refactor_nesting_performance": [
        "tests.benchmark.test_refactor_nesting_performance",
        "",
    ],
}

_EXPORTS: Sequence[str] = [
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestPerformanceBenchmarks",
    "get_memory_usage",
    "test_container_memory",
    "test_container_performance",
    "test_refactor_nesting_performance",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
