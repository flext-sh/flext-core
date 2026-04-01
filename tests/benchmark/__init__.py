# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Benchmark tests for flext container performance and memory usage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from tests.benchmark import (
        test_container_memory,
        test_container_performance,
        test_refactor_nesting_performance,
    )
    from tests.benchmark.test_container_memory import (
        TestContainerMemory,
        get_memory_usage,
    )
    from tests.benchmark.test_container_performance import TestContainerPerformance
    from tests.benchmark.test_refactor_nesting_performance import (
        TestPerformanceBenchmarks,
    )

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "TestContainerMemory": "tests.benchmark.test_container_memory",
    "TestContainerPerformance": "tests.benchmark.test_container_performance",
    "TestPerformanceBenchmarks": "tests.benchmark.test_refactor_nesting_performance",
    "get_memory_usage": "tests.benchmark.test_container_memory",
    "test_container_memory": "tests.benchmark.test_container_memory",
    "test_container_performance": "tests.benchmark.test_container_performance",
    "test_refactor_nesting_performance": "tests.benchmark.test_refactor_nesting_performance",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
