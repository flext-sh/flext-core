# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Benchmark tests for flext container performance and memory usage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.benchmark.test_container_memory import (
        TestContainerMemory,
        get_memory_usage,
    )
    from tests.benchmark.test_container_performance import (
        PerformanceBenchmark,
        TestContainerPerformance,
    )
    from tests.benchmark.test_refactor_nesting_performance import (
        TestPerformanceBenchmarks,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "PerformanceBenchmark": (
        "tests.benchmark.test_container_performance",
        "PerformanceBenchmark",
    ),
    "TestContainerMemory": (
        "tests.benchmark.test_container_memory",
        "TestContainerMemory",
    ),
    "TestContainerPerformance": (
        "tests.benchmark.test_container_performance",
        "TestContainerPerformance",
    ),
    "TestPerformanceBenchmarks": (
        "tests.benchmark.test_refactor_nesting_performance",
        "TestPerformanceBenchmarks",
    ),
    "get_memory_usage": ("tests.benchmark.test_container_memory", "get_memory_usage"),
}

__all__ = [
    "PerformanceBenchmark",
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestPerformanceBenchmarks",
    "get_memory_usage",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
