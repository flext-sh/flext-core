# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Benchmark tests for flext container performance and memory usage."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import FlextTypes
    from tests.benchmark.test_container_memory import (
        TestContainerMemory,
        get_memory_usage,
    )
    from tests.benchmark.test_container_performance import TestContainerPerformance
    from tests.benchmark.test_refactor_nesting_performance import (
        TestPerformanceBenchmarks,
    )

_LAZY_IMPORTS: Mapping[str, tuple[str, str]] = {
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
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestPerformanceBenchmarks",
    "get_memory_usage",
]


_LAZY_CACHE: MutableMapping[str, FlextTypes.ModuleExport] = {}


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562).

    A local cache ``_LAZY_CACHE`` persists resolved objects across repeated
    accesses during process lifetime.

    Args:
        name: Attribute name requested by dir()/import.

    Returns:
        Lazy-loaded module export type.

    Raises:
        AttributeError: If attribute not registered.

    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    value = lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)
    _LAZY_CACHE[name] = value
    return value


def __dir__() -> Sequence[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.

    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
