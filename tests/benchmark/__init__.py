# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core.test_container_memory import TestContainerMemory, get_memory_usage
    from flext_core.test_container_performance import TestContainerPerformance
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_container_memory": (
            "TestContainerMemory",
            "get_memory_usage",
        ),
        ".test_container_performance": ("TestContainerPerformance",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__ = [
    "TestContainerMemory",
    "TestContainerPerformance",
    "get_memory_usage",
]
