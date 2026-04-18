# AUTO-GENERATED FILE — Regenerate with: make gen
"""Benchmark package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_container_memory": ("TestContainerMemory",),
        ".test_container_performance": ("TestContainerPerformance",),
        ".test_lazy_performance": ("TestLazyPerformance",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
