# AUTO-GENERATED FILE — Regenerate with: make gen
"""Benchmark package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_container_memory": ("TestsFlextCoreContainerMemory",),
        ".test_container_performance": ("TestsFlextCoreContainerPerformance",),
        ".test_lazy_performance": ("TestsFlextCoreLazyPerformance",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
