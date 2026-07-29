# AUTO-GENERATED FILE — Regenerate with: make gen
"""Benchmark package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map({
    ".test_container_memory": ("TestsFlextContainerMemory",),
    ".test_container_performance": ("TestsFlextContainerPerformance",),
    ".test_lazy_performance": ("TestsFlextLazyPerformance",),
    "flext_tests": (
        "c",
        "d",
        "e",
        "h",
        "m",
        "p",
        "r",
        "s",
        "t",
        "td",
        "tf",
        "tk",
        "tm",
        "tv",
        "u",
        "x",
    ),
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
