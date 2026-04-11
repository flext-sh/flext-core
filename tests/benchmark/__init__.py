# AUTO-GENERATED FILE — Regenerate with: make gen
"""Benchmark package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_container_memory": ("test_container_memory",),
        ".test_container_performance": ("test_container_performance",),
        ".test_lazy_performance": ("test_lazy_performance",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
