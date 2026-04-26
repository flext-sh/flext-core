# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".errors": ("ExamplesFlextCoreModelsErrors",),
        ".ex00": ("ExamplesFlextCoreModelsEx00",),
        ".ex01": ("ExamplesFlextCoreModelsEx01",),
        ".ex02": ("ExamplesFlextCoreModelsEx02",),
        ".ex03": (
            "Ex03Email",
            "Ex03Money",
            "Ex03Order",
            "Ex03OrderItem",
            "Ex03User",
            "ExamplesFlextCoreModelsEx03",
        ),
        ".ex04": ("ExamplesFlextCoreModelsEx04",),
        ".ex05": ("ExamplesFlextCoreModelsEx05",),
        ".ex07": ("ExamplesFlextCoreModelsEx07",),
        ".ex08": ("ExamplesFlextCoreModelsEx08",),
        ".ex10": ("ExamplesFlextCoreModelsEx10",),
        ".ex11": ("ExamplesFlextCoreModelsEx11",),
        ".ex12": ("ExamplesFlextCoreModelsEx12",),
        ".ex14": ("ExamplesFlextCoreModelsEx14",),
        ".output": ("ExamplesFlextCoreModelsOutput",),
        ".shared": (
            "ExamplesFlextCoreSharedHandle",
            "ExamplesFlextCoreSharedPerson",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
