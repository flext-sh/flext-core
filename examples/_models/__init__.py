# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".errors": ("ExamplesFlextModelsErrors",),
        ".ex00": ("ExamplesFlextModelsEx00",),
        ".ex01": ("ExamplesFlextModelsEx01",),
        ".ex02": ("ExamplesFlextModelsEx02",),
        ".ex03": ("ExamplesFlextModelsEx03",),
        ".ex04": ("ExamplesFlextModelsEx04",),
        ".ex05": ("ExamplesFlextModelsEx05",),
        ".ex07": ("ExamplesFlextModelsEx07",),
        ".ex08": ("ExamplesFlextModelsEx08",),
        ".ex10": ("ExamplesFlextModelsEx10",),
        ".ex11": ("ExamplesFlextModelsEx11",),
        ".ex12": ("ExamplesFlextModelsEx12",),
        ".ex14": ("ExamplesFlextModelsEx14",),
        ".output": ("ExamplesFlextModelsOutput",),
        ".shared": (
            "ExamplesFlextSharedHandle",
            "ExamplesFlextSharedPerson",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
