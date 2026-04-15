# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_base": ("TestsFlextCoreModelsBase",),
        ".test_cqrs": ("TestsFlextCoreModelsCqrs",),
        ".test_entity": ("TestFlextModelsEntity",),
        ".test_exception_params": ("TestFlextModelsExceptionParams",),
        "flext_core": (
            "c",
            "d",
            "e",
            "h",
            "m",
            "p",
            "r",
            "s",
            "t",
            "u",
            "x",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
