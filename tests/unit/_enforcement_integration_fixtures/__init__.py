# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Integration Fixtures package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".bad_module": (
            "TestsFlextBadAccessors",
            "TestsFlextBadAnyField",
            "TestsFlextBadBareCollection",
            "TestsFlextBadConstants",
            "TestsFlextBadFrozen",
            "TestsFlextBadInlineUnion",
            "TestsFlextBadMissingDesc",
            "TestsFlextBadMutableDefault",
            "TestsFlextBadWorkerSettings",
        ),
        ".clean_module": (
            "TestsFlextCleanConstants",
            "TestsFlextCleanModels",
            "TestsFlextCleanProtocols",
            "TestsFlextCleanServiceBase",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
