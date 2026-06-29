# AUTO-GENERATED FILE — Regenerate with: make gen
"""Fixtures package."""

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
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
