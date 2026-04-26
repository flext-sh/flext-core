# AUTO-GENERATED FILE — Regenerate with: make gen
"""Utilities package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_beartype_violation_capture": (
            "TestsFlextCoreUtilitiesBeartypeViolationCapture",
        ),
        ".test_guards": ("TestsFlextCoreUtilitiesGuards",),
        ".test_lightweight_ast": ("TestsFlextCoreUtilitiesLightweightAst",),
        ".test_mapper": ("TestsFlextCoreUtilitiesMapper",),
        ".test_runtime_violation_registry": (
            "TestsFlextCoreUtilitiesRuntimeViolationRegistry",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
