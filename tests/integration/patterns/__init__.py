# AUTO-GENERATED FILE — Regenerate with: make gen
"""Patterns package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_advanced_patterns": ("TestsFlextCoreAdvancedPatterns",),
        ".test_architectural_patterns": ("TestsFlextCoreArchitecturalPatterns",),
        ".test_patterns_commands": ("TestsFlextCorePatternsCommands",),
        ".test_patterns_logging": ("TestsFlextCorePatternsLogging",),
        ".test_patterns_testing": ("TestsFlextCorePatternsTesting",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
