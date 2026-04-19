# AUTO-GENERATED FILE — Regenerate with: make gen
"""Patterns package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_advanced_patterns": ("TestAdvancedPatterns",),
        ".test_architectural_patterns": ("TestArchitecturalPatterns",),
        ".test_patterns_commands": ("TestsFlextCorePatternsCommands",),
        ".test_patterns_logging": ("TestPatternsLogging",),
        ".test_patterns_testing": ("TestPatternsTesting",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
