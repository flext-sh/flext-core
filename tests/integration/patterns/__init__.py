# AUTO-GENERATED FILE — Regenerate with: make gen
"""Patterns package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_advanced_patterns": ("TestsFlextAdvancedPatterns",),
        ".test_architectural_patterns": ("TestsFlextArchitecturalPatterns",),
        ".test_patterns_commands": ("TestsFlextPatternsCommands",),
        ".test_patterns_logging": ("TestsFlextPatternsLogging",),
        ".test_patterns_testing": ("TestsFlextPatternsTesting",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
