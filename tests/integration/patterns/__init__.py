# AUTO-GENERATED FILE — Regenerate with: make gen
"""Patterns package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_advanced_patterns": ("test_advanced_patterns",),
        ".test_architectural_patterns": ("test_architectural_patterns",),
        ".test_patterns_commands": ("test_patterns_commands",),
        ".test_patterns_logging": ("test_patterns_logging",),
        ".test_patterns_testing": ("test_patterns_testing",),
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
