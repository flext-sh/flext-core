# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Parts package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".enforcement_part_01": ("PREDICATE_BINDINGS",),
        ".enforcement_part_05": ("FlextUtilitiesEnforcement",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
