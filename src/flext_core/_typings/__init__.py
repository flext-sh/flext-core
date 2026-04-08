# AUTO-GENERATED FILE — Regenerate with: make gen
"""Typings package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".annotateds": ("FlextTypesAnnotateds",),
        ".base": ("FlextTypingBase",),
        ".containers": ("FlextTypingContainers",),
        ".core": ("FlextTypesCore",),
        ".services": ("FlextTypesServices",),
        ".typeadapters": ("FlextTypesTypeAdapters",),
        ".validation": ("FlextTypesValidation",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
