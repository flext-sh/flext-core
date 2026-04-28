# AUTO-GENERATED FILE — Regenerate with: make gen
"""Beartype package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".attr_visitor": ("FlextUtilitiesBeartypeAttrVisitor",),
        ".class_visitor": ("FlextUtilitiesBeartypeClassVisitor",),
        ".deprecated_visitor": ("FlextUtilitiesBeartypeDeprecatedVisitor",),
        ".field_visitor": ("FlextUtilitiesBeartypeFieldVisitor",),
        ".helpers": ("FlextUtilitiesBeartypeHelpers",),
        ".import_visitor": ("FlextUtilitiesBeartypeImportVisitor",),
        ".method_visitor": ("FlextUtilitiesBeartypeMethodVisitor",),
        ".module_visitor": ("FlextUtilitiesBeartypeModuleVisitor",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
