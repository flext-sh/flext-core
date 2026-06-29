# AUTO-GENERATED FILE — Regenerate with: make gen
"""Beartype package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._class_visitor_parts": ("_class_visitor_parts",),
        "._class_visitor_parts.class_visitor_part_03": (
            "FlextUtilitiesBeartypeClassVisitor",
        ),
        "._helpers_parts": ("_helpers_parts",),
        "._helpers_parts.helpers_part_03": ("FlextUtilitiesBeartypeHelpers",),
        ".attr_visitor": ("FlextUtilitiesBeartypeAttrVisitor",),
        ".deprecated_visitor": ("FlextUtilitiesBeartypeDeprecatedVisitor",),
        ".field_visitor": ("FlextUtilitiesBeartypeFieldVisitor",),
        ".import_visitor": ("FlextUtilitiesBeartypeImportVisitor",),
        ".method_visitor": ("FlextUtilitiesBeartypeMethodVisitor",),
        ".module_visitor": ("FlextUtilitiesBeartypeModuleVisitor",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
