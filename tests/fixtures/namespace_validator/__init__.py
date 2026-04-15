# AUTO-GENERATED FILE — Regenerate with: make gen
"""Namespace Validator package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".rule0_loose_items": ("rule0_loose_items",),
        ".rule0_multiple_classes": ("rule0_multiple_classes",),
        ".rule0_no_class": ("rule0_no_class",),
        ".rule0_valid": ("rule0_valid",),
        ".rule0_wrong_prefix": ("rule0_wrong_prefix",),
        ".rule1_loose_constant": ("rule1_loose_constant",),
        ".rule1_loose_enum": ("rule1_loose_enum",),
        ".rule1_magic_number": ("rule1_magic_number",),
        ".rule1_method_in_constants": ("rule1_method_in_constants",),
        ".rule1_valid_constants": ("rule1_valid_constants",),
        ".rule2_composite_type_loose": ("rule2_composite_type_loose",),
        ".rule2_protocol_in_types": ("rule2_protocol_in_types",),
        ".rule2_typevar_in_class": ("rule2_typevar_in_class",),
        ".rule2_typevar_wrong_module": ("rule2_typevar_wrong_module",),
        ".rule2_valid_types": ("rule2_valid_types",),
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
