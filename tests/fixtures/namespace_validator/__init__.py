# AUTO-GENERATED FILE — Regenerate with: make gen
"""Namespace Validator package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        "namespace_validator.rule0_loose_items": ("rule0_loose_items",),
        "namespace_validator.rule0_multiple_classes": ("rule0_multiple_classes",),
        "namespace_validator.rule0_no_class": ("rule0_no_class",),
        "namespace_validator.rule0_valid": ("rule0_valid",),
        "namespace_validator.rule0_wrong_prefix": ("rule0_wrong_prefix",),
        "namespace_validator.rule1_loose_constant": ("rule1_loose_constant",),
        "namespace_validator.rule1_loose_enum": ("rule1_loose_enum",),
        "namespace_validator.rule1_magic_number": ("rule1_magic_number",),
        "namespace_validator.rule1_method_in_constants": ("rule1_method_in_constants",),
        "namespace_validator.rule1_valid_constants": ("rule1_valid_constants",),
        "namespace_validator.rule2_composite_type_loose": (
            "rule2_composite_type_loose",
        ),
        "namespace_validator.rule2_protocol_in_types": ("rule2_protocol_in_types",),
        "namespace_validator.rule2_typevar_in_class": ("rule2_typevar_in_class",),
        "namespace_validator.rule2_typevar_wrong_module": (
            "rule2_typevar_wrong_module",
        ),
        "namespace_validator.rule2_valid_types": ("rule2_valid_types",),
        "namespace_validator.typings": ("LooseTypeAlias",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
