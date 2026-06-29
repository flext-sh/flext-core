# AUTO-GENERATED FILE — Regenerate with: make gen
"""Exceptions package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._base_parts": ("_base_parts",),
        "._base_parts.flextexceptionsbase_part_01": ("FlextBaseErrorMetadataMixin",),
        "._base_parts.flextexceptionsbase_part_02": ("FlextBaseErrorStateMixin",),
        "._factories_parts": ("_factories_parts",),
        "._factories_parts.flextexceptionsfactories_part_04": (
            "FlextExceptionsFactories",
        ),
        ".base": ("FlextExceptionsBase",),
        ".helpers": ("FlextExceptionsHelpers",),
        ".metrics": ("FlextExceptionsMetrics",),
        ".template": ("FlextExceptionsTemplate",),
        ".types": ("FlextExceptionsTypes",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
