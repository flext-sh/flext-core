# AUTO-GENERATED FILE — Regenerate with: make gen
"""Exceptions package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".base": ("FlextExceptionsBase",),
        ".factories": ("FlextExceptionsFactories",),
        ".helpers": ("FlextExceptionsHelpers",),
        ".metrics": ("FlextExceptionsMetrics",),
        ".template": ("FlextExceptionsTemplate",),
        ".types": ("FlextExceptionsTypes",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
