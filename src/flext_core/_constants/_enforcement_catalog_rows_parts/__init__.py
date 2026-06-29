# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Catalog Rows Parts package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextconstantsenforcementcatalogrows_part_01": (
            "FlextConstantsEnforcementCatalogInfraRows",
        ),
        ".flextconstantsenforcementcatalogrows_part_02": (
            "FlextConstantsEnforcementCatalogSkillRows",
        ),
        ".flextconstantsenforcementcatalogrows_part_03": (
            "FlextConstantsEnforcementCatalogToolRows",
        ),
        ".flextconstantsenforcementcatalogrows_part_04": (
            "FlextConstantsEnforcementCatalogBeartypeRows",
        ),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
