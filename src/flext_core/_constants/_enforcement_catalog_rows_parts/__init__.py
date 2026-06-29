# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Catalog Rows Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_01 import (
        FlextConstantsEnforcementCatalogInfraRows as FlextConstantsEnforcementCatalogInfraRows,
    )
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_02 import (
        FlextConstantsEnforcementCatalogSkillRows as FlextConstantsEnforcementCatalogSkillRows,
    )
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_03 import (
        FlextConstantsEnforcementCatalogToolRows as FlextConstantsEnforcementCatalogToolRows,
    )
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_04 import (
        FlextConstantsEnforcementCatalogBeartypeRows as FlextConstantsEnforcementCatalogBeartypeRows,
    )
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
