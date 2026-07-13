# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Catalog Rows Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_a import (
        INFRA_DETECTOR_ROWS_CORE as INFRA_DETECTOR_ROWS_CORE,
    )
    from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_b import (
        INFRA_DETECTOR_ROWS_PATTERNS as INFRA_DETECTOR_ROWS_PATTERNS,
    )
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
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_05 import (
        FlextConstantsEnforcementCatalogInfraRowsExtended as FlextConstantsEnforcementCatalogInfraRowsExtended,
    )
_LAZY_IMPORTS = build_lazy_import_map({
    "._parts": ("_parts",),
    "._parts.flextconstantsenforcementcatalogrows_part_01_a": (
        "INFRA_DETECTOR_ROWS_CORE",
    ),
    "._parts.flextconstantsenforcementcatalogrows_part_01_b": (
        "INFRA_DETECTOR_ROWS_PATTERNS",
    ),
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
    ".flextconstantsenforcementcatalogrows_part_05": (
        "FlextConstantsEnforcementCatalogInfraRowsExtended",
    ),
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
