# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Enforcement Catalog Rows Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _parts as _parts
    from ._parts.flextconstantsenforcementcatalogrows_part_01_a import (
        INFRA_DETECTOR_ROWS_CORE,
    )
    from ._parts.flextconstantsenforcementcatalogrows_part_01_b import (
        INFRA_DETECTOR_ROWS_PATTERNS,
    )
    from .flextconstantsenforcementcatalogrows_part_01 import (
        FlextConstantsEnforcementCatalogInfraRows,
    )
    from .flextconstantsenforcementcatalogrows_part_02 import (
        FlextConstantsEnforcementCatalogSkillRows,
    )
    from .flextconstantsenforcementcatalogrows_part_03 import (
        FlextConstantsEnforcementCatalogToolRows,
    )
    from .flextconstantsenforcementcatalogrows_part_04 import (
        FlextConstantsEnforcementCatalogBeartypeRows,
    )
    from .flextconstantsenforcementcatalogrows_part_05 import (
        FlextConstantsEnforcementCatalogInfraRowsExtended,
    )
__all__: tuple[str, ...] = (
    "INFRA_DETECTOR_ROWS_CORE",
    "INFRA_DETECTOR_ROWS_PATTERNS",
    "FlextConstantsEnforcementCatalogBeartypeRows",
    "FlextConstantsEnforcementCatalogInfraRows",
    "FlextConstantsEnforcementCatalogInfraRowsExtended",
    "FlextConstantsEnforcementCatalogSkillRows",
    "FlextConstantsEnforcementCatalogToolRows",
    "_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
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
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
