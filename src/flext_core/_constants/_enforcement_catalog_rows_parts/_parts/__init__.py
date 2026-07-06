# AUTO-GENERATED FILE — Regenerate with: make gen
"""Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_a import (
        INFRA_DETECTOR_ROWS_CORE,
    )
    from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_b import (
        INFRA_DETECTOR_ROWS_PATTERNS,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextconstantsenforcementcatalogrows_part_01_a": (
            "INFRA_DETECTOR_ROWS_CORE",
        ),
        ".flextconstantsenforcementcatalogrows_part_01_b": (
            "INFRA_DETECTOR_ROWS_PATTERNS",
        ),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
