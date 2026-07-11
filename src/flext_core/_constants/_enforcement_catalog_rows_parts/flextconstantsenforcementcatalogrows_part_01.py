"""Infrastructure enforcement catalog rows."""

from __future__ import annotations

from typing import Final

from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_a import (
    INFRA_DETECTOR_ROWS_CORE,
)
from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_b import (
    INFRA_DETECTOR_ROWS_PATTERNS,
)
from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_05 import (
    FlextConstantsEnforcementCatalogInfraRowsExtended,
)


class FlextConstantsEnforcementCatalogInfraRows:
    """Infra detector rows for the enforcement catalog."""

    INFRA_DETECTOR_ROWS: Final[
        tuple[tuple[str, str, str, str, tuple[str, ...], bool, str], ...]
    ] = (
        *INFRA_DETECTOR_ROWS_CORE,
        *INFRA_DETECTOR_ROWS_PATTERNS,
        *FlextConstantsEnforcementCatalogInfraRowsExtended.INFRA_DETECTOR_ROWS_EXTENDED,
    )

    # Staged (inert) detector rules: present in the catalog and testable in
    # isolation, but shipped ``enabled=False`` so the workspace gate does not
    # fire on them until their existing offenders are cleared. Move an id out of
    # this set (same cycle the offenders reach zero) to activate the rule.
    STAGED_INFRA_RULE_IDS: Final[frozenset[str]] = frozenset({
        "ENFORCE-098",
    })


__all__ = ["FlextConstantsEnforcementCatalogInfraRows"]
