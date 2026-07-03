"""Infrastructure enforcement catalog rows."""

from __future__ import annotations

from typing import Final

from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_a import (
    INFRA_DETECTOR_ROWS_CORE,
)
from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_b import (
    INFRA_DETECTOR_ROWS_PATTERNS,
)


class FlextConstantsEnforcementCatalogInfraRows:
    """Infra detector rows for the enforcement catalog."""

    INFRA_DETECTOR_ROWS: Final[
        tuple[tuple[str, str, str, str, tuple[str, ...], bool, str], ...]
    ] = (
        *INFRA_DETECTOR_ROWS_CORE,
        *INFRA_DETECTOR_ROWS_PATTERNS,
    )


__all__ = ["FlextConstantsEnforcementCatalogInfraRows"]
