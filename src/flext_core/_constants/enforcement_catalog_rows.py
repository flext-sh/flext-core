"""Canonical enforcement catalog row constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_01 import (
    FlextConstantsEnforcementCatalogInfraRows,
)
from ._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_02 import (
    FlextConstantsEnforcementCatalogSkillRows,
)
from ._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_03 import (
    FlextConstantsEnforcementCatalogToolRows,
)
from ._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_04 import (
    FlextConstantsEnforcementCatalogBeartypeRows,
)


class FlextConstantsEnforcementCatalogRows(
    FlextConstantsEnforcementCatalogInfraRows,
    FlextConstantsEnforcementCatalogSkillRows,
    FlextConstantsEnforcementCatalogToolRows,
    FlextConstantsEnforcementCatalogBeartypeRows,
):
    """Table-driven rows used to build the canonical enforcement catalog."""


__all__ = ["FlextConstantsEnforcementCatalogRows"]
