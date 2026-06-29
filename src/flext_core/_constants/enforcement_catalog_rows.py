"""Canonical enforcement catalog row constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._enforcement_catalog_rows_parts import (
    FlextConstantsEnforcementCatalogBeartypeRows,
    FlextConstantsEnforcementCatalogInfraRows,
    FlextConstantsEnforcementCatalogSkillRows,
    FlextConstantsEnforcementCatalogToolRows,
)


class FlextConstantsEnforcementCatalogRows(
    FlextConstantsEnforcementCatalogInfraRows,
    FlextConstantsEnforcementCatalogSkillRows,
    FlextConstantsEnforcementCatalogToolRows,
    FlextConstantsEnforcementCatalogBeartypeRows,
):
    """Table-driven rows used to build the canonical enforcement catalog."""


__all__ = ["FlextConstantsEnforcementCatalogRows"]
