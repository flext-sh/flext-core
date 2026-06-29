"""Enforcement constants for Pydantic v2 runtime governance.

Constants used by FlextModelsBase.Enforcement to validate class definitions at
import time. Accessed via c.ENFORCEMENT_*.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._constants._enforcement_parts import (
    FlextConstantsEnforcementEnums,
    FlextConstantsEnforcementNamespace,
    FlextConstantsEnforcementRules,
    FlextConstantsEnforcementRuleText,
    FlextConstantsEnforcementRuntime,
    FlextConstantsEnforcementTargets,
)
from flext_core._constants.enforcement_catalog_rows import (
    FlextConstantsEnforcementCatalogRows,
)


class FlextMroViolation(UserWarning):
    """Runtime governance violation emitted by the FLEXT enforcement engine."""


class FlextConstantsEnforcement(
    FlextConstantsEnforcementCatalogRows,
    FlextConstantsEnforcementEnums,
    FlextConstantsEnforcementRuntime,
    FlextConstantsEnforcementNamespace,
    FlextConstantsEnforcementRules,
    FlextConstantsEnforcementRuleText,
    FlextConstantsEnforcementTargets,
):
    """Constants governing Pydantic v2 enforcement behavior."""


__all__: list[str] = ["FlextConstantsEnforcement", "FlextMroViolation"]
