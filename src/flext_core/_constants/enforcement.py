"""Enforcement constants for Pydantic v2 runtime governance.

Constants used by FlextModelsBase.Enforcement to validate class definitions at
import time. Accessed via c.ENFORCEMENT_*.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._enforcement_parts.flextconstantsenforcement_part_01 import (
    FlextConstantsEnforcementEnums,
)
from ._enforcement_parts.flextconstantsenforcement_part_02 import (
    FlextConstantsEnforcementRuntime,
)
from ._enforcement_parts.flextconstantsenforcement_part_03 import (
    FlextConstantsEnforcementNamespace,
)
from ._enforcement_parts.flextconstantsenforcement_part_04 import (
    FlextConstantsEnforcementRules,
)
from ._enforcement_parts.flextconstantsenforcement_part_05 import (
    FlextConstantsEnforcementRuleText,
)
from ._enforcement_parts.flextconstantsenforcement_part_06 import (
    FlextConstantsEnforcementTargets,
)
from ._enforcement_parts.flextconstantsenforcement_part_07 import (
    FlextConstantsEnforcementSmellData,
)
from ._enforcement_parts.flextconstantsenforcement_part_08 import (
    FlextConstantsEnforcementFixActions,
)
from .enforcement_catalog_rows import FlextConstantsEnforcementCatalogRows


class FlextMroViolation(UserWarning):
    """Runtime governance violation emitted by the FLEXT enforcement engine."""


class FlextSmellViolation(FlextMroViolation):
    """Runtime code-smell violation emitted by the FLEXT enforcement engine."""


class FlextConstantsEnforcement(
    FlextConstantsEnforcementCatalogRows,
    FlextConstantsEnforcementEnums,
    FlextConstantsEnforcementRuntime,
    FlextConstantsEnforcementNamespace,
    FlextConstantsEnforcementRules,
    FlextConstantsEnforcementRuleText,
    FlextConstantsEnforcementTargets,
    FlextConstantsEnforcementSmellData,
    FlextConstantsEnforcementFixActions,
):
    """Constants governing Pydantic v2 enforcement behavior."""


# mro-pulj (codex): the declaring module owns every supported root import;
# generated registries must not be required to publish this violation model.
__all__: list[str] = [
    "FlextConstantsEnforcement",
    "FlextMroViolation",
    "FlextSmellViolation",
]
