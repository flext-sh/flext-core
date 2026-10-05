# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Enforcement Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_01 import (
        FlextConstantsEnforcementEnums,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_02 import (
        FlextConstantsEnforcementRuntime,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_03 import (
        FlextConstantsEnforcementNamespace,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_04 import (
        FlextConstantsEnforcementRules,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_06 import (
        FlextConstantsEnforcementTargets,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_07 import (
        FlextConstantsEnforcementSmellData,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_08 import (
        FlextConstantsEnforcementFixActions,
    )


__all__: tuple[str, ...] = (
    "FlextConstantsEnforcementEnums",
    "FlextConstantsEnforcementFixActions",
    "FlextConstantsEnforcementNamespace",
    "FlextConstantsEnforcementRules",
    "FlextConstantsEnforcementRuntime",
    "FlextConstantsEnforcementSmellData",
    "FlextConstantsEnforcementTargets",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextConstantsEnforcementEnums": ".flextconstantsenforcement_part_01",
        "FlextConstantsEnforcementFixActions": ".flextconstantsenforcement_part_08",
        "FlextConstantsEnforcementNamespace": ".flextconstantsenforcement_part_03",
        "FlextConstantsEnforcementRules": ".flextconstantsenforcement_part_04",
        "FlextConstantsEnforcementRuntime": ".flextconstantsenforcement_part_02",
        "FlextConstantsEnforcementSmellData": ".flextconstantsenforcement_part_07",
        "FlextConstantsEnforcementTargets": ".flextconstantsenforcement_part_06",
    }),
    public_exports=__all__,
)
