# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Enforcement Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import build_lazy_import_map, install_lazy_exports

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


__all__: tuple[str, ...] = (
    "FlextConstantsEnforcementEnums",
    "FlextConstantsEnforcementNamespace",
    "FlextConstantsEnforcementRules",
    "FlextConstantsEnforcementRuntime",
    "FlextConstantsEnforcementSmellData",
    "FlextConstantsEnforcementTargets",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".flextconstantsenforcement_part_01": ("FlextConstantsEnforcementEnums",),
            ".flextconstantsenforcement_part_02": ("FlextConstantsEnforcementRuntime",),
            ".flextconstantsenforcement_part_03": (
                "FlextConstantsEnforcementNamespace",
            ),
            ".flextconstantsenforcement_part_04": ("FlextConstantsEnforcementRules",),
            ".flextconstantsenforcement_part_06": ("FlextConstantsEnforcementTargets",),
            ".flextconstantsenforcement_part_07": (
                "FlextConstantsEnforcementSmellData",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
