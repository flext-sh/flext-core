# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_01 import (
        FlextConstantsEnforcementEnums as FlextConstantsEnforcementEnums,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_02 import (
        FlextConstantsEnforcementRuntime as FlextConstantsEnforcementRuntime,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_03 import (
        FlextConstantsEnforcementNamespace as FlextConstantsEnforcementNamespace,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_04 import (
        FlextConstantsEnforcementRules as FlextConstantsEnforcementRules,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_05 import (
        FlextConstantsEnforcementRuleText as FlextConstantsEnforcementRuleText,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_06 import (
        FlextConstantsEnforcementTargets as FlextConstantsEnforcementTargets,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_07 import (
        FlextConstantsEnforcementSmellData as FlextConstantsEnforcementSmellData,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_08 import (
        FlextConstantsEnforcementFixActions as FlextConstantsEnforcementFixActions,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextconstantsenforcement_part_01": ("FlextConstantsEnforcementEnums",),
        ".flextconstantsenforcement_part_02": ("FlextConstantsEnforcementRuntime",),
        ".flextconstantsenforcement_part_03": ("FlextConstantsEnforcementNamespace",),
        ".flextconstantsenforcement_part_04": ("FlextConstantsEnforcementRules",),
        ".flextconstantsenforcement_part_05": ("FlextConstantsEnforcementRuleText",),
        ".flextconstantsenforcement_part_06": ("FlextConstantsEnforcementTargets",),
        ".flextconstantsenforcement_part_07": ("FlextConstantsEnforcementSmellData",),
        ".flextconstantsenforcement_part_08": ("FlextConstantsEnforcementFixActions",),
        ".flextconstantsenforcement_part_09": ("flextconstantsenforcement_part_09",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
