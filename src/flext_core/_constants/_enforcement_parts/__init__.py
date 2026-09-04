# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Enforcement Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextconstantsenforcement_part_01 import FlextConstantsEnforcementEnums
    from .flextconstantsenforcement_part_02 import FlextConstantsEnforcementRuntime
    from .flextconstantsenforcement_part_03 import FlextConstantsEnforcementNamespace
    from .flextconstantsenforcement_part_04 import FlextConstantsEnforcementRules
    from .flextconstantsenforcement_part_05 import FlextConstantsEnforcementRuleText
    from .flextconstantsenforcement_part_06 import FlextConstantsEnforcementTargets
    from .flextconstantsenforcement_part_07 import FlextConstantsEnforcementSmellData
    from .flextconstantsenforcement_part_08 import FlextConstantsEnforcementFixActions
    from .flextconstantsenforcement_part_09 import (
        NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT,
    )
__all__: tuple[str, ...] = (
    "NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT",
    "FlextConstantsEnforcementEnums",
    "FlextConstantsEnforcementFixActions",
    "FlextConstantsEnforcementNamespace",
    "FlextConstantsEnforcementRuleText",
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
            ".flextconstantsenforcement_part_05": (
                "FlextConstantsEnforcementRuleText",
            ),
            ".flextconstantsenforcement_part_06": ("FlextConstantsEnforcementTargets",),
            ".flextconstantsenforcement_part_07": (
                "FlextConstantsEnforcementSmellData",
            ),
            ".flextconstantsenforcement_part_08": (
                "FlextConstantsEnforcementFixActions",
            ),
            ".flextconstantsenforcement_part_09": (
                "NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
