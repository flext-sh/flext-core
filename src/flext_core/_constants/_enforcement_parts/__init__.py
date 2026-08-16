# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Enforcement Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    ".flextconstantsenforcement_part_01": ("FlextConstantsEnforcementEnums",),
    ".flextconstantsenforcement_part_02": ("FlextConstantsEnforcementRuntime",),
    ".flextconstantsenforcement_part_03": ("FlextConstantsEnforcementNamespace",),
    ".flextconstantsenforcement_part_04": ("FlextConstantsEnforcementRules",),
    ".flextconstantsenforcement_part_05": ("FlextConstantsEnforcementRuleText",),
    ".flextconstantsenforcement_part_06": ("FlextConstantsEnforcementTargets",),
    ".flextconstantsenforcement_part_07": ("FlextConstantsEnforcementSmellData",),
    ".flextconstantsenforcement_part_08": ("FlextConstantsEnforcementFixActions",),
    ".flextconstantsenforcement_part_09": ("NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT",),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
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

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
