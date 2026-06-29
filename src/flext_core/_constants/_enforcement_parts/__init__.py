# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Parts package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextconstantsenforcement_part_01": ("FlextConstantsEnforcementEnums",),
        ".flextconstantsenforcement_part_02": ("FlextConstantsEnforcementRuntime",),
        ".flextconstantsenforcement_part_03": ("FlextConstantsEnforcementNamespace",),
        ".flextconstantsenforcement_part_04": ("FlextConstantsEnforcementRules",),
        ".flextconstantsenforcement_part_05": ("FlextConstantsEnforcementRuleText",),
        ".flextconstantsenforcement_part_06": ("FlextConstantsEnforcementTargets",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
