# AUTO-GENERATED FILE — Regenerate with: make gen
"""Constants package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".base": ("FlextConstantsBase",),
        ".cqrs": ("FlextConstantsCqrs",),
        ".enforcement": (
            "FlextConstantsEnforcement",
            "FlextMroViolation",
        ),
        ".environment": ("FlextConstantsEnvironment",),
        ".errors": ("FlextConstantsErrors",),
        ".file": ("FlextConstantsFile",),
        ".guards": ("FlextConstantsGuards",),
        ".infrastructure": ("FlextConstantsInfrastructure",),
        ".logging": ("FlextConstantsLogging",),
        ".mixins": ("FlextConstantsMixins",),
        ".project_metadata": ("FlextConstantsProjectMetadata",),
        ".pydantic": ("FlextConstantsPydantic",),
        ".regex": ("FlextConstantsRegex",),
        ".serialization": ("FlextConstantsSerialization",),
        ".settings": ("FlextConstantsSettings",),
        ".status": ("FlextConstantsStatus",),
        ".timeout": ("FlextConstantsTimeout",),
        ".validation": ("FlextConstantsValidation",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
