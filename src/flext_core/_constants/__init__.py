# AUTO-GENERATED FILE — Regenerate with: make gen
"""Constants package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._enforcement_catalog_rows_parts": ("_enforcement_catalog_rows_parts",),
        "._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_01": (
            "FlextConstantsEnforcementCatalogInfraRows",
        ),
        "._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_02": (
            "FlextConstantsEnforcementCatalogSkillRows",
        ),
        "._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_03": (
            "FlextConstantsEnforcementCatalogToolRows",
        ),
        "._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_04": (
            "FlextConstantsEnforcementCatalogBeartypeRows",
        ),
        "._enforcement_parts": ("_enforcement_parts",),
        "._enforcement_parts.flextconstantsenforcement_part_01": (
            "FlextConstantsEnforcementEnums",
        ),
        "._enforcement_parts.flextconstantsenforcement_part_02": (
            "FlextConstantsEnforcementRuntime",
        ),
        "._enforcement_parts.flextconstantsenforcement_part_03": (
            "FlextConstantsEnforcementNamespace",
        ),
        "._enforcement_parts.flextconstantsenforcement_part_04": (
            "FlextConstantsEnforcementRules",
        ),
        "._enforcement_parts.flextconstantsenforcement_part_05": (
            "FlextConstantsEnforcementRuleText",
        ),
        "._enforcement_parts.flextconstantsenforcement_part_06": (
            "FlextConstantsEnforcementTargets",
        ),
        "._errors_parts": ("_errors_parts",),
        "._errors_parts.flextconstantserrors_part_01": (
            "FlextConstantsErrorsMessages",
        ),
        "._errors_parts.flextconstantserrors_part_02": (
            "FlextConstantsErrorsRuntimeExceptions",
        ),
        "._errors_parts.flextconstantserrors_part_03": (
            "FlextConstantsErrorsValidationExceptions",
        ),
        "._errors_parts.flextconstantserrors_part_04": (
            "FlextConstantsErrorsDomainParser",
        ),
        "._errors_parts.flextconstantserrors_part_05": (
            "FlextConstantsErrorsRuntimeSettings",
        ),
        ".base": ("FlextConstantsBase",),
        ".cqrs": ("FlextConstantsCqrs",),
        ".enforcement": (
            "FlextConstantsEnforcement",
            "FlextMroViolation",
        ),
        ".enforcement_catalog_rows": ("FlextConstantsEnforcementCatalogRows",),
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
