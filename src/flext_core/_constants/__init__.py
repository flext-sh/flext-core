# AUTO-GENERATED FILE — Regenerate with: make gen
"""Constants package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_a import (
        INFRA_DETECTOR_ROWS_CORE,
    )
    from flext_core._constants._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_b import (
        INFRA_DETECTOR_ROWS_PATTERNS,
    )
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_01 import (
        FlextConstantsEnforcementCatalogInfraRows,
    )
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_02 import (
        FlextConstantsEnforcementCatalogSkillRows,
    )
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_03 import (
        FlextConstantsEnforcementCatalogToolRows,
    )
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_04 import (
        FlextConstantsEnforcementCatalogBeartypeRows,
    )
    from flext_core._constants._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_05 import (
        FlextConstantsEnforcementCatalogInfraRowsExtended,
    )
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
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_05 import (
        FlextConstantsEnforcementRuleText,
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
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_09 import (
        NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_01 import (
        FlextConstantsErrorsMessages,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_02 import (
        FlextConstantsErrorsRuntimeExceptions,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_03 import (
        FlextConstantsErrorsValidationExceptions,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_04 import (
        FlextConstantsErrorsDomainParser,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_05 import (
        FlextConstantsErrorsRuntimeSettings,
    )
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs
    from flext_core._constants.enforcement import (
        FlextConstantsEnforcement,
        FlextMroViolation,
    )
    from flext_core._constants.enforcement_catalog_rows import (
        FlextConstantsEnforcementCatalogRows,
    )
    from flext_core._constants.environment import FlextConstantsEnvironment
    from flext_core._constants.errors import FlextConstantsErrors
    from flext_core._constants.file import FlextConstantsFile
    from flext_core._constants.guards import FlextConstantsGuards
    from flext_core._constants.infrastructure import FlextConstantsInfrastructure
    from flext_core._constants.logging import FlextConstantsLogging
    from flext_core._constants.mixins import FlextConstantsMixins
    from flext_core._constants.project_metadata import FlextConstantsProjectMetadata
    from flext_core._constants.pydantic import FlextConstantsPydantic
    from flext_core._constants.regex import FlextConstantsRegex
    from flext_core._constants.serialization import FlextConstantsSerialization
    from flext_core._constants.settings import FlextConstantsSettings
    from flext_core._constants.status import FlextConstantsStatus
    from flext_core._constants.timeout import FlextConstantsTimeout
    from flext_core._constants.validation import FlextConstantsValidation
_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._enforcement_catalog_rows_parts": ("_enforcement_catalog_rows_parts",),
        "._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_a": (
            "INFRA_DETECTOR_ROWS_CORE",
        ),
        "._enforcement_catalog_rows_parts._parts.flextconstantsenforcementcatalogrows_part_01_b": (
            "INFRA_DETECTOR_ROWS_PATTERNS",
        ),
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
        "._enforcement_catalog_rows_parts.flextconstantsenforcementcatalogrows_part_05": (
            "FlextConstantsEnforcementCatalogInfraRowsExtended",
        ),
        "._enforcement_data": ("_enforcement_data",),
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
        "._enforcement_parts.flextconstantsenforcement_part_07": (
            "FlextConstantsEnforcementSmellData",
        ),
        "._enforcement_parts.flextconstantsenforcement_part_08": (
            "FlextConstantsEnforcementFixActions",
        ),
        "._enforcement_parts.flextconstantsenforcement_part_09": (
            "NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT",
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


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
