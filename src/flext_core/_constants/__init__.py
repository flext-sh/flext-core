# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants import (
        _enforcement as _enforcement,
        _enforcement_data as _enforcement_data,
        _errors as _errors,
    )
    from flext_core._constants._enforcement import (
        FlextConstantsEnforcementEnums,
        FlextConstantsEnforcementFixActions,
        FlextConstantsEnforcementNamespace,
        FlextConstantsEnforcementRules,
        FlextConstantsEnforcementRuntime,
        FlextConstantsEnforcementSmellData,
        FlextConstantsEnforcementTargets,
    )
    from flext_core._constants._enforcement_data import (
        FlextConstantsEnforcementData,
        FlextPredicateRule,
        FlextSmellThresholds,
    )
    from flext_core._constants._errors import (
        FlextConstantsErrorsDomainParser,
        FlextConstantsErrorsMessages,
        FlextConstantsErrorsRuntimeExceptions,
        FlextConstantsErrorsRuntimeSettings,
        FlextConstantsErrorsValidationExceptions,
    )
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.config import FlextConstantsConfig
    from flext_core._constants.cqrs import FlextConstantsCqrs
    from flext_core._constants.enforcement import (
        FlextConstantsEnforcement,
        FlextMroViolation,
    )
    from flext_core._constants.environment import FlextConstantsEnvironment
    from flext_core._constants.errors import FlextConstantsErrors
    from flext_core._constants.file import FlextConstantsFile
    from flext_core._constants.guards import FlextConstantsGuards
    from flext_core._constants.infrastructure import FlextConstantsInfrastructure
    from flext_core._constants.loggings import FlextConstantsLogging
    from flext_core._constants.mixins import FlextConstantsMixins
    from flext_core._constants.project_metadata import FlextConstantsProjectMetadata
    from flext_core._constants.pydantic import FlextConstantsPydantic
    from flext_core._constants.regex import FlextConstantsRegex
    from flext_core._constants.serialization import FlextConstantsSerialization
    from flext_core._constants.settings import FlextConstantsSettings
    from flext_core._constants.status import FlextConstantsStatus
    from flext_core._constants.timeout import FlextConstantsTimeout
    from flext_core._constants.validation import FlextConstantsValidation


__all__: tuple[str, ...] = (
    "FlextConstantsBase",
    "FlextConstantsConfig",
    "FlextConstantsCqrs",
    "FlextConstantsEnforcement",
    "FlextConstantsEnforcementData",
    "FlextConstantsEnforcementEnums",
    "FlextConstantsEnforcementFixActions",
    "FlextConstantsEnforcementNamespace",
    "FlextConstantsEnforcementRules",
    "FlextConstantsEnforcementRuntime",
    "FlextConstantsEnforcementSmellData",
    "FlextConstantsEnforcementTargets",
    "FlextConstantsEnvironment",
    "FlextConstantsErrors",
    "FlextConstantsErrorsDomainParser",
    "FlextConstantsErrorsMessages",
    "FlextConstantsErrorsRuntimeExceptions",
    "FlextConstantsErrorsRuntimeSettings",
    "FlextConstantsErrorsValidationExceptions",
    "FlextConstantsFile",
    "FlextConstantsGuards",
    "FlextConstantsInfrastructure",
    "FlextConstantsLogging",
    "FlextConstantsMixins",
    "FlextConstantsProjectMetadata",
    "FlextConstantsPydantic",
    "FlextConstantsRegex",
    "FlextConstantsSerialization",
    "FlextConstantsSettings",
    "FlextConstantsStatus",
    "FlextConstantsTimeout",
    "FlextConstantsValidation",
    "FlextMroViolation",
    "FlextPredicateRule",
    "FlextSmellThresholds",
    "_enforcement",
    "_enforcement_data",
    "_errors",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextConstantsBase": ".base",
        "FlextConstantsConfig": ".config",
        "FlextConstantsCqrs": ".cqrs",
        "FlextConstantsEnforcement": ".enforcement",
        "FlextConstantsEnforcementData": "._enforcement_data.package_data",
        "FlextConstantsEnforcementEnums": (
            "._enforcement.flextconstantsenforcement_part_01"
        ),
        "FlextConstantsEnforcementFixActions": (
            "._enforcement.flextconstantsenforcement_part_08"
        ),
        "FlextConstantsEnforcementNamespace": (
            "._enforcement.flextconstantsenforcement_part_03"
        ),
        "FlextConstantsEnforcementRules": (
            "._enforcement.flextconstantsenforcement_part_04"
        ),
        "FlextConstantsEnforcementRuntime": (
            "._enforcement.flextconstantsenforcement_part_02"
        ),
        "FlextConstantsEnforcementSmellData": (
            "._enforcement.flextconstantsenforcement_part_07"
        ),
        "FlextConstantsEnforcementTargets": (
            "._enforcement.flextconstantsenforcement_part_06"
        ),
        "FlextConstantsEnvironment": ".environment",
        "FlextConstantsErrors": ".errors",
        "FlextConstantsErrorsDomainParser": "._errors.flextconstantserrors_part_04",
        "FlextConstantsErrorsMessages": "._errors.flextconstantserrors_part_01",
        "FlextConstantsErrorsRuntimeExceptions": (
            "._errors.flextconstantserrors_part_02"
        ),
        "FlextConstantsErrorsRuntimeSettings": "._errors.flextconstantserrors_part_05",
        "FlextConstantsErrorsValidationExceptions": (
            "._errors.flextconstantserrors_part_03"
        ),
        "FlextConstantsFile": ".file",
        "FlextConstantsGuards": ".guards",
        "FlextConstantsInfrastructure": ".infrastructure",
        "FlextConstantsLogging": ".loggings",
        "FlextConstantsMixins": ".mixins",
        "FlextConstantsProjectMetadata": ".project_metadata",
        "FlextConstantsPydantic": ".pydantic",
        "FlextConstantsRegex": ".regex",
        "FlextConstantsSerialization": ".serialization",
        "FlextConstantsSettings": ".settings",
        "FlextConstantsStatus": ".status",
        "FlextConstantsTimeout": ".timeout",
        "FlextConstantsValidation": ".validation",
        "FlextMroViolation": ".enforcement",
        "FlextPredicateRule": "._enforcement_data.flextpredicaterule",
        "FlextSmellThresholds": "._enforcement_data.flextsmellthresholds",
        "_enforcement": "._enforcement",
        "_enforcement_data": "._enforcement_data",
        "_errors": "._errors",
    }),
    public_exports=__all__,
)
