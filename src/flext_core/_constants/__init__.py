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
        _enforcement_data,
        _enforcement_parts,
        _errors_parts,
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
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_06 import (
        FlextConstantsEnforcementTargets,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_07 import (
        FlextConstantsEnforcementSmellData,
    )
    from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_08 import (
        FlextConstantsEnforcementFixActions,
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
    "_enforcement_data",
    "_enforcement_parts",
    "_errors_parts",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextConstantsBase": ".base",
        "FlextConstantsConfig": ".config",
        "FlextConstantsCqrs": ".cqrs",
        "FlextConstantsEnforcement": ".enforcement",
        "FlextConstantsEnforcementEnums": (
            "._enforcement_parts.flextconstantsenforcement_part_01"
        ),
        "FlextConstantsEnforcementFixActions": (
            "._enforcement_parts.flextconstantsenforcement_part_08"
        ),
        "FlextConstantsEnforcementNamespace": (
            "._enforcement_parts.flextconstantsenforcement_part_03"
        ),
        "FlextConstantsEnforcementRules": (
            "._enforcement_parts.flextconstantsenforcement_part_04"
        ),
        "FlextConstantsEnforcementRuntime": (
            "._enforcement_parts.flextconstantsenforcement_part_02"
        ),
        "FlextConstantsEnforcementSmellData": (
            "._enforcement_parts.flextconstantsenforcement_part_07"
        ),
        "FlextConstantsEnforcementTargets": (
            "._enforcement_parts.flextconstantsenforcement_part_06"
        ),
        "FlextConstantsEnvironment": ".environment",
        "FlextConstantsErrors": ".errors",
        "FlextConstantsErrorsDomainParser": (
            "._errors_parts.flextconstantserrors_part_04"
        ),
        "FlextConstantsErrorsMessages": "._errors_parts.flextconstantserrors_part_01",
        "FlextConstantsErrorsRuntimeExceptions": (
            "._errors_parts.flextconstantserrors_part_02"
        ),
        "FlextConstantsErrorsRuntimeSettings": (
            "._errors_parts.flextconstantserrors_part_05"
        ),
        "FlextConstantsErrorsValidationExceptions": (
            "._errors_parts.flextconstantserrors_part_03"
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
        "_enforcement_data": "._enforcement_data",
        "_enforcement_parts": "._enforcement_parts",
        "_errors_parts": "._errors_parts",
    }),
    public_exports=__all__,
)
