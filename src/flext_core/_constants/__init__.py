# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants._errors.flextconstantserrors_part_01 import (
        FlextConstantsErrorsMessages,
    )
    from flext_core._constants._errors.flextconstantserrors_part_02 import (
        FlextConstantsErrorsRuntimeExceptions,
    )
    from flext_core._constants._errors.flextconstantserrors_part_03 import (
        FlextConstantsErrorsValidationExceptions,
    )
    from flext_core._constants._errors.flextconstantserrors_part_04 import (
        FlextConstantsErrorsDomainParser,
    )
    from flext_core._constants._errors.flextconstantserrors_part_05 import (
        FlextConstantsErrorsRuntimeSettings,
    )
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.config import FlextConstantsConfig
    from flext_core._constants.cqrs import FlextConstantsCqrs
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
    from flext_core._constants.violations import FlextMroViolation, FlextSmellViolation

__all__: tuple[str, ...] = (
    "FlextConstantsBase",
    "FlextConstantsConfig",
    "FlextConstantsCqrs",
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
    "FlextSmellViolation",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextConstantsBase": ".base",
        "FlextConstantsConfig": ".config",
        "FlextConstantsCqrs": ".cqrs",
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
        "FlextMroViolation": ".violations",
        "FlextSmellViolation": ".violations",
    }),
    public_exports=__all__,
)
