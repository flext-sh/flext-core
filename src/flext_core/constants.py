"""FLEXT Core Constants - Thin MRO Facade.

from flext_core import FlextConstants as Constants

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextConstantsBase,
    FlextConstantsCqrs,
    FlextConstantsEnforcement,
    FlextConstantsEnvironment,
    FlextConstantsErrors,
    FlextConstantsFile,
    FlextConstantsGuards,
    FlextConstantsInfrastructure,
    FlextConstantsLogging,
    FlextConstantsMixins,
    FlextConstantsProjectMetadata,
    FlextConstantsPydantic,
    FlextConstantsRegex,
    FlextConstantsSerialization,
    FlextConstantsSettings,
    FlextConstantsStatus,
    FlextConstantsTimeout,
    FlextConstantsValidation,
    FlextModelsNamespace,
)


class FlextConstants(
    FlextConstantsBase,
    FlextConstantsTimeout,
    FlextConstantsEnvironment,
    FlextConstantsLogging,
    FlextConstantsFile,
    FlextConstantsStatus,
    FlextConstantsRegex,
    FlextConstantsSerialization,
    FlextConstantsValidation,
    FlextConstantsSettings,
    FlextConstantsCqrs,
    FlextConstantsErrors,
    FlextConstantsGuards,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
    FlextConstantsEnforcement,
    FlextConstantsProjectMetadata,
    FlextConstantsPydantic,
    FlextModelsNamespace,
):
    """SSOT facade: all constants flat on c.* via MRO composition."""


__all__: tuple[str, ...] = (
    "FlextConstants",
    "c",
)

c = FlextConstants
