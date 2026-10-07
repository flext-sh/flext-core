"""FLEXT Core Constants - Thin MRO Facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._constants import (
    FlextConstantsBase,
    FlextConstantsConfig,
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
    FlextConstantsConfig,
    FlextConstantsCqrs,
    FlextConstantsErrors,
    FlextConstantsGuards,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
    FlextConstantsEnforcement,
    FlextConstantsProjectMetadata,
    FlextConstantsPydantic,
):
    """SSOT facade: all constants flat on c.* via MRO composition."""


# mro-j47u: publish the canonical constants alias with no stray runtime surface.
c = FlextConstants

__all__ = ("FlextConstants", "FlextConstantsEnforcement", "c")
