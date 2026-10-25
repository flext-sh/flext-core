# AUTO-GENERATED FILE — Regenerate with: make gen
"""FLEXT Core Constants - Thin MRO Facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._constants import (
    FlextConstantsBase,
    FlextConstantsBeartype,
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
    FlextConstantsBeartype,
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
):
    """SSOT facade: all constants flat on c.* via MRO composition.

    The ``c`` letter alias binds to this class; domain-specific constant
    groups (timeouts, environment, logging, status, errors, ...) each live in
    their own module and flatten onto this single namespace at import time.
    """


c = FlextConstants

__all__ = ("FlextConstants", "FlextConstantsEnforcement", "c")
