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
    FlextConstantsNetwork,
    FlextConstantsPagination,
    FlextConstantsPlatform,
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
    FlextConstantsNetwork,
    FlextConstantsTimeout,
    FlextConstantsEnvironment,
    FlextConstantsLogging,
    FlextConstantsFile,
    FlextConstantsStatus,
    FlextConstantsRegex,
    FlextConstantsSerialization,
    FlextConstantsPagination,
    FlextConstantsBase,
    FlextConstantsValidation,
    FlextConstantsSettings,
    FlextConstantsPlatform,
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
    """Centralized constants for the FLEXT ecosystem (Layer 0).

    This class acts as a facade, composing all constant subclasses via MRO.
    All constants are accessible via inheritance—do not duplicate parent attributes.
    """


__all__: list[str] = [
    "FlextConstants",
    "c",
]

c = FlextConstants
