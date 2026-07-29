"""FLEXT Core Constants - Thin MRO Facade.

from flext_core import FlextConstants as Constants

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._constants.base import FlextConstantsBase
from flext_core._constants.config import FlextConstantsConfig
from flext_core._constants.cqrs import FlextConstantsCqrs
from flext_core._constants.enforcement import FlextConstantsEnforcement
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

__all__: tuple[str, ...] = ("FlextConstants", "FlextConstantsEnforcement", "c")
