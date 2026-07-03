"""FLEXT Core Constants - Thin MRO Facade.

from flext_core import FlextConstants as Constants

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._constants.base import FlextConstantsBase
from ._constants.cqrs import FlextConstantsCqrs
from ._constants.enforcement import FlextConstantsEnforcement
from ._constants.environment import FlextConstantsEnvironment
from ._constants.errors import FlextConstantsErrors
from ._constants.file import FlextConstantsFile
from ._constants.guards import FlextConstantsGuards
from ._constants.infrastructure import FlextConstantsInfrastructure
from ._constants.logging import FlextConstantsLogging
from ._constants.mixins import FlextConstantsMixins
from ._constants.project_metadata import FlextConstantsProjectMetadata
from ._constants.pydantic import FlextConstantsPydantic
from ._constants.regex import FlextConstantsRegex
from ._constants.serialization import FlextConstantsSerialization
from ._constants.settings import FlextConstantsSettings
from ._constants.status import FlextConstantsStatus
from ._constants.timeout import FlextConstantsTimeout
from ._constants.validation import FlextConstantsValidation


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
):
    """SSOT facade: all constants flat on c.* via MRO composition."""


__all__: tuple[str, ...] = (
    "FlextConstants",
    "FlextConstantsEnforcement",
    "c",
)

c = FlextConstants
