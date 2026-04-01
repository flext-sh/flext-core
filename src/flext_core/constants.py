"""FLEXT Core Constants - Thin MRO Facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._constants.base import FlextConstantsBase
from flext_core._constants.cqrs import FlextConstantsCqrs
from flext_core._constants.domain import FlextConstantsDomain
from flext_core._constants.errors import FlextConstantsErrors
from flext_core._constants.infrastructure import FlextConstantsInfrastructure
from flext_core._constants.mixins import FlextConstantsMixins
from flext_core._constants.platform import FlextConstantsPlatform
from flext_core._constants.settings import FlextConstantsSettings
from flext_core._constants.validation import FlextConstantsValidation


class FlextConstants(
    FlextConstantsBase,
    FlextConstantsValidation,
    FlextConstantsSettings,
    FlextConstantsPlatform,
    FlextConstantsDomain,
    FlextConstantsCqrs,
    FlextConstantsErrors,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
):
    """Centralized constants for the FLEXT ecosystem (Layer 0).

    This class acts as a facade, composing all constant subclasses via MRO.
    All constants are accessible via inheritance—do not duplicate parent attributes.
    """


c = FlextConstants
__all__ = [
    "FlextConstants",
    "c",
]
