"""FLEXT Core Constants - Thin MRO Facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextConstantsBase,
    FlextConstantsCqrs,
    FlextConstantsDomain,
    FlextConstantsErrors,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
    FlextConstantsPlatform,
    FlextConstantsSettings,
    FlextConstantsValidation,
)
from flext_core._constants.enforcement import FlextConstantsEnforcement
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement


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
    FlextConstantsEnforcement,
):
    """Centralized constants for the FLEXT ecosystem (Layer 0).

    This class acts as a facade, composing all constant subclasses via MRO.
    All constants are accessible via inheritance—do not duplicate parent attributes.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce constants governance on subclasses."""
        super().__init_subclass__(**kwargs)
        FlextUtilitiesEnforcement.run_constants(cls)


c = FlextConstants
__all__ = [
    "FlextConstants",
    "c",
]
