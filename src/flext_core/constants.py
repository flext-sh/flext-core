"""FLEXT Core Constants - Thin MRO Facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextConstantsBase,
    FlextConstantsCqrs,
    FlextConstantsDomain,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
    FlextConstantsPlatform,
    FlextConstantsSettings,
    FlextConstantsValidation,
)


class FlextConstants(
    FlextConstantsBase,
    FlextConstantsValidation,
    FlextConstantsSettings,
    FlextConstantsPlatform,
    FlextConstantsDomain,
    FlextConstantsCqrs,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
):
    """Centralized constants for the FLEXT ecosystem (Layer 0).

    This class acts as a facade, composing all constant subclasses via MRO.
    All constants are accessible via inheritance—do not duplicate parent attributes.
    """

    # NOTE: Both FlextConstantsDomain and FlextConstantsCqrs define Status (StrEnum).
    # The MRO resolves to FlextConstantsDomain.Status first, which is correct.
    Status = FlextConstantsDomain.Status

    pass


c = FlextConstants
__all__ = [
    "FlextConstants",
    "c",
]
