"""FLEXT Core Constants - Thin MRO Facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextConstantsBase,
    FlextConstantsCqrs,
    FlextConstantsDomain,
    FlextConstantsEnforcement,
    FlextConstantsErrors,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
    FlextConstantsOutput,
    FlextConstantsPlatform,
    FlextConstantsPydantic,
    FlextConstantsSettings,
    FlextConstantsValidation,
    FlextModelsNamespace,
)


class FlextConstants(
    FlextConstantsBase,
    FlextConstantsValidation,
    FlextConstantsSettings,
    FlextConstantsPlatform,
    FlextConstantsDomain,
    FlextConstantsCqrs,
    FlextConstantsErrors,
    FlextConstantsInfrastructure,
    FlextConstantsOutput,
    FlextConstantsMixins,
    FlextConstantsEnforcement,
    FlextConstantsPydantic,
    FlextModelsNamespace,
):
    """Centralized constants for the FLEXT ecosystem (Layer 0).

    This class acts as a facade, composing all constant subclasses via MRO.
    All constants are accessible via inheritance—do not duplicate parent attributes.
    """


c = FlextConstants
__all__: list[str] = [
    "FlextConstants",
    "c",
]
