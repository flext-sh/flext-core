"""FlextModelsExceptionParams - validated params for typed exception hierarchy.

Canonical home for exception parameter models. Used by:
- FlextExceptions (exceptions.py) for __init__ validation
- FlextModelsSettings ErrorConfig models (settings.py) via inheritance

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models._exception_params_parts.flextmodelsexceptionparams_part_03 import (
    FlextModelsExceptionParams as FlextModelsExceptionParamsPartFinal,
)


class FlextModelsExceptionParams(FlextModelsExceptionParamsPartFinal):
    """Public facade for FlextModelsExceptionParams."""


__all__: list[str] = ["FlextModelsExceptionParams"]
