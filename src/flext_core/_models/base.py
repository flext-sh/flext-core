"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelsBase
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models._base_parts.flextmodelsbase_part_03 import (
    FlextModelsBase as FlextModelsBasePartFinal,
)


class FlextModelsBase(FlextModelsBasePartFinal):
    """Public facade for FlextModelsBase."""


__all__: list[str] = ["FlextModelsBase"]
