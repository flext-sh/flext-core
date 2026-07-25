"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._cqrs_parts.flextmodelscqrs_part_02 import (
    FlextModelsCqrs as FlextModelsCqrsPartFinal,
)


class FlextModelsCqrs(FlextModelsCqrsPartFinal):
    """Public facade for FlextModelsCqrs."""


__all__: list[str] = ["FlextModelsCqrs"]
