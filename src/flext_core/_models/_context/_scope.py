"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .__scope_parts.flextmodelscontextscope_part_03 import (
    FlextModelsContextScope as FlextModelsContextScopePartFinal,
)


class FlextModelsContextScope(FlextModelsContextScopePartFinal):
    """Public facade for FlextModelsContextScope."""


__all__: list[str] = ["FlextModelsContextScope"]
