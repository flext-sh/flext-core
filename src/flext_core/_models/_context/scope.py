"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models import flextmodelscontextscope_part_03


class FlextModelsContextScope(
    flextmodelscontextscope_part_03.FlextModelsContextScopePart03
):
    """Public facade for FlextModelsContextScope."""


__all__: list[str] = ["FlextModelsContextScope"]
