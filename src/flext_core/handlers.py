"""CQRS handler foundation facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._handlers_parts.flexthandlers_part_07 import FlextHandlersPart07


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersPart07[MessageT_contra, ResultT]
):
    """Public CQRS handler facade."""


h = FlextHandlers

__all__: list[str] = ["FlextHandlers", "h"]
