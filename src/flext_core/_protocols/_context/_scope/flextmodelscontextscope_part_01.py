# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/_scope/flextmodelscontextscope_part_01
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import t


class FlextProtocolsContextScopeFlextmodelscontextscopePart01:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextStatistics(Protocol):
        """Structural data contract projected from the owning model."""

        sets: t.NonNegativeInt
        gets: t.NonNegativeInt
        removes: t.NonNegativeInt
        clears: t.NonNegativeInt
        operations: t.JsonMapping


__all__: list[str] = [
    "FlextProtocolsContextScopeFlextmodelscontextscopePart01",
]
