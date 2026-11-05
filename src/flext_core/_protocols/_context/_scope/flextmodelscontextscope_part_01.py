# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/_scope/flextmodelscontextscope_part_01
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import flext_core
from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    pass


class FlextProtocolsContextScopeFlextmodelscontextscopePart01:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextStatistics(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def sets(self) -> flext_core.t.NonNegativeInt:
            """Number of set operations."""
            ...

        @property
        def gets(self) -> flext_core.t.NonNegativeInt:
            """Number of get operations."""
            ...

        @property
        def removes(self) -> flext_core.t.NonNegativeInt:
            """Number of remove operations."""
            ...

        @property
        def clears(self) -> flext_core.t.NonNegativeInt:
            """Number of clear operations."""
            ...

        @property
        def operations(self) -> flext_core.t.JsonMapping:
            """Additional metric counters and timing values grouped by metric key."""
            ...


__all__: list[str] = [
    "FlextProtocolsContextScopeFlextmodelscontextscopePart01",
]
