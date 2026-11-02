# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/_scope/flextmodelscontextscope_part_03
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsContainer


class FlextProtocolsContextScopeFlextmodelscontextscopePart03:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextContainerState(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def container(self) -> FlextProtocolsContainer.Container | None:
            """Container configured for service namespace resolution."""
            ...


__all__: list[str] = [
    "FlextProtocolsContextScopeFlextmodelscontextscopePart03",
]
