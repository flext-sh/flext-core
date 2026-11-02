# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/flext_context
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsContainers


class FlextProtocolsFlextContext:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextState(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def data(self) -> FlextProtocolsContainers.ConfigMap:
            """Scoped key-value payload for this context instance."""
            ...

        @property
        def metadata(self) -> FlextProtocolsBase.Metadata:
            """Correlation and service metadata snapshot."""
            ...


__all__: list[str] = [
    "FlextProtocolsFlextContext",
]
