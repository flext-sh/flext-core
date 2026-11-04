# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/export
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsEntity

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsBase, FlextProtocolsContainers
    from flext_core._typings import FlextTypesServices, FlextTypingBase


class FlextProtocolsContextExport:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextExport(FlextProtocolsEntity.Value, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def data(self) -> Mapping[str, FlextTypesServices.JsonPayload]:
            """All context data from all scopes."""
            ...

        @property
        def metadata(
            self,
        ) -> FlextProtocolsBase.Metadata | FlextProtocolsContainers.Dict | None:
            """Context metadata (creation info, source, etc.)."""
            ...

        @property
        def statistics(self) -> FlextTypingBase.JsonMapping:
            """Usage statistics (operation counts, timing info)."""
            ...


__all__: list[str] = [
    "FlextProtocolsContextExport",
]
