# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_cqrs/flextmodelscqrs_part_01
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsCqrsFlextmodelscqrsPart01:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Pagination(FlextProtocolsBase.FlexibleInternalModel, Protocol):
        """Structural data contract projected from the owning model."""

        page: t.PositiveInt
        size: t.PositiveInt

        @property
        def limit(self) -> int:
            """Limit alias for size."""
            ...

        @property
        def offset(self) -> int:
            """Offset from page and size."""
            ...


__all__: list[str] = [
    "FlextProtocolsCqrsFlextmodelscqrsPart01",
]
