# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_cqrs/flextmodelscqrs_part_01
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import t


class FlextProtocolsCqrsFlextmodelscqrsPart01:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Pagination(Protocol):
        """Structural data contract projected from the owning model."""

        page: t.PositiveInt
        size: t.PositiveInt

        @property
        def limit(self) -> int: ...
        @property
        def offset(self) -> int: ...


__all__: list[str] = [
    "FlextProtocolsCqrsFlextmodelscqrsPart01",
]
