# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex08
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from examples import t
    from flext_core import p


class ExamplesFlextProtocolsEx08:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class UserEntity(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        version: t.NonNegativeInt
        unique_id: t.NonEmptyStr

        @property
        def created_at(self) -> datetime:
            """Creation timestamp (configured timezone)."""
            ...

        updated_at: datetime | None

        @property
        def domain_events(self) -> Sequence[p.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

        name: str

    @runtime_checkable
    class OrderEntity(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        version: t.NonNegativeInt
        unique_id: t.NonEmptyStr

        @property
        def created_at(self) -> datetime:
            """Creation timestamp (configured timezone)."""
            ...

        updated_at: datetime | None

        @property
        def domain_events(self) -> Sequence[p.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

        status: str


__all__: list[str] = [
    "ExamplesFlextProtocolsEx08",
]
