# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex00
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from examples import c, t
    from flext_core import p


class ExamplesFlextProtocolsEx00:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class UserProfile(FlextProtocols.BaseModel, Protocol):
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
        email: str
        status: c.Status

        def activate(self) -> p.Result[bool]:
            """Activate user once."""
            ...

    @runtime_checkable
    class UserInput(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """User display name."""
            ...

        @property
        def email(self) -> str:
            """User email address."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx00",
]
