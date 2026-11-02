# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex08
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx08:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class UserEntity(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def domain_events(self) -> Sequence[p.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

        @property
        def name(self) -> str:
            """User display name."""
            ...

    @runtime_checkable
    class OrderEntity(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def domain_events(self) -> Sequence[p.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

        @property
        def status(self) -> str:
            """Order lifecycle status."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx08",
]
