# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex03
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx03:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Email(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def value(self) -> str:
            """Email address value."""
            ...

    @runtime_checkable
    class Money(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def amount(self) -> float:
            """Monetary amount."""
            ...

        @property
        def currency(self) -> str:
            """ISO 4217 currency code."""
            ...

    @runtime_checkable
    class OrderItem(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def sku(self) -> str:
            """Stock-keeping unit identifier."""
            ...

        @property
        def quantity(self) -> int:
            """Number of units ordered."""
            ...

    @runtime_checkable
    class Order(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def domain_events(self) -> Sequence[p.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

        @property
        def status(self) -> str:
            """Order lifecycle status."""
            ...

    @runtime_checkable
    class EmailUser(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def domain_events(self) -> Sequence[p.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

        @property
        def email(self) -> str:
            """User email address."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx03",
]
