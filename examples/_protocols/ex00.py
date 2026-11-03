# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex00
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from examples import c
from flext_core import FlextProtocols

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx00:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class UserProfile(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def domain_events(self) -> Sequence[p.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

        @property
        def name(self) -> str:
            """User display name."""
            ...

        @property
        def email(self) -> str:
            """User email address."""
            ...

        @property
        def status(self) -> c.Status:
            """User account status."""
            ...

        def activate(self) -> p.Result[bool]:
            """Activate user once."""
            ...

    @runtime_checkable
    class UserInput(FlextProtocols.Model, Protocol):
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
