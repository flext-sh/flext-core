# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex01
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from examples import c, t
    from flext_core import p


class ExamplesFlextProtocolsEx01:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class User(FlextProtocols.BaseModel, Protocol):
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

    @runtime_checkable
    class DemonstrationResult(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def demonstrations_completed(self) -> int:
            """Count of completed demonstrations."""
            ...

        @property
        def patterns_covered(self) -> tuple[str, ...]:
            """Tuple of covered pattern names."""
            ...

        @property
        def completed_at(self) -> str:
            """ISO 8601 completion timestamp."""
            ...

    @runtime_checkable
    class RunDemonstrationCommand(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr
        command_id: t.NonEmptyStr
        issuer_id: t.NonEmptyStr | None
        operation: str

    @runtime_checkable
    class ValidPersonPayload(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Person name."""
            ...

        @property
        def age(self) -> int:
            """Person age in years."""
            ...

    @runtime_checkable
    class InvalidPersonPayload(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Person name."""
            ...

        @property
        def age(self) -> str:
            """Invalid age (string instead of int) for testing."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx01",
]
