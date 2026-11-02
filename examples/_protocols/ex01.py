# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex01
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols, c, t

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx01:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class User(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def domain_events(self) -> Sequence[p.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

        name: str
        email: str

    @runtime_checkable
    class DemonstrationResult(FlextProtocols.Model, Protocol):
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
    class RunDemonstrationCommand(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        @property
        def command_type(self) -> t.NonEmptyStr:
            """Command type identifier."""
            ...

        @property
        def command_id(self) -> t.NonEmptyStr:
            """Unique command identifier used for tracing and idempotency checks."""
            ...

        @property
        def issuer_id(self) -> t.NonEmptyStr | None:
            """Identity of the principal that issued this command."""
            ...

        operation: str

    @runtime_checkable
    class ValidPersonPayload(FlextProtocols.Model, Protocol):
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
    class InvalidPersonPayload(FlextProtocols.Model, Protocol):
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
