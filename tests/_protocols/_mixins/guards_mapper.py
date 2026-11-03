# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/guards_mapper
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols

if TYPE_CHECKING:
    from flext_tests import p

    from tests import c, t


class TestsFlextProtocolsMixinsGuardsMapper:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class PortModel(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        port: int
        nested: t.JsonMapping

    @runtime_checkable
    class MaybeModel(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        x: str | None

    @runtime_checkable
    class GroupModel(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        kind: str | None

    @runtime_checkable
    class UserCreatedEvent(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerType.EVENT]:
            """Immutable event discriminator for union routing."""
            ...

        event_type: t.NonEmptyStr
        aggregate_id: t.NonEmptyStr

        @property
        def data(self) -> p.ConfigMap:
            """Event data container."""
            ...

        user_id: str
        user_name: str
        timestamp: float


__all__: list[str] = [
    "TestsFlextProtocolsMixinsGuardsMapper",
]
