# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/guards_mapper
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols

from tests import c, t

if TYPE_CHECKING:
    from flext_tests import p


class TestsFlextProtocolsMixinsGuardsMapper:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class PortModel(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        port: int
        nested: t.JsonMapping

    @runtime_checkable
    class MaybeModel(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        x: str | None

    @runtime_checkable
    class GroupModel(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        kind: str | None

    @runtime_checkable
    class UserCreatedEvent(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerType.EVENT]:
            """Immutable event discriminator for union routing."""
            ...

        @property
        def event_type(self) -> t.NonEmptyStr:
            """Domain event type identifier for subscriber routing."""
            ...

        @property
        def aggregate_id(self) -> t.NonEmptyStr:
            """Identifier of the aggregate root that produced this event."""
            ...

        @property
        def data(self) -> p.ConfigMap:
            """Event data container."""
            ...

        @property
        def user_id(self) -> str:
            """Identifier of the created user."""
            ...

        @property
        def user_name(self) -> str:
            """Name assigned to the new user."""
            ...

        @property
        def timestamp(self) -> float:
            """POSIX timestamp when the event fired."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsGuardsMapper",
]
