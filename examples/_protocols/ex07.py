# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex07
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from examples import c, t
    from flext_core import p


class ExamplesFlextProtocolsEx07:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class CreateUserCommand(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr
        command_id: t.NonEmptyStr
        issuer_id: t.NonEmptyStr | None
        name: str
        email: str

    @runtime_checkable
    class UserCreatedEvent(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerType.EVENT]:
            """Message type discriminator (always 'event')."""
            ...

        event_type: str
        aggregate_id: str
        event_id: t.NonEmptyStr
        data: Mapping[str, t.Scalar]
        metadata: Mapping[str, t.Scalar]
        name: str

    @runtime_checkable
    class GetUserQuery(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.QUERY]:
            """Message type discriminator."""
            ...

        filters: Mapping[str, t.Scalar]

        @property
        def pagination(self) -> p.Pagination:
            """Structural member projected from the owning model.

            Pagination settings controlling page number and page size for query
            results.
            """
            ...

        query_id: t.NonEmptyStr
        query_type: str | None
        user_id: str

    @runtime_checkable
    class DemoPlugin(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Plugin name."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx07",
]
