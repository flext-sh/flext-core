# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex07
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols, c, t

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx07:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class CreateUserCommand(FlextProtocols.Model, Protocol):
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

        name: str
        email: str

    @runtime_checkable
    class UserCreatedEvent(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerType.EVENT]:
            """Message type discriminator (always 'event')."""
            ...

        event_type: str
        aggregate_id: str

        @property
        def event_id(self) -> t.NonEmptyStr:
            """Unique event identifier used for deduplication and observability."""
            ...

        @property
        def data(self) -> Mapping[str, t.Scalar]:
            """Event payload data."""
            ...

        @property
        def metadata(self) -> Mapping[str, t.Scalar]:
            """Event metadata (timestamps, correlation IDs, etc.)."""
            ...

        name: str

    @runtime_checkable
    class GetUserQuery(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.QUERY]:
            """Message type discriminator."""
            ...

        @property
        def filters(self) -> Mapping[str, t.Scalar]:
            """Structural member projected from the owning model.

            Filter values that restrict which records are returned by the query.
            """
            ...

        @property
        def pagination(self) -> p.Pagination:
            """Structural member projected from the owning model.

            Pagination settings controlling page number and page size for query
            results.
            """
            ...

        @property
        def query_id(self) -> t.NonEmptyStr:
            """Unique query identifier used for tracing and cache correlation."""
            ...

        @property
        def query_type(self) -> str | None:
            """Query type identifier for dispatcher routing."""
            ...

        user_id: str

    @runtime_checkable
    class DemoPlugin(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Plugin name."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx07",
]
