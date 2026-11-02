# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex04
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols, c, t

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx04:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class _CommandEnvelope(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        @property
        def command_type(self) -> t.NonEmptyStr:
            """Command type identifier for the ex04 operation."""
            ...

        @property
        def command_id(self) -> t.NonEmptyStr:
            """Unique command identifier used for tracing and idempotency checks."""
            ...

        @property
        def issuer_id(self) -> t.NonEmptyStr | None:
            """Identity of the principal that issued this command."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

        @property
        def event_type(self) -> str:
            """Event type placeholder for the ex04 operation."""
            ...

    @runtime_checkable
    class _QueryEnvelope(FlextProtocols.Model, Protocol):
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
            """Query type identifier for the ex04 operation."""
            ...

        @property
        def command_type(self) -> str:
            """Command type placeholder for the ex04 operation."""
            ...

        @property
        def event_type(self) -> str:
            """Event type placeholder for the ex04 operation."""
            ...

    @runtime_checkable
    class _EventEnvelope(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerType.EVENT]:
            """Message type discriminator (always 'event')."""
            ...

        @property
        def event_type(self) -> str:
            """Event type identifier for the ex04 operation."""
            ...

        @property
        def aggregate_id(self) -> str:
            """Aggregate ID for events."""
            ...

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

        @property
        def command_type(self) -> str:
            """Command type placeholder for the ex04 operation."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

    @runtime_checkable
    class CreateUser(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr

        @property
        def command_id(self) -> t.NonEmptyStr:
            """Unique command identifier used for tracing and idempotency checks."""
            ...

        @property
        def issuer_id(self) -> t.NonEmptyStr | None:
            """Identity of the principal that issued this command."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

        event_type: str

        @property
        def username(self) -> str:
            """Username for the user to create."""
            ...

    @runtime_checkable
    class GetUser(FlextProtocols.Model, Protocol):
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

        query_type: str | None

        @property
        def command_type(self) -> str:
            """Command type placeholder for the ex04 operation."""
            ...

        event_type: str

        @property
        def username(self) -> str:
            """Username to retrieve."""
            ...

    @runtime_checkable
    class DeleteUser(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr

        @property
        def command_id(self) -> t.NonEmptyStr:
            """Unique command identifier used for tracing and idempotency checks."""
            ...

        @property
        def issuer_id(self) -> t.NonEmptyStr | None:
            """Identity of the principal that issued this command."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

        event_type: str

        @property
        def username(self) -> str:
            """Username of the user to delete."""
            ...

    @runtime_checkable
    class FailingDelete(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr

        @property
        def command_id(self) -> t.NonEmptyStr:
            """Unique command identifier used for tracing and idempotency checks."""
            ...

        @property
        def issuer_id(self) -> t.NonEmptyStr | None:
            """Identity of the principal that issued this command."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

        event_type: str

        @property
        def username(self) -> str:
            """Username for the failing delete operation."""
            ...

    @runtime_checkable
    class AutoCommand(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr

        @property
        def command_id(self) -> t.NonEmptyStr:
            """Unique command identifier used for tracing and idempotency checks."""
            ...

        @property
        def issuer_id(self) -> t.NonEmptyStr | None:
            """Identity of the principal that issued this command."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

        event_type: str

        @property
        def payload(self) -> str:
            """Payload data for the auto command."""
            ...

    @runtime_checkable
    class Ping(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr

        @property
        def command_id(self) -> t.NonEmptyStr:
            """Unique command identifier used for tracing and idempotency checks."""
            ...

        @property
        def issuer_id(self) -> t.NonEmptyStr | None:
            """Identity of the principal that issued this command."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

        event_type: str

        @property
        def value(self) -> str:
            """Value data for the ping command."""
            ...

    @runtime_checkable
    class UnknownQuery(FlextProtocols.Model, Protocol):
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

        query_type: str | None

        @property
        def command_type(self) -> str:
            """Command type placeholder for the ex04 operation."""
            ...

        event_type: str

        @property
        def payload(self) -> str:
            """Payload data for the unknown query."""
            ...

    @runtime_checkable
    class UserCreated(FlextProtocols.Model, Protocol):
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

        @property
        def command_type(self) -> str:
            """Command type placeholder for the ex04 operation."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

        @property
        def username(self) -> str:
            """Username of the user that was created."""
            ...

    @runtime_checkable
    class NoSubscriberEvent(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerType.EVENT]:
            """Message type discriminator (always 'event')."""
            ...

        event_type: str

        @property
        def aggregate_id(self) -> str:
            """Aggregate ID for events."""
            ...

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

        @property
        def command_type(self) -> str:
            """Command type placeholder for the ex04 operation."""
            ...

        @property
        def query_type(self) -> str:
            """Query type placeholder for the ex04 operation."""
            ...

        @property
        def marker(self) -> str:
            """Marker identifying the event as having no subscribers."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx04",
]
