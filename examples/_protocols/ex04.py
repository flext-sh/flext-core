# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex04
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from examples import c, t
    from flext_core import p


class ExamplesFlextProtocolsEx04:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class _CommandEnvelope(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr
        command_id: t.NonEmptyStr
        issuer_id: t.NonEmptyStr | None
        query_type: str
        event_type: str

    @runtime_checkable
    class _QueryEnvelope(FlextProtocols.BaseModel, Protocol):
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
        command_type: str
        event_type: str

    @runtime_checkable
    class _EventEnvelope(FlextProtocols.BaseModel, Protocol):
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
        command_type: str
        query_type: str

    @runtime_checkable
    class CreateUser(_CommandEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        username: str

    @runtime_checkable
    class GetUser(_QueryEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        username: str

    @runtime_checkable
    class DeleteUser(_CommandEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        username: str

    @runtime_checkable
    class FailingDelete(_CommandEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        username: str

    @runtime_checkable
    class AutoCommand(_CommandEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        payload: str

    @runtime_checkable
    class Ping(_CommandEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        value: str

    @runtime_checkable
    class UnknownQuery(_QueryEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        payload: str

    @runtime_checkable
    class UserCreated(_EventEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        username: str

    @runtime_checkable
    class NoSubscriberEvent(_EventEnvelope, Protocol):
        """Structural data contract projected from the owning model."""

        marker: str


__all__: list[str] = [
    "ExamplesFlextProtocolsEx04",
]
