# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex10
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from examples import c, t
from flext_core import FlextProtocols

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx10:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Message(FlextProtocols.Model, Protocol):
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

        @property
        def text(self) -> str:
            """Message text content."""
            ...

    @runtime_checkable
    class DerivedMessage(FlextProtocols.Model, Protocol):
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

        @property
        def text(self) -> str:
            """Message text content."""
            ...

    @runtime_checkable
    class ContextPayload(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def text(self) -> str:
            """Text payload for context."""
            ...

    @runtime_checkable
    class Entity(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def unique_id(self) -> str:
            """Unique identifier for entity."""
            ...

    @runtime_checkable
    class ProcessorGood(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def marker(self) -> str:
            """Marker indicating successful processing."""
            ...

        @staticmethod
        def process() -> bool:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class ProcessorBad(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def marker(self) -> str:
            """Marker indicating failed processing."""
            ...

    @runtime_checkable
    class ProtocolHandler(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> type[p.Command]:
            """Message type for protocol handler."""
            ...

        @staticmethod
        def handle(message: ExamplesFlextProtocolsEx10.Message) -> p.Result[str]:
            """Structural member projected from the owning model."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx10",
]
