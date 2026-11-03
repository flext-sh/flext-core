# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex10
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from examples import c, t
    from flext_core import p


class ExamplesFlextProtocolsEx10:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Message(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr
        command_id: t.NonEmptyStr
        issuer_id: t.NonEmptyStr | None
        text: str

    @runtime_checkable
    class DerivedMessage(Message, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class ContextPayload(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def text(self) -> str:
            """Text payload for context."""
            ...

    @runtime_checkable
    class Entity(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def unique_id(self) -> str:
            """Unique identifier for entity."""
            ...

    @runtime_checkable
    class ProcessorGood(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def marker(self) -> str:
            """Marker indicating successful processing."""
            ...

    @runtime_checkable
    class ProcessorBad(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def marker(self) -> str:
            """Marker indicating failed processing."""
            ...

    @runtime_checkable
    class ProtocolHandler(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        message_type: type[p.Command]


__all__: list[str] = [
    "ExamplesFlextProtocolsEx10",
]
