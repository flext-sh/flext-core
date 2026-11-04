# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex12
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from examples import c, t
    from flext_core import p


class ExamplesFlextProtocolsEx12:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class RegistryLookupOptions(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def registry(self) -> p.Registry:
            """Registry instance for plugin lookup."""
            ...

        category: str
        plugin_name: str

    @runtime_checkable
    class CommandA(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: str
        command_id: t.NonEmptyStr
        issuer_id: t.NonEmptyStr | None
        query_type: str
        event_type: str
        value: str

    @runtime_checkable
    class CommandB(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: str
        command_id: t.NonEmptyStr
        issuer_id: t.NonEmptyStr | None
        query_type: str
        event_type: str
        amount: int


__all__: list[str] = [
    "ExamplesFlextProtocolsEx12",
]
