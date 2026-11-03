# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex12
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from examples import c, t
from flext_core import FlextProtocols

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx12:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class RegistryLookupOptions(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def registry(self) -> p.Registry:
            """Registry instance for plugin lookup."""
            ...

        @property
        def category(self) -> str:
            """Plugin category for registry lookup."""
            ...

        @property
        def plugin_name(self) -> str:
            """Plugin name for registry lookup."""
            ...

    @runtime_checkable
    class CommandA(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        @property
        def command_type(self) -> str:
            """Command type identifier for command A."""
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
            """Query type placeholder for command A."""
            ...

        @property
        def event_type(self) -> str:
            """Event type for command A."""
            ...

        @property
        def value(self) -> str:
            """Command A payload."""
            ...

    @runtime_checkable
    class CommandB(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        @property
        def command_type(self) -> str:
            """Command type identifier for command B."""
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
            """Query type placeholder for command B."""
            ...

        @property
        def event_type(self) -> str:
            """Event type for command B."""
            ...

        @property
        def amount(self) -> int:
            """Command B payload."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx12",
]
