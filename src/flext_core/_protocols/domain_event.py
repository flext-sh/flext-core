# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/domain_event
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._constants import FlextConstantsCqrs
    from flext_core._protocols import FlextProtocolsContainers
    from flext_core._typings import FlextTypesAnnotateds


class FlextProtocolsDomainEvent:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class DomainEvent(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[FlextConstantsCqrs.HandlerType.EVENT]:
            """Immutable event discriminator for union routing."""
            ...

        @property
        def event_type(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Domain event type identifier for subscriber routing."""
            ...

        @property
        def aggregate_id(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Identifier of the aggregate root that produced this event."""
            ...

        @property
        def data(self) -> FlextProtocolsContainers.ConfigMap:
            """Event data container."""
            ...


__all__: list[str] = [
    "FlextProtocolsDomainEvent",
]
