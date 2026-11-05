# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/domain_event
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import flext_core
from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsContainers


class FlextProtocolsDomainEvent:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class DomainEvent(FlextProtocolsBase.TimestampedModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[flext_core.c.HandlerType.EVENT]:
            """Immutable event discriminator for union routing."""
            ...

        event_type: flext_core.t.NonEmptyStr
        aggregate_id: flext_core.t.NonEmptyStr

        @property
        def data(self) -> FlextProtocolsContainers.ConfigMap:
            """Event data container."""
            ...


__all__: list[str] = [
    "FlextProtocolsDomainEvent",
]
