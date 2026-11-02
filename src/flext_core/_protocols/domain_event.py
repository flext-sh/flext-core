# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/domain_event
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import c, t

if TYPE_CHECKING:
    from flext_core.protocols import p


class FlextProtocolsDomainEvent:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class DomainEvent(Protocol):
        """Structural data contract projected from the owning model."""

        message_type: Literal[c.HandlerType.EVENT]
        event_type: t.NonEmptyStr
        aggregate_id: t.NonEmptyStr
        data: p.ConfigMap


__all__: list[str] = [
    "FlextProtocolsDomainEvent",
]
