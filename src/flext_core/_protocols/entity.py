# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/entity
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core import t
    from flext_core._protocols import FlextProtocolsDomainEvent


class FlextProtocolsEntity:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Entity(FlextProtocolsBase.TimestampedModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def domain_events(self) -> Sequence[FlextProtocolsDomainEvent.DomainEvent]:
            """List of uncommitted domain events for event sourcing."""
            ...

    @runtime_checkable
    class Value(FlextProtocolsBase.FrozenValueModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class AggregateRoot(Entity, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class TransformOptions(Value, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def normalize(self) -> bool:
            """Normalize values to metadata form."""
            ...

        @property
        def strip_none(self) -> bool:
            """Strip ``None`` values."""
            ...

        @property
        def strip_empty(self) -> bool:
            """Strip empty values."""
            ...

        @property
        def map_keys(self) -> Mapping[str, str] | None:
            """Optional old-key to new-key mapping."""
            ...

    @runtime_checkable
    class ExtractResolvePathPartContext(Value, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def path_context(self) -> str:
            """Resolved parent path context."""
            ...

        @property
        def default(self) -> t.JsonPayload | None:
            """Default fallback payload."""
            ...

        @property
        def required(self) -> bool:
            """Whether missing values are fatal."""
            ...


__all__: list[str] = [
    "FlextProtocolsEntity",
]
