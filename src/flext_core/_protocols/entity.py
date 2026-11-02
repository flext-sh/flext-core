# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/entity
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core import p


class FlextProtocolsEntity:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Entity(Protocol):
        """Structural data contract projected from the owning model."""

        domain_events: MutableSequence[p.DomainEvent]

    @runtime_checkable
    class Value(Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class AggregateRoot(Protocol):
        """Structural data contract projected from the owning model."""

        domain_events: MutableSequence[p.DomainEvent]

    @runtime_checkable
    class TransformOptions(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def normalize(self) -> bool: ...
        @property
        def strip_none(self) -> bool: ...
        @property
        def strip_empty(self) -> bool: ...
        @property
        def map_keys(self) -> Mapping[str, str] | None: ...

    @runtime_checkable
    class ExtractResolvePathPartContext(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def path_context(self) -> str: ...
        @property
        def default(self) -> t.JsonPayload | None: ...
        @property
        def required(self) -> bool: ...


__all__: list[str] = [
    "FlextProtocolsEntity",
]
