"""Entity and DDD patterns for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, protocols, runtime only.
Imports m and adds only DDD-specific data classes.

The DomainEvent is defined in domain_event.py and imported here to
avoid Pydantic forward-reference issues with nested class annotations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    MutableSequence,
)
from typing import Annotated, override

from pydantic import Field

from flext_core import (
    FlextModelsBase as m,
    FlextModelsDomainEvent,
    FlextUtilitiesDomain as u,
    FlextUtilitiesGenerators,
    t,
)


class FlextModelsEntity:
    """Entity and DDD pattern container class.

    This class acts as a namespace container for Entity and related DDD patterns.
    Uses m for all base classes (Tier 0).

    DomainEvent is imported from FlextModelsDomainEvent to break
    the forward-reference cycle that Pydantic cannot resolve.
    """

    class Entity(
        m.TimestampedModel,
        m.IdentifiableMixin,
        m.VersionableMixin,
    ):
        """Entity implementation - base class for domain entities with identity.

        Combines TimestampedModel, IdentifiableMixin, and VersionableMixin to provide:
        - unique_id: Unique identifier (from IdentifiableMixin)
        - created_at/updated_at: Timestamps (from TimestampedModel)
        - version: Optimistic locking (from VersionableMixin)
        - domain_events: Event sourcing support

        ``domain_events`` is an intentionally mutable in-process buffer; the
        field contract is a ``MutableSequence`` because the event-sourcing API
        appends new entries during the entity lifecycle.
        """

        domain_events: Annotated[
            MutableSequence[FlextModelsDomainEvent.Entry],
            Field(
                default_factory=list,
                description="List of uncommitted domain events for event sourcing",
            ),
        ]

        @override
        def __eq__(self, other: object) -> bool:
            """Identity-based equality for entities."""
            if not isinstance(other, m.EnforcedModel):
                return NotImplemented
            return u.compare_entities_by_id(self, other)

        def __hash__(self) -> int:
            """Identity-based hash for entities."""
            return u.hash_entity_by_id(self)

        @override
        def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
            """Post-initialization hook to set updated_at timestamp when absent."""
            if self.updated_at is None:
                self.updated_at = FlextUtilitiesGenerators.generate_datetime_utc()

    class Value(m.ContractModel):
        """Base class for value objects - immutable and compared by value."""

        @override
        def __eq__(self, other: object) -> bool:
            """Compare by value."""
            if not isinstance(other, m.EnforcedModel):
                return NotImplemented
            return u.compare_value_objects_by_value(self, other)

        def __hash__(self) -> int:
            """Hash based on values for use in sets/dicts."""
            return u.hash_value_object_by_value(self)

    class AggregateRoot(Entity):
        """Aggregate-root marker class (DDD consistency boundary)."""


__all__: list[str] = ["FlextModelsEntity"]
