"""Entity and DDD patterns for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, protocols, runtime only.
Imports FlextModelsBase and adds only DDD-specific data classes.

The DomainEvent is defined in domain_event.py and imported here to
avoid Pydantic forward-reference issues with nested class annotations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from typing import Annotated, ClassVar, Self, override

from pydantic import BaseModel, Field, computed_field, model_validator

from flext_core import FlextUtilitiesDomain, FlextUtilitiesGenerators, c, p, r, t
from flext_core._models.base import FlextModelsBase
from flext_core._models.domain_event import FlextModelsDomainEvent
from flext_core.loggings import FlextLogger


class FlextModelsEntity:
    """Entity and DDD pattern container class.

    This class acts as a namespace container for Entity and related DDD patterns.
    Uses FlextModelsBase for all base classes (Tier 0).

    DomainEvent is imported from FlextModelsDomainEvent to break
    the forward-reference cycle that Pydantic cannot resolve.
    """

    class Entity(
        FlextModelsBase.TimestampedModel,
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.VersionableMixin,
    ):
        """Entity implementation - base class for domain entities with identity.

        Combines TimestampedModel, IdentifiableMixin, and VersionableMixin to provide:
        - unique_id: Unique identifier (from IdentifiableMixin)
        - entity_id: Entity identifier property (alias for unique_id)
        - created_at/updated_at: Timestamps (from TimestampedModel)
        - version: Optimistic locking (from VersionableMixin)
        - domain_events: Event sourcing support
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
            if not isinstance(other, BaseModel):
                return NotImplemented
            return FlextUtilitiesDomain.compare_entities_by_id(self, other)

        def __hash__(self) -> int:
            """Identity-based hash for entities."""
            return FlextUtilitiesDomain.hash_entity_by_id(self)

        @computed_field
        @property
        def entity_id(self) -> str:
            """Entity identifier property - alias for unique_id."""
            return self.unique_id

        @property
        def logger(self) -> p.Logger:
            """Get the shared structlog logger for entity instances."""
            return FlextLogger.get_logger(__name__)

        @computed_field
        @property
        def uncommitted_events(self: Self) -> Sequence[FlextModelsDomainEvent.Entry]:
            """Get uncommitted domain events without clearing them."""
            return list(self.domain_events)

        def add_domain_event(
            self: Self,
            event_type: str,
            data: t.ConfigMap | Mapping[str, t.MetadataOrValue | None] | None = None,
        ) -> r[FlextModelsDomainEvent.Entry]:
            """Add a domain event to this entity.

            Args:
                event_type: Type/name of the event
                data: Event data mapping

            Returns:
                r with the created DomainEvent or error

            """
            if not event_type:
                return r[FlextModelsDomainEvent.Entry].fail(
                    "Domain event name must be a non-empty string",
                )
            if len(self.domain_events) >= c.HTTP_STATUS_MIN:
                return r[FlextModelsDomainEvent.Entry].fail(
                    f"Cannot add event: would exceed max events limit of {c.HTTP_STATUS_MIN}",
                )
            data_map = FlextModelsDomainEvent.ComparableConfigMap(
                root=dict(FlextUtilitiesDomain.normalize_domain_event_data(data)),
            )
            event = FlextModelsDomainEvent.Entry(
                event_type=event_type,
                aggregate_id=self.unique_id,
                data=data_map,
            )
            self.domain_events.append(event)
            handler_event_type_raw = data_map.get("event_type", event_type)
            handler_event_type = (
                handler_event_type_raw
                if isinstance(handler_event_type_raw, str) and handler_event_type_raw
                else event_type
            )
            handler_name = f"_apply_{handler_event_type}"
            handler = getattr(self, handler_name, None)
            if handler is not None and callable(handler):
                with contextlib.suppress(Exception):
                    _ = handler(data_map.root)
            return r[FlextModelsDomainEvent.Entry].ok(event)

        def add_domain_events_bulk(
            self: Self,
            events: Sequence[tuple[str, t.ConfigMap | None]],
        ) -> r[Sequence[FlextModelsDomainEvent.Entry]]:
            """Add multiple domain events in bulk.

            Args:
                events: Sequence of (event_type, data) tuples

            Returns:
                r with list of created DomainEvents or error

            """
            if not isinstance(events, (list, tuple)):
                return r[Sequence[FlextModelsDomainEvent.Entry]].fail(
                    "Events must be a list or tuple",
                )
            event_items = list(events)
            total_after = len(self.domain_events) + len(event_items)
            if total_after > c.HTTP_STATUS_MIN:
                return r[Sequence[FlextModelsDomainEvent.Entry]].fail(
                    f"Cannot add {len(events)} events: would exceed max events limit of {c.HTTP_STATUS_MIN}",
                )
            for event_type, _ in event_items:
                if not event_type:
                    return r[Sequence[FlextModelsDomainEvent.Entry]].fail(
                        "Event name must be non-empty string",
                    )
            created_events: MutableSequence[FlextModelsDomainEvent.Entry] = []
            for event_type, data in event_items:
                event = FlextModelsDomainEvent.Entry(
                    event_type=event_type,
                    aggregate_id=self.unique_id,
                    data=FlextModelsDomainEvent.ComparableConfigMap(
                        root=dict(
                            FlextUtilitiesDomain.normalize_domain_event_data(data)
                        ),
                    ),
                )
                self.domain_events.append(event)
                created_events.append(event)
            return r[Sequence[FlextModelsDomainEvent.Entry]].ok(created_events)

        def clear_domain_events(self: Self) -> Sequence[FlextModelsDomainEvent.Entry]:
            """Clear and return domain events."""
            events = list(self.domain_events)
            self.domain_events.clear()
            return events

        def mark_events_as_committed(
            self: Self,
        ) -> r[Sequence[FlextModelsDomainEvent.Entry]]:
            """Mark all uncommitted events as committed and return them.

            Returns:
                r with list of committed events

            """
            events = list(self.domain_events)
            self.domain_events.clear()
            return r[Sequence[FlextModelsDomainEvent.Entry]].ok(events)

        @override
        def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
            """Post-initialization hook to set updated_at timestamp."""
            self.updated_at = FlextUtilitiesGenerators.generate_datetime_utc()

    class Value(FlextModelsBase.ContractModel):
        """Base class for value objects - immutable and compared by value."""

        @override
        def __eq__(self, other: object) -> bool:
            """Compare by value."""
            if not isinstance(other, BaseModel):
                return NotImplemented
            return FlextUtilitiesDomain.compare_value_objects_by_value(self, other)

        def __hash__(self) -> int:
            """Hash based on values for use in sets/dicts."""
            return FlextUtilitiesDomain.hash_value_object_by_value(self)

    class AggregateRoot(Entity):
        """Base class for aggregate roots - consistency boundaries."""

        _invariants: ClassVar[Sequence[Callable[[], bool]]] = []

        def check_invariants(self) -> None:
            """Check all business invariants."""
            for invariant in self._invariants:
                if not invariant():
                    msg = f"Invariant violated: {invariant.__name__}"
                    raise ValueError(msg)

        @model_validator(mode="after")
        def validate_aggregate_consistency(self) -> Self:
            invariant_result = r[None].ok(None).map(lambda _: self.check_invariants())
            if invariant_result.is_failure:
                error_msg = invariant_result.error or "invariant check failed"
                msg = f"Aggregate invariant violation: {error_msg}"
                raise ValueError(msg)
            if len(self.domain_events) > c.HTTP_STATUS_MIN:
                max_events = c.HTTP_STATUS_MIN
                event_count = len(self.domain_events)
                msg = f"Too many uncommitted domain events: {event_count} (max: {max_events})"
                raise ValueError(msg)
            return self


__all__ = ["FlextModelsEntity"]
