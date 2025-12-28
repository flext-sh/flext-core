"""Entity and DDD patterns for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, protocols, runtime only.
Imports FlextModelsBase and adds only DDD-specific data classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import ClassVar, Self, override

from pydantic import Field

from flext_core._models.base import FlextModelsBase
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextModelsEntity:
    """Entity and DDD pattern container class.

    This class acts as a namespace container for Entity and related DDD patterns.
    Uses FlextModelsBase for all base classes (Tier 0).
    """

    class DomainEvent(
        FlextModelsBase.ArbitraryTypesModel,
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.TimestampableMixin,
    ):
        """Base class for domain events."""

        message_type: c.Cqrs.EventMessageTypeLiteral = Field(
            default="event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
        )

        event_type: str
        aggregate_id: str
        # Use t.PydanticConfigDict - Pydantic-safe type that avoids schema recursion
        data: t.PydanticConfigDict = Field(
            default_factory=dict,
            description="Event data - Pydantic-safe config dict",
        )
        metadata: t.PydanticConfigDict = Field(
            default_factory=dict,
            description="Event metadata - Pydantic-safe config dict",
        )

    class Entry(
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

        domain_events: list[FlextModelsEntity.DomainEvent] = Field(
            default_factory=list,
            description="List of uncommitted domain events for event sourcing",
        )

        @property
        def entity_id(self) -> str:
            """Entity identifier property - alias for unique_id."""
            return self.unique_id

        @property
        def logger(self) -> p.Log.StructlogLogger:
            """Get logger instance."""
            return FlextRuntime.get_logger(__name__)

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Post-initialization hook to set updated_at timestamp."""
            if self.updated_at is None:
                self.updated_at = FlextRuntime.generate_datetime_utc()

        @override
        def __eq__(self, other: object) -> bool:
            """Identity-based equality for entities."""
            if not FlextRuntime.is_base_model(other):
                return NotImplemented
            # Type narrowed to BaseModel via TypeGuard (part of GeneralValueType)
            return FlextRuntime.compare_entities_by_id(self, other)

        def __hash__(self) -> int:
            """Identity-based hash for entities."""
            return FlextRuntime.hash_entity_by_id(self)

        @property
        def uncommitted_events(self: Self) -> list[FlextModelsEntity.DomainEvent]:
            """Get uncommitted domain events without clearing them."""
            return list(self.domain_events)

        def clear_domain_events(self: Self) -> list[FlextModelsEntity.DomainEvent]:
            """Clear and return domain events."""
            events = list(self.domain_events)
            self.domain_events.clear()
            return events

        def add_domain_event(
            self: Self,
            event_type: str,
            data: t.EventDataMapping | None = None,
        ) -> r[FlextModelsEntity.DomainEvent]:
            """Add a domain event to this entity.

            Args:
                event_type: Type/name of the event
                data: Event data mapping

            Returns:
                FlextResult with the created DomainEvent or error

            """
            if not event_type:
                return r[FlextModelsEntity.DomainEvent].fail(
                    "Domain event name must be a non-empty string",
                )

            if len(self.domain_events) >= c.Validation.MAX_UNCOMMITTED_EVENTS:
                return r[FlextModelsEntity.DomainEvent].fail(
                    f"Cannot add event: would exceed max events limit of "
                    f"{c.Validation.MAX_UNCOMMITTED_EVENTS}",
                )

            data_dict = FlextUtilitiesModel.normalize_to_pydantic_dict(data)
            event = FlextModelsEntity.DomainEvent(
                event_type=event_type,
                aggregate_id=self.unique_id,
                data=data_dict,
            )
            self.domain_events.append(event)

            # Call event handler if defined
            # Use event_type from data if present, otherwise use argument
            handler_event_type = data_dict.get("event_type", event_type)
            if isinstance(handler_event_type, str):
                handler_name = f"_apply_{handler_event_type}"
                handler = getattr(self, handler_name, None)
                if handler is not None and callable(handler):
                    # Swallow handler exceptions - event is still added
                    with contextlib.suppress(Exception):
                        handler(data_dict)

            return r[FlextModelsEntity.DomainEvent].ok(event)

        def add_domain_events_bulk(
            self: Self,
            events: Sequence[tuple[str, t.EventDataMapping | None]],
        ) -> r[list[FlextModelsEntity.DomainEvent]]:
            """Add multiple domain events in bulk.

            Args:
                events: Sequence of (event_type, data) tuples

            Returns:
                FlextResult with list of created DomainEvents or error

            """
            # Validate input is a valid sequence (list or tuple)
            if not isinstance(events, (list, tuple)):
                return r[list[FlextModelsEntity.DomainEvent]].fail(
                    "Events must be a list or tuple",
                )

            # Convert to list for iteration (ensures proper type)
            event_items = list(events)

            # Check if adding all events would exceed limit
            total_after = len(self.domain_events) + len(event_items)
            if total_after > c.Validation.MAX_UNCOMMITTED_EVENTS:
                return r[list[FlextModelsEntity.DomainEvent]].fail(
                    f"Cannot add {len(events)} events: would exceed max events "
                    f"limit of {c.Validation.MAX_UNCOMMITTED_EVENTS}",
                )

            # Validate all event names first
            for event_type, _ in event_items:
                if not event_type:
                    return r[list[FlextModelsEntity.DomainEvent]].fail(
                        "Event name must be non-empty string",
                    )

            # Add all events
            created_events: list[FlextModelsEntity.DomainEvent] = []
            for event_type, data in event_items:
                event = FlextModelsEntity.DomainEvent(
                    event_type=event_type,
                    aggregate_id=self.unique_id,
                    data=FlextUtilitiesModel.normalize_to_pydantic_dict(data),
                )
                self.domain_events.append(event)
                created_events.append(event)

            return r[list[FlextModelsEntity.DomainEvent]].ok(created_events)

        def mark_events_as_committed(
            self: Self,
        ) -> r[list[FlextModelsEntity.DomainEvent]]:
            """Mark all uncommitted events as committed and return them.

            Returns:
                FlextResult with list of committed events

            """
            events = list(self.domain_events)
            self.domain_events.clear()
            return r[list[FlextModelsEntity.DomainEvent]].ok(events)

    class Value(FlextModelsBase.FrozenStrictModel):
        """Base class for value objects - immutable and compared by value."""

        @override
        def __eq__(self: Self, other: object) -> bool:
            """Compare by value."""
            if not FlextRuntime.is_base_model(other):
                return NotImplemented
            # Type narrowed to BaseModel via TypeGuard (part of GeneralValueType)
            return FlextRuntime.compare_value_objects_by_value(self, other)

        def __hash__(self) -> int:
            """Hash based on values for use in sets/dicts."""
            return FlextRuntime.hash_value_object_by_value(self)

    class AggregateRoot(Entry):
        """Base class for aggregate roots - consistency boundaries."""

        _invariants: ClassVar[list[Callable[[], bool]]] = []

        def check_invariants(self) -> None:
            """Check all business invariants."""
            for invariant in self._invariants:
                if not invariant():
                    msg = f"Invariant violated: {invariant.__name__}"
                    raise ValueError(msg)

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Post-init hook to check invariants."""
            super().model_post_init(__context)
            self.check_invariants()


# DISABLED: model_rebuild() causes circular import issues with GeneralValueType
# Models use arbitrary_types_allowed=True and work without rebuild
# FlextModelsEntity.DomainEvent.model_rebuild()
# FlextModelsEntity.Entry.model_rebuild()
# FlextModelsEntity.Value.model_rebuild()
# FlextModelsEntity.AggregateRoot.model_rebuild()

__all__ = ["FlextModelsEntity"]
