"""Entity and DDD patterns for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, protocols, runtime only.
Imports FlextModelsBase and adds only DDD-specific data classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar, Self, override

from pydantic import Field, field_validator
from structlog.typing import BindableLogger

from flext_core._models.base import FlextModelsBase
from flext_core.constants import c
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


def _to_config_map(data: t.ConfigMap | None) -> _ComparableConfigMap:
    if not data:
        return _ComparableConfigMap(root={})
    return _ComparableConfigMap(
        root={
            str(key): FlextRuntime.normalize_to_metadata_value(value)
            for key, value in data.items()
        }
    )


class _ComparableConfigMap(t.ConfigMap):
    def __eq__(self, other: t.GuardInputValue) -> bool:
        if other.__class__ is dict or (Mapping in other.__class__.__mro__):
            return self.root == dict(other.items())
        return super().__eq__(other)

    __hash__ = t.ConfigMap.__hash__


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
        data: _ComparableConfigMap = Field(
            default_factory=_ComparableConfigMap,
            description="Event data container",
        )
        metadata: t.ConfigMap = Field(
            default_factory=t.ConfigMap,
            description="Event metadata container",
        )

        @field_validator("data", mode="before")
        @classmethod
        def normalize_event_data(
            cls,
            value: t.GuardInputValue,
        ) -> _ComparableConfigMap:
            if _ComparableConfigMap in value.__class__.__mro__:
                return value
            if t.ConfigMap in value.__class__.__mro__:
                return _ComparableConfigMap(root=dict(value.items()))
            if value.__class__ is dict or (Mapping in value.__class__.__mro__):
                normalized: t.ConfigMap = t.ConfigMap(
                    root={
                        str(k): FlextRuntime.normalize_to_metadata_value(v)
                        for k, v in value.items()
                    }
                )
                return _ComparableConfigMap(root=normalized.root)
            return _ComparableConfigMap(root={})

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
        def logger(self) -> BindableLogger:
            """Get logger instance."""
            return FlextRuntime.get_logger(__name__)

        @override
        def model_post_init(self, __context: t.GuardInputValue, /) -> None:
            """Post-initialization hook to set updated_at timestamp."""
            if self.updated_at is None:
                self.updated_at = FlextRuntime.generate_datetime_utc()

        @override
        def __eq__(self, other: t.GuardInputValue) -> bool:
            """Identity-based equality for entities."""
            if not FlextRuntime.is_base_model(other):
                return NotImplemented
            # Type narrowed to BaseModel via TypeGuard (part of PayloadValue)
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
            data: t.ConfigMap | None = None,
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

            data_map = _to_config_map(data)
            event = FlextModelsEntity.DomainEvent(
                event_type=event_type,
                aggregate_id=self.unique_id,
                data=data_map,
            )
            self.domain_events.append(event)

            # Call event handler if defined
            # Use event_type from data if present, otherwise use argument
            handler_event_type = data_map.get("event_type", event_type)
            if handler_event_type.__class__ is str:
                handler_name = f"_apply_{handler_event_type}"
                handler = getattr(self, handler_name, None)
                if handler is not None and callable(handler):
                    # Swallow handler exceptions - event is still added
                    with contextlib.suppress(Exception):
                        handler(data_map.root)

            return r[FlextModelsEntity.DomainEvent].ok(event)

        def add_domain_events_bulk(
            self: Self,
            events: Sequence[tuple[str, t.ConfigMap | None]],
        ) -> r[list[FlextModelsEntity.DomainEvent]]:
            """Add multiple domain events in bulk.

            Args:
                events: Sequence of (event_type, data) tuples

            Returns:
                FlextResult with list of created DomainEvents or error

            """
            # Validate input is a valid sequence (list or tuple)
            if events.__class__ not in (list, tuple):
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
                    data=_to_config_map(data),
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
        def __eq__(self: Self, other: t.GuardInputValue) -> bool:
            """Compare by value."""
            if not FlextRuntime.is_base_model(other):
                return NotImplemented
            # Type narrowed to BaseModel via TypeGuard (part of PayloadValue)
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
        def model_post_init(self, __context: t.GuardInputValue, /) -> None:
            """Post-init hook to check invariants."""
            super().model_post_init(__context)
            self.check_invariants()


# Resolve forward references created by `from __future__ import annotations`.
# `Entry.domain_events` references `FlextModelsEntity.DomainEvent` which is
# unavailable during class body execution. Now that the outer class is fully
# defined, Pydantic can resolve the ForwardRef from module globals.
FlextModelsEntity.Entry.model_rebuild()
FlextModelsEntity.AggregateRoot.model_rebuild()

__all__ = ["FlextModelsEntity"]
