"""Entity and DDD patterns for FLEXT ecosystem.

TIER 1: Usa base.py (Tier 0) + pode usar result, exceptions, runtime.
Importa FlextModelsBase e adiciona apenas classes DDD específicas.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, Self, cast, override

from pydantic import Field

from flext_core._models.base import FlextModelsBase
from flext_core._models.validation import FlextModelsValidation
from flext_core._utilities.domain import FlextDomain
from flext_core._utilities.generators import FlextGenerators
from flext_core.constants import c
from flext_core.exceptions import FlextExceptions
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextModelsEntity:
    """Entity and DDD pattern container class.

    This class acts as a namespace container for Entity and related DDD patterns.
    Uses FlextModelsBase for all base classes (Tier 0).
    """

    # Constants
    EVENT_TUPLE_SIZE: ClassVar[int] = 2

    class DomainEvent(
        FlextModelsBase.ArbitraryTypesModel,
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.TimestampableMixin,
    ):
        """Base class for domain events."""

        message_type: c.Cqrs.EventMessageTypeLiteral = Field(
            default=c.Cqrs.HandlerType.EVENT,
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
        )

        event_type: str
        aggregate_id: str
        data: t.Types.EventDataMapping = Field(
            default_factory=dict,
            description="Event data - maps to EventDataMapping",
        )
        metadata: t.Types.FieldMetadataMapping = Field(
            default_factory=dict,
            description="Event metadata - maps to FieldMetadataMapping",
        )

    class Core(
        FlextModelsBase.TimestampedModel,
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.VersionableMixin,
    ):
        """Core Entity implementation - base class for domain entities with identity.

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
        def logger(self) -> p.StructlogLogger:
            """Get logger instance."""
            return FlextRuntime.get_logger(__name__)

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Post-initialization hook to set updated_at timestamp."""
            if self.updated_at is None:
                self.updated_at = FlextGenerators.generate_datetime_utc()

        @override
        def __eq__(self, other: object) -> bool:
            """Identity-based equality for entities (using uDomain).

            Returns:
                True if entities have the same unique_id, False otherwise.
                NotImplemented if other is not a compatible type.

            """
            # Type narrowing: other must implement HasModelDump protocol for comparison
            if not isinstance(other, p.HasModelDump):
                return NotImplemented
            return FlextDomain.compare_entities_by_id(self, other)

        def __hash__(self) -> int:
            """Identity-based hash for entities (using uDomain).

            Returns:
                Hash value based on entity's unique_id.

            """
            return FlextDomain.hash_entity_by_id(self)

        def _validate_event_input(
            self: Self,
            event_name: str,
            data: t.GeneralValueType | None,
        ) -> r[bool]:
            """Validate event input parameters."""
            if not event_name:
                return r[bool].fail(
                    "Domain event name must be a non-empty string",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            if data is not None and not FlextRuntime.is_dict_like(data):
                return r[bool].fail(
                    "Domain event data must be a dictionary or None",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            if len(self.domain_events) >= c.Validation.MAX_UNCOMMITTED_EVENTS:
                return r[bool].fail(
                    f"Maximum uncommitted events reached: {c.Validation.MAX_UNCOMMITTED_EVENTS}",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            return r[bool].ok(True)

        def _create_and_validate_event(
            self: Self,
            event_name: str,
            data: t.GeneralValueType | None,
        ) -> r[FlextModelsEntity.DomainEvent]:
            """Create and validate domain event."""
            try:
                # Fast fail: data must be dict (None not allowed for domain events)
                # Type narrowing: if dict-like, treat as Mapping; else use empty dict
                if isinstance(data, dict):
                    event_data: t.Types.EventDataMapping = data
                else:
                    event_data = {}

                domain_event = FlextModelsEntity.DomainEvent(
                    event_type=event_name,
                    aggregate_id=self.unique_id,
                    data=event_data,
                )

                # Validate domain event using FlextModelsValidation
                # Convert DomainEvent (BaseModel) to dict for validation
                event_dict: t.GeneralValueType = domain_event.model_dump()
                validation_result = FlextModelsValidation.validate_domain_event(
                    event_dict,
                )
                if validation_result.is_failure:
                    return r[FlextModelsEntity.DomainEvent].fail(
                        f"Domain event validation failed: {validation_result.error}",
                        error_code=c.Errors.VALIDATION_ERROR,
                    )

                return r[FlextModelsEntity.DomainEvent].ok(domain_event)

            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return r[FlextModelsEntity.DomainEvent].fail(
                    f"Failed to create domain event: {e}",
                    error_code=c.Errors.DOMAIN_EVENT_ERROR,
                )

        def _execute_event_handler(
            self: Self,
            event_name: str,
            data: t.GeneralValueType | None,
        ) -> None:
            """Execute event handler if available."""
            # Fast fail: event_type must be str
            event_type: str = ""
            if FlextRuntime.is_dict_like(data):
                # Type narrowing: is_dict_like ensures data is Mapping-like
                event_type_raw = data.get("event_type")
                event_type = "" if event_type_raw is None else str(event_type_raw)
            if event_type:
                handler_method_name = f"_apply_{str(event_type).lower()}"
                if hasattr(self, handler_method_name):
                    try:
                        handler_method = getattr(self, handler_method_name)
                        handler_method(data)
                        self.logger.debug(
                            "Domain event handler executed",
                            event_type=event_name,
                            handler=handler_method_name,
                            aggregate_id=self.unique_id,
                        )
                    except (
                        AttributeError,
                        TypeError,
                        ValueError,
                        RuntimeError,
                        KeyError,
                    ) as e:
                        self.logger.warning(
                            f"Domain event handler {handler_method_name} failed for event {event_name}: {e!s}",
                            handler=handler_method_name,
                            event=event_name,
                            error=str(e),
                        )

        def add_domain_event(
            self: Self,
            event_name: str,
            data: t.Types.EventDataMapping | None,
        ) -> r[bool]:
            """Add a domain event to be dispatched with enhanced validation."""
            # Validate input
            validation_result = self._validate_event_input(event_name, data)
            if validation_result.is_failure:
                return validation_result

            # Create and validate event
            event_result = self._create_and_validate_event(event_name, data)
            if event_result.is_failure:
                base_msg = "Event creation failed"
                error_msg = (
                    f"{base_msg}: {event_result.error}"
                    if event_result.error
                    else f"{base_msg} (domain event creation failed)"
                )
                return r[bool].fail(
                    error_msg,
                    error_code=c.Errors.DOMAIN_EVENT_ERROR,
                )

            domain_event = event_result.unwrap()

            # Add event and track
            self.domain_events.append(domain_event)

            FlextRuntime.Integration.track_domain_event(
                event_name=event_name,
                aggregate_id=self.unique_id,
                event_data=data,
            )

            self.logger.debug(
                "Domain event added",
                event_type=event_name,
                aggregate_id=self.unique_id,
                aggregate_type=self.__class__.__name__,
                event_id=domain_event.unique_id,
                data_keys=list(data.keys()) if data else [],
            )

            # Execute handler if available
            self._execute_event_handler(event_name, data)

            # Update entity state
            self.increment_version()
            self.update_timestamp()

            return r[bool].ok(True)

        @property
        def uncommitted_events(
            self: Self,
        ) -> list[FlextModelsEntity.DomainEvent]:
            """Get uncommitted domain events without clearing them."""
            return list(self.domain_events)

        def mark_events_as_committed(
            self: Self,
        ) -> r[list[FlextModelsEntity.DomainEvent]]:
            """Mark all domain events as committed and return them."""
            try:
                events = self.uncommitted_events

                if events:
                    # NOTE: Cannot use u.map() here due to circular import
                    # (utilities.py -> _utilities/context.py -> _models/context.py -> entity.py)
                    event_types = [e.event_type for e in events]
                    self.logger.info(
                        "Domain events committed",
                        aggregate_id=self.unique_id,
                        aggregate_type=self.__class__.__name__,
                        event_count=len(events),
                        event_types=event_types,
                    )

                self.domain_events.clear()

                return r[list[FlextModelsEntity.DomainEvent]].ok(events)

            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return r[list[FlextModelsEntity.DomainEvent]].fail(
                    f"Failed to commit domain events: {e}",
                    error_code=c.Errors.DOMAIN_EVENT_ERROR,
                )

        def clear_domain_events(
            self: Self,
        ) -> list[FlextModelsEntity.DomainEvent]:
            """Clear and return domain events."""
            events = list(self.domain_events)

            if events:
                self.logger.debug(
                    "Domain events cleared",
                    aggregate_id=self.unique_id,
                    aggregate_type=self.__class__.__name__,
                    event_count=len(events),
                    # NOTE: Cannot use u.map() here due to circular import
                    event_types=[e.event_type for e in events],
                )

            self.domain_events.clear()
            return events

        def _validate_bulk_events_input(
            self: Self,
            events: t.GeneralValueType | None,
        ) -> r[int]:
            """Validate bulk events input and return total count.

            Returns:
                r with total event count after add, or error

            """
            if not events:
                return r[int].ok(0)  # Signal empty - caller handles

            if not isinstance(events, (list, tuple)):
                return r[int].fail(
                    "Events must be a list of tuples",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            total_after_add = len(self.domain_events) + len(events)
            if total_after_add > c.Validation.MAX_UNCOMMITTED_EVENTS:
                return r[int].fail(
                    f"Bulk add would exceed max events: {total_after_add} > {c.Validation.MAX_UNCOMMITTED_EVENTS}",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            return r[int].ok(total_after_add)

        @staticmethod
        def validate_and_collect_events(
            events: Sequence[tuple[str, t.Types.EventDataMapping | None]],
        ) -> r[list[tuple[str, t.Types.EventDataMapping]]]:
            """Validate and collect events for bulk add.

            Returns:
                r with validated events list or error

            """
            validated_events: list[tuple[str, t.Types.EventDataMapping]] = []

            if not isinstance(events, (list, tuple)):
                return r[list[tuple[str, t.Types.EventDataMapping]]].fail(
                    "Events must be a list or tuple",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            for i, event_item in enumerate(events):
                event_tuple_size = 2  # Event tuple: (name, data)
                if len(event_item) != event_tuple_size:
                    return r[list[tuple[str, t.Types.EventDataMapping]]].fail(
                        f"Event {i}: must have exactly 2 elements (name, data)",
                        error_code=c.Errors.VALIDATION_ERROR,
                    )

                event_name, data = event_item

                if not event_name:
                    return r[list[tuple[str, t.Types.EventDataMapping]]].fail(
                        f"Event {i}: name must be non-empty string",
                        error_code=c.Errors.VALIDATION_ERROR,
                    )
                if data is not None and not FlextRuntime.is_dict_like(data):
                    return r[list[tuple[str, t.Types.EventDataMapping]]].fail(
                        f"Event {i}: data must be dict[str, t.GeneralValueType] or None",
                        error_code=c.Errors.VALIDATION_ERROR,
                    )
                # Fast fail: data must be dict (None not allowed)
                # Type narrowing: if dict-like, treat as Mapping; else use empty dict
                if isinstance(data, dict):
                    event_data: t.Types.EventDataMapping = data
                else:
                    event_data = {}
                validated_events.append((event_name, event_data))

            return r[list[tuple[str, t.Types.EventDataMapping]]].ok(
                validated_events,
            )

        def _add_validated_events_bulk(
            self: Self,
            validated_events: list[tuple[str, t.Types.EventDataMapping]],
        ) -> r[bool]:
            """Add validated events in bulk.

            Args:
                validated_events: Already validated list of (event_name, data) tuples

            Returns:
                r indicating success or error

            """
            try:
                for event_name, data in validated_events:
                    domain_event = FlextModelsEntity.DomainEvent(
                        event_type=event_name,
                        aggregate_id=self.unique_id,
                        data=data,
                    )
                    self.domain_events.append(domain_event)
                    self.increment_version()

                self.update_timestamp()

                self.logger.info(
                    "Bulk domain events added",
                    aggregate_id=self.unique_id,
                    aggregate_type=self.__class__.__name__,
                    event_count=len(validated_events),
                    # NOTE: Cannot use u.map() here due to circular import
                    event_types=[name for name, _ in validated_events],
                )

                return r[bool].ok(True)

            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return r[bool].fail(
                    f"Failed to add bulk domain events: {e}",
                    error_code=c.Errors.DOMAIN_EVENT_ERROR,
                )

        def add_domain_events_bulk(
            self: Self,
            events: Sequence[tuple[str, t.Types.EventDataMapping | None]],
        ) -> r[bool]:
            """Add multiple domain events in bulk with validation."""
            # Validate input
            count_result = self._validate_bulk_events_input(events)
            if count_result.is_failure:
                base_msg = "Input validation failed"
                error_msg = (
                    f"{base_msg}: {count_result.error}"
                    if count_result.error
                    else f"{base_msg} (count validation failed)"
                )
                return r[bool].fail(
                    error_msg,
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            # Handle empty case
            if count_result.unwrap() == 0:
                return r[bool].ok(True)

            # Validate and collect events
            validated_result = FlextModelsEntity.Core.validate_and_collect_events(
                events,
            )
            if validated_result.is_failure:
                base_msg = "Event validation failed"
                error_msg = (
                    f"{base_msg}: {validated_result.error}"
                    if validated_result.error
                    else f"{base_msg} (event validation rule failed)"
                )
                return r[bool].fail(
                    error_msg,
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            # Add validated events
            return self._add_validated_events_bulk(validated_result.unwrap())

        def validate_consistency(self: Self) -> r[bool]:
            """Validate entity consistency using centralized validation."""
            # Cast self to GeneralValueType for validation
            entity_typed: t.GeneralValueType = cast(
                "t.GeneralValueType",
                self,
            )
            entity_result = FlextModelsValidation.validate_entity_relationships(
                entity_typed,
            )
            if entity_result.is_failure:
                return r[bool].fail(
                    f"Entity validation failed: {entity_result.error}",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

            for event in self.domain_events:
                # Convert DomainEvent (BaseModel) to dict for validation
                event_dict: t.GeneralValueType = event.model_dump()
                event_result = FlextModelsValidation.validate_domain_event(event_dict)
                if event_result.is_failure:
                    return r[bool].fail(
                        f"Domain event validation failed: {event_result.error}",
                        error_code=c.Errors.VALIDATION_ERROR,
                    )

            return r[bool].ok(True)

    class Value(FlextModelsBase.FrozenStrictModel):
        """Base class for value objects - immutable and compared by value."""

        @override
        def __eq__(self: Self, other: object) -> bool:
            """Compare by value (using uDomain)."""
            if not isinstance(other, p.HasModelDump):
                return NotImplemented
            return FlextDomain.compare_value_objects_by_value(self, other)

        def __hash__(self) -> int:
            """Hash based on values for use in sets/dicts (using uDomain)."""
            return FlextDomain.hash_value_object_by_value(self)

    # Alias for backward compatibility - Entity is the same as Core
    Entity = Core

    class AggregateRoot(Core):
        """Base class for aggregate roots - consistency boundaries."""

        _invariants: ClassVar[list[Callable[[], bool]]] = []

        def check_invariants(self) -> None:
            """Check all business invariants."""
            for invariant in self._invariants:
                if not invariant():
                    msg = f"Invariant violated: {invariant.__name__}"
                    raise e.ValidationError(
                        msg,
                        error_code=c.Errors.VALIDATION_ERROR,
                    )

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Post-init hook to check invariants."""
            super().model_post_init(__context)
            self.check_invariants()


__all__ = ["FlextModelsEntity"]

# Rebuild models to resolve forward references within the module
# This is required because Core references DomainEvent which is defined in the same class
_ = FlextModelsEntity.Core.model_rebuild()
_ = FlextModelsEntity.AggregateRoot.model_rebuild()
