"""Entity and DDD patterns for FLEXT ecosystem.

TIER 1: Usa base.py (Tier 0) + pode usar result, exceptions, runtime.
Importa FlextModelsBase e adiciona apenas classes DDD específicas.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar, Self, TypeAlias, cast, override

from pydantic import Field

from flext_core._models.base import FlextModelsBase
from flext_core._models.validation import FlextModelsValidation
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime, StructlogLogger
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

# Constants for event validation
_EVENT_TUPLE_LENGTH = 2


def _is_dict_like(value: object) -> bool:
    """Check if value is dict-like (inline to avoid extra import)."""
    return isinstance(value, Mapping)


class FlextModelsEntity:
    """Entity and DDD pattern container class.

    This class acts as a namespace container for Entity and related DDD patterns.
    Uses FlextModelsBase for all base classes (Tier 0).
    """

    # Type aliases for base classes (mypy-compatible)
    ArbitraryTypesModel: TypeAlias = FlextModelsBase.ArbitraryTypesModel
    FrozenStrictModel: TypeAlias = FlextModelsBase.FrozenStrictModel
    IdentifiableMixin: TypeAlias = FlextModelsBase.IdentifiableMixin
    TimestampableMixin: TypeAlias = FlextModelsBase.TimestampableMixin
    VersionableMixin: TypeAlias = FlextModelsBase.VersionableMixin
    TimestampedModel: TypeAlias = FlextModelsBase.TimestampedModel

    class DomainEvent(
        FlextModelsBase.ArbitraryTypesModel,
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.TimestampableMixin,
    ):
        """Base class for domain events."""

        message_type: FlextConstants.Cqrs.EventMessageTypeLiteral = Field(
            default=FlextConstants.Cqrs.HandlerType.EVENT,
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
        )

        event_type: str
        aggregate_id: str
        data: FlextTypes.Types.EventDataMapping = Field(
            default_factory=dict,
            description="Event data - maps to EventDataMapping",
        )
        metadata: FlextTypes.Types.FieldMetadataMapping = Field(
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
        def logger(self) -> StructlogLogger:
            """Get logger instance."""
            return FlextRuntime.get_logger(__name__)

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Post-initialization hook to set updated_at timestamp."""
            if self.updated_at is None:
                self.updated_at = FlextUtilities.Generators.generate_datetime_utc()

        @override
        def __eq__(self, other: object) -> bool:
            """Identity-based equality for entities (using FlextUtilities.Domain)."""
            return FlextUtilities.Domain.compare_entities_by_id(self, other)

        def __hash__(self) -> int:
            """Identity-based hash for entities (using FlextUtilities.Domain)."""
            return FlextUtilities.Domain.hash_entity_by_id(self)

        def _validate_event_input(
            self: Self,
            event_name: str,
            data: object,
        ) -> FlextResult[bool]:
            """Validate event input parameters."""
            if not event_name or not isinstance(event_name, str):
                return FlextResult[bool].fail(
                    "Domain event name must be a non-empty string",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            if data is not None and not _is_dict_like(data):
                return FlextResult[bool].fail(
                    "Domain event data must be a dictionary or None",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            if (
                len(self.domain_events)
                >= FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS
            ):
                return FlextResult[bool].fail(
                    f"Maximum uncommitted events reached: {FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return FlextResult[bool].ok(True)

        def _create_and_validate_event(
            self: Self,
            event_name: str,
            data: object,
        ) -> FlextResult[FlextModelsEntity.DomainEvent]:
            """Create and validate domain event."""
            try:
                # Fast fail: data must be dict (None not allowed for domain events)
                # Type narrowing: if dict-like, treat as Mapping; else use empty dict
                if isinstance(data, dict):
                    event_data: FlextTypes.Types.EventDataMapping = data
                else:
                    event_data = {}

                domain_event = FlextModelsEntity.DomainEvent(
                    event_type=event_name,
                    aggregate_id=self.unique_id,
                    data=event_data,
                )

                # Validate domain event using FlextUtilitiesValidation
                # (imported at module level - safe because validation.py uses ResultProtocol)
                validation_result = FlextUtilities.Validation.validate_domain_event(
                    domain_event,
                )
                if validation_result.is_failure:
                    return FlextResult[FlextModelsEntity.DomainEvent].fail(
                        f"Domain event validation failed: {validation_result.error}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[FlextModelsEntity.DomainEvent].ok(domain_event)

            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return FlextResult[FlextModelsEntity.DomainEvent].fail(
                    f"Failed to create domain event: {e}",
                    error_code=FlextConstants.Errors.DOMAIN_EVENT_ERROR,
                )

        def _execute_event_handler(
            self: Self,
            event_name: str,
            data: object,
        ) -> None:
            """Execute event handler if available."""
            # Fast fail: event_type must be str
            event_type: str = ""
            if _is_dict_like(data):
                data_dict = cast("Mapping[str, object]", data)
                event_type_raw = data_dict.get("event_type")
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
                            "Domain event handler %s failed for event %s: %s",
                            handler_method_name,
                            event_name,
                            e,
                        )

        def add_domain_event(
            self: Self,
            event_name: str,
            data: FlextTypes.Types.EventDataMapping | None,
        ) -> FlextResult[bool]:
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
                return FlextResult[bool].fail(
                    error_msg,
                    error_code=FlextConstants.Errors.DOMAIN_EVENT_ERROR,
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

            return FlextResult[bool].ok(True)

        @property
        def uncommitted_events(
            self: Self,
        ) -> list[FlextModelsEntity.DomainEvent]:
            """Get uncommitted domain events without clearing them."""
            return list(self.domain_events)

        def mark_events_as_committed(
            self: Self,
        ) -> FlextResult[list[FlextModelsEntity.DomainEvent]]:
            """Mark all domain events as committed and return them."""
            try:
                events = self.uncommitted_events

                if events:
                    event_types = [e.event_type for e in events]
                    self.logger.info(
                        "Domain events committed",
                        aggregate_id=self.unique_id,
                        aggregate_type=self.__class__.__name__,
                        event_count=len(events),
                        event_types=event_types,
                    )

                self.domain_events.clear()

                return FlextResult[list[FlextModelsEntity.DomainEvent]].ok(events)

            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return FlextResult[list[FlextModelsEntity.DomainEvent]].fail(
                    f"Failed to commit domain events: {e}",
                    error_code=FlextConstants.Errors.DOMAIN_EVENT_ERROR,
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
                    event_types=[e.event_type for e in events],
                )

            self.domain_events.clear()
            return events

        def _validate_bulk_events_input(
            self: Self,
            events: object,
        ) -> FlextResult[int]:
            """Validate bulk events input and return total count.

            Returns:
                FlextResult with total event count after add, or error

            """
            if not events:
                return FlextResult[int].ok(0)  # Signal empty - caller handles

            if not isinstance(events, (list, tuple)):
                return FlextResult[int].fail(
                    "Events must be a list of tuples",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            total_after_add = len(self.domain_events) + len(events)
            if total_after_add > FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS:
                return FlextResult[int].fail(
                    f"Bulk add would exceed max events: {total_after_add} > {FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return FlextResult[int].ok(total_after_add)

        def _validate_and_collect_events(
            self: Self,
            events: Sequence[tuple[str, FlextTypes.Types.EventDataMapping | None]],
        ) -> FlextResult[list[tuple[str, FlextTypes.Types.EventDataMapping]]]:
            """Validate and collect events for bulk add.

            Returns:
                FlextResult with validated events list or error

            """
            validated_events: list[tuple[str, FlextTypes.Types.EventDataMapping]] = []

            if not isinstance(events, (list, tuple)):
                return FlextResult[
                    list[tuple[str, FlextTypes.Types.EventDataMapping]]
                ].fail(
                    "Events must be a list or tuple",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            for i, event_item in enumerate(events):
                if not isinstance(event_item, tuple):
                    return FlextResult[
                        list[tuple[str, FlextTypes.Types.EventDataMapping]]
                    ].fail(
                        f"Event {i}: must be a tuple of (name, data)",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                if len(event_item) != _EVENT_TUPLE_LENGTH:
                    return FlextResult[
                        list[tuple[str, FlextTypes.Types.EventDataMapping]]
                    ].fail(
                        f"Event {i}: must have exactly 2 elements (name, data)",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                event_name, data = event_item

                if not isinstance(event_name, str) or not event_name:
                    return FlextResult[
                        list[tuple[str, FlextTypes.Types.EventDataMapping]]
                    ].fail(
                        f"Event {i}: name must be non-empty string",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                if data is not None and not _is_dict_like(data):
                    return FlextResult[
                        list[tuple[str, FlextTypes.Types.EventDataMapping]]
                    ].fail(
                        f"Event {i}: data must be dict[str, FlextTypes.GeneralValueType] or None",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                # Fast fail: data must be dict (None not allowed)
                # Type narrowing: if dict-like, treat as Mapping; else use empty dict
                if isinstance(data, dict):
                    event_data: FlextTypes.Types.EventDataMapping = data
                else:
                    event_data = {}
                validated_events.append((event_name, event_data))

            return FlextResult[list[tuple[str, FlextTypes.Types.EventDataMapping]]].ok(
                validated_events,
            )

        def _add_validated_events_bulk(
            self: Self,
            validated_events: list[tuple[str, FlextTypes.Types.EventDataMapping]],
        ) -> FlextResult[bool]:
            """Add validated events in bulk.

            Args:
                validated_events: Already validated list of (event_name, data) tuples

            Returns:
                FlextResult indicating success or error

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
                    event_types=[name for name, _ in validated_events],
                )

                return FlextResult[bool].ok(True)

            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return FlextResult[bool].fail(
                    f"Failed to add bulk domain events: {e}",
                    error_code=FlextConstants.Errors.DOMAIN_EVENT_ERROR,
                )

        def add_domain_events_bulk(
            self: Self,
            events: Sequence[tuple[str, FlextTypes.Types.EventDataMapping | None]],
        ) -> FlextResult[bool]:
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
                return FlextResult[bool].fail(
                    error_msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Handle empty case
            if count_result.unwrap() == 0:
                return FlextResult[bool].ok(True)

            # Validate and collect events
            validated_result = self._validate_and_collect_events(events)
            if validated_result.is_failure:
                base_msg = "Event validation failed"
                error_msg = (
                    f"{base_msg}: {validated_result.error}"
                    if validated_result.error
                    else f"{base_msg} (event validation rule failed)"
                )
                return FlextResult[bool].fail(
                    error_msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Add validated events
            return self._add_validated_events_bulk(validated_result.unwrap())

        def validate_consistency(self: Self) -> FlextResult[bool]:
            """Validate entity consistency using centralized validation."""
            entity_result = FlextModelsValidation.validate_entity_relationships(self)
            if entity_result.is_failure:
                return FlextResult[bool].fail(
                    f"Entity validation failed: {entity_result.error}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            for event in self.domain_events:
                event_result = FlextUtilities.Validation.validate_domain_event(event)
                if event_result.is_failure:
                    return FlextResult[bool].fail(
                        f"Domain event validation failed: {event_result.error}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            return FlextResult[bool].ok(True)

    class Value(FlextModelsBase.FrozenStrictModel):
        """Base class for value objects - immutable and compared by value."""

        @override
        def __eq__(self: Self, other: object) -> bool:
            """Compare by value (using FlextUtilities.Domain)."""
            return FlextUtilities.Domain.compare_value_objects_by_value(self, other)

        def __hash__(self) -> int:
            """Hash based on values for use in sets/dicts (using FlextUtilities.Domain)."""
            return FlextUtilities.Domain.hash_value_object_by_value(self)

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
                    raise FlextExceptions.ValidationError(
                        message=msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Post-init hook to check invariants."""
            super().model_post_init(__context)
            self.check_invariants()


__all__ = ["FlextModelsEntity"]
