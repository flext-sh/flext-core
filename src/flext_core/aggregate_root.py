"""Domain-Driven Design aggregate root implementation.

Provides enterprise-grade aggregate root patterns following DDD principles
with event sourcing, transactional boundaries, and clean architecture patterns.
Integrated with FLEXT protocols for type-safe domain modeling and event handling.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast, override

from pydantic import ConfigDict

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.models import FlextEntity
from flext_core.payload import FlextEvent, FlextPayload
from flext_core.result import FlextResult
from flext_core.root_models import (
    FlextEntityId,
    FlextEventList,
    FlextMetadata,
    FlextTimestamp,
    FlextVersion,
)
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


def _get_exception_class(name: str) -> type[Exception]:
    """Get dynamically created exception class with type safety."""
    return cast("type[Exception]", getattr(FlextExceptions, name))


class FlextAggregateRoot(FlextEntity):
    """DDD aggregate root with transactional boundaries and event management.

    Extends FlextEntity with specific aggregate capabilities for business operations,
    domain events, and transactional consistency. Implements domain event protocols
    for event sourcing and clean architecture patterns.

    This class follows Clean Architecture principles with:
        - Event sourcing through FlextProtocols.Domain.DomainEvent patterns
        - Type-safe domain event management with FlextResult railways
        - Transactional consistency boundaries for business operations
        - Immutable aggregate state with event-driven state changes

    Architecture:
        - Domain Layer: Business logic and domain events
        - Application Layer: Event handling and persistence coordination
        - Infrastructure Layer: Event storage and retrieval (via protocols)

    Examples:
        Creating an aggregate with domain events::

            aggregate = FlextAggregateRoot(
                entity_id="user-123",
                version=1,
                name="John Doe",
                email="john@example.com",
            )

        Adding domain events for event sourcing::

            result = aggregate.add_domain_event(
                "UserActivated",
                {"timestamp": datetime.now(), "reason": "Email verified"},
            )
            if result.success:
                events = aggregate.get_domain_events()
                # Process events through event store

    """

    model_config = ConfigDict(
        # Type safety and validation
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        # Serialization
        arbitrary_types_allowed=True,
        validate_default=True,
        # Immutable aggregate root for consistency
        frozen=True,
    )

    def __init__(
        self,
        entity_id: FlextTypes.Domain.EntityId | None = None,
        version: FlextTypes.Domain.EntityVersion = 1,
        **data: object,
    ) -> None:
        """Initialize with an empty event list."""
        actual_id = self._resolve_entity_id(entity_id, data)
        domain_events_objects = self._process_domain_events(data)
        created_at = self._extract_created_at(data)
        entity_data = {
            k: v
            for k, v in data.items()
            if k not in {"created_at", "_domain_event_objects"}
        }

        try:
            self._initialize_parent(
                actual_id,
                version,
                created_at,
                domain_events_objects,
                entity_data,
            )
            # Initialize a domain event objects list for aggregate root functionality
            # Use object.__setattr__ because the model is frozen
            object.__setattr__(self, "_domain_event_objects", [])
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            # REAL SOLUTION: Proper error handling for initialization failures
            error_msg = f"Failed to initialize aggregate root with provided data: {e}"
            validation_error = _get_exception_class("FlextExceptions.ValidationError")
            raise validation_error(error_msg) from e

    @staticmethod
    def _coerce_metadata_to_root(
        meta: object | None,
    ) -> FlextMetadata | None:
        """Coerce various metadata inputs into FlextMetadata or return None.

        This helper centralizes the conversion and keeps _initialize_parent small.
        """
        if meta is None:
            return None

        # Use FlextUtilities for safe dict conversion
        coerced_dict = FlextUtilities.SimpleTypeAdapters.to_dict_safe(meta)
        if coerced_dict:
            # Normalize keys to strings
            normalized_dict: dict[str, object] = {
                str(k): v for k, v in coerced_dict.items()
            }
            return FlextMetadata(normalized_dict)

        # Fallback for non-convertible objects
        return FlextMetadata(
            {FlextConstants.InfrastructureMessages.SERIALIZATION_WARNING: str(meta)}
        )

    @staticmethod
    def _normalize_domain_event_list(
        raw: FlextTypes.Core.List,
    ) -> list[FlextTypes.Core.Dict]:
        """Normalize raw domain events list into FlextProtocols.Domain.DomainEvent compatible format.

        Converts various event formats (dict, model, object) into standardized
        dictionary format compatible with FlextProtocols.Domain.DomainEvent protocol.
        Ensures type safety and consistent event structure for event sourcing.
        """
        normalized: list[FlextTypes.Core.Dict] = []
        for ev in raw:
            if isinstance(ev, dict):
                normalized.append(
                    {str(k): v for k, v in cast("FlextTypes.Core.Dict", ev).items()},
                )
                continue

            model_dump_fn = getattr(ev, "model_dump", None)
            if callable(model_dump_fn):
                dumped = model_dump_fn()
                if isinstance(dumped, dict):
                    normalized.append(
                        {
                            str(k): v
                            for k, v in cast("FlextTypes.Core.Dict", dumped).items()
                        },
                    )
                    continue

            normalized.append({"event": str(ev)})
        return normalized

    def _resolve_entity_id(
        self,
        entity_id: FlextTypes.Domain.EntityId | None,
        data: FlextTypes.Core.Dict,
    ) -> FlextTypes.Domain.EntityId:
        """Resolve the actual entity ID from parameters and data."""
        provided_id = data.pop("id", None)
        if provided_id is not None and isinstance(provided_id, str):
            return provided_id
        if entity_id is not None:
            return entity_id
        return FlextUtilities.Generators.generate_uuid()

    def _process_domain_events(
        self,
        data: FlextTypes.Core.Dict,
    ) -> list[FlextTypes.Core.Dict]:
        """Process and convert domain events to the proper format."""
        domain_events_raw: object = data.pop("domain_events", [])
        if isinstance(domain_events_raw, list):
            return self._normalize_domain_event_list(
                cast("FlextTypes.Core.List", domain_events_raw)
            )
        return []

    def _extract_created_at(
        self, data: FlextTypes.Core.Dict
    ) -> FlextTypes.Domain.EntityTimestamp | None:
        """Extract and validate created_at datetime from data."""
        created_at = data.get("created_at")
        if isinstance(created_at, datetime):
            return created_at
        return None

    def _initialize_parent(
        self,
        actual_id: FlextTypes.Domain.EntityId,
        version: FlextTypes.Domain.EntityVersion,
        created_at: FlextTypes.Domain.EntityTimestamp | None,
        domain_events_objects: list[FlextTypes.Core.Dict],
        entity_data: FlextTypes.Core.Dict,
    ) -> None:
        """Initialize parent class with all parameters."""
        # Explicit type conversion of expected types - always create new instances
        # to avoid isinstance issues
        id_value = FlextEntityId(actual_id)
        version_value = FlextVersion(version)
        created_at_value: FlextTimestamp | None = None
        if created_at is not None:
            created_at_value = FlextTimestamp(created_at)

        # domain_events_objects is expected to be normalized by _process_domain_events
        domain_events_value = FlextEventList(domain_events_objects)

        # Convert metadata if present
        metadata_value = None
        if "metadata" in entity_data:
            meta = entity_data.pop("metadata")
            metadata_value = self._coerce_metadata_to_root(meta)

        # Call parent with core strongly-typed parameters.
        # Provide created_at, updated_at and metadata explicitly to satisfy
        # static analysers (Pylance) which expect these arguments. Use the
        # coerced values when available, otherwise fall back to sensible
        # defaults defined by the root models.
        super().__init__(
            id=id_value,
            version=version_value,
            created_at=(
                created_at_value
                if created_at_value is not None
                else FlextTimestamp(datetime.now(UTC))
            ),
            updated_at=FlextTimestamp(datetime.now(UTC)),
            domain_events=domain_events_value,
            metadata=(
                metadata_value
                if metadata_value is not None
                else FlextMetadata.model_construct(root={})
            ),
            **entity_data,
        )
        # The _apply_entity_data function was removed as data is now passed directly.
        # _apply_entity_data(self, entity_data)

    @override
    def add_domain_event(
        self,
        event_type_or_dict: FlextTypes.Domain.EventType | FlextTypes.Core.Dict,
        event_data: FlextTypes.Domain.EventData | None = None,
    ) -> FlextResult[None]:
        """Add domain event following FlextProtocols.Domain.DomainEvent pattern.

        Creates and stores domain events compatible with event sourcing protocols.
        Events are validated and formatted according to DDD patterns with proper
        type safety through FlextResult railway-oriented programming.

        Args:
            event_type_or_dict: Event type string or complete event dictionary
                following FlextProtocols.Domain.DomainEvent structure
            event_data: Event payload data (used when first arg is event type string)

        Returns:
            FlextResult[None]: Success/failure status with structured error handling
                - Success: Event added to internal event list and domain_events
                - Failure: Validation error with FlextConstants.Errors codes

        Examples:
            Adding event with string type::

                result = aggregate.add_domain_event(
                    "OrderPlaced", {"order_id": "123", "amount": 99.99}
                )

            Adding event with dictionary::

                event_dict = {
                    "type": "OrderPlaced",
                    "data": {"order_id": "123", "amount": 99.99},
                }
                result = aggregate.add_domain_event(event_dict)

        """
        try:
            # Handle both signatures
            if isinstance(event_type_or_dict, str):
                event_type = event_type_or_dict
                data = event_data or {}
            else:
                # Assume it's a dict
                event_dict = event_type_or_dict
                event_type = str(
                    event_dict.get("type", FlextConstants.Errors.UNKNOWN_ERROR)
                )
                data_raw = event_dict.get("data", {})
                data = (
                    cast("FlextTypes.Core.Dict", data_raw)
                    if isinstance(data_raw, dict)
                    else {}
                )

            event_result: FlextResult[FlextEvent] = cast(
                "FlextResult[FlextEvent]",
                FlextPayload.create_event(
                    event_type=event_type,
                    event_data=data,
                    aggregate_id=str(self.id),
                    version=int(self.version),
                ),
            )
            if event_result.is_failure:
                return FlextResult[None].fail(
                    f"{FlextConstants.Handlers.EVENT_PROCESSING_FAILED}: {event_result.error}",
                    error_code=FlextConstants.Errors.EVENT_ERROR,
                )

            event: FlextEvent = event_result.value
            current_events: list[FlextEvent] = getattr(
                self,
                "_domain_event_objects",
                [],
            )
            new_events: list[FlextEvent] = [*current_events, event]
            object.__setattr__(self, "_domain_event_objects", new_events)
            current_dict_events: list[FlextTypes.Core.Dict] = list(
                self.domain_events.root
            )
            new_dict_events: list[FlextTypes.Core.Dict] = [
                *current_dict_events,
                event.model_dump(),
            ]
            object.__setattr__(self, "domain_events", FlextEventList(new_dict_events))
            return FlextResult[None].ok(None)
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            return FlextResult[None].fail(
                f"{FlextConstants.Handlers.EVENT_PROCESSING_FAILED}: {e}",
                error_code=FlextConstants.Errors.EXCEPTION_ERROR,
            )

    def add_typed_domain_event(
        self,
        event_type: FlextTypes.Domain.EventType,
        event_data: FlextTypes.Domain.EventData,
    ) -> FlextResult[None]:
        """Add strongly-typed domain event with FlextProtocols validation.

        Creates domain events with strict type checking following
        FlextProtocols.Domain.DomainEvent protocol. Provides enhanced
        type safety compared to add_domain_event() method.

        Args:
            event_type: Strongly-typed event type identifier
            event_data: Validated event payload data

        Returns:
            FlextResult[None]: Type-safe result with validation
                - Success: Event validated and added to aggregate
                - Failure: Type validation or creation error

        Note:
            This method provides stricter type checking than add_domain_event()
            and should be preferred for new code following FLEXT patterns.

        """
        try:
            event_result: FlextResult[FlextEvent] = cast(
                "FlextResult[FlextEvent]",
                FlextPayload.create_event(
                    event_type=event_type,
                    event_data=event_data,
                    aggregate_id=str(self.id),
                    version=int(self.version),
                ),
            )
            if event_result.is_failure:
                return FlextResult[None].fail(
                    f"{FlextConstants.Handlers.EVENT_PROCESSING_FAILED}: {event_result.error}",
                    error_code=FlextConstants.Errors.EVENT_ERROR,
                )

            event: FlextEvent = event_result.value
            current_events: list[FlextEvent] = getattr(
                self,
                "_domain_event_objects",
                [],
            )
            new_events: list[FlextEvent] = [*current_events, event]
            object.__setattr__(self, "_domain_event_objects", new_events)
            event_dict: FlextTypes.Core.Dict = event.model_dump()
            self.domain_events.root.append(event_dict)
            return FlextResult[None].ok(None)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[None].fail(
                f"{FlextConstants.Handlers.EVENT_PROCESSING_FAILED}: {e}",
                error_code=FlextConstants.Errors.EXCEPTION_ERROR,
            )

    def add_event_object(self, event: FlextEvent) -> None:
        """Add FlextEvent object directly following FlextProtocols.Domain.DomainEvent.

        Convenience method for adding pre-constructed FlextEvent objects that
        already comply with FlextProtocols.Domain.DomainEvent protocol structure.
        Bypasses event creation but maintains aggregate consistency.

        Args:
            event: Pre-constructed FlextEvent following domain event protocol

        Note:
            Event object must be compatible with FlextProtocols.Domain.DomainEvent
            protocol for proper event sourcing integration.

        """
        current_events: list[FlextEvent] = getattr(self, "_domain_event_objects", [])
        new_events: list[FlextEvent] = [*current_events, event]
        object.__setattr__(self, "_domain_event_objects", new_events)
        current_dict_events: list[FlextTypes.Core.Dict] = list(self.domain_events.root)
        new_dict_events: list[FlextTypes.Core.Dict] = [
            *current_dict_events,
            event.model_dump(),
        ]
        object.__setattr__(self, "domain_events", FlextEventList(new_dict_events))

    def get_domain_events(self) -> list[FlextEvent]:
        """Get unpublished domain events compatible with FlextProtocols.Domain.EventStore.

        Returns domain events that follow FlextProtocols.Domain.DomainEvent protocol
        for integration with event sourcing infrastructure. Events are returned as
        FlextEvent objects maintaining type safety and protocol compliance.

        Returns:
            list[FlextEvent]: Domain events ready for event store persistence
                Each event implements FlextProtocols.Domain.DomainEvent protocol

        Integration:
            Events can be directly passed to FlextProtocols.Domain.EventStore.save_events()
            for persistence in event sourcing systems.

        """
        if hasattr(self, "_domain_event_objects"):
            return list(getattr(self, "_domain_event_objects", []))
        return []

    def clear_domain_events(self) -> list[FlextTypes.Core.Dict]:
        """Clear domain events after FlextProtocols.Domain.EventStore persistence.

        Clears the aggregate's domain events after they have been successfully
        persisted through FlextProtocols.Domain.EventStore.save_events().
        Returns cleared events for confirmation or additional processing.

        Returns:
            list[FlextTypes.Core.Dict]: Cleared domain events as dictionaries
                Each dictionary follows FlextProtocols.Domain.DomainEvent structure

        Usage Pattern:
            1. Get events: events = aggregate.get_domain_events()
            2. Persist: event_store.save_events(aggregate_id, events, version)
            3. Clear: cleared = aggregate.clear_domain_events()

        """
        if hasattr(self, "_domain_event_objects"):
            events: list[FlextEvent] = list(getattr(self, "_domain_event_objects", []))
            # Use object.__setattr__ because the model is frozen
            object.__setattr__(self, "_domain_event_objects", [])
            object.__setattr__(
                self,
                "domain_events",
                FlextEventList([]),
            )  # Also clear the dict version
            # Convert FlextEvent objects to dictionaries
            return [
                cast("FlextTypes.Core.Dict", event.model_dump())
                if hasattr(event, "model_dump")
                else cast("FlextTypes.Core.Dict", {"event": str(event)})
                for event in events
            ]
        return []

    def has_domain_events(self) -> bool:
        """Check if aggregate has unpublished events for FlextProtocols.Domain.EventStore.

        Determines if the aggregate has domain events pending persistence
        through FlextProtocols.Domain.EventStore infrastructure patterns.

        Returns:
            bool: True if events need to be persisted via event store

        """
        return bool(self.domain_events)


# Export API
__all__ = ["FlextAggregateRoot"]
