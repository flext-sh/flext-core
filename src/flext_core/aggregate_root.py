"""Domain-Driven Design aggregate root implementation.

Provides a DDD aggregate root pattern with transactional boundaries,
domain event management, and business logic encapsulation for the
FLEXT data integration ecosystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pydantic import ConfigDict

from flext_core.exceptions import FlextValidationError
from flext_core.models import FlextEntity
from flext_core.payload import FlextEvent
from flext_core.result import FlextResult
from flext_core.utilities import FlextGenerators

if TYPE_CHECKING:
    from datetime import datetime

    from flext_core.typings import TAnyDict


class FlextAggregateRoot(FlextEntity):
    """DDD aggregate root with transactional boundaries and event management.

    Extends FlextEntity with aggregate-specific capabilities for business
    operations, domain events, and transactional consistency.
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
        entity_id: str | None = None,
        version: int = 1,
        **data: object,
    ) -> None:
        """Initialize with an empty event list."""
        actual_id = self._resolve_entity_id(entity_id, data)
        domain_events_objects = self._process_domain_events(data)
        created_at = self._extract_created_at(data)
        entity_data = {k: v for k, v in data.items() if k != "created_at"}

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
            raise FlextValidationError(
                error_msg,
                validation_details={
                    "aggregate_id": actual_id,
                    "initialization_error": str(e),
                    "provided_entity_data": list(entity_data.keys()),
                    "provided_version": version,
                },
            ) from e

    def _resolve_entity_id(self, entity_id: str | None, data: dict[str, object]) -> str:
        """Resolve the actual entity ID from parameters and data."""
        provided_id = data.pop("id", None)
        if provided_id is not None and isinstance(provided_id, str):
            return provided_id
        if entity_id is not None:
            return entity_id
        return FlextGenerators.generate_uuid()

    def _process_domain_events(self, data: dict[str, object]) -> list[object]:
        """Process and convert domain events to the proper format."""
        domain_events_raw = data.pop("domain_events", [])
        domain_events_objects: list[object] = []
        if isinstance(domain_events_raw, list):
            for ev in domain_events_raw:
                if isinstance(ev, dict):
                    domain_events_objects.append(ev)
                elif hasattr(ev, "model_dump"):
                    dumped = ev.model_dump()
                    if isinstance(dumped, dict):
                        domain_events_objects.append(dumped)
                else:
                    domain_events_objects.append(ev)
        return domain_events_objects

    def _extract_created_at(self, data: dict[str, object]) -> datetime | None:
        """Extract and validate created_at datetime from data."""
        if "created_at" in data and hasattr(data["created_at"], "year"):
            return cast("datetime", data["created_at"])
        return None

    def _initialize_parent(
        self,
        actual_id: str,
        version: int,
        created_at: datetime | None,
        domain_events_objects: list[object],
        entity_data: dict[str, object],
    ) -> None:
        """Initialize parent class with all parameters."""
        # Initialize base entity first with all parameters
        init_params: dict[str, object] = {"id": actual_id, "version": version}
        if created_at is not None:
            init_params["created_at"] = created_at
        # Add all other entity data
        init_params.update(entity_data)
        super().__init__(**init_params)  # type: ignore[arg-type]
        # Set domain events
        object.__setattr__(self, "domain_events", list(domain_events_objects))

    def add_domain_event(self, *args: object) -> FlextResult[None]:
        """Add domain event for event sourcing.

        Supports both legacy signature (event: dict) and modern
        (event_type: str, data: dict).
        """
        try:
            legacy_args_count = 1
            modern_args_count = 2

            if len(args) == legacy_args_count and isinstance(args[0], dict):
                # Legacy signature: event dict
                event_dict = args[0]
                event_type = event_dict.get("type", "unknown")
                event_data = event_dict.get("data", {})
            elif (
                len(args) == modern_args_count
                and isinstance(args[0], str)
                and isinstance(args[1], dict)
            ):
                # Modern signature: event_type, event_data
                event_type = args[0]
                event_data = args[1]
            else:
                return FlextResult.fail("Invalid event arguments")

            # Create FlextEvent instance for proper type support
            event_result = FlextEvent.create_event(
                event_type=str(event_type),
                event_data=cast("TAnyDict", event_data),
                aggregate_id=self.id,
                version=self.version,
            )
            if event_result.is_failure:
                return FlextResult.fail(f"Failed to create event: {event_result.error}")

            # Store the FlextEvent directly and also in domain_events for compatibility
            event = event_result.unwrap()
            # Store as FlextEvent instance in a separate list for aggregate root tests
            # Use object.__setattr__ because the model is frozen
            current_events = getattr(self, "_domain_event_objects", [])
            new_events = [*current_events, event]
            object.__setattr__(self, "_domain_event_objects", new_events)
            # Store as dict for FlextEntity compatibility
            current_dict_events = list(self.domain_events)
            new_dict_events = [*current_dict_events, event.model_dump()]
            object.__setattr__(self, "domain_events", new_dict_events)
            return FlextResult.ok(None)
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            return FlextResult.fail(f"Failed to add domain event: {e}")

    def add_typed_domain_event(
        self,
        event_type: str,
        event_data: TAnyDict,
    ) -> FlextResult[None]:
        """Add typed domain event with validation.

        Args:
            event_type: Type of domain event
            event_data: Event data

        Returns:
            Result of adding event

        """
        try:
            event_result = FlextEvent.create_event(
                event_type=event_type,
                event_data=event_data,
                aggregate_id=self.id,
                version=self.version,
            )
            if event_result.is_failure:
                return FlextResult.fail(f"Failed to create event: {event_result.error}")

            # Convert FlextEvent to dict for compatibility with FlextEntity
            event = event_result.unwrap()
            self.domain_events.append(event.model_dump())
            return FlextResult.ok(None)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(f"Failed to add domain event: {e}")

    def add_event_object(self, event: FlextEvent) -> None:
        """Add a domain event object directly (convenience method).

        Args:
            event: Domain event object to add

        """
        # Store as FlextEvent instance in a separate list for aggregate root tests
        # Use object.__setattr__ because the model is frozen
        current_events = getattr(self, "_domain_event_objects", [])
        new_events = [*current_events, event]
        object.__setattr__(self, "_domain_event_objects", new_events)
        # Convert FlextEvent to dict for compatibility with FlextEntity
        current_dict_events = list(self.domain_events)
        new_dict_events = [*current_dict_events, event.model_dump()]
        object.__setattr__(self, "domain_events", new_dict_events)

    def get_domain_events(self) -> list[FlextEvent]:
        """Get all unpublished domain events as FlextEvent objects.

        Returns:
            List of domain events as FlextEvent instances

        """
        if hasattr(self, "_domain_event_objects"):
            return list(self._domain_event_objects)
        return []

    def clear_domain_events(self) -> list[dict[str, object]]:
        """Clear all domain events after publishing.

        Returns:
            List of domain events that were cleared as dictionaries

        """
        if hasattr(self, "_domain_event_objects"):
            events = list(getattr(self, "_domain_event_objects", []))
            # Use object.__setattr__ because the model is frozen
            object.__setattr__(self, "_domain_event_objects", [])
            object.__setattr__(self, "domain_events", [])  # Also clear the dict version
            # Convert FlextEvent objects to dictionaries
            return [
                event.model_dump()
                if hasattr(event, "model_dump")
                else event.__dict__
                if hasattr(event, "__dict__")
                else {"event": str(event)}
                for event in events
            ]
        return []

    def has_domain_events(self) -> bool:
        """Check if aggregate has unpublished events."""
        return bool(self.domain_events)


# =============================================================================
# MODEL REBUILDS - Resolve forward references for Pydantic
# =============================================================================

# Rebuild models to resolve forward references after import
# Note: model_rebuild() disabled due to TAnyDict circular reference issues
# The models work correctly without an explicit rebuild as Pydantic handles
# forward references automatically during runtime validation
# FlextAggregateRoot.model_rebuild()

# Export API
__all__: list[str] = ["FlextAggregateRoot"]
