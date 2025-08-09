"""Domain-Driven Design aggregate root implementation.

Provides DDD aggregate root pattern with transactional boundaries,
domain event management, and business logic encapsulation for the
FLEXT data integration ecosystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

    def __init__(
        self,
        entity_id: str | None = None,
        version: int = 1,
        **data: object,
    ) -> None:
        """Initialize with empty event list."""
        # Handle id from data or entity_id parameter
        provided_id = data.pop("id", None)
        # Ensure actual_id is a string
        if provided_id is not None and isinstance(provided_id, str):
            actual_id = provided_id
        elif entity_id is not None:
            actual_id = entity_id
        else:
            actual_id = FlextGenerators.generate_uuid()

        # Initialize domain events list
        domain_events_raw = data.pop("domain_events", [])
        # Ensure domain_events is properly typed as list[FlextEvent]
        domain_events = domain_events_raw if isinstance(domain_events_raw, list) else []

        # Only add created_at if it's a proper datetime
        created_at = None
        if "created_at" in data and hasattr(data["created_at"], "year"):
            created_at = cast("datetime", data["created_at"])

        # Remove created_at from data to avoid duplicate argument
        entity_data = {k: v for k, v in data.items() if k != "created_at"}

        # Initialize parent FlextEntity with proper arguments
        # Pass through all additional fields for subclass-specific attributes
        init_kwargs = {
            "id": actual_id,
            "version": version,
            "domain_events": domain_events,
        }

        # Only add created_at if it's not None
        if created_at is not None:
            init_kwargs["created_at"] = created_at

        # Add all remaining entity data for subclass fields
        init_kwargs.update(entity_data)

        try:
            # Pydantic BaseModel accepts **kwargs, but MyPy needs explicit params
            # Extract known FlextEntity parameters with proper typing
            entity_id_raw = init_kwargs.pop("id", actual_id)
            entity_id = cast("str", entity_id_raw if entity_id_raw else actual_id)

            entity_version_raw = init_kwargs.pop("version", version)
            entity_version = cast(
                "int",
                entity_version_raw if entity_version_raw else version,
            )

            entity_domain_events_raw = init_kwargs.pop("domain_events", domain_events)
            entity_domain_events = cast(
                "list[dict[str, object]]",
                entity_domain_events_raw if entity_domain_events_raw else domain_events,
            )

            # Initialize parent class with explicit parameters
            super().__init__(
                id=entity_id,
                version=entity_version,
                domain_events=entity_domain_events,
            )

            # Set any additional attributes after initialization
            for key, value in init_kwargs.items():
                setattr(self, key, value)
        except Exception as e:
            # REAL SOLUTION: Proper error handling for initialization failures
            error_msg = f"Failed to initialize aggregate root with provided data: {e}"
            raise FlextValidationError(
                error_msg,
                validation_details={
                    "aggregate_id": actual_id,
                    "initialization_error": str(e),
                    "provided_kwargs": list(init_kwargs.keys()),
                },
            ) from e

    def add_domain_event(self, event: dict[str, object]) -> None:  # type: ignore[override]
        """Add domain event to be published after persistence.

        Override from FlextEntity to maintain compatibility.

        Args:
            event: Event dictionary

        """
        self.domain_events.append(event)

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
        """Add domain event object directly (convenience method).

        Args:
            event: Domain event object to add

        """
        # Convert FlextEvent to dict for compatibility with FlextEntity
        self.domain_events.append(event.model_dump())

    def get_domain_events(self) -> list[dict[str, object]]:
        """Get all unpublished domain events.

        Returns:
            List of domain events as dictionaries

        """
        return list(self.domain_events)

    def clear_domain_events(self) -> list[dict[str, object]]:
        """Clear all domain events after publishing.

        Returns:
            The cleared domain events list

        """
        events = self.domain_events.copy()
        self.domain_events.clear()
        return events

    def has_domain_events(self) -> bool:
        """Check if aggregate has unpublished events."""
        return bool(self.domain_events)


# =============================================================================
# MODEL REBUILDS - Resolve forward references for Pydantic
# =============================================================================

# Rebuild models to resolve forward references after import
# Note: model_rebuild() disabled due to TAnyDict circular reference issues
# The models work correctly without explicit rebuild as Pydantic handles
# forward references automatically during runtime validation
# FlextAggregateRoot.model_rebuild()

# Export API
__all__: list[str] = ["FlextAggregateRoot"]
