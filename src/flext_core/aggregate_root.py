"""Domain-Driven Design aggregate root implementation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast, override

from pydantic import ConfigDict

from flext_core.exceptions import FlextValidationError
from flext_core.models import FlextEntity
from flext_core.payload import FlextEvent
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.root_models import (
    FlextEntityId,
    FlextEventList,
    FlextMetadata,
    FlextTimestamp,
    FlextVersion,
)
from flext_core.utilities import FlextGenerators

# Type aliases for unified approach with FlextProtocols integration - Python 3.13+ syntax
type DomainServiceProtocol = FlextProtocols.Domain.Service


def _coerce_metadata_to_root(meta: object | None) -> FlextMetadata | None:
    """Coerce various metadata inputs into FlextMetadata or return None.

    This helper centralizes the conversion and keeps _initialize_parent small.
    """
    if meta is None:
        return None
    if isinstance(meta, FlextMetadata):
        return meta
    if isinstance(meta, dict):
        coerced = {str(k): v for k, v in cast("dict[str, object]", meta).items()}
        return FlextMetadata(coerced)
    try:
        # Use cast to handle dict conversion from object
        coerced = dict(cast("dict[str, object]", meta))
        coerced_meta = {str(k): v for k, v in coerced.items()}
        return FlextMetadata(coerced_meta)
    except Exception:
        return FlextMetadata({"raw": str(meta)})


def _normalize_domain_event_list(raw: list[object]) -> list[dict[str, object]]:
    """Normalize a raw domain events list into list[dict[str, object]]."""
    normalized: list[dict[str, object]] = []
    for ev in raw:
        if isinstance(ev, dict):
            normalized.append(
                {str(k): v for k, v in cast("dict[object, object]", ev).items()},
            )
            continue

        model_dump_fn = getattr(ev, "model_dump", None)
        if callable(model_dump_fn):
            dumped = model_dump_fn()
            if isinstance(dumped, dict):
                normalized.append(
                    {
                        str(k): v
                        for k, v in cast("dict[object, object]", dumped).items()
                    },
                )
                continue

        normalized.append({"event": str(ev)})
    return normalized


class FlextAggregateRoot(FlextEntity):
    """DDD aggregate root with transactional boundaries and event management.

    Extends FlextEntity with specific aggregate capabilities for business operations,
    domain events, and transactional consistency.

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

    def _process_domain_events(
        self,
        data: dict[str, object],
    ) -> list[dict[str, object]]:
        """Process and convert domain events to the proper format."""
        domain_events_raw: object = data.pop("domain_events", [])
        if isinstance(domain_events_raw, list):
            return _normalize_domain_event_list(cast("list[object]", domain_events_raw))
        return []

    def _extract_created_at(self, data: dict[str, object]) -> datetime | None:
        """Extract and validate created_at datetime from data."""
        created_at = data.get("created_at")
        if isinstance(created_at, datetime):
            return created_at
        return None

    def _initialize_parent(
        self,
        actual_id: str,
        version: int,
        created_at: datetime | None,
        domain_events_objects: list[dict[str, object]],
        entity_data: dict[str, object],
    ) -> None:
        """Initialize parent class with all parameters."""
        # Conversão explícita dos tipos esperados - sempre criar novos para
        # evitar isinstance issues
        id_value = FlextEntityId(actual_id)
        version_value = FlextVersion(version)
        created_at_value: FlextTimestamp | None = None
        if created_at is not None:
            created_at_value = FlextTimestamp(created_at)

        # domain_events_objects is expected to be normalized by _process_domain_events
        domain_events_value = FlextEventList(domain_events_objects)

        # Se houver metadata, converter
        metadata_value = None
        if "metadata" in entity_data:
            meta = entity_data.pop("metadata")
            metadata_value = _coerce_metadata_to_root(meta)

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
        event_type_or_dict: str | dict[str, object],
        event_data: dict[str, object] | None = None,
    ) -> FlextResult[None]:
        """Add domain event for event sourcing.

        Args:
            event_type_or_dict: Event type string or event dictionary
            event_data: Event data (used when first arg is string)

        Returns:
            FlextResult indicating success or failure

        """
        try:
            # Handle both signatures
            if isinstance(event_type_or_dict, str):
                event_type = event_type_or_dict
                data = event_data or {}
            else:
                # Assume it's a dict
                event_dict = event_type_or_dict
                event_type = str(event_dict.get("type", "unknown"))
                data_raw = event_dict.get("data", {})
                data = (
                    cast("dict[str, object]", data_raw)
                    if isinstance(data_raw, dict)
                    else {}
                )

            event_result: FlextResult[FlextEvent] = FlextEvent.create_event(
                event_type=event_type,
                event_data=data,
                aggregate_id=str(self.id),
                version=int(self.version),
            )
            if event_result.is_failure:
                return FlextResult[None].fail(
                    f"Failed to create event: {event_result.error}",
                )

            event: FlextEvent = event_result.value
            current_events: list[FlextEvent] = getattr(
                self,
                "_domain_event_objects",
                [],
            )
            new_events: list[FlextEvent] = [*current_events, event]
            object.__setattr__(self, "_domain_event_objects", new_events)
            current_dict_events: list[dict[str, object]] = list(self.domain_events.root)
            new_dict_events: list[dict[str, object]] = [
                *current_dict_events,
                event.model_dump(),
            ]
            object.__setattr__(self, "domain_events", FlextEventList(new_dict_events))
            return FlextResult[None].ok(None)
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            return FlextResult[None].fail(f"Failed to add domain event: {e}")

    def add_typed_domain_event(
        self,
        event_type: str,
        event_data: dict[str, object],
    ) -> FlextResult[None]:
        """Add typed domain event with validation.

        Args:
            event_type: Type of domain event
            event_data: Event data

        Returns:
            Result of adding event

        """
        try:
            event_result: FlextResult[FlextEvent] = FlextEvent.create_event(
                event_type=event_type,
                event_data=event_data,
                aggregate_id=str(self.id),
                version=int(self.version),
            )
            if event_result.is_failure:
                return FlextResult[None].fail(
                    f"Failed to create event: {event_result.error}",
                )

            event: FlextEvent = event_result.value
            event_dict: dict[str, object] = event.model_dump()
            self.domain_events.root.append(event_dict)
            return FlextResult[None].ok(None)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult[None].fail(f"Failed to add domain event: {e}")

    def add_event_object(self, event: FlextEvent) -> None:
        """Add a domain event object directly (convenience method).

        Args:
            event: Domain event object to add

        """
        current_events: list[FlextEvent] = getattr(self, "_domain_event_objects", [])
        new_events: list[FlextEvent] = [*current_events, event]
        object.__setattr__(self, "_domain_event_objects", new_events)
        current_dict_events: list[dict[str, object]] = list(self.domain_events.root)
        new_dict_events: list[dict[str, object]] = [
            *current_dict_events,
            event.model_dump(),
        ]
        object.__setattr__(self, "domain_events", FlextEventList(new_dict_events))

    def get_domain_events(self) -> list[FlextEvent]:
        """Get all unpublished domain events as FlextEvent objects.

        Returns:
            List of domain events as FlextEvent instances

        """
        if hasattr(self, "_domain_event_objects"):
            return list(getattr(self, "_domain_event_objects", []))
        return []

    def clear_domain_events(self) -> list[dict[str, object]]:
        """Clear all domain events after publishing.

        Returns:
            List of domain events that were cleared as dictionaries

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
                event.model_dump()
                if hasattr(event, "model_dump")
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
# Note: model_rebuild() disabled due to FlextTypes.Core.Dict circular reference issues
# The models work correctly without an explicit rebuild as Pydantic handles
# forward references automatically during runtime validation
# FlextAggregateRoot.model_rebuild()

# Export API
__all__ = ["FlextAggregateRoot"]
