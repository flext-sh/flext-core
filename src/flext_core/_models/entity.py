"""Entity and DDD patterns extracted from FlextModels.

This module contains the FlextModelsEntity class with all related DDD patterns
as nested classes. It should NOT be imported directly - use FlextModels.Entity instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import ClassVar, override

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

import flext_core._models.validation as _validation_module
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime


class FlextModelsEntity:
    """Entity and DDD pattern container class.

    This class acts as a namespace container for Entity and related DDD patterns.
    All nested classes are accessed via FlextModels.Entity.* in the main models.py.
    """

    class IdentifiableMixin(BaseModel):
        """Mixin for models with unique identifiers."""

        # Explicit construction for type safety (avoiding TypedDict spreading issues)
        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
            json_schema_extra={
                "title": "IdentifiableMixin",
                "description": "Mixin providing unique identifier fields",
            },
        )

        unique_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    class TimestampableMixin(BaseModel):
        """Mixin for models with creation and update timestamps."""

        # Explicit construction for type safety (avoiding TypedDict spreading issues)
        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
            json_schema_extra={
                "title": "TimestampableMixin",
                "description": "Mixin providing timestamp fields and serialization",
            },
        )

        created_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp when the model was created (UTC timezone)",
            examples=[
                datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
                datetime(2025, 10, 12, 15, 30, 0, tzinfo=UTC),
            ],
        )
        updated_at: datetime | None = Field(
            default=None,
            description="Timestamp when the model was last updated (UTC timezone)",
            examples=[
                None,
                datetime(2025, 1, 2, 14, 30, 0, tzinfo=UTC),
            ],
        )

        @field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(self, value: datetime | None) -> str | None:
            """Serialize timestamps to ISO 8601 format for JSON."""
            return value.isoformat() if value else None

        @computed_field
        def is_modified(self) -> bool:
            """Check if the model has been modified after creation."""
            return self.updated_at is not None

        def update_timestamp(self) -> None:
            """Update the updated_at timestamp to current UTC time."""
            self.updated_at = datetime.now(UTC)

    class VersionableMixin(BaseModel):
        """Mixin for models with versioning support."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "VersionableMixin",
                "description": "Mixin providing version fields and optimistic locking",
            },
        )

        version: int = Field(
            default=FlextConstants.Performance.DEFAULT_VERSION,
            ge=FlextConstants.Performance.MIN_VERSION,
            description="Version number for optimistic locking - increments with each update",
            examples=[1, 5, 42, 100],
        )

        @computed_field
        def is_initial_version(self) -> bool:
            """Check if this is the initial version (version 1)."""
            return self.version == FlextConstants.Performance.DEFAULT_VERSION

        def increment_version(self) -> None:
            """Increment the version number for optimistic locking."""
            self.version += 1

    class ArbitraryTypesModel(BaseModel):
        """Base model with arbitrary types support."""

        model_config = ConfigDict(
            validate_assignment=True,
            extra="forbid",
            arbitrary_types_allowed=True,
            use_enum_values=True,
            json_schema_extra={
                "title": "ArbitraryTypesModel",
                "description": "Base model with arbitrary types support and comprehensive validation",
            },
        )

    class FrozenStrictModel(BaseModel):
        """Immutable base model."""

        # Explicit construction for type safety (avoiding TypedDict spreading issues)
        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
            frozen=True,
            json_schema_extra={
                "title": "FrozenStrictModel",
                "description": "Immutable base model with strict validation and frozen state",
            },
        )

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Base class for models with timestamp fields."""

    class DomainEvent(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Base class for domain events."""

        message_type: FlextConstants.Cqrs.EventMessageTypeLiteral = Field(
            default="event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
        )

        event_type: str
        aggregate_id: str
        data: dict[str, object] = Field(default_factory=dict)
        metadata: dict[str, str | int | float] = Field(default_factory=dict)

    class Core(TimestampedModel, IdentifiableMixin, VersionableMixin):
        """Core Entity implementation - base class for domain entities with identity.

        Combines TimestampedModel, IdentifiableMixin, and VersionableMixin to provide:
        - unique_id: Unique identifier (from IdentifiableMixin)
        - entity_id: Entity identifier property (alias for unique_id)
        - created_at/updated_at: Timestamps (from TimestampedModel)
        - version: Optimistic locking (from VersionableMixin)
        - domain_events: Event sourcing support
        """

        _internal_logger: ClassVar[FlextLogger | None] = None

        @property
        def entity_id(self) -> str:
            """Entity identifier property - alias for unique_id."""
            return self.unique_id

        @classmethod
        def _get_logger(cls) -> FlextLogger:
            """Get or create the internal logger (lazy initialization)."""
            if cls._internal_logger is None:
                cls._internal_logger = FlextLogger(__name__)
            return cls._internal_logger

        domain_events: list[FlextModelsEntity.DomainEvent] = Field(default_factory=list)

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Post-initialization hook to set updated_at timestamp."""
            if self.updated_at is None:
                self.updated_at = datetime.now(UTC)

        @override
        def __eq__(self, other: object) -> bool:
            """Identity-based equality for entities (using FlextUtilities.Domain)."""
            return FlextUtilitiesDomain.compare_entities_by_id(self, other)

        @override
        def __hash__(self) -> int:
            """Identity-based hash for entities (using FlextUtilities.Domain)."""
            return FlextUtilitiesDomain.hash_entity_by_id(self)

        def _validate_event_input(
            self, event_name: str, data: dict[str, object]
        ) -> FlextResult[bool]:
            """Validate event input parameters."""
            if not event_name or not isinstance(event_name, str):
                return FlextResult[bool].fail(
                    "Domain event name must be a non-empty string",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            if data is not None and not FlextRuntime.is_dict_like(data):
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
            self, event_name: str, data: dict[str, object]
        ) -> FlextResult[FlextModelsEntity.DomainEvent]:
            """Create and validate domain event."""
            try:
                # Fast fail: data must be dict (None not allowed for domain events)
                event_data: dict[str, object] = data if isinstance(data, dict) else {}
                domain_event = FlextModelsEntity.DomainEvent(
                    event_type=event_name,
                    aggregate_id=self.unique_id,
                    data=event_data,
                )

                validation_result = (
                    _validation_module.FlextModelsValidation.validate_domain_event(
                        domain_event
                    )
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
            self, event_name: str, data: dict[str, object]
        ) -> None:
            """Execute event handler if available."""
            # Fast fail: event_type must be str or None
            event_type_value: str = ""
            if isinstance(data, dict):
                event_type_raw = data.get("event_type")
                event_type_value = (
                    str(event_type_raw) if event_type_raw is not None else ""
                )
            event_type: str = event_type_value
            if event_type:
                handler_method_name = f"_apply_{str(event_type).lower()}"
                if hasattr(self, handler_method_name):
                    try:
                        handler_method = getattr(self, handler_method_name)
                        handler_method(data)
                        self._get_logger().debug(
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
                        self._get_logger().warning(
                            f"Domain event handler {handler_method_name} failed for event {event_name}: {e}"
                        )

        def add_domain_event(
            self, event_name: str, data: dict[str, object]
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

            self._get_logger().debug(
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
        def uncommitted_events(self) -> list[FlextModelsEntity.DomainEvent]:
            """Get uncommitted domain events without clearing them."""
            return list(self.domain_events)

        def mark_events_as_committed(
            self,
        ) -> FlextResult[list[FlextModelsEntity.DomainEvent]]:
            """Mark all domain events as committed and return them."""
            try:
                events = self.uncommitted_events

                if events:
                    event_types = [e.event_type for e in events]
                    self._get_logger().info(
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

        def clear_domain_events(self) -> list[FlextModelsEntity.DomainEvent]:
            """Clear and return domain events."""
            events = self.domain_events.copy()

            if events:
                domain_events = list(events)
                self._get_logger().debug(
                    "Domain events cleared",
                    aggregate_id=self.unique_id,
                    aggregate_type=self.__class__.__name__,
                    event_count=len(domain_events),
                    event_types=[e.event_type for e in domain_events]
                    if domain_events
                    else [],
                )

            self.domain_events.clear()
            return events

        def _validate_bulk_events_input(
            self, events: list[tuple[str, dict[str, object]]]
        ) -> FlextResult[int]:
            """Validate bulk events input and return total count.

            Returns:
                FlextResult with total event count after add, or error

            """
            if not events:
                return FlextResult[int].ok(0)  # Signal empty - caller handles

            if not FlextRuntime.is_list_like(events):
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
            self, events: list[tuple[str, dict[str, object]]]
        ) -> FlextResult[list[tuple[str, dict[str, object]]]]:
            """Validate and collect events for bulk add.

            Returns:
                FlextResult with validated events list or error

            """
            validated_events: list[tuple[str, dict[str, object]]] = []

            for i, (event_name, data) in enumerate(events):
                if not isinstance(event_name, str) or not event_name:
                    return FlextResult[list[tuple[str, dict[str, object]]]].fail(
                        f"Event {i}: name must be non-empty string",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                if data is not None and not FlextRuntime.is_dict_like(data):
                    return FlextResult[list[tuple[str, dict[str, object]]]].fail(
                        f"Event {i}: data must be dict[str, object] or None",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                # Fast fail: data must be dict (None not allowed)
                event_data: dict[str, object] = data if isinstance(data, dict) else {}
                validated_events.append((event_name, event_data))

            return FlextResult[list[tuple[str, dict[str, object]]]].ok(validated_events)

        def _add_validated_events_bulk(
            self, validated_events: list[tuple[str, dict[str, object]]]
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

                self._get_logger().info(
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
            self, events: list[tuple[str, dict[str, object]]]
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

        def validate_consistency(self) -> FlextResult[bool]:
            """Validate entity consistency using centralized validation."""
            entity_result = (
                _validation_module.FlextModelsValidation.validate_entity_relationships(
                    self
                )
            )
            if entity_result.is_failure:
                return FlextResult[bool].fail(
                    f"Entity validation failed: {entity_result.error}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            for event in self.domain_events:
                event_result = (
                    _validation_module.FlextModelsValidation.validate_domain_event(
                        event
                    )
                )
                if event_result.is_failure:
                    return FlextResult[bool].fail(
                        f"Domain event validation failed: {event_result.error}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            return FlextResult[bool].ok(True)

    class Value(FrozenStrictModel):
        """Base class for value objects - immutable and compared by value."""

        @override
        def __eq__(self, other: object) -> bool:
            """Compare by value (using FlextUtilities.Domain)."""
            return FlextUtilitiesDomain.compare_value_objects_by_value(self, other)

        @override
        def __hash__(self) -> int:
            """Hash based on values for use in sets/dicts (using FlextUtilities.Domain)."""
            return FlextUtilitiesDomain.hash_value_object_by_value(self)

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
