"""FLEXT Core Aggregate Root Module.

Comprehensive Domain-Driven Design (DDD) aggregate root implementation with
transactional boundaries, domain event management, and consistency enforcement.
Implements consolidated architecture with entity inheritance and event sourcing
patterns.

Architecture:
    - Domain-Driven Design aggregate root patterns with transactional boundaries
    - Entity inheritance providing identity, versioning, and validation capabilities
    - Domain event collection and management for event sourcing patterns
    - Transactional consistency enforcement within aggregate boundaries
    - Event publishing coordination for cross-aggregate communication
    - Immutable design with controlled mutation through business operations

Aggregate Root System Components:
    - FlextAggregateRoot: Primary aggregate root with event management
    - Domain event collection: Event storage and lifecycle management
    - Transactional boundary: Consistency enforcement within aggregate
    - Event publishing: Coordination of domain events for external systems
    - Business operation methods: Encapsulated aggregate behaviors

Maintenance Guidelines:
    - Create domain aggregates by inheriting from FlextAggregateRoot
    - Implement business operations as methods on the aggregate root
    - Use add_domain_event for collecting events during business operations
    - Clear events after publishing to external systems or event store
    - Maintain aggregate invariants through validation and business rules
    - Follow DDD principles with bounded context respect

Design Decisions:
    - Inheritance from FlextEntity for identity and versioning capabilities
    - Internal events list excluded from serialization for performance
    - Event collection separate from event publishing for flexibility
    - Immutable aggregate design with controlled mutation patterns
    - Business operation encapsulation within aggregate boundaries
    - FlextResult pattern integration for type-safe error handling

Domain-Driven Design Features:
    - Transactional boundaries defining consistency scope
    - Aggregate invariant enforcement through business operations
    - Domain event collection for cross-aggregate communication
    - Business logic encapsulation within aggregate root
    - Entity coordination and lifecycle management
    - Bounded context integration and aggregate relationships

Event Sourcing Integration:
    - Domain event collection during aggregate operations
    - Event lifecycle management from creation to publishing
    - Event clearing for batch publishing coordination
    - Event object support for rich event implementations
    - Cross-aggregate communication through published events
    - Event store integration patterns for persistence

Dependencies:
    - entities: FlextEntity inheritance for identity and versioning
    - result: FlextResult pattern for consistent error handling
    - utilities: FlextGenerators for ID generation and utility functions
    - pydantic: Field configuration for event management and serialization

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flext_core.entities import FlextEntity
from flext_core.payload import FlextEvent
from flext_core.result import FlextResult
from flext_core.utilities import FlextGenerators

if TYPE_CHECKING:
    from datetime import datetime

    from flext_core.types import TAnyDict


class FlextAggregateRoot(FlextEntity):
    """DDD aggregate root with transactional boundaries and event sourcing capabilities.

    Comprehensive aggregate root implementation providing transactional consistency,
    domain event management, and business logic encapsulation. Extends FlextEntity with
    aggregate-specific capabilities including event collection, publishing coordination,
    and invariant enforcement.

    Architecture:
        - Inheritance from FlextEntity for identity, versioning, and validation
        - Domain event collection with internal events list management
        - Transactional boundary enforcement for aggregate consistency
        - Business operation encapsulation within aggregate boundaries
        - Event publishing coordination for cross-aggregate communication

    Transactional Boundaries:
        - Aggregate consistency enforcement through business operations
        - Invariant validation across all entities within aggregate
        - Atomic changes within aggregate boundary scope
        - Cross-aggregate communication through domain events
        - Business rule enforcement at aggregate level

    Domain Event Management:
        - Event collection during business operation execution
        - Event lifecycle management from creation to publishing
        - Event clearing after successful publishing to external systems
        - Support for both structured events and event objects
        - Batch event publishing coordination for performance

    Usage Patterns:
        # Define domain aggregate
        class Order(FlextAggregateRoot):
            customer_id: str
            items: list[OrderItem]
            status: str = "DRAFT"
            total_amount: float = 0.0

            def validate_domain_rules(self) -> FlextResult[None]:
                if not self.items:
                    return FlextResult.fail("Order must have items")
                if self.total_amount < 0:
                    return FlextResult.fail("Total amount cannot be negative")
                return FlextResult.ok(None)

            def add_item(self, item: OrderItem) -> FlextResult[Order]:
                if self.status != "DRAFT":
                    return FlextResult.fail("Cannot modify confirmed order")

                # Create new aggregate with added item
                new_items = self.items + [item]
                new_total = self.total_amount + item.price

                result = self.copy_with(
                    items=new_items,
                    total_amount=new_total
                )

                if result.is_success:
                    updated_order = result.data
                    # Add domain event
                    updated_order.add_domain_event(
                        "OrderItemAdded",
                        {
                            "order_id": self.id,
                            "item_id": item.id,
                            "new_total": new_total
                        }
                    )
                    return FlextResult.ok(updated_order)

                return result

        # Create and use aggregate
        order = Order(
            customer_id="cust_123",
            items=[],
            status="DRAFT"
        )

        # Execute business operations
        item = OrderItem(id="item_456", price=29.99)
        updated_result = order.add_item(item)

        if updated_result.is_success:
            updated_order = updated_result.data

            # Collect events for publishing
            events = updated_order.get_domain_events()
            # Publish events to event store/message bus

            # Clear events after publishing
            updated_order.clear_domain_events()

    Event Publishing Integration:
        - Event collection during aggregate operations
        - Batch publishing for performance optimization
        - Event clearing after successful publishing
        - Event store integration for persistence
        - Message bus integration for cross-aggregate communication

    Consistency Management:
        - Aggregate invariant enforcement through validation
        - Business rule validation during operations
        - Version tracking for optimistic concurrency control
        - Atomic changes within aggregate boundaries
        - Cross-aggregate eventual consistency through events
    """

    # Domain events inherited from FlextEntity: list[FlextEvent]

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

        # Pydantic handles initialization with explicit arguments and remaining data
        init_data = {
            "id": actual_id,
            "version": version,
            "domain_events": domain_events,
            **entity_data,  # Pass remaining entity-specific fields
        }

        if created_at is not None:
            init_data["created_at"] = created_at

        # Use type: ignore for complex init data passing to Pydantic BaseModel
        super().__init__(**init_data)  # type: ignore[arg-type]

    def add_domain_event(
        self,
        event_type: str,
        event_data: TAnyDict,
    ) -> FlextResult[None]:
        """Add domain event to be published after persistence.

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

            self.domain_events.append(event_result.unwrap())
            return FlextResult.ok(None)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(f"Failed to add domain event: {e}")

    def add_event_object(self, event: FlextEvent) -> None:
        """Add domain event object directly (convenience method).

        Args:
            event: Domain event object to add

        """
        self.domain_events.append(event)

    def get_domain_events(self) -> list[FlextEvent]:
        """Get all unpublished domain events.

        Returns:
            List of domain events

        """
        return list(self.domain_events)

    def clear_domain_events(self) -> None:
        """Clear all domain events after publishing."""
        self.domain_events.clear()

    def has_domain_events(self) -> bool:
        """Check if aggregate has unpublished events."""
        return bool(self.domain_events)


# Export API
__all__ = ["FlextAggregateRoot"]
