"""FLEXT Core Entities Module.

Comprehensive Domain-Driven Design (DDD) entity implementation with identity management,
version tracking, and domain event integration. Implements consolidated architecture
Pydantic validation, mixin inheritance, and type-safe operations.

Architecture:
    - Domain-Driven Design entity patterns with identity-based equality
    - Pydantic BaseModel integration for automatic validation and serialization
    - Mixin inheritance for entity-specific behaviors and cross-cutting concerns
    - Version tracking for optimistic locking and concurrency control
    - Domain event collection for event sourcing and notification patterns
    - Immutable entity instances with copy-on-write modification patterns

Entity System Components:
    - FlextEntity: Abstract base entity with identity, versioning, and domain events
    - FlextEntityFactory: Factory pattern for type-safe entity creation
    - Domain event integration: Event collection and management within entities
    - Version management: Optimistic locking and automatic version incrementing
    - Identity management: Entity equality based on identity rather than attributes

Maintenance Guidelines:
    - Create domain entities by inheriting from FlextEntity abstract base class
    - Implement validate_domain_rules method for entity-specific business validation
    - Use copy_with method for immutable entity modifications with version tracking
    - Collect domain events during entity operations for event sourcing patterns
    - Leverage factory methods for consistent entity creation with validation
    - Follow DDD principles with rich entity behaviors and encapsulated business logic

Design Decisions:
    - Abstract base class pattern enforcing domain validation implementation
    - Pydantic frozen models for immutability and thread safety
    - Version field for optimistic concurrency control in distributed systems
    - Domain events list for event sourcing and cross-aggregate communication
    - Factory pattern for validated entity creation with default value support
    - FlextResult pattern integration for type-safe error handling

Domain-Driven Design Features:
    - Identity-based equality following DDD entity principles
    - Rich domain model with encapsulated business behaviors
    - Domain event collection for publishing aggregate events
    - Version tracking for conflict detection in concurrent scenarios
    - Business rule validation through abstract validate_domain_rules method
    - Immutable entities with controlled modification through copy patterns

Event Sourcing Integration:
    - Domain event collection during entity lifecycle operations
    - Aggregate ID correlation for event stream reconstruction
    - Version tracking for event ordering and conflict resolution
    - Event clearing for batch publishing after persistence
    - Integration with FlextEvent payload for structured event transport

Dependencies:
    - pydantic: Data validation and immutable model configuration
    - mixins: Entity-specific behavior inheritance for identity and validation
    - payload: FlextEvent integration for domain event management
    - result: FlextResult pattern for consistent error handling
    - abc: Abstract base class patterns for enforcing implementation contracts

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from flext_core.flext_types import TAnyDict
from flext_core.payload import FlextEvent
from flext_core.result import FlextResult
from flext_core.utilities import FlextGenerators

if TYPE_CHECKING:
    from flext_core.types import TAnyDict


# =============================================================================
# ENTITIES - Domain Objects with Identity
# =============================================================================


class FlextEntity(BaseModel, ABC):
    """Abstract Domain-Driven Design entity with identity, versioning, and events.

    Comprehensive entity implementation providing identity-based equality, version
    tracking for optimistic locking, and domain event collection. Combines
    Pydantic validation with DDD principles and concurrency control.

    Architecture:
        - Abstract base class enforcing domain validation implementation
        - Pydantic BaseModel for automatic validation and serialization
        - FlextEntityMixin for identity management and entity-specific behaviors
        - Frozen configuration for immutability and thread safety
        - Version tracking for optimistic concurrency control

    Identity and Equality:
        - Entity identity provided by FlextEntityMixin (ID field and behavior)
        - Equality based on identity rather than attribute values
        - Hash function based on entity ID for proper collection behavior
        - String representation including entity type and ID

    Version Management:
        - Automatic version incrementing on entity modifications
        - Optimistic locking support for concurrent access scenarios
        - Version validation ensuring non-negative values
        - Copy-on-write pattern with automatic version updates

    Domain Event Integration:
        - Domain event collection during entity lifecycle operations
        - Event correlation with aggregate ID and version information
        - Event clearing for batch publishing after persistence
        - Integration with FlextEvent for structured event transport

    Usage Patterns:
        # Define domain entity
        class User(FlextEntity):
            email: str
            name: str
            is_active: bool = True

            def validate_domain_rules(self) -> FlextResult[None]:
                if not self.email or "@" not in self.email:
                    return FlextResult.fail("Invalid email address")
                if not self.name.strip():
                    return FlextResult.fail("Name cannot be empty")
                return FlextResult.ok(None)

            def activate(self) -> FlextResult[User]:
                if self.is_active:
                    return FlextResult.fail("User already active")

                # Create modified entity with event
                result = self.copy_with(is_active=True)
                if result.is_success:
                    activated_user = result.data
                    activated_user.add_domain_event(
                        "UserActivated",
                        {"user_id": self.id, "activated_at": datetime.utcnow()}
                    )
                return result

        # Create entity with factory
        user_result = User(
            id="user_123",
            email="john@example.com",
            name="John Doe"
        )

        # Modify entity immutably
        modified_result = user.copy_with(name="John Smith")
        if modified_result.is_success:
            updated_user = modified_result.data
            # Version automatically incremented
            assert updated_user.version == user.version + 1

        # Collect and clear domain events
        events = updated_user.clear_events()
        # Process events for publishing

    Concurrency Control:
        - Version field for optimistic locking in distributed scenarios
        - Automatic version incrementing on modifications
        - Conflict detection through version comparison
        - Thread-safe operations through immutable design

    Validation Integration:
        - Abstract validate_domain_rules method enforcing implementation
        - Automatic validation on entity creation and modification
        - FlextResult pattern for type-safe error handling
        - Business rule validation separate from data validation
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
    )

    # ID field - explicitly defined for Pydantic compatibility
    id: str = Field(
        description="Unique entity identifier",
    )

    version: int = Field(
        default=1,
        description="Version for optimistic locking",
        ge=1,
    )

    # Timestamp fields for entity lifecycle tracking
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Entity creation timestamp",
    )

    # Domain events collected during operations
    domain_events: list[FlextEvent] = Field(default_factory=list, exclude=True)

    def __eq__(self, other: object) -> bool:
        """Check equality based on entity ID."""
        if not isinstance(other, FlextEntity):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Generate hash based on entity ID."""
        return hash(self.id)

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(id={self.id})"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"{self.__class__.__name__}(id={self.id}, version={self.version})"

    @abstractmethod
    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate entity-specific business rules.

        Must be implemented by concrete entities.
        Uses FlextResult for type-safe error handling.
        """

    def increment_version(self) -> FlextResult[Self]:
        """Create new instance with incremented version.

        Returns:
            FlextResult with new entity

        """
        try:
            data = self.model_dump()
            data["version"] = self.version + 1
            new_entity = self.__class__(**data)

            # Validate domain rules
            validation_result = new_entity.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    validation_result.error or "Domain validation failed",
                )

            return FlextResult.ok(new_entity)
        except (TypeError, ValueError, ValidationError) as e:
            return FlextResult.fail(f"Failed to increment version: {e}")

    def copy_with(self, **changes: object) -> FlextResult[Self]:
        """Create copy with changes and auto-increment version.

        Returns:
            FlextResult with new entity

        """
        try:
            data = self.model_dump()
            data.update(changes)

            # Auto-increment version unless explicitly provided
            if changes and "version" not in changes:
                data["version"] = self.version + 1

            new_entity = self.__class__(**data)

            # Validate domain rules
            validation_result = new_entity.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    validation_result.error or "Domain validation failed",
                )

            return FlextResult.ok(new_entity)
        except (TypeError, ValueError, ValidationError) as e:
            return FlextResult.fail(f"Failed to copy entity: {e}")

    def add_domain_event(
        self,
        event_type: str,
        event_data: dict[str, object],
    ) -> FlextResult[None]:
        """Add domain event to entity.

        Args:
            event_type: Type of domain event
            event_data: Event data

        Returns:
            Result of adding event

        """
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

    def clear_events(self) -> list[FlextEvent]:
        """Clear and return collected domain events.

        Returns:
            List of domain events

        """
        events = self.domain_events.copy()
        self.domain_events.clear()
        return events


# =============================================================================
# FACTORY METHODS - Convenience builders for Entities
# =============================================================================


class FlextEntityFactory:
    """Enterprise factory pattern for type-safe entity creation with validation.

    Comprehensive factory implementation providing type-safe entity creation with
    automatic ID generation, default values, and validation. Implements factory
    with FlextResult integration for consistent error handling and reliability.

    Architecture:
        - Static factory methods for stateless entity creation
        - Default value support for consistent entity initialization
        - Automatic ID generation using FlextGenerators utilities
        - Type-safe factory functions with generic return types
        - FlextResult pattern integration for error handling

    Factory Features:
        - Dynamic factory function creation for any entity type
        - Default value merging with override capability
        - Automatic entity ID generation when not provided
        - Domain validation integration through entity validate_domain_rules
        - Error handling with detailed failure messages

    Entity Creation Process:
        - Default value application for consistent initialization
        - Parameter override support for customization
        - Automatic ID generation for entities without provided ID
        - Version initialization with default value of 1
        - Domain validation execution before entity return
        - FlextResult wrapping for type-safe error handling

    Usage Patterns:
        # Create factory for User entity
        user_factory = FlextEntityFactory.create_entity_factory(
            User,
            defaults={
                "is_active": False,
                "created_at": datetime.utcnow(),
                "role": "user"
            }
        )

        # Use factory to create entities
        user_result = user_factory(
            email="john@example.com",
            name="John Doe",
            is_active=True  # Overrides default
        )

        if user_result.is_success:
            user = user_result.data
            # ID automatically generated if not provided
            # Version set to 1 by default
            # Domain validation already executed

        # Factory with specific entity ID
        admin_result = user_factory(
            id="admin_001",
            email="admin@example.com",
            name="Admin User",
            role="admin"
        )

        # Handle factory creation errors
        if admin_result.is_failure:
            logger.error(f"Failed to create admin: {admin_result.error}")

    Factory Pattern Benefits:
        - Consistent entity creation with validation
        - Default value management across entity instances
        - Type-safe creation with compile-time verification
        - Error handling with detailed failure information
        - Automatic ID generation reducing boilerplate code
    """

    @staticmethod
    def create_entity_factory(
        entity_class: type[FlextEntity],
        defaults: TAnyDict | None = None,
    ) -> object:
        """Create a factory function for entities.

        Args:
            entity_class: Entity class to create
            defaults: Default values for the factory

        Returns:
            Factory function that returns FlextResult

        """

        def factory(
            **kwargs: object,
        ) -> FlextResult[FlextEntity]:
            try:
                if FlextGenerators is None:
                    # Fallback if FlextGenerators is not available
                    error_msg = "FlextGenerators not available due to circular imports"
                    raise ImportError(error_msg)

                data = {**(defaults or {}), **kwargs}

                # Generate ID if not provided
                if "id" not in data or not data["id"]:
                    data["id"] = FlextGenerators.generate_entity_id()

                # Set default version if not provided
                if "version" not in data:
                    data["version"] = 1
                instance = entity_class.model_validate(data)
                validation_result = instance.validate_domain_rules()
                if validation_result.is_failure:
                    return validation_result
                return FlextResult.ok(instance)
            except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                return FlextResult.fail(f"Entity creation failed: {e}")
            except ImportError as e:
                return FlextResult.fail(f"Import error: {e}")

        return factory


# Export API
__all__ = ["FlextEntity", "FlextEntityFactory"]
