"""FLEXT Core Entities - Domain Layer Entity Implementation.

Domain-Driven Design (DDD) entity implementation providing identity management, version
tracking, and domain event integration across the 32-project FLEXT ecosystem. Foundation
for business entity modeling with optimistic concurrency control and event sourcing
patterns in data integration domains.

Module Role in Architecture:
    Domain Layer → Entity Modeling → Business Logic Foundation

    This module provides DDD entity patterns used throughout FLEXT projects:
    - Identity-based equality distinguishing entities from value objects
    - Version tracking for optimistic concurrency control in distributed systems
    - Domain event integration for event sourcing and CQRS patterns
    - Business logic encapsulation within entity boundaries

Entity Architecture Patterns:
    Identity Management: Unique identifier generation and identity-based equality
    Version Tracking: Automatic versioning for optimistic concurrency control
    Domain Events: Event collection for cross-aggregate communication
    Business Validation: Domain rule enforcement through validate_domain_rules

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Identity management, version tracking, Pydantic validation
    🚧 Active Development: Event sourcing integration (Priority 1 - September 2025)
    📋 TODO Integration: Complete domain event persistence (Priority 1)

Domain Entity Features:
    FlextEntity: Abstract base with identity-based equality and version tracking
    Identity Management: Unique identifier generation with collision prevention
    Version Tracking: Automatic increment on modification for concurrency control
    Domain Events: Event collection during business operations for event sourcing

Ecosystem Usage Patterns:
    # FLEXT Service Entities
    class User(FlextEntity):
        name: str
        email: str

        def validate_domain_rules(self) -> FlextResult[None]:
            if '@' not in self.email:
                return FlextResult.fail("Invalid email format")
            return FlextResult.ok(None)

    # Singer Tap/Target Entities
    class OracleTable(FlextEntity):
        schema_name: str
        table_name: str
        column_count: int

        def validate_domain_rules(self) -> FlextResult[None]:
            if self.column_count <= 0:
                return FlextResult.fail("Table must have columns")
            return FlextResult.ok(None)

    # client-a Migration Entities
    class LdapUser(FlextEntity):
        dn: str
        cn: str
        uid: str

Entity Lifecycle Management:
    - Creation: Identity assignment and initial validation
    - Modification: Version increment and domain event generation
    - Persistence: Optimistic concurrency control with version checking
    - Event Sourcing: Domain event collection for state reconstruction

Quality Standards:
    - All entities must implement validate_domain_rules for business validation
    - Identity must be immutable after initial assignment
    - Version tracking must increment on any business state change
    - Domain events must be collected during business operations

See Also:
    docs/TODO.md: Priority 1 - Event sourcing foundation implementation
    aggregate_root.py: Aggregate root patterns with transactional boundaries
    value_objects.py: Value object patterns with attribute-based equality

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from flext_core.exceptions import FlextValidationError
from flext_core.fields import FlextFields
from flext_core.flext_types import TAnyDict
from flext_core.payload import FlextEvent
from flext_core.result import FlextResult
from flext_core.utilities import FlextGenerators

if TYPE_CHECKING:
    from flext_core.flext_types import TAnyDict

# =============================================================================
# DOMAIN-SPECIFIC TYPES - Entity Pattern Specializations
# =============================================================================

# Entity pattern specific types for better domain modeling
type TEntityVersion = int  # Entity version for optimistic locking
type TEntityTimestamp = datetime  # Entity timestamp fields
type TDomainEventType = str  # Domain event type identifier
type TDomainEventData = TAnyDict  # Domain event payload data
type TAggregateId = str  # Aggregate root identifier
type TEntityRule = str  # Domain rule identifier for validation
type TEntityState = str  # Entity state for state machines
type TEntityMetadata = TAnyDict  # Entity metadata for extensions

# Factory and creation types
type TEntityDefaults = TAnyDict  # Default values for entity creation
type TEntityChanges = TAnyDict  # Changes for entity updates
type TFactoryResult[T] = FlextResult[T]  # Factory creation result

# Event sourcing types
type TDomainEvents = list[FlextEvent]  # Collection of domain events
type TEventStream = list[FlextEvent]  # Entity event stream
type TEventVersion = int  # Event version for ordering

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
        # Use shared domain entity with factory pattern


        # Create entity using SharedDomainFactory for consistency
        user_result = SharedDomainFactory.create_user(
            name="John Doe",
            email="john@example.com",
            age=30
        )

        if user_result.is_success:
            user = user_result.data
            # Entity has built-in validation and business rules
            validation = user.validate_domain_rules()

            # User entity includes standard business operations
            if user.status.value == "pending":
                activation_result = user.activate()
                if activation_result.is_success:
                    activated_user = activation_result.data
                    print(f"User {activated_user.name} activated")

        # Example of entity inheritance for custom behavior
        class CustomUser(User):
            department: str = "general"

            def validate_domain_rules(self) -> FlextResult[None]:
                # Call parent validation first
                result = super().validate_domain_rules()
                if result.is_failure:
                    return result

                # Add custom validation
                if not self.department.strip():
                    return FlextResult.fail("Department cannot be empty")
                return FlextResult.ok(None)

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
            if updated_user.version != user.version + 1:
                expected_version = user.version + 1
                raise AssertionError(
                    f"Expected {expected_version}, got {updated_user.version}"
                )

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

    version: TEntityVersion = Field(
        default=1,
        description="Version for optimistic locking",
        ge=1,
    )

    # Timestamp fields for entity lifecycle tracking
    created_at: TEntityTimestamp = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Entity creation timestamp",
    )

    # Domain events collected during operations
    domain_events: TDomainEvents = Field(default_factory=list, exclude=True)

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
        # Include all model fields in representation
        fields = []
        for name, value in self.model_dump().items():
            if name != "domain_events":  # Exclude domain events from repr
                if isinstance(value, str):
                    fields.append(f"{name}={value}")
                else:
                    fields.append(f"{name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"

    @field_validator("id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity ID field."""
        if not v or not v.strip():
            msg = "Entity ID cannot be empty"
            raise FlextValidationError(
                msg,
                validation_details={"field": "id", "value": v},
            )
        return v.strip()

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
        event_type: TDomainEventType,
        event_data: TDomainEventData,
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

    def clear_events(self) -> TEventStream:
        """Clear and return collected domain events.

        Returns:
            List of domain events

        """
        events = self.domain_events.copy()
        self.domain_events.clear()
        return events

    def validate_field(self, field_name: str, field_value: object) -> FlextResult[None]:
        """Validate a specific field using the fields system.

        Args:
            field_name: Name of the field to validate
            field_value: Value to validate

        Returns:
            Result of field validation

        """
        try:
            # Get field definition from registry
            field_result = FlextFields.get_field_by_name(field_name)
            if field_result.is_success:
                field_def = field_result.unwrap()
                validation_result = field_def.validate_value(field_value)
                if validation_result.is_success:
                    return FlextResult.ok(None)
                return FlextResult.fail(
                    validation_result.error or "Field validation failed",
                )

            # If no field definition found, return success (allow other validation)
            return FlextResult.ok(None)
        except (AttributeError, RuntimeError, ImportError) as e:
            return FlextResult.fail(f"Field validation error: {e}")

    def validate_all_fields(self) -> FlextResult[None]:
        """Validate all entity fields using the fields system.

        Automatically validates all model fields that have corresponding
        field definitions in the fields registry.

        Returns:
            Result of comprehensive field validation

        """
        errors = []

        # Get all model fields and their values
        model_data = self.model_dump()

        for field_name, field_value in model_data.items():
            # Skip internal fields
            if field_name.startswith("_") or field_name == "domain_events":
                continue

            validation_result = self.validate_field(field_name, field_value)
            if validation_result.is_failure:
                errors.append(f"{field_name}: {validation_result.error}")

        if errors:
            return FlextResult.fail(f"Field validation errors: {'; '.join(errors)}")

        return FlextResult.ok(None)

    def with_version(self, version: TEntityVersion) -> Self:
        """Create new instance with specific version number.

        Args:
            version: New version number for the entity

        Returns:
            New entity instance with updated version

        Raises:
            ValueError: If version is not greater than current version

        """
        if version <= self.version:
            msg = "New version must be greater than current version"
            raise FlextValidationError(
                msg,
                validation_details={
                    "field": "version",
                    "value": version,
                    "current_version": self.version,
                },
            )

        try:
            data = self.model_dump()
            data["version"] = version
            new_entity = self.__class__(**data)

            # Validate domain rules
            validation_result = new_entity.validate_domain_rules()
            if validation_result.is_failure:
                msg = validation_result.error or "Domain validation failed"
                raise FlextValidationError(
                    msg,
                    validation_details={"field": "domain_rules", "entity_id": self.id},
                )
            return new_entity
        except (TypeError, ValidationError) as e:
            msg = f"Failed to set version: {e}"
            raise FlextValidationError(
                msg,
                validation_details={
                    "field": "version",
                    "value": version,
                    "error": str(e),
                },
            ) from e


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
                "created_at": datetime.now(timezone.utc),
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
        REDACTED_LDAP_BIND_PASSWORD_result = user_factory(
            id="REDACTED_LDAP_BIND_PASSWORD_001",
            email="REDACTED_LDAP_BIND_PASSWORD@example.com",
            name="Admin User",
            role="REDACTED_LDAP_BIND_PASSWORD"
        )

        # Handle factory creation errors
        if REDACTED_LDAP_BIND_PASSWORD_result.is_failure:
            logger.error(f"Failed to create REDACTED_LDAP_BIND_PASSWORD: {REDACTED_LDAP_BIND_PASSWORD_result.error}")

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
                    return FlextResult.fail(
                        validation_result.error or "Domain validation failed",
                    )
                return FlextResult.ok(instance)
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                ImportError,
            ) as e:
                return FlextResult.fail(f"Entity creation failed: {e}")

        return factory


# Export API
__all__ = ["FlextEntity", "FlextEntityFactory"]
