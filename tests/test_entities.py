"""Comprehensive tests for entities module."""

from __future__ import annotations

import time
from datetime import UTC
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator

from flext_core.domain import FlextAggregateRoot
from flext_core.domain import FlextEntity
from flext_core.domain import FlextValueObject


class SampleDomainEvent(BaseModel):
    """Simple domain event for testing with Pydantic validation."""

    model_config = ConfigDict(frozen=True)  # Make immutable

    aggregate_id: str = Field(
        ...,
        min_length=1,
        description="Aggregate identifier",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description="Event type identifier",
    )
    event_id: str = Field(default_factory=lambda: str(time.time()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = Field(default=1, ge=1)
    action: str = Field(default="")
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("aggregate_id")
    @classmethod
    def validate_aggregate_id(cls, v: str) -> str:
        """Validate aggregate_id is not empty."""
        if not v.strip():
            msg = "Aggregate ID cannot be empty"
            raise ValueError(msg)
        return v

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event_type is not empty."""
        if not v.strip():
            msg = "Event type cannot be empty"
            raise ValueError(msg)
        return v


class SampleEntity(FlextEntity):
    """Test entity implementation."""

    name: str
    status: str = "active"

    def __init__(self, **data: Any) -> None:
        """Initialize entity with provided data."""
        super().__init__(**data)

    def validate_domain_rules(self) -> None:
        """Validate test entity domain rules."""
        if not self.name.strip():
            msg = "Entity name cannot be empty"
            raise ValueError(msg)


class SampleValueObject(FlextValueObject):
    """Test value object implementation."""

    amount: int
    currency: str = "USD"

    def validate_domain_rules(self) -> None:
        """Validate test value object domain rules."""
        if self.amount < 0:
            msg = "Amount cannot be negative"
            raise ValueError(msg)


class SampleAggregateRoot(FlextAggregateRoot):
    """Test aggregate root implementation."""

    title: str
    description: str = ""

    def validate_domain_rules(self) -> None:
        """Validate test aggregate root domain rules."""
        if not self.title.strip():
            msg = "Aggregate title cannot be empty"
            raise ValueError(msg)

    def perform_action(self, action: str) -> None:
        """Test method that raises domain event."""
        event = SampleDomainEvent(
            aggregate_id=self.id,
            event_type=f"test.{action}",
            action=action,
            details={"title": self.title},
        )
        self.add_domain_event(event)


class TestFlextEntity:
    """Test FlextEntity base class."""

    def test_entity_creation_with_auto_id(self) -> None:
        """Test entity creation with auto-generated ID."""
        entity = SampleEntity(name="Test Entity")

        assert entity.name == "Test Entity"
        assert entity.status == "active"
        assert entity.id is not None
        assert len(entity.id) == 36  # UUID length
        assert entity.created_at is not None

    def test_entity_creation_with_custom_id(self) -> None:
        """Test entity creation with custom ID."""
        custom_id = "custom-entity-123"
        entity = SampleEntity(id=custom_id, name="Test Entity")

        assert entity.id == custom_id
        assert entity.name == "Test Entity"

    def test_entity_creation_with_timestamps(self) -> None:
        """Test entity creation with custom timestamps."""
        now = datetime.now(UTC)
        entity = SampleEntity(
            name="Test Entity",
            created_at=now,
        )

        assert entity.created_at == now

    def test_entity_immutability(self) -> None:
        """Test that entities are immutable."""
        entity = SampleEntity(name="Test Entity")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            entity.name = "New Name"  # type: ignore[misc]

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            entity.id = "new-id"  # type: ignore[misc]

    def test_entity_equality_by_id(self) -> None:
        """Test that entities are equal based on ID."""
        entity_id = "test-entity-123"
        entity1 = SampleEntity(id=entity_id, name="Entity 1")
        entity2 = SampleEntity(id=entity_id, name="Entity 2")  # Different name
        entity3 = SampleEntity(id="different-id", name="Entity 1")

        assert entity1 == entity2  # Same ID
        assert entity1 != entity3  # Different ID
        assert entity2 != entity3  # Different ID

    def test_entity_hash_consistency(self) -> None:
        """Test that entity hash is consistent with equality."""
        entity_id = "test-entity-123"
        entity1 = SampleEntity(id=entity_id, name="Entity 1")
        entity2 = SampleEntity(id=entity_id, name="Entity 2")

        assert hash(entity1) == hash(entity2)

        # Test in set/dict
        entity_set = {entity1, entity2}
        assert len(entity_set) == 1  # Only one entity due to same ID

    def test_entity_string_representation(self) -> None:
        """Test entity string representation."""
        entity = SampleEntity(id="test-123", name="Test Entity")

        str_repr = str(entity)
        repr_str = repr(entity)

        assert "test-123" in str_repr
        assert "SampleEntity" in repr_str

    def test_entity_model_dump(self) -> None:
        """Test entity serialization."""
        entity = SampleEntity(name="Test Entity", status="active")

        data = entity.model_dump()

        assert isinstance(data, dict)
        assert data["name"] == "Test Entity"
        assert data["status"] == "active"
        assert "id" in data
        assert "created_at" in data

    def test_entity_validation_error(self) -> None:
        """Test entity validation with invalid data."""
        with pytest.raises(ValidationError):
            SampleEntity()  # Missing required 'name' field


class TestFlextValueObject:
    """Test FlextValueObject base class."""

    def test_value_object_creation(self) -> None:
        """Test value object creation."""
        vo = SampleValueObject(amount=100, currency="EUR")

        assert vo.amount == 100
        assert vo.currency == "EUR"

    def test_value_object_with_defaults(self) -> None:
        """Test value object creation with default values."""
        vo = SampleValueObject(amount=50)

        assert vo.amount == 50
        assert vo.currency == "USD"  # Default value

    def test_value_object_immutability(self) -> None:
        """Test that value objects are immutable."""
        vo = SampleValueObject(amount=100)

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            vo.amount = 200  # type: ignore[misc]

    def test_value_object_equality_by_value(self) -> None:
        """Test that value objects are equal based on all attributes."""
        vo1 = SampleValueObject(amount=100, currency="USD")
        vo2 = SampleValueObject(amount=100, currency="USD")
        vo3 = SampleValueObject(amount=100, currency="EUR")
        vo4 = SampleValueObject(amount=200, currency="USD")

        assert vo1 == vo2  # Same values
        assert vo1 != vo3  # Different currency
        assert vo1 != vo4  # Different amount

    def test_value_object_hash_consistency(self) -> None:
        """Test that value object hash is consistent with equality."""
        vo1 = SampleValueObject(amount=100, currency="USD")
        vo2 = SampleValueObject(amount=100, currency="USD")
        vo3 = SampleValueObject(amount=100, currency="EUR")

        assert hash(vo1) == hash(vo2)
        assert hash(vo1) != hash(vo3)

        # Test in set
        vo_set = {vo1, vo2, vo3}
        assert len(vo_set) == 2  # vo1 and vo2 are the same

    def test_value_object_model_dump(self) -> None:
        """Test value object serialization."""
        vo = SampleValueObject(amount=100, currency="EUR")

        data = vo.model_dump()

        assert isinstance(data, dict)
        assert data["amount"] == 100
        assert data["currency"] == "EUR"


class TestFlextDomainEvent:
    """Test FlextDomainEvent base class."""

    def test_domain_event_creation(self) -> None:
        """Test domain event creation."""
        aggregate_id = "aggregate-123"
        event = SampleDomainEvent(
            aggregate_id=aggregate_id,
            event_type="test.created",
            action="create",
            details={"key": "value"},
        )

        assert event.aggregate_id == aggregate_id
        assert event.event_type == "test.created"
        assert event.action == "create"
        assert event.details == {"key": "value"}
        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.version == 1

    def test_domain_event_with_custom_values(self) -> None:
        """Test domain event with custom ID and timestamp."""
        custom_id = "event-456"
        custom_time = datetime.now(UTC)

        event = SampleDomainEvent(
            aggregate_id="aggregate-789",
            event_type="test.updated",
            event_id=custom_id,
            timestamp=custom_time,
            version=5,
            action="update",
        )

        assert event.event_id == custom_id
        assert event.timestamp == custom_time
        assert event.version == 5

    def test_domain_event_immutability(self) -> None:
        """Test that domain events are immutable."""
        event = SampleDomainEvent(
            aggregate_id="aggregate-123",
            event_type="test.created",
            action="create",
        )

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            event.action = "update"  # type: ignore[misc]

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            event.timestamp = datetime.now(UTC)  # type: ignore[misc]

    def test_domain_event_validation(self) -> None:
        """Test domain event validation."""
        with pytest.raises(ValidationError):
            SampleDomainEvent(
                aggregate_id="",  # Empty aggregate ID
                event_type="test.created",
                action="create",
            )

        with pytest.raises(ValidationError):
            SampleDomainEvent(
                aggregate_id="aggregate-123",
                event_type="",  # Empty event type
                action="create",
            )

    def test_domain_event_ordering(self) -> None:
        """Test domain event ordering by timestamp."""
        event1 = SampleDomainEvent(
            aggregate_id="aggregate-123",
            event_type="test.first",
            action="first",
        )

        # Create second event with later timestamp

        time.sleep(0.001)  # Ensure different timestamp

        event2 = SampleDomainEvent(
            aggregate_id="aggregate-123",
            event_type="test.second",
            action="second",
        )

        assert event1.timestamp < event2.timestamp

    def test_domain_event_model_dump(self) -> None:
        """Test domain event serialization."""
        event = SampleDomainEvent(
            aggregate_id="aggregate-123",
            event_type="test.created",
            action="create",
            details={"key": "value"},
        )

        data = event.model_dump()

        assert isinstance(data, dict)
        assert data["aggregate_id"] == "aggregate-123"
        assert data["event_type"] == "test.created"
        assert data["action"] == "create"
        assert data["details"] == {"key": "value"}
        assert "event_id" in data
        assert "timestamp" in data
        assert "version" in data


class TestFlextAggregateRoot:
    """Test FlextAggregateRoot base class."""

    def test_aggregate_root_creation(self) -> None:
        """Test aggregate root creation."""
        aggregate = SampleAggregateRoot(
            title="Test Aggregate",
            description="Test description",
        )

        assert aggregate.title == "Test Aggregate"
        assert aggregate.description == "Test description"
        assert aggregate.id is not None
        assert aggregate.version == 1
        assert len(aggregate.get_domain_events()) == 0

    def test_aggregate_root_with_custom_version(self) -> None:
        """Test aggregate root with custom version."""
        aggregate = SampleAggregateRoot(
            title="Test Aggregate",
            version=5,
        )

        assert aggregate.version == 5

    def test_aggregate_root_raise_event(self) -> None:
        """Test raising domain events."""
        aggregate = SampleAggregateRoot(title="Test Aggregate")

        # Initially no events
        assert len(aggregate.get_domain_events()) == 0

        # Perform action that raises event
        aggregate.perform_action("created")

        # Check event was raised
        assert len(aggregate.get_domain_events()) == 1
        event = aggregate.get_domain_events()[0]
        assert isinstance(event, SampleDomainEvent)
        assert event.aggregate_id == aggregate.id
        assert event.event_type == "test.created"
        assert event.action == "created"

    def test_aggregate_root_multiple_events(self) -> None:
        """Test raising multiple domain events."""
        aggregate = SampleAggregateRoot(title="Test Aggregate")

        aggregate.perform_action("created")
        aggregate.perform_action("updated")
        aggregate.perform_action("activated")

        assert len(aggregate.get_domain_events()) == 3

        # Check event types
        event_types = [
            event.event_type for event in aggregate.get_domain_events()
        ]
        assert "test.created" in event_types
        assert "test.updated" in event_types
        assert "test.activated" in event_types

    def test_aggregate_root_clear_events(self) -> None:
        """Test clearing pending events."""
        aggregate = SampleAggregateRoot(title="Test Aggregate")

        # Raise some events
        aggregate.perform_action("created")
        aggregate.perform_action("updated")
        assert len(aggregate.get_domain_events()) == 2

        # Clear events
        aggregate.clear_domain_events()
        assert len(aggregate.get_domain_events()) == 0

    def test_aggregate_root_immutability(self) -> None:
        """Test that aggregate root is immutable."""
        aggregate = SampleAggregateRoot(title="Test Aggregate")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            aggregate.title = "New Title"  # type: ignore[misc]

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            aggregate.version = 2  # type: ignore[misc]

    def test_aggregate_root_inheritance_from_entity(self) -> None:
        """Test that aggregate root inherits entity behavior."""
        aggregate = SampleAggregateRoot(title="Test Aggregate")

        # Should have entity properties
        assert hasattr(aggregate, "id")
        assert hasattr(aggregate, "created_at")

        # Should support entity equality
        same_id_aggregate = SampleAggregateRoot(
            id=aggregate.id,
            title="Different Title",
        )
        assert aggregate == same_id_aggregate

    def test_aggregate_root_model_dump(self) -> None:
        """Test aggregate root serialization."""
        aggregate = SampleAggregateRoot(
            title="Test Aggregate",
            description="Test description",
        )

        # Add some events
        aggregate.perform_action("created")

        data = aggregate.model_dump()

        assert isinstance(data, dict)
        assert data["title"] == "Test Aggregate"
        assert data["description"] == "Test description"
        assert "id" in data
        assert "version" in data
        # Domain events are excluded from serialization
        assert "domain_events" not in data
        # But we can verify events exist through the method
        assert len(aggregate.get_domain_events()) == 1


class TestEntitiesIntegration:
    """Integration tests for entities."""

    def test_entity_value_object_composition(self) -> None:
        """Test composing entities with value objects."""
        # This would be a more complex test in a real scenario
        entity = SampleEntity(name="Test Entity")
        value_obj = SampleValueObject(amount=100, currency="USD")

        # Test that both can be serialized together
        composite_data = {
            "entity": entity.model_dump(),
            "value": value_obj.model_dump(),
        }

        assert composite_data["entity"]["name"] == "Test Entity"
        assert composite_data["value"]["amount"] == 100

    def test_aggregate_event_sourcing_pattern(self) -> None:
        """Test basic event sourcing pattern."""
        aggregate = SampleAggregateRoot(title="Test Aggregate")

        # Simulate business operations
        aggregate.perform_action("created")
        aggregate.perform_action("updated")
        aggregate.perform_action("activated")

        # Collect events (would normally be persisted)
        events = list(aggregate.get_domain_events())
        assert len(events) == 3

        # Clear events (simulate after persistence)
        aggregate.clear_domain_events()
        assert len(aggregate.get_domain_events()) == 0

        # Events should contain full audit trail
        assert events[0].action == "created"
        assert events[1].action == "updated"
        assert events[2].action == "activated"

        # All events should be for same aggregate
        aggregate_ids = [event.aggregate_id for event in events]
        assert all(aid == aggregate.id for aid in aggregate_ids)

    def test_polymorphic_entity_behavior(self) -> None:
        """Test polymorphic behavior of entities."""

        class SpecialEntity(SampleEntity):
            special_field: str = "special"

        entity = SpecialEntity(name="Special Entity")

        # Should behave as both SpecialEntity and FlextEntity
        assert isinstance(entity, SpecialEntity)
        assert isinstance(entity, SampleEntity)
        assert isinstance(entity, FlextEntity)

        # Should have all properties
        assert entity.name == "Special Entity"
        assert entity.special_field == "special"
        assert entity.id is not None

    @pytest.mark.parametrize(
        "entity_class",
        [SampleEntity, SampleAggregateRoot],
    )
    def test_entity_types_consistency(self, entity_class: type) -> None:
        """Test consistency across different entity types."""
        if entity_class is SampleEntity:
            entity = entity_class(name="Test")
        else:  # SampleAggregateRoot
            entity = entity_class(title="Test")

        # All entities should have these base properties
        assert hasattr(entity, "id")
        assert hasattr(entity, "created_at")

        # All should be immutable
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            entity.id = "new-id"

        # All should be serializable
        data = entity.model_dump()
        assert isinstance(data, dict)
        assert "id" in data
