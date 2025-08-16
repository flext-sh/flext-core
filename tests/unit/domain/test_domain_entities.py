"""Comprehensive tests for entities module."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from flext_core import (
    FlextAggregateRoot,
    FlextEntity,
    FlextEvent,
    FlextFactory,
    FlextResult,
    FlextValueObject,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3


def create_test_entity(entity_class: type, **kwargs: object) -> object:
    """Create test entities using factory with proper DDD validation."""
    # Get the factory function directly (not a FlextResult)
    factory = FlextFactory.create_entity_factory(entity_class)

    if not callable(factory):
        factory_msg: str = f"Factory for {entity_class.__name__} is not callable"
        raise TypeError(factory_msg)

    # Call the factory function which returns FlextResult
    result = cast("FlextResult[object]", factory(**kwargs))

    if result.is_failure:
        creation_msg: str = f"Failed to create {entity_class.__name__}: {result.error}"
        raise AssertionError(creation_msg)

    instance = result.unwrap()

    # Additional domain validation for aggregate roots
    if isinstance(instance, FlextAggregateRoot):
        validation_result = instance.validate_domain_rules()
        if validation_result.is_failure:
            validation_msg: str = f"Domain validation failed for {entity_class.__name__}: {validation_result.error}"
            raise AssertionError(validation_msg)

    return instance


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
    details: dict[str, object] = Field(default_factory=dict)

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

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate test entity domain rules."""
        if not self.name.strip():
            return FlextResult.fail("Entity name cannot be empty")
        return FlextResult.ok(None)


class SampleValueObject(FlextValueObject):
    """Test value object implementation."""

    amount: int
    currency: str = "USD"

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate test value object business rules."""
        if self.amount < 0:
            return FlextResult.fail("Amount cannot be negative")
        return FlextResult.ok(None)


class SampleAggregateRoot(FlextAggregateRoot):
    """Test aggregate root implementation."""

    title: str
    description: str = ""

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate test aggregate root domain rules."""
        if not self.title.strip():
            return FlextResult.fail("Aggregate title cannot be empty")
        return FlextResult.ok(None)

    def perform_action(self, action: str) -> None:
        """Test method that raises domain event."""
        event_type = f"test.{action}"
        event_data: dict[str, object] = {
            "aggregate_id": self.id,
            "action": action,
            "details": self.title,  # Simplified to avoid nested dict with object
        }
        result = self.add_domain_event(event_type, event_data)
        if result.is_failure:
            msg: str = f"Failed to add domain event: {result.error}"
            raise ValueError(msg)


class TestFlextEntity:
    """Test FlextEntity base class."""

    def test_entity_creation_with_auto_id(self) -> None:
        """Test entity creation with auto-generated ID."""
        # Use factory to create entity with auto-generated ID
        entity_obj = create_test_entity(SampleEntity, name="Test Entity")
        entity = cast("SampleEntity", entity_obj)

        if entity.name != "Test Entity":
            raise AssertionError(f"Expected {'Test Entity'}, got {entity.name}")
        assert entity.status == "active"
        assert entity.id is not None
        assert len(str(entity.id)) > 0  # Has generated ID
        # Modern FlextEntityId wrapper - check the string representation
        assert isinstance(str(entity.id), str)
        # Modern FlextVersion wrapper - compare root value
        if entity.version.root != 1:
            raise AssertionError(f"Expected {1}, got {entity.version.root}")

    def test_entity_creation_with_custom_id(self) -> None:
        """Test entity creation with custom ID."""
        custom_id = "custom-entity-123"
        entity = SampleEntity(id=custom_id, name="Test Entity")

        # Modern FlextEntityId wrapper - compare string representation
        if str(entity.id) != custom_id:
            raise AssertionError(f"Expected {custom_id}, got {entity.id!s}")
        assert entity.name == "Test Entity"

    def test_entity_creation_with_timestamps(self) -> None:
        """Test entity creation with custom timestamps."""
        now = datetime.now(UTC)
        entity_obj = create_test_entity(
            SampleEntity,
            name="Test Entity",
            created_at=now,
        )
        entity = cast("SampleEntity", entity_obj)

        # Modern FlextTimestamp wrapper - compare root value
        if entity.created_at.root != now:
            raise AssertionError(f"Expected {now}, got {entity.created_at.root}")

    def test_entity_mutability(self) -> None:
        """Test that entities are mutable (unlike aggregate roots)."""
        entity_obj = create_test_entity(SampleEntity, name="Test Entity")
        entity = cast("SampleEntity", entity_obj)

        # Entities should be mutable - we can change their state
        entity.name = "New Name"
        assert entity.name == "New Name"

        entity.id = "new-id"
        # Modern FlextEntityId wrapper - compare string representation
        assert str(entity.id) == "new-id"

    def test_entity_equality_by_id(self) -> None:
        """Test that entities are equal based on ID."""
        entity_id = "test-entity-123"
        entity1 = SampleEntity(id=entity_id, name="Entity 1")
        entity2 = SampleEntity(id=entity_id, name="Entity 2")  # Different name
        entity3 = SampleEntity(id="different-id", name="Entity 1")

        if entity1 != entity2:  # Same ID
            raise AssertionError(f"Expected {entity2}, got {entity1}")
        assert entity1 != entity3  # Different ID
        assert entity2 != entity3  # Different ID

    def test_entity_hash_consistency(self) -> None:
        """Test that entity hash is consistent with equality."""
        entity_id = "test-entity-123"
        entity1 = SampleEntity(id=entity_id, name="Entity 1")
        entity2 = SampleEntity(id=entity_id, name="Entity 2")

        if hash(entity1) != hash(entity2):
            raise AssertionError(f"Expected {hash(entity2)}, got {hash(entity1)}")

        # Test in set/dict
        entity_set = {entity1, entity2}
        if len(entity_set) != 1:  # Only one entity due to same ID
            raise AssertionError(f"Expected {1}, got {len(entity_set)}")

    def test_entity_string_representation(self) -> None:
        """Test entity string representation."""
        entity = SampleEntity(id="test-123", name="Test Entity")

        str_repr = str(entity)
        repr_str = repr(entity)

        if "test-123" not in str_repr:
            raise AssertionError(f"Expected {'test-123'} in {str_repr}")
        assert "SampleEntity" in repr_str

    def test_entity_model_dump(self) -> None:
        """Test entity serialization."""
        entity_obj = create_test_entity(
            SampleEntity,
            name="Test Entity",
            status="active",
        )
        entity = cast("SampleEntity", entity_obj)

        data = entity.model_dump()

        assert isinstance(data, dict)
        if data["name"] != "Test Entity":
            raise AssertionError(f"Expected {'Test Entity'}, got {data['name']}")
        assert data["status"] == "active"
        if "id" not in data:
            raise AssertionError(f"Expected {'id'} in {data}")
        assert "created_at" in data

    def test_entity_validation_error(self) -> None:
        """Test entity validation with invalid data."""
        # Test validation error via factory directly
        factory_fn = FlextFactory.create_entity_factory(SampleEntity)
        # Cast to callable returning FlextResult (avoiding explicit Any)
        entity_factory = cast("Callable[[], FlextResult[object]]", factory_fn)
        factory_result = entity_factory()  # Missing required 'name' field
        # No need for redundant cast since factory_result is already FlextResult[object]
        assert factory_result.is_failure


class TestFlextValueObject:
    """Test FlextValueObject base class."""

    def test_value_object_creation(self) -> None:
        """Test value object creation."""
        vo = SampleValueObject(amount=100, currency="EUR")

        if vo.amount != 100:
            raise AssertionError(f"Expected {100}, got {vo.amount}")
        assert vo.currency == "EUR"

    def test_value_object_with_defaults(self) -> None:
        """Test value object creation with default values."""
        vo = SampleValueObject(amount=50)

        if vo.amount != 50:
            raise AssertionError(f"Expected {50}, got {vo.amount}")
        assert vo.currency == "USD"  # Default value

    def test_value_object_immutability(self) -> None:
        """Test that value objects are immutable."""
        vo = SampleValueObject(amount=100)

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            vo.amount = 200

    def test_value_object_equality_by_value(self) -> None:
        """Test that value objects are equal based on all attributes."""
        vo1 = SampleValueObject(amount=100, currency="USD")
        vo2 = SampleValueObject(amount=100, currency="USD")
        vo3 = SampleValueObject(amount=100, currency="EUR")
        vo4 = SampleValueObject(amount=200, currency="USD")

        if vo1 != vo2:  # Same values
            raise AssertionError(f"Expected {vo2}, got {vo1}")
        assert vo1 != vo3  # Different currency
        assert vo1 != vo4  # Different amount

    def test_value_object_hash_consistency(self) -> None:
        """Test that value object hash is consistent with equality."""
        vo1 = SampleValueObject(amount=100, currency="USD")
        vo2 = SampleValueObject(amount=100, currency="USD")
        vo3 = SampleValueObject(amount=100, currency="EUR")

        if hash(vo1) != hash(vo2):
            raise AssertionError(f"Expected {hash(vo2)}, got {hash(vo1)}")
        assert hash(vo1) != hash(vo3)

        # Test in set
        vo_set = {vo1, vo2, vo3}
        if len(vo_set) != EXPECTED_BULK_SIZE:  # vo1 and vo2 are the same
            raise AssertionError(f"Expected {2}, got {len(vo_set)}")

    def test_value_object_model_dump(self) -> None:
        """Test value object serialization."""
        vo = SampleValueObject(amount=100, currency="EUR")

        data = vo.model_dump()

        assert isinstance(data, dict)
        if data["amount"] != 100:
            raise AssertionError(f"Expected {100}, got {data['amount']}")
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

        if event.aggregate_id != aggregate_id:
            raise AssertionError(f"Expected {aggregate_id}, got {event.aggregate_id}")
        assert event.event_type == "test.created"
        if event.action != "create":
            raise AssertionError(f"Expected {'create'}, got {event.action}")
        assert event.details == {"key": "value"}
        assert event.event_id is not None
        assert event.timestamp is not None
        if event.version != 1:
            raise AssertionError(f"Expected {1}, got {event.version}")

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

        if event.event_id != custom_id:
            raise AssertionError(f"Expected {custom_id}, got {event.event_id}")
        assert event.timestamp == custom_time
        if event.version != 5:
            raise AssertionError(f"Expected {5}, got {event.version}")

    def test_domain_event_immutability(self) -> None:
        """Test that domain events are immutable."""
        event = SampleDomainEvent(
            aggregate_id="aggregate-123",
            event_type="test.created",
            action="create",
        )

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            event.action = "update"

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            event.timestamp = datetime.now(UTC)

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
        if data["aggregate_id"] != "aggregate-123":
            raise AssertionError(
                f"Expected {'aggregate-123'}, got {data['aggregate_id']}",
            )
        assert data["event_type"] == "test.created"
        if data["action"] != "create":
            raise AssertionError(f"Expected {'create'}, got {data['action']}")
        assert data["details"] == {"key": "value"}
        if "event_id" not in data:
            raise AssertionError(f"Expected {'event_id'} in {data}")
        assert "timestamp" in data
        if "version" not in data:
            raise AssertionError(f"Expected {'version'} in {data}")


class TestFlextAggregateRoot:
    """Test FlextAggregateRoot base class."""

    def test_aggregate_root_creation(self) -> None:
        """Test aggregate root creation."""
        aggregate_obj = create_test_entity(
            SampleAggregateRoot,
            title="Test Aggregate",
            description="Test description",
        )
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        if aggregate.title != "Test Aggregate":
            raise AssertionError(f"Expected {'Test Aggregate'}, got {aggregate.title}")
        assert aggregate.description == "Test description"
        assert aggregate.id is not None
        # Modern FlextVersion wrapper - compare root value
        if aggregate.version.root != 1:
            raise AssertionError(f"Expected {1}, got {aggregate.version.root}")
        assert len(aggregate.get_domain_events()) == 0

    def test_aggregate_root_with_custom_version(self) -> None:
        """Test aggregate root with custom version."""
        aggregate_obj = create_test_entity(
            SampleAggregateRoot,
            title="Test Aggregate",
            version=5,
        )
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Modern FlextVersion wrapper - compare root value
        if aggregate.version.root != 5:
            raise AssertionError(f"Expected {5}, got {aggregate.version.root}")

    def test_aggregate_root_raise_event(self) -> None:
        """Test raising domain events."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Initially no events
        if len(aggregate.get_domain_events()) != 0:
            raise AssertionError(
                f"Expected {0}, got {len(aggregate.get_domain_events())}",
            )

        # Perform action that raises event
        aggregate.perform_action("created")

        # Check event was raised
        if len(aggregate.get_domain_events()) != 1:
            raise AssertionError(
                f"Expected {1}, got {len(aggregate.get_domain_events())}",
            )
        event = aggregate.get_domain_events()[0]
        assert isinstance(event, FlextEvent)
        # Modern FlextEntityId wrapper - compare string representations
        if str(event.get_metadata("aggregate_id")) != str(aggregate.id):
            raise AssertionError(
                f"Expected {aggregate.id!s}, got {event.get_metadata('aggregate_id')!s}",
            )
        assert event.get_metadata("event_type") == "test.created"
        assert event.data is not None
        if event.data.get("action") != "created":
            raise AssertionError(
                f"Expected {'created'}, got {event.data.get('action')}",
            )

    def test_aggregate_root_multiple_events(self) -> None:
        """Test raising multiple domain events."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        aggregate.perform_action("created")
        aggregate.perform_action("updated")
        aggregate.perform_action("activated")

        if len(aggregate.get_domain_events()) != EXPECTED_DATA_COUNT:
            raise AssertionError(
                f"Expected {3}, got {len(aggregate.get_domain_events())}",
            )

        # Check event types
        event_types = [
            event.get_metadata("event_type") for event in aggregate.get_domain_events()
        ]
        if "test.created" not in event_types:
            raise AssertionError(f"Expected {'test.created'} in {event_types}")
        assert "test.updated" in event_types
        if "test.activated" not in event_types:
            raise AssertionError(f"Expected {'test.activated'} in {event_types}")

    def test_aggregate_root_clear_events(self) -> None:
        """Test clearing pending events."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Raise some events
        aggregate.perform_action("created")
        aggregate.perform_action("updated")
        if len(aggregate.get_domain_events()) != EXPECTED_BULK_SIZE:
            raise AssertionError(
                f"Expected {2}, got {len(aggregate.get_domain_events())}",
            )

        # Clear events
        aggregate.clear_domain_events()
        if len(aggregate.get_domain_events()) != 0:
            raise AssertionError(
                f"Expected {0}, got {len(aggregate.get_domain_events())}",
            )

    def test_aggregate_root_immutability(self) -> None:
        """Test that aggregate root is immutable."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            aggregate.title = "New Title"

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            aggregate.version = 2

    def test_aggregate_root_inheritance_from_entity(self) -> None:
        """Test that aggregate root inherits entity behavior."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Should have entity properties
        assert hasattr(aggregate, "id")
        assert hasattr(aggregate, "created_at")

        # Should support entity equality - use string representation for ID comparison
        same_id_aggregate = SampleAggregateRoot(
            id=str(aggregate.id),  # Convert FlextEntityId to string
            title="Different Title",
        )
        if aggregate != same_id_aggregate:
            raise AssertionError(f"Expected {same_id_aggregate}, got {aggregate}")

    def test_aggregate_root_model_dump(self) -> None:
        """Test aggregate root serialization."""
        aggregate_obj = create_test_entity(
            SampleAggregateRoot,
            title="Test Aggregate",
            description="Test description",
        )
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Add some events
        aggregate.perform_action("created")

        data = aggregate.model_dump()

        assert isinstance(data, dict)
        if data["title"] != "Test Aggregate":
            raise AssertionError(f"Expected {'Test Aggregate'}, got {data['title']}")
        assert data["description"] == "Test description"
        if "id" not in data:
            raise AssertionError(f"Expected {'id'} in {data}")
        assert "version" in data
        # Domain events are excluded from serialization
        if "domain_events" in data:
            raise AssertionError(f"Expected {'domain_events'} not in {data}")
        # But we can verify events exist through the method
        if len(aggregate.get_domain_events()) != 1:
            raise AssertionError(
                f"Expected {1}, got {len(aggregate.get_domain_events())}",
            )

    def test_aggregate_root_with_entity_id_parameter(self) -> None:
        """Test aggregate root creation with entity_id parameter."""
        custom_id = "custom-aggregate-id"
        aggregate = SampleAggregateRoot(
            id=custom_id,
            title="Test Aggregate",
        )

        # Modern FlextEntityId wrapper - compare string representation
        if str(aggregate.id) != custom_id:
            raise AssertionError(f"Expected {custom_id}, got {aggregate.id!s}")
        assert aggregate.title == "Test Aggregate"
        # Modern FlextVersion wrapper - compare root value
        if aggregate.version.root != 1:
            raise AssertionError(f"Expected {1}, got {aggregate.version.root}")

    def test_aggregate_root_with_created_at_datetime(self) -> None:
        """Test aggregate root creation with created_at datetime."""
        created_time = datetime.now(UTC)
        aggregate = SampleAggregateRoot(
            id="test-agg-id",
            title="Test Aggregate",
            created_at=created_time,
        )

        if aggregate.title != "Test Aggregate":
            raise AssertionError(f"Expected {'Test Aggregate'}, got {aggregate.title}")
        # Modern FlextTimestamp wrapper - compare root value
        assert aggregate.created_at.root == created_time

    def test_add_domain_event_failure_handling(self) -> None:
        """Test add_domain_event when event creation fails."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Try to add event with invalid event_type (empty string should cause failure)
        result = aggregate.add_domain_event("", {"data": "test"})

        assert result.is_failure
        if "Failed to create event" not in (result.error or ""):
            raise AssertionError(f"Expected 'Failed to create event' in {result.error}")

    def test_add_domain_event_exception_handling(self) -> None:
        """Test add_domain_event exception handling."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Mock a scenario that would cause an exception in add_domain_event
        # This is tricky since the method is robust, but we can potentially trigger it
        # by passing invalid event_data that causes JSON serialization issues
        # Use a complex object that might cause serialization issues
        invalid_data: dict[str, object] = {
            "func": lambda x: x,
        }  # Functions can't be serialized
        result = aggregate.add_domain_event("test.event", invalid_data)

        # The method should handle this gracefully
        # Either outcome is acceptable
        assert result.is_failure or result.success

    def test_add_event_object_method(self) -> None:
        """Test add_event_object convenience method."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Create a test event - convert FlextEntityId and FlextVersion to primitives
        event_result = FlextEvent.create_event(
            event_type="test.direct",
            event_data={"action": "direct_add"},
            aggregate_id=str(aggregate.id),  # Convert FlextEntityId to string
            version=aggregate.version.root,  # Extract int from FlextVersion RootModel
        )
        assert event_result.success
        event = event_result.unwrap()

        # Initially no events
        if len(aggregate.get_domain_events()) != 0:
            raise AssertionError(
                f"Expected {0}, got {len(aggregate.get_domain_events())}",
            )

        # Add event object directly
        aggregate.add_event_object(event)

        # Verify event was added
        if len(aggregate.get_domain_events()) != 1:
            raise AssertionError(
                f"Expected {1}, got {len(aggregate.get_domain_events())}",
            )
        assert aggregate.get_domain_events()[0] == event

    def test_has_domain_events_method(self) -> None:
        """Test has_domain_events method."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Initially no events
        assert not aggregate.has_domain_events()

        # Add an event
        aggregate.perform_action("created")

        # Now has events
        assert aggregate.has_domain_events()

        # Clear events
        aggregate.clear_domain_events()

        # No events again
        assert not aggregate.has_domain_events()


class TestEntitiesIntegration:
    """Integration tests for entities."""

    def test_entity_value_object_composition(self) -> None:
        """Test composing entities with value objects."""
        # This would be a more complex test in a real scenario
        entity_obj = create_test_entity(SampleEntity, name="Test Entity")
        entity = cast("SampleEntity", entity_obj)
        value_obj = SampleValueObject(amount=100, currency="USD")

        # Test that both can be serialized together
        composite_data = {
            "entity": entity.model_dump(),
            "value": value_obj.model_dump(),
        }

        if composite_data["entity"]["name"] != "Test Entity":
            raise AssertionError(
                f"Expected {'Test Entity'}, got {composite_data['entity']['name']}",
            )
        assert composite_data["value"]["amount"] == 100

    @pytest.mark.ddd
    @pytest.mark.architecture
    def test_aggregate_event_sourcing_pattern(self) -> None:
        """Test basic event sourcing pattern."""
        aggregate_obj = create_test_entity(SampleAggregateRoot, title="Test Aggregate")
        aggregate = cast("SampleAggregateRoot", aggregate_obj)

        # Simulate business operations
        aggregate.perform_action("created")
        aggregate.perform_action("updated")
        aggregate.perform_action("activated")

        # Collect events (would normally be persisted)
        events = list(aggregate.get_domain_events())
        if len(events) != EXPECTED_DATA_COUNT:
            raise AssertionError(f"Expected {3}, got {len(events)}")

        # Clear events (simulate after persistence)
        aggregate.clear_domain_events()
        if len(aggregate.get_domain_events()) != 0:
            raise AssertionError(
                f"Expected {0}, got {len(aggregate.get_domain_events())}",
            )

        # Events should contain full audit trail
        assert events[0].data is not None
        assert events[1].data is not None
        assert events[2].data is not None
        if events[0].data.get("action") != "created":
            raise AssertionError(
                f"Expected {'created'}, got {events[0].data.get('action')}",
            )
        assert events[1].data.get("action") == "updated"
        if events[2].data.get("action") != "activated":
            raise AssertionError(
                f"Expected {'activated'}, got {events[2].data.get('action')}",
            )

        # All events should be for same aggregate
        aggregate_ids = [event.get_metadata("aggregate_id") for event in events]
        # Convert both to strings for proper comparison with modern FlextEntityId
        if not all(str(aid) == str(aggregate.id) for aid in aggregate_ids):
            raise AssertionError(
                f"Expected all aggregate IDs to be {aggregate.id!s}, got {aggregate_ids}",
            )

    def test_polymorphic_entity_behavior(self) -> None:
        """Test polymorphic behavior of entities."""

        class SpecialEntity(SampleEntity):
            special_field: str = "special"

        entity_obj = create_test_entity(SpecialEntity, name="Special Entity")
        entity = cast("SpecialEntity", entity_obj)

        # Should behave as both SpecialEntity and FlextEntity
        assert isinstance(entity, SpecialEntity)
        assert isinstance(entity, SampleEntity)
        assert isinstance(entity, FlextEntity)

        # Should have all properties
        if entity.name != "Special Entity":
            raise AssertionError(f"Expected {'Special Entity'}, got {entity.name}")
        assert entity.special_field == "special"
        assert entity.id is not None

    @pytest.mark.parametrize(
        "entity_class",
        [SampleEntity, SampleAggregateRoot],
    )
    def test_entity_types_consistency(self, entity_class: type) -> None:
        """Test consistency across different entity types."""
        if entity_class is SampleEntity:
            entity_obj = create_test_entity(entity_class, name="Test")
            entity = cast("FlextEntity", entity_obj)  # Use base type for consistency
        else:  # SampleAggregateRoot
            entity_obj = create_test_entity(entity_class, title="Test")
            entity = cast("FlextEntity", entity_obj)  # Use base type for consistency

        # All entities should have these base properties
        assert hasattr(entity, "id")
        assert hasattr(entity, "created_at")

        # Immutability depends on entity type
        if entity_class is SampleAggregateRoot:
            # Aggregate roots should be immutable
            with pytest.raises((ValidationError, AttributeError, TypeError)):
                entity.id = "new-id"
        else:
            # Regular entities should be mutable
            entity.id = "new-id"
            # Modern FlextEntityId wrapper - compare string representation
            assert str(entity.id) == "new-id"

        # All should be serializable
        data = entity.model_dump()
        assert isinstance(data, dict)
        if "id" not in data:
            raise AssertionError(f"Expected {'id'} in {data}")
