"""Comprehensive tests for pydantic_base.py - 100% coverage requirement.

This module provides complete test coverage for all base classes in pydantic_base.py,
ensuring enterprise-grade quality and adherence to PEP standards.

Test Coverage:
- DomainBaseModel: 5 test scenarios
- DomainValueObject: 6 test scenarios
- DomainEntity: 8 test scenarios
- DomainAggregateRoot: 10 test scenarios
- DomainCommand: 5 test scenarios
- DomainQuery: 5 test scenarios
- DomainEvent: 7 test scenarios
- DomainSpecification + compositions: 12 test scenarios

Total: 58 comprehensive test scenarios for 100% coverage
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from flx_core.domain.pydantic_base import (
    AndSpecification,
    DomainAggregateRoot,
    DomainBaseModel,
    DomainCommand,
    DomainEntity,
    DomainEvent,
    DomainQuery,
    DomainSpecification,
    DomainValueObject,
    NotSpecification,
    OrSpecification,
)
from pydantic import ValidationError


class TestDomainBaseModel:
    """Test suite for DomainBaseModel - 5 scenarios."""

    def test_model_creation_with_defaults(self) -> None:
        """Test creating DomainBaseModel with default configuration."""

        class TestModel(DomainBaseModel):
            name: str
            value: int = 42

        model = TestModel(name="test")
        assert model.name == "test"
        assert model.value == 42

    def test_model_validation_assignment(self) -> None:
        """Test validate_assignment configuration works."""

        class TestModel(DomainBaseModel):
            name: str

        model = TestModel(name="test")
        model.name = "updated"
        assert model.name == "updated"

    def test_model_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden per configuration."""

        class TestModel(DomainBaseModel):
            name: str

        with pytest.raises(ValidationError):
            TestModel(name="test", extra_field="not_allowed")

    def test_model_dump_json_safe(self) -> None:
        """Test JSON-safe serialization method."""

        class TestModel(DomainBaseModel):
            name: str
            timestamp: datetime
            optional_field: str | None = None

        model = TestModel(
            name="test",
            timestamp=datetime.now(UTC),
        )

        json_data = model.model_dump_json_safe()
        assert isinstance(json_data, dict)
        assert json_data["name"] == "test"
        assert "timestamp" in json_data
        assert "optional_field" not in json_data  # Excluded due to None

    def test_model_str_strip_whitespace(self) -> None:
        """Test string whitespace stripping configuration."""

        class TestModel(DomainBaseModel):
            name: str

        model = TestModel(name="  test  ")
        assert model.name == "test"


class TestDomainValueObject:
    """Test suite for DomainValueObject - 6 scenarios."""

    def test_value_object_immutability(self) -> None:
        """Test that value objects are immutable (frozen)."""

        class TestValueObject(DomainValueObject):
            value: int
            name: str

        vo = TestValueObject(value=42, name="test")

        with pytest.raises(ValidationError):
            vo.value = 100  # Should fail due to frozen=True

    def test_value_object_equality_by_value(self) -> None:
        """Test value-based equality for value objects."""

        class TestValueObject(DomainValueObject):
            value: int
            name: str

        vo1 = TestValueObject(value=42, name="test")
        vo2 = TestValueObject(value=42, name="test")
        vo3 = TestValueObject(value=43, name="test")

        assert vo1 == vo2  # Same values
        assert vo1 != vo3  # Different values
        assert vo1 != "not_a_value_object"  # Different type

    def test_value_object_hashing(self) -> None:
        """Test value-based hashing for value objects."""

        class TestValueObject(DomainValueObject):
            value: int
            name: str

        vo1 = TestValueObject(value=42, name="test")
        vo2 = TestValueObject(value=42, name="test")

        assert hash(vo1) == hash(vo2)  # Same hash for same values

        # Test in set/dict usage
        vo_set = {vo1, vo2}
        assert len(vo_set) == 1  # Should deduplicate

    def test_value_object_validation(self) -> None:
        """Test Pydantic validation in value objects."""

        class TestValueObject(DomainValueObject):
            positive_number: int

            @pytest.fixture
            def validate_positive(cls, v: int) -> int:
                if v <= 0:
                    raise ValueError("Must be positive")
                return v

        vo = TestValueObject(positive_number=42)
        assert vo.positive_number == 42

    def test_value_object_serialization(self) -> None:
        """Test value object serialization."""

        class TestValueObject(DomainValueObject):
            value: int
            name: str

        vo = TestValueObject(value=42, name="test")
        json_data = vo.model_dump_json_safe()

        assert json_data["value"] == 42
        assert json_data["name"] == "test"

    def test_value_object_extra_forbidden(self) -> None:
        """Test that extra fields are forbidden in value objects."""

        class TestValueObject(DomainValueObject):
            value: int

        with pytest.raises(ValidationError):
            TestValueObject(value=42, extra_field="not_allowed")


class TestDomainEntity:
    """Test suite for DomainEntity - 8 scenarios."""

    def test_entity_auto_generated_fields(self) -> None:
        """Test entity auto-generated fields (id, created_at, version)."""

        class TestEntity(DomainEntity):
            name: str

        entity = TestEntity(name="test")

        assert isinstance(entity.id, UUID)
        assert isinstance(entity.created_at, datetime)
        assert entity.created_at.tzinfo == UTC
        assert entity.version == 1
        assert entity.updated_at is None

    def test_entity_identity_based_equality(self) -> None:
        """Test identity-based equality for entities."""

        class TestEntity(DomainEntity):
            name: str

        entity1 = TestEntity(name="test1")
        entity2 = TestEntity(name="test2")
        entity3 = TestEntity(id=entity1.id, name="different_name")

        assert entity1 != entity2  # Different IDs
        assert entity1 == entity3  # Same ID (identity-based)
        assert entity1 != "not_an_entity"  # Different type

    def test_entity_identity_based_hashing(self) -> None:
        """Test identity-based hashing for entities."""

        class TestEntity(DomainEntity):
            name: str

        entity1 = TestEntity(name="test1")
        entity2 = TestEntity(id=entity1.id, name="different_name")

        assert hash(entity1) == hash(entity2)  # Same ID = same hash

    def test_entity_updated_at_automatic_setting(self) -> None:
        """Test automatic updated_at timestamp setting."""

        class TestEntity(DomainEntity):
            name: str

        # Create entity without updated_at
        entity_data = {"name": "test"}
        entity = TestEntity.model_validate(entity_data)

        assert entity.updated_at is not None
        assert isinstance(entity.updated_at, datetime)
        assert entity.updated_at.tzinfo == UTC

    def test_entity_custom_id(self) -> None:
        """Test entity with custom ID."""

        class TestEntity(DomainEntity):
            name: str

        custom_id = uuid4()
        entity = TestEntity(id=custom_id, name="test")

        assert entity.id == custom_id

    def test_entity_custom_timestamps(self) -> None:
        """Test entity with custom timestamps."""

        class TestEntity(DomainEntity):
            name: str

        custom_created = datetime.now(UTC)
        custom_updated = datetime.now(UTC)

        entity = TestEntity(
            name="test",
            created_at=custom_created,
            updated_at=custom_updated,
        )

        assert entity.created_at == custom_created
        assert entity.updated_at == custom_updated

    def test_entity_version_field(self) -> None:
        """Test entity version field for optimistic locking."""

        class TestEntity(DomainEntity):
            name: str

        entity = TestEntity(name="test", version=5)
        assert entity.version == 5

    def test_entity_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden in entities."""

        class TestEntity(DomainEntity):
            name: str

        with pytest.raises(ValidationError):
            TestEntity(name="test", extra_field="not_allowed")


class TestDomainAggregateRoot:
    """Test suite for DomainAggregateRoot - 10 scenarios."""

    def test_aggregate_root_inherits_entity(self) -> None:
        """Test that aggregate root inherits all entity functionality."""

        class TestAggregate(DomainAggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Entity functionality
        assert isinstance(aggregate.id, UUID)
        assert isinstance(aggregate.created_at, datetime)
        assert aggregate.version == 1

        # Aggregate-specific functionality
        assert aggregate.aggregate_version == 1
        assert isinstance(aggregate.domain_events_list, list)
        assert len(aggregate.domain_events_list) == 0

    def test_aggregate_root_domain_events_list_excluded(self) -> None:
        """Test that domain_events_list is excluded from serialization."""

        class TestAggregate(DomainAggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")
        json_data = aggregate.model_dump()

        assert "domain_events_list" not in json_data
        assert "name" in json_data

    def test_aggregate_add_domain_event(self) -> None:
        """Test adding domain events to aggregate."""

        class TestAggregate(DomainAggregateRoot):
            name: str

        class TestEvent(DomainEvent):
            aggregate_id: UUID
            event_type: str = "test_event"

        aggregate = TestAggregate(name="test")
        event = TestEvent(aggregate_id=aggregate.id)

        initial_version = aggregate.aggregate_version
        aggregate.add_domain_event(event)

        assert len(aggregate.domain_events_list) == 1
        assert aggregate.domain_events_list[0] == event
        assert aggregate.aggregate_version == initial_version + 1

    def test_aggregate_clear_domain_events(self) -> None:
        """Test clearing domain events and returning them."""

        class TestAggregate(DomainAggregateRoot):
            name: str

        class TestEvent(DomainEvent):
            aggregate_id: UUID
            event_type: str = "test_event"

        aggregate = TestAggregate(name="test")
        event1 = TestEvent(aggregate_id=aggregate.id)
        event2 = TestEvent(aggregate_id=aggregate.id)

        aggregate.add_domain_event(event1)
        aggregate.add_domain_event(event2)

        events = aggregate.clear_domain_events()

        assert len(events) == 2
        assert event1 in events
        assert event2 in events
        assert len(aggregate.domain_events_list) == 0

    def test_aggregate_domain_events_property(self) -> None:
        """Test domain_events property returns copy."""

        class TestAggregate(DomainAggregateRoot):
            name: str

        class TestEvent(DomainEvent):
            aggregate_id: UUID
            event_type: str = "test_event"

        aggregate = TestAggregate(name="test")
        event = TestEvent(aggregate_id=aggregate.id)
        aggregate.add_domain_event(event)

        events_copy = aggregate.domain_events
        assert len(events_copy) == 1
        assert events_copy[0] == event

        # Modifying copy shouldn't affect original
        events_copy.clear()
        assert len(aggregate.domain_events_list) == 1

    def test_aggregate_multiple_events(self) -> None:
        """Test handling multiple domain events."""

        class TestAggregate(DomainAggregateRoot):
            name: str

        class TestEvent(DomainEvent):
            aggregate_id: UUID
            event_type: str = "test_event"
            event_data: dict[str, Any] = {}

        aggregate = TestAggregate(name="test")

        for i in range(5):
            event = TestEvent(
                aggregate_id=aggregate.id,
                event_data={"sequence": i},
            )
            aggregate.add_domain_event(event)

        assert len(aggregate.domain_events) == 5
        assert aggregate.aggregate_version == 6  # Started at 1, added 5

    def test_aggregate_version_tracking(self) -> None:
        """Test aggregate version tracking with events."""

        class TestAggregate(DomainAggregateRoot):
            name: str

        class TestEvent(DomainEvent):
            aggregate_id: UUID
            event_type: str = "test_event"

        aggregate = TestAggregate(name="test", aggregate_version=10)
        event = TestEvent(aggregate_id=aggregate.id)

        aggregate.add_domain_event(event)
        assert aggregate.aggregate_version == 11

    def test_aggregate_arbitrary_types_allowed(self) -> None:
        """Test that arbitrary types are allowed in aggregate configuration."""

        class CustomType:
            def __init__(self, value: str) -> None:
                self.value = value

        class TestAggregate(DomainAggregateRoot):
            name: str
            custom: CustomType

        custom_obj = CustomType("test")
        aggregate = TestAggregate(name="test", custom=custom_obj)

        assert aggregate.custom.value == "test"

    def test_aggregate_event_sourcing_workflow(self) -> None:
        """Test complete event sourcing workflow."""

        class TestAggregate(DomainAggregateRoot):
            name: str
            counter: int = 0

        class CounterIncrementedEvent(DomainEvent):
            aggregate_id: UUID
            event_type: str = "counter_incremented"
            event_data: dict[str, Any] = {}

        # Create aggregate
        aggregate = TestAggregate(name="test")
        initial_version = aggregate.aggregate_version

        # Business operation that generates events
        aggregate.counter += 1
        event = CounterIncrementedEvent(
            aggregate_id=aggregate.id,
            event_data={"new_value": aggregate.counter},
        )
        aggregate.add_domain_event(event)

        # Verify state
        assert aggregate.counter == 1
        assert aggregate.aggregate_version == initial_version + 1
        assert len(aggregate.domain_events) == 1

        # Simulate event processing
        events = aggregate.clear_domain_events()
        assert len(events) == 1
        assert events[0].event_data["new_value"] == 1
        assert len(aggregate.domain_events) == 0

    def test_aggregate_validation_assignment(self) -> None:
        """Test that validate_assignment works for aggregates."""

        class TestAggregate(DomainAggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")
        aggregate.name = "updated"
        assert aggregate.name == "updated"


class TestDomainCommand:
    """Test suite for DomainCommand - 5 scenarios."""

    def test_command_auto_generated_fields(self) -> None:
        """Test command auto-generated fields."""

        class TestCommand(DomainCommand):
            action: str

        command = TestCommand(action="test_action")

        assert isinstance(command.command_id, UUID)
        assert isinstance(command.issued_at, datetime)
        assert command.issued_at.tzinfo == UTC
        assert command.correlation_id is None
        assert command.issued_by is None
        assert isinstance(command.metadata, dict)
        assert len(command.metadata) == 0

    def test_command_with_optional_fields(self) -> None:
        """Test command with all optional fields provided."""

        class TestCommand(DomainCommand):
            action: str

        correlation_id = uuid4()
        metadata = {"source": "test", "version": 1}

        command = TestCommand(
            action="test_action",
            correlation_id=correlation_id,
            issued_by="test_user",
            metadata=metadata,
        )

        assert command.correlation_id == correlation_id
        assert command.issued_by == "test_user"
        assert command.metadata == metadata

    def test_command_custom_command_id(self) -> None:
        """Test command with custom command ID."""

        class TestCommand(DomainCommand):
            action: str

        custom_id = uuid4()
        command = TestCommand(action="test_action", command_id=custom_id)

        assert command.command_id == custom_id

    def test_command_serialization(self) -> None:
        """Test command serialization."""

        class TestCommand(DomainCommand):
            action: str
            target_id: UUID

        target_id = uuid4()
        command = TestCommand(action="test_action", target_id=target_id)

        json_data = command.model_dump_json_safe()

        assert json_data["action"] == "test_action"
        assert "command_id" in json_data
        assert "issued_at" in json_data
        assert json_data["target_id"] == str(target_id)

    def test_command_metadata_validation(self) -> None:
        """Test command metadata type validation."""

        class TestCommand(DomainCommand):
            action: str

        # Valid metadata types
        valid_metadata = {
            "string": "value",
            "int": 42,
            "bool": True,
            "float": 3.14,
            "none": None,
        }

        command = TestCommand(action="test_action", metadata=valid_metadata)
        assert command.metadata == valid_metadata


class TestDomainQuery:
    """Test suite for DomainQuery - 5 scenarios."""

    def test_query_auto_generated_fields(self) -> None:
        """Test query auto-generated fields."""

        class TestQuery(DomainQuery):
            filter_name: str

        query = TestQuery(filter_name="test_filter")

        assert isinstance(query.query_id, UUID)
        assert isinstance(query.issued_at, datetime)
        assert query.issued_at.tzinfo == UTC
        assert query.correlation_id is None
        assert query.issued_by is None
        assert query.limit is None
        assert query.offset is None

    def test_query_pagination_fields(self) -> None:
        """Test query pagination fields with validation."""

        class TestQuery(DomainQuery):
            filter_name: str

        query = TestQuery(
            filter_name="test_filter",
            limit=100,
            offset=0,
        )

        assert query.limit == 100
        assert query.offset == 0

    def test_query_limit_validation(self) -> None:
        """Test query limit validation constraints."""

        class TestQuery(DomainQuery):
            filter_name: str

        # Valid limits
        query = TestQuery(filter_name="test", limit=1)
        assert query.limit == 1

        query = TestQuery(filter_name="test", limit=1000)
        assert query.limit == 1000

        # Invalid limits
        with pytest.raises(ValidationError):
            TestQuery(filter_name="test", limit=0)  # Must be >= 1

        with pytest.raises(ValidationError):
            TestQuery(filter_name="test", limit=1001)  # Must be <= 1000

    def test_query_offset_validation(self) -> None:
        """Test query offset validation constraints."""

        class TestQuery(DomainQuery):
            filter_name: str

        # Valid offsets
        query = TestQuery(filter_name="test", offset=0)
        assert query.offset == 0

        query = TestQuery(filter_name="test", offset=100)
        assert query.offset == 100

        # Invalid offset
        with pytest.raises(ValidationError):
            TestQuery(filter_name="test", offset=-1)  # Must be >= 0

    def test_query_with_correlation(self) -> None:
        """Test query with correlation tracking."""

        class TestQuery(DomainQuery):
            search_term: str

        correlation_id = uuid4()
        query = TestQuery(
            search_term="test",
            correlation_id=correlation_id,
            issued_by="user123",
        )

        assert query.correlation_id == correlation_id
        assert query.issued_by == "user123"


class TestDomainEvent:
    """Test suite for DomainEvent - 7 scenarios."""

    def test_event_required_fields(self) -> None:
        """Test event with required fields."""

        class TestEvent(DomainEvent):
            pass

        aggregate_id = uuid4()
        event = TestEvent(
            aggregate_id=aggregate_id,
            event_type="test_event",
        )

        assert isinstance(event.event_id, UUID)
        assert event.aggregate_id == aggregate_id
        assert event.event_type == "test_event"
        assert event.event_version == 1
        assert isinstance(event.occurred_at, datetime)
        assert event.occurred_at.tzinfo == UTC
        assert event.correlation_id is None
        assert event.causation_id is None
        assert isinstance(event.event_data, dict)
        assert len(event.event_data) == 0
        assert isinstance(event.metadata, dict)
        assert len(event.metadata) == 0

    def test_event_with_all_fields(self) -> None:
        """Test event with all optional fields provided."""

        class TestEvent(DomainEvent):
            pass

        aggregate_id = uuid4()
        correlation_id = uuid4()
        causation_id = uuid4()
        event_data = {"key": "value", "count": 42}
        metadata = {"source": "test", "version": 2}

        event = TestEvent(
            aggregate_id=aggregate_id,
            event_type="test_event",
            event_version=3,
            correlation_id=correlation_id,
            causation_id=causation_id,
            event_data=event_data,
            metadata=metadata,
        )

        assert event.event_version == 3
        assert event.correlation_id == correlation_id
        assert event.causation_id == causation_id
        assert event.event_data == event_data
        assert event.metadata == metadata

    def test_event_immutability(self) -> None:
        """Test that events are immutable (frozen)."""

        class TestEvent(DomainEvent):
            pass

        event = TestEvent(
            aggregate_id=uuid4(),
            event_type="test_event",
        )

        with pytest.raises(ValidationError):
            event.event_type = "modified"  # Should fail due to frozen=True

    def test_event_stream_id_computed_field(self) -> None:
        """Test event stream ID computed field."""

        class TestEvent(DomainEvent):
            pass

        aggregate_id = uuid4()
        event = TestEvent(
            aggregate_id=aggregate_id,
            event_type="test_event",
        )

        expected_stream_id = f"test_event-{aggregate_id}"
        assert event.event_stream_id == expected_stream_id

    def test_event_serialization(self) -> None:
        """Test event serialization includes all fields."""

        class TestEvent(DomainEvent):
            pass

        aggregate_id = uuid4()
        event_data = {"action": "created", "name": "test"}

        event = TestEvent(
            aggregate_id=aggregate_id,
            event_type="entity_created",
            event_data=event_data,
        )

        json_data = event.model_dump_json_safe()

        assert "event_id" in json_data
        assert json_data["aggregate_id"] == str(aggregate_id)
        assert json_data["event_type"] == "entity_created"
        assert json_data["event_data"] == event_data
        assert "occurred_at" in json_data
        assert "event_stream_id" in json_data

    def test_event_inheritance(self) -> None:
        """Test event inheritance from DomainValueObject."""

        class TestEvent(DomainEvent):
            pass

        # Should inherit value object behavior
        aggregate_id = uuid4()
        event1 = TestEvent(
            aggregate_id=aggregate_id,
            event_type="test_event",
            event_id=uuid4(),
        )
        event2 = TestEvent(
            aggregate_id=aggregate_id,
            event_type="test_event",
            event_id=event1.event_id,
            occurred_at=event1.occurred_at,
        )

        # Should be equal based on all field values
        assert event1 == event2

    def test_event_data_and_metadata_types(self) -> None:
        """Test event_data and metadata type validation."""

        class TestEvent(DomainEvent):
            pass

        complex_event_data = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None,
        }

        complex_metadata = {
            "string": "value",
            "int": 42,
            "bool": True,
            "float": 3.14,
            "none": None,
        }

        event = TestEvent(
            aggregate_id=uuid4(),
            event_type="complex_event",
            event_data=complex_event_data,
            metadata=complex_metadata,
        )

        assert event.event_data == complex_event_data
        assert event.metadata == complex_metadata


class TestDomainSpecification:
    """Test suite for DomainSpecification and compositions - 12 scenarios."""

    def test_base_specification_creation(self) -> None:
        """Test base specification creation and default behavior."""

        class TestSpecification(DomainSpecification):
            specification_name: str = "test_spec"

        spec = TestSpecification()

        assert spec.specification_name == "test_spec"
        assert spec.description is None
        assert isinstance(spec.created_at, datetime)
        assert spec.created_at.tzinfo == UTC

    def test_base_specification_is_satisfied_by_default(self) -> None:
        """Test base specification default is_satisfied_by method."""

        class TestSpecification(DomainSpecification):
            specification_name: str = "test_spec"

        spec = TestSpecification()

        # Default implementation returns True
        assert spec.is_satisfied_by("any_candidate") is True
        assert spec.is_satisfied_by(42) is True
        assert spec.is_satisfied_by(None) is True

    def test_custom_specification_implementation(self) -> None:
        """Test custom specification implementation."""

        class PositiveNumberSpecification(DomainSpecification):
            specification_name: str = "positive_number"

            def is_satisfied_by(self, candidate: object) -> bool:
                """Check if candidate is a positive number."""
                return isinstance(candidate, (int, float)) and candidate > 0

        spec = PositiveNumberSpecification()

        assert spec.is_satisfied_by(42) is True
        assert spec.is_satisfied_by(3.14) is True
        assert spec.is_satisfied_by(0) is False
        assert spec.is_satisfied_by(-5) is False
        assert spec.is_satisfied_by("not_a_number") is False

    def test_and_specification_composition(self) -> None:
        """Test AND specification composition."""

        class PositiveSpecification(DomainSpecification):
            specification_name: str = "positive"

            def is_satisfied_by(self, candidate: object) -> bool:
                return isinstance(candidate, (int, float)) and candidate > 0

        class EvenSpecification(DomainSpecification):
            specification_name: str = "even"

            def is_satisfied_by(self, candidate: object) -> bool:
                return isinstance(candidate, int) and candidate % 2 == 0

        positive_spec = PositiveSpecification()
        even_spec = EvenSpecification()

        # Test __and__ operator
        positive_and_even = positive_spec & even_spec

        assert isinstance(positive_and_even, AndSpecification)
        assert positive_and_even.left == positive_spec
        assert positive_and_even.right == even_spec
        assert positive_and_even.specification_name == "and_specification"

        # Test combined behavior
        assert positive_and_even.is_satisfied_by(4) is True  # Positive and even
        assert positive_and_even.is_satisfied_by(3) is False  # Positive but odd
        assert positive_and_even.is_satisfied_by(-2) is False  # Even but negative
        assert positive_and_even.is_satisfied_by(-3) is False  # Neither

    def test_or_specification_composition(self) -> None:
        """Test OR specification composition."""

        class ZeroSpecification(DomainSpecification):
            specification_name: str = "zero"

            def is_satisfied_by(self, candidate: object) -> bool:
                return candidate == 0

        class PositiveSpecification(DomainSpecification):
            specification_name: str = "positive"

            def is_satisfied_by(self, candidate: object) -> bool:
                return isinstance(candidate, (int, float)) and candidate > 0

        zero_spec = ZeroSpecification()
        positive_spec = PositiveSpecification()

        # Test __or__ operator
        zero_or_positive = zero_spec | positive_spec

        assert isinstance(zero_or_positive, OrSpecification)
        assert zero_or_positive.left == zero_spec
        assert zero_or_positive.right == positive_spec
        assert zero_or_positive.specification_name == "or_specification"

        # Test combined behavior
        assert zero_or_positive.is_satisfied_by(0) is True  # Zero
        assert zero_or_positive.is_satisfied_by(5) is True  # Positive
        assert zero_or_positive.is_satisfied_by(-3) is False  # Neither

    def test_not_specification_composition(self) -> None:
        """Test NOT specification composition."""

        class PositiveSpecification(DomainSpecification):
            specification_name: str = "positive"

            def is_satisfied_by(self, candidate: object) -> bool:
                return isinstance(candidate, (int, float)) and candidate > 0

        positive_spec = PositiveSpecification()

        # Test __invert__ operator
        not_positive = ~positive_spec

        assert isinstance(not_positive, NotSpecification)
        assert not_positive.specification == positive_spec
        assert not_positive.specification_name == "not_specification"

        # Test negated behavior
        assert not_positive.is_satisfied_by(5) is False  # Positive (negated)
        assert not_positive.is_satisfied_by(0) is True  # Not positive
        assert not_positive.is_satisfied_by(-3) is True  # Not positive

    def test_complex_specification_composition(self) -> None:
        """Test complex specification composition with multiple operators."""

        class NumberSpecification(DomainSpecification):
            specification_name: str = "number"

            def is_satisfied_by(self, candidate: object) -> bool:
                return isinstance(candidate, (int, float))

        class PositiveSpecification(DomainSpecification):
            specification_name: str = "positive"

            def is_satisfied_by(self, candidate: object) -> bool:
                return isinstance(candidate, (int, float)) and candidate > 0

        class EvenSpecification(DomainSpecification):
            specification_name: str = "even"

            def is_satisfied_by(self, candidate: object) -> bool:
                return isinstance(candidate, int) and candidate % 2 == 0

        number_spec = NumberSpecification()
        positive_spec = PositiveSpecification()
        even_spec = EvenSpecification()

        # Complex composition: (number AND positive) OR even
        complex_spec = (number_spec & positive_spec) | even_spec

        assert complex_spec.is_satisfied_by(5) is True  # Number and positive
        assert (
            complex_spec.is_satisfied_by(4) is True
        )  # Even (and also number/positive)
        assert complex_spec.is_satisfied_by(-2) is True  # Even (but not positive)
        assert complex_spec.is_satisfied_by(-3) is False  # Not even, not positive
        assert complex_spec.is_satisfied_by("text") is False  # Not number, not even

    def test_specification_with_description(self) -> None:
        """Test specification with description field."""

        class TestSpecification(DomainSpecification):
            specification_name: str = "test_spec"

        spec = TestSpecification(
            description="This is a test specification for validation",
        )

        assert spec.description == "This is a test specification for validation"

    def test_specification_serialization(self) -> None:
        """Test specification serialization."""

        class TestSpecification(DomainSpecification):
            specification_name: str = "test_spec"
            custom_field: int = 42

        spec = TestSpecification(
            description="Test description",
            custom_field=100,
        )

        json_data = spec.model_dump_json_safe()

        assert json_data["specification_name"] == "test_spec"
        assert json_data["description"] == "Test description"
        assert json_data["custom_field"] == 100
        assert "created_at" in json_data

    def test_specification_validation(self) -> None:
        """Test specification field validation."""

        class TestSpecification(DomainSpecification):
            specification_name: str

        # Valid specification
        spec = TestSpecification(specification_name="valid_name")
        assert spec.specification_name == "valid_name"

        # Whitespace should be stripped
        spec = TestSpecification(specification_name="  trimmed  ")
        assert spec.specification_name == "trimmed"

    def test_specification_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden in specifications."""

        class TestSpecification(DomainSpecification):
            specification_name: str = "test_spec"

        with pytest.raises(ValidationError):
            TestSpecification(extra_field="not_allowed")

    def test_nested_specification_composition(self) -> None:
        """Test deeply nested specification composition."""

        class TrueSpecification(DomainSpecification):
            specification_name: str = "always_true"

            def is_satisfied_by(self, candidate: object) -> bool:
                return True

        class FalseSpecification(DomainSpecification):
            specification_name: str = "always_false"

            def is_satisfied_by(self, candidate: object) -> bool:
                return False

        true_spec = TrueSpecification()
        false_spec = FalseSpecification()

        # Test: NOT((True AND False) OR False) should always be True
        complex_nested = ~((true_spec & false_spec) | false_spec)

        assert complex_nested.is_satisfied_by("anything") is True
        assert complex_nested.is_satisfied_by(42) is True
        assert complex_nested.is_satisfied_by(None) is True


# Integration tests for cross-class functionality
class TestCrossClassIntegration:
    """Integration tests for interactions between base classes."""

    def test_entity_with_command_workflow(self) -> None:
        """Test workflow with entity and command interaction."""

        class User(DomainEntity):
            name: str
            email: str

        class CreateUserCommand(DomainCommand):
            name: str
            email: str

        # Create command
        command = CreateUserCommand(name="John Doe", email="john@example.com")

        # Create entity from command
        user = User(
            name=command.name,
            email=command.email,
        )

        assert user.name == command.name
        assert user.email == command.email
        assert isinstance(user.id, UUID)
        assert isinstance(command.command_id, UUID)

    def test_aggregate_with_events_workflow(self) -> None:
        """Test complete aggregate with events workflow."""

        class Order(DomainAggregateRoot):
            customer_name: str
            total_amount: float = 0.0
            status: str = "pending"

        class OrderCreatedEvent(DomainEvent):
            event_type: str = "order_created"

        class OrderConfirmedEvent(DomainEvent):
            event_type: str = "order_confirmed"

        # Create order
        order = Order(customer_name="Jane Doe")

        # Add creation event
        created_event = OrderCreatedEvent(
            aggregate_id=order.id,
            event_data={"customer_name": order.customer_name},
        )
        order.add_domain_event(created_event)

        # Confirm order
        order.status = "confirmed"
        confirmed_event = OrderConfirmedEvent(
            aggregate_id=order.id,
            event_data={"status": order.status},
        )
        order.add_domain_event(confirmed_event)

        # Verify state
        assert order.customer_name == "Jane Doe"
        assert order.status == "confirmed"
        assert len(order.domain_events) == 2
        assert order.aggregate_version == 3  # Started at 1, added 2 events

        # Process events
        events = order.clear_domain_events()
        assert len(events) == 2
        assert events[0].event_type == "order_created"
        assert events[1].event_type == "order_confirmed"

    def test_specification_with_entities(self) -> None:
        """Test specifications working with entities."""

        class User(DomainEntity):
            name: str
            age: int
            active: bool = True

        class AdultUserSpecification(DomainSpecification):
            specification_name: str = "adult_user"

            def is_satisfied_by(self, candidate: object) -> bool:
                return (
                    isinstance(candidate, User)
                    and candidate.age >= 18
                    and candidate.active
                )

        adult_spec = AdultUserSpecification()

        adult_user = User(name="Adult User", age=25, active=True)
        minor_user = User(name="Minor User", age=16, active=True)
        inactive_user = User(name="Inactive User", age=30, active=False)

        assert adult_spec.is_satisfied_by(adult_user) is True
        assert adult_spec.is_satisfied_by(minor_user) is False
        assert adult_spec.is_satisfied_by(inactive_user) is False
