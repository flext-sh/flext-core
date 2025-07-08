"""Domain core tests - 100% coverage.

Tests for all base domain classes and patterns matching actual implementation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError as PydanticValidationError

from flext_core.domain.core import (
    AggregateRoot,
    DomainError,
    DomainEvent,
    Entity,
    NotFoundError,
    Repository,
    RepositoryError,
    ServiceResult,
    ValidationError,
    ValueObject,
)


class TestValueObject:
    """Test ValueObject base class."""

    def test_value_object_equality(self) -> None:
        """Test value objects are equal by value."""

        # Create test value object
        class TestVO(ValueObject):
            value: str
            number: int

        vo1 = TestVO(value="test", number=42)
        vo2 = TestVO(value="test", number=42)
        vo3 = TestVO(value="other", number=42)

        assert vo1 == vo2
        assert vo1 != vo3
        assert hash(vo1) == hash(vo2)
        assert hash(vo1) != hash(vo3)

    def test_value_object_immutable(self) -> None:
        """Test value objects are immutable (frozen=True)."""

        class TestVO(ValueObject):
            value: str

        vo = TestVO(value="test")

        # Pydantic with frozen=True raises validation error
        with pytest.raises(PydanticValidationError):
            vo.__dict__["value"] = "changed"  # Bypass type checking for test

    def test_value_object_config(self) -> None:
        """Test ValueObject configuration."""

        class TestVO(ValueObject):
            name: str

        # Test extra fields are forbidden
        with pytest.raises(PydanticValidationError):
            TestVO(name="test", extra="field")  # type: ignore

        # Test whitespace stripping
        vo = TestVO(name="  test  ")
        assert vo.name == "test"


class TestEntity:
    """Test Entity base class."""

    def test_entity_lifecycle(self) -> None:
        """Test entity creation and updates."""

        # Create test entity
        class TestEntity(Entity[str]):
            id: str
            name: str

        # Test creation
        entity = TestEntity(id="test-1", name="Test Entity")
        assert entity.id == "test-1"
        assert entity.name == "Test Entity"
        assert entity.created_at <= datetime.now(UTC)
        assert entity.updated_at is None  # Not set on creation

    def test_entity_auto_id(self) -> None:
        """Test entity with auto-generated ID."""

        class TestEntity(Entity[UUID]):
            name: str

        # Don't provide ID, should auto-generate
        entity = TestEntity(name="Test")
        assert isinstance(entity.id, UUID)
        assert entity.name == "Test"

    def test_entity_equality_by_id(self) -> None:
        """Test entities are equal by ID only."""

        class TestEntity(Entity[str]):
            id: str
            name: str

        entity1 = TestEntity(id="same-id", name="Name 1")
        entity2 = TestEntity(id="same-id", name="Name 2")
        entity3 = TestEntity(id="other-id", name="Name 1")

        # Same ID = equal, even with different data
        assert entity1 == entity2
        assert entity1 != entity3
        assert hash(entity1) == hash(entity2)

        # Test with non-entity
        assert entity1 != "not an entity"
        assert entity1 != 42


class TestAggregateRoot:
    """Test AggregateRoot with events."""

    def test_aggregate_event_handling(self) -> None:
        """Test aggregate can collect domain events."""

        # Create test event
        class TestEvent(DomainEvent):
            aggregate_id: str
            data: str

        # Create test aggregate
        class TestAggregate(AggregateRoot[str]):
            id: str

            def do_something(self) -> None:
                """Business operation that emits event."""
                self.add_event(TestEvent(aggregate_id=self.id, data="something happened"))

        # Test
        agg = TestAggregate(id="agg-1")
        # Clear any existing class-level events
        TestAggregate.get_events()

        agg.do_something()
        events = TestAggregate.get_events()
        assert len(events) == 1
        assert isinstance(events[0], TestEvent)
        assert events[0].data == "something happened"
        assert events[0].aggregate_id == "agg-1"

        # After get_events, list should be empty
        events = TestAggregate.get_events()
        assert len(events) == 0

    def test_aggregate_multiple_events(self) -> None:
        """Test aggregate with multiple events."""

        class Event1(DomainEvent):
            data: str

        class Event2(DomainEvent):
            value: int

        class TestAggregate(AggregateRoot[str]):
            id: str

            def operation1(self) -> None:
                self.add_event(Event1(data="op1"))

            def operation2(self) -> None:
                self.add_event(Event2(value=42))

        agg = TestAggregate(id="test")
        TestAggregate.get_events()  # Clear

        agg.operation1()
        agg.operation2()

        events = TestAggregate.get_events()
        assert len(events) == 2
        assert isinstance(events[0], Event1)
        assert isinstance(events[1], Event2)
        assert events[0].data == "op1"
        assert events[1].value == 42


class TestDomainEvent:
    """Test DomainEvent base class."""

    def test_domain_event_basic(self) -> None:
        """Test events have automatic fields."""

        class TestEvent(DomainEvent):
            message: str

        event = TestEvent(message="test")
        assert event.occurred_at <= datetime.now(UTC)
        assert event.message == "test"
        assert isinstance(event.event_id, UUID)
        assert event.event_type == "TestEvent"  # Auto-set from class name

    def test_domain_event_immutable(self) -> None:
        """Test domain events are immutable."""

        class TestEvent(DomainEvent):
            data: str

        event = TestEvent(data="original")
        with pytest.raises(PydanticValidationError):
            event.__dict__["data"] = "modified"  # Bypass type checking for test

    def test_domain_event_subclass(self) -> None:
        """Test event type is set correctly for subclasses."""

        class UserCreatedEvent(DomainEvent):
            user_id: str

        class UserDeletedEvent(DomainEvent):
            user_id: str

        event1 = UserCreatedEvent(user_id="123")
        event2 = UserDeletedEvent(user_id="456")

        assert event1.event_type == "UserCreatedEvent"
        assert event2.event_type == "UserDeletedEvent"


class TestServiceResult:
    """Test ServiceResult pattern."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = ServiceResult.ok("success data")

        assert result.success is True
        assert result.error is None
        assert result.unwrap() == "success data"
        assert result.data == "success data"

    def test_failure_result(self) -> None:
        """Test failure result."""
        result: ServiceResult[str] = ServiceResult.fail("error message")

        assert result.success is False
        assert result.data is None
        assert result.error == "error message"

        with pytest.raises(RuntimeError, match="Cannot unwrap failed result: error message"):
            result.unwrap()

    def test_service_result_initialization(self) -> None:
        """Test ServiceResult direct initialization."""
        # Success with data
        result1 = ServiceResult(success=True, data="test")
        assert result1.success is True
        assert result1.data == "test"
        assert result1.error is None

        # Failure with error
        result2: ServiceResult[str] = ServiceResult(success=False, error="failed")
        assert result2.success is False
        assert result2.data is None
        assert result2.error == "failed"

        # Edge case: success with no data
        result3: ServiceResult[str] = ServiceResult(success=True)
        assert result3.success is True
        assert result3.data is None
        with pytest.raises(RuntimeError, match="Cannot unwrap failed result"):
            result3.unwrap()  # Should fail since data is None

    def test_service_result_type_safety(self) -> None:
        """Test ServiceResult with different types."""
        # With int
        int_result = ServiceResult.ok(42)
        assert int_result.unwrap() == 42

        # With list
        list_result = ServiceResult.ok([1, 2, 3])
        assert list_result.unwrap() == [1, 2, 3]

        # With custom object
        class CustomData:
            value: int = 10

        obj_result = ServiceResult.ok(CustomData())
        assert obj_result.unwrap().value == 10


class TestDomainExceptions:
    """Test domain exception hierarchy."""

    def test_exception_hierarchy(self) -> None:
        """Test exception inheritance."""
        base_error = DomainError("base")
        validation_error = ValidationError("validation")
        repo_error = RepositoryError("repo")
        not_found_error = NotFoundError("not found")

        # Check inheritance
        assert isinstance(validation_error, DomainError)
        assert isinstance(repo_error, DomainError)
        assert isinstance(not_found_error, RepositoryError)
        assert isinstance(not_found_error, DomainError)

        # Check messages
        assert str(base_error) == "base"
        assert str(validation_error) == "validation"
        assert str(repo_error) == "repo"
        assert str(not_found_error) == "not found"

    def test_exception_raising(self) -> None:
        """Test raising domain exceptions."""
        with pytest.raises(DomainError, match="domain error"):
            raise DomainError("domain error")

        with pytest.raises(ValidationError, match="invalid data"):
            raise ValidationError("invalid data")

        with pytest.raises(NotFoundError, match="entity not found"):
            raise NotFoundError("entity not found")


class TestRepositoryProtocol:
    """Test Repository protocol definition."""

    def test_repository_protocol_structure(self) -> None:
        """Test repository protocol has required methods."""
        # Just verify the protocol exists and has the right structure
        assert hasattr(Repository, "save")
        assert hasattr(Repository, "get")
        assert hasattr(Repository, "delete")
        # Note: Repository in core.py only has these 3 abstract methods

    async def test_repository_implementation(self) -> None:
        """Test that Repository can be implemented."""

        class TestRepo(Repository[str, int]):
            async def save(self, entity: str) -> str:
                return entity

            async def get(self, entity_id: int) -> str | None:
                return "test" if entity_id == 1 else None

            async def delete(self, entity_id: int) -> bool:
                return entity_id == 1

        repo = TestRepo()
        assert await repo.save("test") == "test"
        assert await repo.get(1) == "test"
        assert await repo.get(2) is None
        assert await repo.delete(1) is True
        assert await repo.delete(2) is False
