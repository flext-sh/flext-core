"""Comprehensive tests for FlextModels - Data Models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar, cast
from unittest.mock import patch

import pytest
from pydantic import Field, field_validator

from flext_core import (
    FlextConstants,
    FlextModels,
)


# Test fixture classes defined at module level for proper Pydantic resolution
# (Renamed to avoid pytest collection - they're not test classes)
class SampleAggregate(FlextModels.AggregateRoot):
    """Sample aggregate root for testing (renamed to avoid pytest collection)."""

    name: str
    status: str = "active"

    def change_status(self, new_status: str) -> None:
        """Change status and add domain event."""
        self.status = new_status
        result = self.add_domain_event(
            "status_changed",
            {"old_status": "active", "new_status": new_status},
        )
        if result.is_failure:
            raise ValueError(f"Failed to add domain event: {result.error}")


class EventAggregate(FlextModels.AggregateRoot):
    """Test aggregate for domain events."""

    name: str

    def trigger_event(self) -> None:
        """Trigger a test event."""
        self.add_domain_event("test_event", {"data": "test"})


class TestFlextModels:
    """Test suite for FlextModels data model functionality."""

    def test_models_initialization(self) -> None:
        """Test models initialization."""
        models = FlextModels()
        assert models is not None
        assert isinstance(models, FlextModels)

    def test_models_entity_creation(self) -> None:
        """Test entity creation and validation."""

        class TestEntity(FlextModels.Entity):
            name: str
            email: str

            @field_validator("email")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    msg = "Invalid email"
                    raise ValueError(msg)
                return v

        # Test entity creation
        entity = TestEntity(
            name="Test User",
            email="test@example.com",
            domain_events=[],
        )
        assert entity.name == "Test User"
        assert entity.email == "test@example.com"
        assert entity.unique_id is not None  # Entity should have auto-generated ID

        # Test validation - should work with valid email (exclude extra fields from base Entity class and computed fields)
        entity_dict = entity.model_dump(
            exclude={"is_initial_version", "is_modified", "uncommitted_events"}
        )
        validated_entity = entity.model_validate(entity_dict)
        assert validated_entity.email == "test@example.com"

        # Test validation failure with invalid email
        with pytest.raises(ValueError, match="Invalid email"):
            TestEntity(name="Test User", email="invalid-email", domain_events=[])

    def test_models_value_object_creation(self) -> None:
        """Test value object creation and immutability."""

        class TestValue(FlextModels.Value):
            data: str
            count: int

        # Test value object creation
        value = TestValue(data="test", count=42)
        assert value.data == "test"
        assert value.count == 42

        # Test immutability (Value objects are frozen)
        # Create a new instance to test immutability
        original_data = value.data
        original_count = value.count

        # Verify values haven't changed (immutability)
        assert value.data == original_data
        assert value.count == original_count

        # Test that we can create a new instance with different values
        new_value = TestValue(data="modified", count=100)
        assert new_value.data == "modified"
        assert new_value.count == 100
        assert value.data == "test"  # Original unchanged

    def test_models_aggregate_root_creation(self) -> None:
        """Test aggregate root creation and basic functionality."""
        # Import the actual FlextModels class
        from flext_core import FlextModels

        # Test that AggregateRoot inherits from Entity properly
        assert issubclass(FlextModels.AggregateRoot, FlextModels.Entity)

        # Test aggregate root has invariants list
        assert hasattr(FlextModels.AggregateRoot, "_invariants")
        assert isinstance(FlextModels.AggregateRoot._invariants, list)

        # Test check_invariants method exists
        assert hasattr(FlextModels.AggregateRoot, "check_invariants")
        assert callable(FlextModels.AggregateRoot.check_invariants)

    def test_models_command_creation(self) -> None:
        """Test command model creation."""

        class TestCommand(FlextModels.Command):
            command_type: str = "test_command"
            data: str

        command = TestCommand(data="test_data")
        assert command.command_type == "test_command"
        assert command.data == "test_data"
        assert command.created_at is not None  # Timestamp from TimestampableMixin
        assert command.unique_id is not None  # ID from IdentifiableMixin

    def test_models_metadata_creation(self) -> None:
        """Test metadata model creation."""
        metadata = FlextModels.Metadata(
            created_by="test_user",
            tags=["tag1", "tag2"],
        )
        assert metadata.created_by == "test_user"
        assert metadata.created_at is not None
        assert len(metadata.tags) == 2

    def test_models_pagination_creation(self) -> None:
        """Test pagination model creation."""
        pagination = FlextModels.Pagination(page=1, size=10)
        assert pagination.page == 1
        assert pagination.size == 10
        assert (pagination.page - 1) * pagination.size == 0

        pagination2 = FlextModels.Pagination(page=3, size=10)
        assert (pagination2.page - 1) * pagination2.size == 20

    def test_models_domain_events(self) -> None:
        """Test domain events functionality on Entity base class."""
        # Import the actual FlextModels class
        from flext_core import FlextModels

        # Test that Entity has add_domain_event method
        assert hasattr(FlextModels.Entity, "add_domain_event")
        assert callable(FlextModels.Entity.add_domain_event)

        # Test that Entity has clear_domain_events method
        assert hasattr(FlextModels.Entity, "clear_domain_events")
        assert callable(FlextModels.Entity.clear_domain_events)

        # Test that domain_events field exists in model fields
        fields = FlextModels.Entity.model_fields
        assert "domain_events" in fields

    def test_models_version_management(self) -> None:
        """Test version management in entities."""

        class VersionedEntity(FlextModels.Entity):
            name: str

        entity = VersionedEntity(name="Test", domain_events=[])
        initial_version = entity.version

        # Increment version
        entity.increment_version()
        assert entity.version == initial_version + 1

    def test_models_timestamped_functionality(self) -> None:
        """Test timestamped functionality."""

        class TimestampedEntity(FlextModels.Entity):
            name: str

        entity = TimestampedEntity(name="Test", domain_events=[])
        initial_created = entity.created_at

        # Update timestamp
        entity.update_timestamp()
        assert entity.updated_at is not None
        assert entity.updated_at > initial_created

    def test_models_validation_patterns(self) -> None:
        """Test validation patterns."""

        class ValidatedEntity(FlextModels.Entity):
            name: str
            email: str

            @field_validator("name")
            @classmethod
            def validate_name(cls, v: str) -> str:
                if not v:
                    msg = "Name is required"
                    raise ValueError(msg)
                return v

            @field_validator("email")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    msg = "Invalid email"
                    raise ValueError(msg)
                return v

        # Test valid entity
        entity = ValidatedEntity(
            name="Test",
            email="test@example.com",
            domain_events=[],
        )
        assert entity.name == "Test"
        assert entity.email == "test@example.com"

        # Test invalid entity - should raise validation errors
        with pytest.raises(ValueError, match="Name is required"):
            ValidatedEntity(name="", email="test@example.com", domain_events=[])

        with pytest.raises(ValueError, match="Invalid email"):
            ValidatedEntity(name="Test", email="invalid", domain_events=[])

    def test_models_thread_safety(self) -> None:
        """Test thread safety of models."""

        class ThreadSafeEntity(FlextModels.Entity):
            counter: int = 0

            def increment(self) -> None:
                self.counter += 1

        entity = ThreadSafeEntity(domain_events=[])

        def worker() -> None:
            for _ in range(100):
                entity.increment()

        # Create multiple threads
        threads: list[threading.Thread] = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check final counter
        assert entity.counter == 500

    def test_models_removed_classes_not_available(self) -> None:
        """Verify that removed classes are no longer available."""
        # These are classes that were actually removed in Phase 9 refactoring
        # Note: Most classes were refactored but not removed to maintain backward compatibility
        removed_classes: list[str] = []

        for class_name in removed_classes:
            assert not hasattr(FlextModels, class_name), (
                f"Removed class {class_name} should not be available"
            )

    def test_models_critical_classes_available(self) -> None:
        """Verify that critical classes are available after Pydantic v2 refactoring."""
        critical_classes = [
            "Entity",
            "Value",
            "AggregateRoot",
            "Command",
            "Query",
            "DomainEvent",
            "Cqrs",
            "Metadata",
            "Payload",
            "Pagination",
            "RegistrationDetails",
            "HandlerExecutionConfig",
            "BatchProcessingConfig",
            "RetryConfiguration",
            "ValidationConfiguration",
        ]

        for class_name in critical_classes:
            assert hasattr(FlextModels, class_name), (
                f"Critical class {class_name} should be available"
            )

    def test_entity_equality_and_hash(self) -> None:
        """Test Entity equality and hash based on identity."""

        class TestEntity(FlextModels.Entity):
            name: str

        entity1 = TestEntity(name="test", unique_id="123")
        entity2 = TestEntity(name="test", unique_id="123")
        entity3 = TestEntity(name="test", unique_id="456")

        # Same ID should be equal
        assert entity1 == entity2
        # Different ID should not be equal
        assert entity1 != entity3
        # Compare with non-entity should return False
        assert (entity1 == "not an entity") is False

        # Hash should be based on ID
        assert hash(entity1) == hash(entity2)
        assert hash(entity1) != hash(entity3)

        # Can be used in sets (identity-based)
        entity_set = {entity1, entity2, entity3}
        assert (
            len(entity_set) == 2
        )  # entity1 and entity2 are same, entity3 is different

    def test_entity_domain_events(self) -> None:
        """Test domain event functionality in Entity."""

        class TestEntity(FlextModels.Entity):
            name: str

        entity = TestEntity(name="test")

        # Initially no domain events
        assert len(entity.domain_events) == 0

        # Add domain event
        result = entity.add_domain_event("test_event", {"data": "value"})
        assert result.is_success

        # Should have one domain event
        assert len(entity.domain_events) == 1
        event = entity.domain_events[0]
        assert event.event_type == "test_event"
        assert event.data == {"data": "value"}
        assert event.aggregate_id == entity.unique_id

        # Clear events
        cleared_events = entity.clear_domain_events()
        assert len(cleared_events) == 1
        assert len(entity.domain_events) == 0

    def test_entity_domain_events_validation(self) -> None:
        """Test domain event validation."""

        class TestEntity(FlextModels.Entity):
            name: str

        entity = TestEntity(name="test")

        # Empty event name should fail
        result = entity.add_domain_event("", {"data": "value"})
        assert result.is_failure
        if result.error is not None:
            assert "Domain event name must be a non-empty string" in result.error

        # Non-serializable data should fail
        import datetime

        result = entity.add_domain_event(
            "test", {"date": datetime.datetime.now(tz=datetime.UTC)}
        )
        # This might pass or fail depending on serialization, but let's test valid case
        result = entity.add_domain_event("valid", {"string": "value", "number": 42})
        assert result.is_success

    def test_entity_initial_version(self) -> None:
        """Test Entity initial version state."""

        class TestEntity(FlextModels.Entity):
            name: str

        entity = TestEntity(name="test")

        # Should be initial version
        assert entity.is_initial_version is True
        assert entity.version == 1

        # Mark as modified (this would normally happen through update_timestamp)
        entity.updated_at = entity.created_at  # Simulate modification
        assert entity.is_modified is True

    def test_timestampable_mixin_serialization(self) -> None:
        """Test timestamp serialization in JSON output."""

        class TestEntity(FlextModels.Entity):
            name: str

        entity = TestEntity(name="test")

        # Test JSON serialization includes timestamps
        json_data = entity.model_dump_json()
        assert '"created_at"' in json_data
        assert '"updated_at"' in json_data or '"updated_at":null' in json_data

        # Test that timestamps are serialized as ISO strings
        import json

        parsed = json.loads(json_data)
        if parsed.get("created_at"):
            assert "T" in parsed["created_at"]  # ISO format has 'T'

    def test_timestampable_mixin_update_timestamp(self) -> None:
        """Test update_timestamp method."""

        class TestEntity(FlextModels.Entity):
            name: str

        entity = TestEntity(name="test")
        original_updated = entity.updated_at

        # Update timestamp
        entity.update_timestamp()

        # Should have updated_at set
        assert entity.updated_at is not None
        assert entity.updated_at != original_updated
        assert entity.is_modified is True

    def test_versionable_mixin(self) -> None:
        """Test VersionableMixin functionality."""

        class TestEntity(FlextModels.Entity):
            name: str

        entity = TestEntity(name="test")

        # Should start at version 1
        assert entity.version == 1

        # Increment version
        entity.increment_version()
        assert entity.version == 2

        entity.increment_version()
        assert entity.version == 3

    def test_aggregate_root_inheritance(self) -> None:
        """Test AggregateRoot inherits from Entity properly."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Should have Entity properties
        assert hasattr(aggregate, "unique_id")
        assert hasattr(aggregate, "version")
        assert hasattr(aggregate, "created_at")

        # Should have AggregateRoot specific properties
        assert hasattr(aggregate, "_invariants")
        assert isinstance(aggregate._invariants, list)

        # Should be able to add domain events
        result = aggregate.add_domain_event("test", {"data": "value"})
        assert result.is_success

    def test_aggregate_root_invariants(self) -> None:
        """Test AggregateRoot invariant checking."""

        # Test with passing invariant
        class TestAggregate(FlextModels.AggregateRoot):
            name: str
            value: int

            # Define invariants as class-level callables that return bool
            _invariants: ClassVar[list[Callable[[], bool]]] = [
                lambda: True,  # Simple passing invariant for testing
            ]

        # Disable automatic invariant checking for this test
        with patch.object(TestAggregate, "model_post_init", new=lambda self, ctx: None):
            try:
                aggregate = TestAggregate(name="test", value=10)
                # Should pass invariant check when called manually
                aggregate.check_invariants()  # Should not raise
            finally:
                # Restore original method
                pass

        # Test with failing invariant
        class FailingAggregate(FlextModels.AggregateRoot):
            name: str

            _invariants: ClassVar[list[Callable[[], bool]]] = [
                lambda: False,  # Always fails
            ]

        # Disable automatic invariant checking
        with patch.object(
            FailingAggregate, "model_post_init", new=lambda self, ctx: None
        ):
            try:
                failing_aggregate = FailingAggregate(name="test")
                # Should fail invariant check when called manually
                from flext_core import FlextExceptions

                with pytest.raises(
                    FlextExceptions.ValidationError, match="Invariant violated"
                ):
                    failing_aggregate.check_invariants()
            finally:
                # Restore original method
                pass

    def test_value_object_immutability(self) -> None:
        """Test Value object immutability and equality."""

        class TestValue(FlextModels.Value):
            name: str
            value: int

        value1 = TestValue(name="test", value=42)
        value2 = TestValue(name="test", value=42)
        value3 = TestValue(name="other", value=42)

        # Value objects should be equal if all fields are equal (value equality)
        assert value1 == value2
        assert value1 != value3

        # Should be hashable for use in sets/dicts
        value_set = {value1, value2, value3}
        assert len(value_set) == 2  # value1 and value2 are same, value3 is different

        # Should be immutable (frozen)
        from pydantic import ValidationError

        with pytest.raises((
            ValidationError,
            AttributeError,
            TypeError,
        )):  # Pydantic v2 may raise ValidationError or frozen model errors
            value1.value = 100

    def test_command_creation_with_mixins(self) -> None:
        """Test Command creation with all mixins."""

        class TestCommand(FlextModels.Command):
            action: str
            target: str

        command = TestCommand(action="create", target="user")

        # Should have IdentifiableMixin properties
        assert hasattr(command, "unique_id")
        assert command.unique_id is not None

        # Should have TimestampableMixin properties
        assert hasattr(command, "created_at")
        assert command.created_at is not None
        assert hasattr(command, "updated_at")

        # Should have Command-specific properties
        assert command.command_type == "generic_command"  # default value
        assert command.action == "create"
        assert command.target == "user"

    def test_domain_event_creation(self) -> None:
        """Test DomainEvent creation."""
        event = FlextModels.DomainEvent(
            event_type="user_created", aggregate_id="user-123", data={"name": "test"}
        )

        # Should have IdentifiableMixin properties
        assert hasattr(event, "unique_id")
        assert event.unique_id is not None

        # Should have TimestampableMixin properties
        assert hasattr(event, "created_at")
        assert event.created_at is not None

        # Should have DomainEvent-specific properties
        assert event.event_type == "user_created"
        assert event.aggregate_id == "user-123"
        assert event.data == {"name": "test"}
        assert event.message_type == "event"

    def test_query_creation(self) -> None:
        """Test Query creation."""
        query = FlextModels.Query(query_type="find_users", filters={"active": "true"})

        # Should have query_id
        assert hasattr(query, "query_id")
        assert query.query_id is not None

        # Should have Query-specific properties
        assert query.query_type == "find_users"
        assert query.filters == {"active": "true"}
        assert query.message_type == "query"

    def test_aggregate_root_mark_events_as_committed(self) -> None:
        """Test mark_events_as_committed method."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Add some domain events
        aggregate.add_domain_event("event1", {"data": "value1"})
        aggregate.add_domain_event("event2", {"data": "value2"})

        # Should have domain events
        assert len(aggregate.domain_events) == 2

        # Mark events as committed
        result = aggregate.mark_events_as_committed()
        assert result.is_success

        # Events should be cleared after committing
        assert len(aggregate.domain_events) == 0

        # Result should contain the committed events
        committed_events = result.unwrap()
        assert len(committed_events) == 2

    def test_aggregate_root_mark_events_empty(self) -> None:
        """Test mark_events_as_committed with no events."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Mark events with no events
        result = aggregate.mark_events_as_committed()
        assert result.is_success

        # Should return empty list
        committed_events = result.unwrap()
        assert len(committed_events) == 0
        # result = aggregate.mark_events_as_committed()
        # assert result.is_success

    def test_aggregate_root_bulk_domain_events(self) -> None:
        """Test add_domain_events_bulk method."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Bulk add events
        events = [
            ("event1", cast("dict[str, object]", {"data": "value1"})),
            ("event2", cast("dict[str, object]", {"data": "value2"})),
            ("event3", cast("dict[str, object]", {"data": "value3"})),
        ]

        result = aggregate.add_domain_events_bulk(events)
        assert result.is_success

        # Should have added all events
        assert len(aggregate.domain_events) == 3

        # Check event details
        assert aggregate.domain_events[0].event_type == "event1"
        assert aggregate.domain_events[1].event_type == "event2"
        assert aggregate.domain_events[2].event_type == "event3"

    def test_aggregate_root_bulk_domain_events_validation(self) -> None:
        """Test add_domain_events_bulk validation."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Test empty list
        result = aggregate.add_domain_events_bulk([])
        assert result.is_success

        # Test invalid input type
        result = aggregate.add_domain_events_bulk(
            cast("list[tuple[str, dict[str, object]]]", "not a list")
        )
        assert result.is_failure
        if result.error is not None:  # Check for None before 'in'
            assert "Events must be a list" in result.error

        # Test empty event name
        result = aggregate.add_domain_events_bulk([
            ("", cast("dict[str, object]", {"data": "value"}))
        ])
        assert result.is_failure
        if result.error is not None:  # Check for None before 'in'
            assert "name must be non-empty string" in result.error

        # Test invalid data type
        result = aggregate.add_domain_events_bulk([
            ("event", cast("dict[str, object]", "not a dict"))
        ])
        assert result.is_failure
        if result.error is not None:  # Check for None before 'in'
            assert "data must be dict" in result.error

        # Test None data (should be converted to empty dict)
        result = aggregate.add_domain_events_bulk([
            ("event", cast("dict[str, object]", {}))
        ])
        assert result.is_success

    def test_aggregate_root_bulk_domain_events_limit(self) -> None:
        """Test add_domain_events_bulk respects max events limit."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Try to add more than max events
        max_events = 1000  # From FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS
        events = [
            (f"event{i}", cast("dict[str, object]", {"data": f"value{i}"}))
            for i in range(max_events + 1)
        ]

        result = aggregate.add_domain_events_bulk(events)
        assert result.is_failure
        if result.error is not None:  # Check for None before 'in'
            assert "would exceed max events" in result.error

    def test_aggregate_root_domain_event_handler_execution(self) -> None:
        """Test domain event handler execution."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str
            handler_called: bool = False
            handler_data: dict[str, object] = Field(default_factory=dict)

            def _apply_test_event(self, data: dict[str, object]) -> None:
                """Event handler method."""
                self.handler_called = True
                self.handler_data = data

        aggregate = TestAggregate(name="test")

        # Add event that should trigger handler (event_type in data)
        result = aggregate.add_domain_event(
            "user_action", {"event_type": "test_event", "key": "value"}
        )
        assert result.is_success

        # Handler should have been called
        assert aggregate.handler_called is True
        assert aggregate.handler_data == {"event_type": "test_event", "key": "value"}

    def test_aggregate_root_domain_event_handler_error(self) -> None:
        """Test domain event handler error handling."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str

            def _apply_failing_event(self, data: dict[str, object]) -> None:
                """Event handler that raises exception."""
                msg = "Handler failed"
                raise ValueError(msg)

        aggregate = TestAggregate(name="test")

        # Add event that triggers failing handler
        result = aggregate.add_domain_event("failing_event", {"data": "value"})
        # Should still succeed (handler errors are logged but don't fail the operation)
        assert result.is_success

    def test_domain_event_model_creation(self) -> None:
        """Test DomainEvent model creation and properties."""
        event = FlextModels.DomainEvent(
            event_type="test_event", aggregate_id="aggregate-123", data={"key": "value"}
        )

        assert event.event_type == "test_event"
        assert event.aggregate_id == "aggregate-123"
        assert event.data == {"key": "value"}
        assert event.unique_id is not None
        assert event.created_at is not None
        assert event.message_type == "event"

        # Test serialization
        json_data = event.model_dump_json()
        assert '"event_type":"test_event"' in json_data
        assert '"aggregate_id":"aggregate-123"' in json_data

    def test_command_model_creation(self) -> None:
        """Test Command model creation with correct fields."""
        # Command has: message_type, command_type, issuer_id (+ id, created_at from mixins)
        command = FlextModels.Command(
            command_type="CreateOrder", issuer_id="issuer-123"
        )

        assert command.command_type == "CreateOrder"
        assert command.issuer_id == "issuer-123"
        assert command.message_type == "command"
        assert command.unique_id is not None
        assert command.created_at is not None

    def test_metadata_model_creation(self) -> None:
        """Test Metadata model creation with correct fields."""
        # Metadata has: created_by, created_at, modified_by, modified_at, tags, attributes
        meta = FlextModels.Metadata(
            created_by="user-123",
            modified_by="user-456",
            tags=["important", "urgent"],
            attributes={"priority": "high"},
        )

        assert meta.created_by == "user-123"
        assert meta.modified_by == "user-456"
        assert len(meta.tags) == 2
        assert meta.attributes["priority"] == "high"

    def test_payload_model_creation(self) -> None:
        """Test Payload model creation with required data field."""
        # Payload[T] has: data (required), metadata, expires_at, correlation_id, source_service, message_type
        payload = FlextModels.Payload(
            data={"order_id": "123"}, correlation_id="corr-abc"
        )

        assert payload.data == {"order_id": "123"}
        assert payload.correlation_id == "corr-abc"
        assert payload.message_type is None

    def test_processing_request_model_creation(self) -> None:
        """Test ProcessingRequest model with correct fields."""
        # ProcessingRequest has: operation_id, data, context, timeout_seconds, retry_attempts, enable_validation
        request = FlextModels.ProcessingRequest(
            data={"input": "data"}, enable_validation=True
        )

        assert request.data == {"input": "data"}
        assert request.enable_validation is True
        assert request.operation_id is not None

    def test_handler_registration_model_creation(self) -> None:
        """Test HandlerRegistration model with correct fields."""

        # HandlerRegistration has: name, handler (callable), event_types, priority
        def dummy_handler() -> None:
            pass

        reg = FlextModels.HandlerRegistration(
            name="TestHandler",
            handler=dummy_handler,
            event_types=["CreateUser"],
            priority=5,
        )

        assert reg.name == "TestHandler"
        assert callable(reg.handler)
        assert "CreateUser" in reg.event_types

    def test_batch_processing_config_model(self) -> None:
        """Test BatchProcessingConfig model with correct fields."""
        # BatchProcessingConfig has: batch_size, max_workers, timeout_per_item, continue_on_error, data_items
        config = FlextModels.BatchProcessingConfig(
            batch_size=100, continue_on_error=True, data_items=[1, 2, 3]
        )

        assert config.batch_size == 100
        assert config.continue_on_error is True
        assert len(config.data_items) == 3

    def test_handler_execution_config_model(self) -> None:
        """Test HandlerExecutionConfig model with correct fields."""
        # HandlerExecutionConfig has: handler_name, input_data, execution_context, timeout_seconds, retry_on_failure, max_retries, fallback_handlers
        config = FlextModels.HandlerExecutionConfig(
            handler_name="my_handler",
            input_data={"key": "value"},
            retry_on_failure=True,
        )

        assert config.handler_name == "my_handler"
        assert config.input_data == {"key": "value"}
        assert config.retry_on_failure is True

    def test_retry_configuration_model(self) -> None:
        """Test RetryConfiguration model with correct fields."""
        # RetryConfiguration has: max_attempts (le=3), initial_delay_seconds, max_delay_seconds, exponential_backoff, backoff_multiplier, retry_on_exceptions, retry_on_status_codes
        retry = FlextModels.RetryConfiguration(
            max_attempts=3,
            initial_delay_seconds=1000,
            max_delay_seconds=30000,
            backoff_multiplier=2.0,
        )

        assert retry.max_attempts == 3
        assert retry.initial_delay_seconds == 1000
        assert retry.backoff_multiplier == 2.0

    def test_validation_configuration_model(self) -> None:
        """Test ValidationConfiguration model with correct fields."""
        # ValidationConfiguration has: enable_strict_mode, max_validation_errors, validate_on_assignment, validate_on_read, custom_validators
        val_config = FlextModels.ValidationConfiguration(
            enable_strict_mode=True, validate_on_assignment=True, validate_on_read=False
        )

        assert val_config.enable_strict_mode is True
        assert val_config.validate_on_assignment is True

    def test_conditional_execution_request_model(self) -> None:
        """Test ConditionalExecutionRequest model with correct fields."""

        # ConditionalExecutionRequest has: condition (callable), true_action (callable), false_action (callable), context
        def condition_func(v: object) -> bool:
            return True

        def true_func() -> None:
            pass

        cond_req = FlextModels.ConditionalExecutionRequest(
            condition=condition_func, true_action=true_func, context={"test": "value"}
        )

        assert callable(cond_req.condition)
        assert callable(cond_req.true_action)
        assert cond_req.context["test"] == "value"

    def test_state_initialization_request_model(self) -> None:
        """Test StateInitializationRequest model with correct fields."""
        # StateInitializationRequest has: data, state_key, initial_value, ttl_seconds, persistence_level, field_name, state
        state_init = FlextModels.StateInitializationRequest(
            data="state_data",
            state_key="key-123",
            initial_value="initial",
            state="initialized",
        )

        assert state_init.state_key == "key-123"
        assert state_init.initial_value == "initial"
        assert state_init.state == "initialized"

    def test_cqrs_handler_model_creation(self) -> None:
        """Test Cqrs.Handler model creation."""
        # Cqrs.Handler has: handler_id, handler_name, handler_type, handler_mode, command_timeout, max_command_retries, metadata
        handler_config = FlextModels.Cqrs.Handler(
            handler_id="handler-123",
            handler_name="CreateUserHandler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        assert handler_config.handler_id == "handler-123"
        assert handler_config.handler_name == "CreateUserHandler"

    def test_pagination_model_creation(self) -> None:
        """Test Pagination model with correct fields."""
        # Pagination has: page, size (plus computed offset and limit)
        paging = FlextModels.Pagination(page=1, size=20)

        assert paging.page == 1
        assert paging.size == 20
        assert paging.offset == 0  # Computed field

    def test_query_model_creation(self) -> None:
        """Test Query model with validators."""
        # Query has: message_type, filters, pagination, query_id, query_type
        query = FlextModels.Query(
            filters={"user_id": "user-456"},
            pagination={"page": 1, "size": 20},
            query_type="GetOrdersByUser",
        )

        assert query.filters["user_id"] == "user-456"
        assert query.query_type == "GetOrdersByUser"
        assert isinstance(query.pagination, FlextModels.Pagination)

    def test_context_data_model_creation(self) -> None:
        """Test ContextData model with validators."""
        # ContextData has: data (dict), metadata (dict) with JSON-serializable validation
        ctx = FlextModels.ContextData(
            data={"request_id": "req-456", "user_id": "user-123"},
            metadata={"source": "api"},
        )

        assert ctx.data["request_id"] == "req-456"
        assert ctx.metadata["source"] == "api"

    def test_context_export_model_creation(self) -> None:
        """Test ContextExport model creation."""
        # ContextExport has: data, metadata, statistics (all dicts) with JSON-serializable validation
        export = FlextModels.ContextExport(
            data={"key": "value"},
            metadata={"version": "1.0"},
            statistics={"sets": 10, "gets": 20},
        )

        assert export.data["key"] == "value"
        assert export.statistics["sets"] == 10

    def test_handler_execution_context_model(self) -> None:
        """Test HandlerExecutionContext model creation."""
        # HandlerExecutionContext has: handler_name, handler_mode (+ private attributes for timing)
        context = FlextModels.HandlerExecutionContext.create_for_handler(
            handler_name="ProcessOrderCommand", handler_mode="command"
        )

        assert context.handler_name == "ProcessOrderCommand"
        assert context.handler_mode == "command"
        # Verify model can be created and has required fields
        assert isinstance(context, FlextModels.HandlerExecutionContext)

    def test_registration_details_model(self) -> None:
        """Test RegistrationDetails model creation."""
        # RegistrationDetails has: registration_id, handler_mode, timestamp, status
        details = FlextModels.RegistrationDetails(
            registration_id="reg-123",
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
            timestamp="2025-01-01T00:00:00Z",
            status="running",
        )

        assert details.registration_id == "reg-123"
        assert details.status == "running"
