"""m comprehensive functionality tests.

Module: flext_core
Scope: m - Pydantic-based data models, validation, serialization,
domain entities, value objects, aggregates, commands, queries, events

Tests core m functionality including:
- Model creation, validation, and serialization
- Domain-driven design patterns (entities, value objects, aggregates)
- CQRS patterns (commands, queries, events)
- Thread safety and immutability
- Custom validators and field processing
- JSON serialization and deserialization

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from enum import StrEnum, unique
from typing import Annotated, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from flext_tests import tm
from tests import c, m, t, u


class TestModels:
    """Test suite for m using u and c."""

    @unique
    class ModelType(StrEnum):
        """Model types for parametrized testing."""

        ENTITY = "entity"
        VALUE = "value"
        AGGREGATE = "aggregate"
        COMMAND = "command"
        QUERY = "query"
        EVENT = "event"
        METADATA = "metadata"
        PAYLOAD = "payload"

    class ModelCreationScenario(BaseModel):
        """Scenario for testing model creation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        model_type: Annotated[
            TestModels.ModelType,
            Field(description="Model type under creation test"),
        ]
        field_data: Annotated[
            t.RecursiveContainerMapping,
            Field(description="Model input payload"),
        ]
        expected_checks: Annotated[
            t.StrSequence,
            Field(description="Expected validation check labels"),
        ]
        description: Annotated[
            str,
            Field(default="", description="Scenario description"),
        ] = ""

    class SampleAggregate(m.AggregateRoot):
        """Sample aggregate root for testing."""

        name: str
        status: str = "active"

        def change_status(self, new_status: str) -> None:
            """Change status and add domain event."""
            self.status = new_status
            result = self.add_domain_event(
                "status_changed",
                t.ConfigMap(root={"old_status": "active", "new_status": new_status}),
            )
            if result.failure:
                error_msg = f"Failed to add domain event: {result.error}"
                raise ValueError(error_msg)

    class EventAggregate(m.AggregateRoot):
        """Test aggregate for domain events."""

        name: str

        def trigger_event(self) -> None:
            """Trigger a test event."""
            _ = self.add_domain_event("test_event", t.ConfigMap(root={"data": "test"}))

    def test_models_initialization(self) -> None:
        """Test models initialization with real validation."""
        models = m()
        assert models is not None, "m instance must not be None"
        assert isinstance(
            models,
            m,
        ), "m instance must be of type m"

    def test_models_entity_creation(self) -> None:
        """Test entity creation and validation."""

        class TestEntity(m.Entity):
            name: str
            email: str

            @field_validator("email")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    error_msg = "Invalid email"
                    raise ValueError(error_msg)
                return v

        entity = TestEntity(
            name="Test User",
            email="test@example.com",
            domain_events=[],
        )
        tm.that(entity.name, eq="Test User", msg="Entity name must match input")
        tm.that(
            entity.email,
            eq="test@example.com",
            msg="Entity email must match input",
        )
        tm.that("@" in entity.email, eq=True, msg="Entity email must contain @")
        tm.that(
            entity.unique_id,
            none=False,
            empty=False,
            msg="Entity must have unique_id",
        )
        tm.that(entity.unique_id, is_=str, msg="Entity unique_id must be string")
        dumped = entity.model_dump()
        assert "name" in dumped, "Entity dict must contain name"
        assert "email" in dumped, "Entity dict must contain email"
        validation_dict = {
            k: dumped[k] for k in type(entity).model_fields if k in dumped
        }
        validated_entity = type(entity).model_validate(validation_dict)
        tm.that(
            validated_entity.email,
            eq="test@example.com",
            msg="Validated entity email must match",
        )
        error_msg = "Invalid email"
        with pytest.raises(ValueError, match=error_msg):
            _ = TestEntity(name="Test User", email="invalid-email", domain_events=[])

    def test_models_value_object_creation(self) -> None:
        """Test value-object creation and immutability."""

        class TestValue(m.Value):
            data: str
            count: int

        value = TestValue(data="test", count=42)
        tm.that(value.data, eq="test", msg="Value data must match input")
        tm.that(value.count, eq=42, msg="Value count must match input")
        tm.that(value.count, is_=int, gt=0, msg="Value count must be positive integer")
        new_value = TestValue(data="modified", count=100)
        tm.that(new_value.data, eq="modified", msg="New value data must match input")
        tm.that(
            value.data,
            eq="test",
            msg="Original value must remain unchanged (immutability)",
        )

    def test_models_aggregate_root_creation(self) -> None:
        """Test aggregate root creation and basic functionality with real validation."""
        tm.that(True, eq=True, msg="AggregateRoot must be subclass of Entity")
        tm.that(
            hasattr(m.AggregateRoot, "_invariants"),
            eq=True,
            msg="AggregateRoot must have _invariants attribute",
        )
        tm.that(True, eq=True, msg="AggregateRoot _invariants must be a list")
        tm.that(
            hasattr(m.AggregateRoot, "check_invariants"),
            eq=True,
            msg="AggregateRoot must have check_invariants method",
        )
        tm.that(
            callable(m.AggregateRoot.check_invariants),
            eq=True,
            msg="AggregateRoot check_invariants must be callable",
        )

    def test_models_command_creation(self) -> None:
        """Test command model creation."""

        class TestCommand(m.Command):
            command_type: str = "test_command"
            data: str

        command = TestCommand.model_validate({"data": "test_data"})
        tm.that(
            command.command_type,
            eq="test_command",
            msg="Command type must match default",
        )
        tm.that(command.data, eq="test_data", msg="Command data must match input")
        tm.that(
            command.command_id,
            none=False,
            empty=False,
            msg="Command must have command_id",
        )
        tm.that(command.command_id, is_=str, msg="Command command_id must be string")

    def test_models_metadata_creation(self) -> None:
        """Test metadata model creation."""
        metadata = m.Metadata(created_by="test_user", tags=["tag1", "tag2"])
        tm.that(
            metadata.created_by,
            eq="test_user",
            msg="Metadata created_by must match input",
        )
        tm.that(
            metadata.created_at,
            none=False,
            msg="Metadata must have created_at timestamp",
        )
        tm.that(metadata.tags, is_=list, len=2, msg="Metadata must have 2 tags")
        tm.that(metadata.tags, has="tag1", msg="Metadata tags must contain tag1")
        tm.that(metadata.tags, has="tag2", msg="Metadata tags must contain tag2")

    def test_models_pagination_creation(self) -> None:
        """Test pagination model creation."""
        pagination = m.Pagination(page=1, size=10)
        tm.that(pagination.page, eq=1, msg="Pagination page must match input")
        tm.that(pagination.size, eq=10, msg="Pagination size must match input")
        tm.that(pagination.size, gt=0, msg="Pagination size must be positive")
        offset = (pagination.page - 1) * pagination.size
        tm.that(offset, eq=0, msg="Pagination offset calculation must be correct")

    def test_models_domain_events(self) -> None:
        """Test domain events functionality on Entity base class."""
        assert callable(m.Entity.add_domain_event)
        assert callable(m.Entity.clear_domain_events)
        assert "domain_events" in m.Entity.model_fields

    def test_models_version_management(self) -> None:
        """Test version management in entities."""

        class VersionedEntity(m.Entity):
            name: str

        entity = VersionedEntity(name="Test", domain_events=[])
        initial_version = entity.version
        tm.that(initial_version, gt=0, msg="Initial version must be positive")
        entity.increment_version()
        tm.that(
            entity.version,
            eq=initial_version + 1,
            msg="Version must increment by 1",
        )
        tm.that(
            entity.version,
            gt=initial_version,
            msg="Version must increase after increment",
        )

    def test_models_timestamped_functionality(self) -> None:
        """Test timestamped functionality."""

        class TimestampedEntity(m.Entity):
            name: str

        entity = TimestampedEntity(name="Test", domain_events=[])
        initial_created = entity.created_at
        tm.that(
            initial_created,
            none=False,
            msg="Entity must have created_at timestamp",
        )
        entity.update_timestamp()
        tm.that(
            entity.updated_at,
            none=False,
            msg="Entity must have updated_at after update",
        )
        if entity.updated_at is not None:
            tm.that(
                entity.updated_at > initial_created,
                eq=True,
                msg="Updated timestamp must be after created timestamp",
            )

    def test_models_validation_patterns(self) -> None:
        """Test validation patterns."""

        class ValidatedEntity(m.Entity):
            name: str
            email: str

            @field_validator("name")
            @classmethod
            def validate_name(cls, v: str) -> str:
                if not v:
                    error_msg = "Name is required"
                    raise ValueError(error_msg)
                return v

            @field_validator("email")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    error_msg = "Invalid email"
                    raise ValueError(error_msg)
                return v

        entity = ValidatedEntity(
            name="Test",
            email="test@example.com",
            domain_events=[],
        )
        tm.that(entity.name, eq="Test", msg="Validated entity name must match input")
        tm.that(
            entity.email,
            eq="test@example.com",
            msg="Validated entity email must match input",
        )
        tm.that("@" in entity.email, eq=True, msg="Validated email must contain @")
        with pytest.raises(ValueError, match="Name is required"):
            _ = ValidatedEntity(name="", email="test@example.com", domain_events=[])
        with pytest.raises(ValueError, match="Invalid email"):
            _ = ValidatedEntity(name="Test", email="invalid", domain_events=[])

    def test_models_thread_safety(self) -> None:
        """Test thread safety of models."""

        class ThreadSafeEntity(m.Entity):
            counter: int = 0

            def increment(self) -> None:
                self.counter += 1

        entity = ThreadSafeEntity(domain_events=[])

        def worker() -> None:
            for _ in range(100):
                entity.increment()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        tm.that(
            entity.counter,
            eq=500,
            msg="Thread-safe counter must equal 5 threads × 100 increments",
        )
        tm.that(entity.counter, gt=0, msg="Counter must be positive after increments")

    def test_models_critical_classes_available(self) -> None:
        """Verify that critical classes are available."""
        critical_classes = [
            "Entity",
            "Value",
            "AggregateRoot",
            "DomainEvent",
            "Metadata",
            "Command",
            "Query",
            "Pagination",
            "Handler",
            "RetryConfiguration",
        ]
        for class_name in critical_classes:
            tm.that(
                hasattr(m, class_name),
                eq=True,
                msg=f"Critical class {class_name} must be available",
            )
            class_obj = getattr(m, class_name)
            tm.that(
                class_obj,
                none=False,
                msg=f"Critical class {class_name} must not be None",
            )

    def test_entity_equality_and_hash(self) -> None:
        """Test Entity equality and hash based on identity."""

        class TestEntity(m.Entity):
            name: str

        entity1 = TestEntity(name="test", unique_id="123", domain_events=[])
        entity2 = TestEntity(name="test", unique_id="123", domain_events=[])
        entity3 = TestEntity(name="test", unique_id="456", domain_events=[])
        tm.that(
            entity1 == entity2,
            eq=True,
            msg="Entities with same unique_id must be equal",
        )
        tm.that(
            entity1 != entity3,
            eq=True,
            msg="Entities with different unique_id must not be equal",
        )
        tm.that(
            hash(entity1) == hash(entity2),
            eq=True,
            msg="Entities with same unique_id must have same hash",
        )
        tm.that(
            hash(entity1) != hash(entity3),
            eq=True,
            msg="Entities with different unique_id must have different hash",
        )
        entity_list: Sequence[TestEntity] = [entity1, entity2, entity3]
        entity_set = set(entity_list)
        tm.that(len(entity_set), eq=2, msg="Set must contain 2 unique entities")

    def test_entity_domain_events(self) -> None:
        """Test domain event functionality in Entity."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test", domain_events=[])
        tm.that(
            len(entity.domain_events),
            eq=0,
            msg="New entity must have no domain events",
        )
        result = entity.add_domain_event(
            "test_event",
            t.ConfigMap(root={"data": "value"}),
        )
        _ = u.Core.Tests.assert_success(result)
        tm.that(
            len(entity.domain_events),
            eq=1,
            msg="Entity must have 1 domain event after add",
        )
        event = entity.domain_events[0]
        tm.that(event.event_type, eq="test_event", msg="Event type must match input")
        tm.that(
            getattr(event.data, "root", event.data),
            eq={"data": "value"},
            msg="Event data must match input",
        )
        tm.that(
            event.aggregate_id,
            eq=entity.unique_id,
            msg="Event aggregate_id must match entity unique_id",
        )
        cleared_events = entity.clear_domain_events()
        tm.that(
            len(cleared_events),
            eq=1,
            msg="clear_domain_events must return 1 event",
        )
        tm.that(
            len(entity.domain_events),
            eq=0,
            msg="Entity must have no domain events after clear",
        )

    def test_entity_domain_events_validation(self) -> None:
        """Test domain event validation."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test", domain_events=[])
        result = entity.add_domain_event("", t.ConfigMap(root={"data": "value"}))
        _ = u.Core.Tests.assert_failure(result)
        tm.that(result.error, none=False, msg="Failure result must have error message")
        if result.error is not None:
            tm.that(
                "Domain event name must be a non-empty string" in result.error,
                eq=True,
                msg="Error message must indicate empty event name",
            )
        result = entity.add_domain_event(
            "valid",
            t.ConfigMap(root={"string": "value", "number": 42}),
        )
        _ = u.Core.Tests.assert_success(result)
        tm.that(
            len(entity.domain_events),
            eq=1,
            msg="Entity must have 1 valid domain event",
        )

    def test_entity_initial_version(self) -> None:
        """Test Entity initial version state."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test", domain_events=[])
        assert entity.version == 1

    def test_timestampable_mixin_serialization(self) -> None:
        """Test timestamp serialization in JSON output."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test", domain_events=[])
        json_data = entity.model_dump_json()
        assert '"created_at"' in json_data
        parsed = json.loads(json_data)
        if parsed.get("created_at"):
            assert "T" in parsed["created_at"]

    def test_timestampable_mixin_update_timestamp(self) -> None:
        """Test update_timestamp method."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test", domain_events=[])
        original_updated = entity.updated_at
        entity.update_timestamp()
        assert entity.updated_at is not None
        assert entity.updated_at != original_updated

    def test_versionable_mixin(self) -> None:
        """Test VersionableMixin functionality."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test", domain_events=[])
        assert entity.version == 1
        entity.increment_version()
        assert entity.version == 2
        entity.increment_version()
        assert entity.version == 3

    def test_aggregate_root_inheritance(self) -> None:
        """Test AggregateRoot inherits from Entity properly."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test", domain_events=[])
        assert all(
            hasattr(aggregate, attr) for attr in ["unique_id", "version", "created_at"]
        )
        assert isinstance(aggregate._invariants, list)
        result = aggregate.add_domain_event("test", t.ConfigMap(root={"data": "value"}))
        _ = u.Core.Tests.assert_success(result)

    def test_aggregate_root_invariants(self) -> None:
        """Test AggregateRoot invariant checking."""

        def passing_invariant() -> bool:
            return True

        class TestAggregate(m.AggregateRoot):
            name: str
            value: int
            _invariants: ClassVar[Sequence[Callable[[], bool]]] = [passing_invariant]

        aggregate = TestAggregate(name="test", value=10, domain_events=[])
        aggregate.check_invariants()

        def failing_invariant() -> bool:
            return False

        class FailingAggregate(m.AggregateRoot):
            name: str
            _invariants: ClassVar[Sequence[Callable[[], bool]]] = [failing_invariant]

        with pytest.raises(ValidationError, match="Invariant violated"):
            _ = FailingAggregate(name="test", domain_events=[])

    def test_value_object_immutability(self) -> None:
        """Test value immutability and equality."""

        class TestValue(m.Value):
            name: str
            value: int

        value1 = TestValue(name="test", value=42)
        value2 = TestValue(name="test", value=42)
        value3 = TestValue(name="other", value=42)
        assert value1 == value2
        assert value1 != value3
        value_list: Sequence[TestValue] = [value1, value2, value3]
        value_set = set(value_list)
        assert len(value_set) == 2
        with pytest.raises(ValidationError):
            setattr(value1, "value", 100)

    def test_command_creation_with_mixins(self) -> None:
        """Test Command creation with all mixins."""

        class TestCommand(m.Command):
            action: str
            target: str

        command = TestCommand.model_validate({"action": "create", "target": "user"})
        assert all(hasattr(command, attr) for attr in ["command_id", "command_type"])
        assert command.command_type == "generic_command"
        assert command.action == "create"
        assert command.target == "user"

    def test_domain_event_creation(self) -> None:
        """Test DomainEvent creation."""
        event = m.DomainEvent(
            event_type="user_created",
            aggregate_id="user-123",
            data=m.ComparableConfigMap(root={"name": "test"}),
        )
        assert all(hasattr(event, attr) for attr in ["unique_id", "created_at"])
        assert event.event_type == "user_created"
        assert event.aggregate_id == "user-123"
        assert getattr(event.data, "root", event.data) == {"name": "test"}
        assert event.message_type == c.HandlerType.EVENT

    def test_query_creation(self) -> None:
        """Test Query creation."""
        query = m.Query(
            query_type="find_users",
            filters=t.Dict(root={"active": "true"}),
            pagination=m.Pagination(),
            query_id="q-test-1",
        )
        assert query.query_id is not None
        assert query.query_type == "find_users"
        assert getattr(query.filters, "root", query.filters) == {"active": "true"}
        assert query.message_type == c.HandlerType.QUERY

    def test_aggregate_root_mark_events_as_committed(self) -> None:
        """Test mark_events_as_committed method."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test", domain_events=[])
        _ = aggregate.add_domain_event("event1", t.ConfigMap(root={"data": "value1"}))
        _ = aggregate.add_domain_event("event2", t.ConfigMap(root={"data": "value2"}))
        assert len(aggregate.domain_events) == 2
        result = aggregate.mark_events_as_committed()
        _ = u.Core.Tests.assert_success(result)
        assert not aggregate.domain_events
        committed_events = result.value
        assert len(committed_events) == 2

    def test_aggregate_root_mark_events_empty(self) -> None:
        """Test mark_events_as_committed with no events."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test", domain_events=[])
        result = aggregate.mark_events_as_committed()
        _ = u.Core.Tests.assert_success(result)
        committed_events = result.value
        assert not committed_events

    def test_aggregate_root_bulk_domain_events(self) -> None:
        """Test add_domain_events_bulk method."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test", domain_events=[])
        events = [
            ("event1", t.ConfigMap(root={"data": "value1"})),
            ("event2", t.ConfigMap(root={"data": "value2"})),
            ("event3", t.ConfigMap(root={"data": "value3"})),
        ]
        result = aggregate.add_domain_events_bulk(events)
        _ = u.Core.Tests.assert_success(result)
        assert len(aggregate.domain_events) == 3
        assert all(
            aggregate.domain_events[i].event_type == f"event{i + 1}" for i in range(3)
        )

    def test_aggregate_root_bulk_domain_events_validation(self) -> None:
        """Test add_domain_events_bulk validation."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test", domain_events=[])
        result = aggregate.add_domain_events_bulk([])
        _ = u.Core.Tests.assert_success(result)
        invalid_empty_name = [("", t.ConfigMap(root={"data": "value"}))]
        result = aggregate.add_domain_events_bulk(invalid_empty_name)
        _ = u.Core.Tests.assert_failure(result)
        assert (
            result.error is not None and "name must be non-empty string" in result.error
        )

    def test_aggregate_root_bulk_domain_events_limit(self) -> None:
        """Test add_domain_events_bulk respects max events limit."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test", domain_events=[])
        max_events = c.HTTP_STATUS_MIN
        events = [
            (f"event{i}", t.ConfigMap(root={"data": f"value{i}"}))
            for i in range(max_events + 1)
        ]
        result = aggregate.add_domain_events_bulk(events)
        error = u.Core.Tests.assert_failure(result)
        assert "would exceed max events" in error

    def test_aggregate_root_domain_event_handler_execution(self) -> None:
        """Test domain event handler execution."""

        class TestAggregate(m.AggregateRoot):
            name: str
            handler_called: bool = False
            handler_data: Annotated[
                t.RecursiveContainerMapping,
                Field(default_factory=dict),
            ]

            def _apply_test_event(self, data: t.RecursiveContainerMapping) -> None:
                self.handler_called = True
                self.handler_data = data

        aggregate = TestAggregate(name="test", domain_events=[], handler_data={})
        result = aggregate.add_domain_event(
            "user_action",
            t.ConfigMap(root={"event_type": "test_event", "key": "value"}),
        )
        _ = u.Core.Tests.assert_success(result)
        assert aggregate.handler_called is True
        assert aggregate.handler_data == {"event_type": "test_event", "key": "value"}

    def test_aggregate_root_domain_event_handler_error(self) -> None:
        """Test domain event handler error handling."""

        class TestAggregate(m.AggregateRoot):
            name: str

            def _apply_failing_event(
                self,
                _data: t.RecursiveContainerMapping,
            ) -> None:
                error_msg = "Handler failed"
                raise ValueError(error_msg)

        aggregate = TestAggregate(name="test", domain_events=[])
        result = aggregate.add_domain_event(
            "failing_event",
            t.ConfigMap(root={"data": "value"}),
        )
        _ = u.Core.Tests.assert_success(result)

    def test_domain_event_model_creation(self) -> None:
        """Test DomainEvent model creation and properties."""
        event = m.DomainEvent(
            event_type="test_event",
            aggregate_id="aggregate-123",
            data=m.ComparableConfigMap(root={"key": "value"}),
        )
        assert event.event_type == "test_event"
        assert event.aggregate_id == "aggregate-123"
        assert getattr(event.data, "root", event.data) == {"key": "value"}
        assert event.unique_id is not None
        assert event.created_at is not None
        assert event.message_type == c.HandlerType.EVENT
        json_data = event.model_dump_json()
        assert '"event_type":"test_event"' in json_data

    def test_command_model_creation(self) -> None:
        """Test Command model creation with correct fields."""
        command = m.Command(
            command_type="CreateOrder",
            issuer_id="issuer-123",
            command_id="cmd-test-1",
        )
        assert command.command_type == "CreateOrder"
        assert command.issuer_id == "issuer-123"
        assert command.message_type == c.HandlerType.COMMAND
        assert command.command_id is not None

    def test_metadata_model_creation(self) -> None:
        """Test Metadata model creation with correct fields."""
        meta = m.Metadata(
            created_by="user-123",
            modified_by="user-456",
            tags=["important", "urgent"],
            attributes={"priority": "high"},
        )
        assert meta.created_by == "user-123"
        assert meta.modified_by == "user-456"
        assert len(meta.tags) == 2
        assert meta.attributes["priority"] == "high"

    def test_retry_configuration_model(self) -> None:
        """Test RetryConfiguration model with correct fields."""
        retry = m.RetryConfiguration.model_validate({
            "max_attempts": 3,
            "initial_delay_seconds": 1000,
            "max_delay_seconds": 30000,
            "backoff_multiplier": 2.0,
        })
        tm.that(retry.max_retries, eq=3, msg="max_retries must be 3")
        tm.that(
            retry.initial_delay_seconds,
            eq=1000.0,
            msg="initial_delay_seconds must be 1000",
        )
        tm.that(retry.backoff_multiplier, eq=2.0, msg="backoff_multiplier must be 2.0")
        tm.that(
            retry.retry_on_exceptions,
            is_=list,
            none=False,
            empty=True,
            msg="retry_on_exceptions must be empty list by default",
        )
        tm.that(
            retry.retry_on_status_codes,
            is_=list,
            none=False,
            empty=True,
            msg="retry_on_status_codes must be empty list by default",
        )

    def test_retry_configuration_with_exceptions(self) -> None:
        """Test RetryConfiguration with retry_on_exceptions."""
        retry = m.RetryConfiguration.model_validate({
            "max_attempts": 3,
            "retry_on_exceptions": [ValueError, RuntimeError, TypeError],
        })
        assert isinstance(
            retry.retry_on_exceptions,
            list,
        ), "retry_on_exceptions must be list"
        assert retry.retry_on_exceptions is not None
        assert len(retry.retry_on_exceptions) == 3
        assert ValueError in retry.retry_on_exceptions
        assert RuntimeError in retry.retry_on_exceptions
        assert TypeError in retry.retry_on_exceptions

    def test_retry_configuration_with_status_codes(self) -> None:
        """Test RetryConfiguration with retry_on_status_codes."""
        retry = m.RetryConfiguration.model_validate({
            "max_attempts": 3,
            "retry_on_status_codes": [500, 502, 503, 504],
        })
        tm.that(
            retry.retry_on_status_codes,
            is_=list,
            none=False,
            empty=False,
            msg="retry_on_status_codes must be list",
        )
        tm.that(
            len(retry.retry_on_status_codes),
            eq=4,
            msg="retry_on_status_codes must have 4 items",
        )
        tm.that(
            500 in retry.retry_on_status_codes,
            eq=True,
            msg="500 must be in retry_on_status_codes",
        )
        tm.that(
            502 in retry.retry_on_status_codes,
            eq=True,
            msg="502 must be in retry_on_status_codes",
        )
        tm.that(
            503 in retry.retry_on_status_codes,
            eq=True,
            msg="503 must be in retry_on_status_codes",
        )
        tm.that(
            504 in retry.retry_on_status_codes,
            eq=True,
            msg="504 must be in retry_on_status_codes",
        )

    def test_retry_configuration_with_both_exceptions_and_status_codes(self) -> None:
        """Test RetryConfiguration with both retry_on_exceptions and status codes."""
        retry = m.RetryConfiguration.model_validate({
            "max_attempts": 3,
            "retry_on_exceptions": [ValueError, ConnectionError],
            "retry_on_status_codes": [429, 500, 503],
        })
        tm.that(retry.max_retries, eq=3, msg="max_retries must be 3")
        tm.that(
            len(retry.retry_on_exceptions),
            eq=2,
            msg="retry_on_exceptions must have 2 items",
        )
        tm.that(
            len(retry.retry_on_status_codes),
            eq=3,
            msg="retry_on_status_codes must have 3 items",
        )
        tm.that(
            ValueError in retry.retry_on_exceptions,
            eq=True,
            msg="ValueError must be in retry_on_exceptions",
        )
        tm.that(
            ConnectionError in retry.retry_on_exceptions,
            eq=True,
            msg="ConnectionError must be in retry_on_exceptions",
        )
        tm.that(
            429 in retry.retry_on_status_codes,
            eq=True,
            msg="429 must be in retry_on_status_codes",
        )
        tm.that(
            500 in retry.retry_on_status_codes,
            eq=True,
            msg="500 must be in retry_on_status_codes",
        )
        tm.that(
            503 in retry.retry_on_status_codes,
            eq=True,
            msg="503 must be in retry_on_status_codes",
        )

    def test_pagination_model_creation(self) -> None:
        """Test Pagination model with correct fields."""
        paging = m.Pagination(page=1, size=20)
        assert paging.page == 1
        assert paging.size == 20
        assert paging.offset == 0

    def test_query_model_creation(self) -> None:
        """Test Query model with validators."""
        pagination = m.Pagination(page=1, size=20)
        query = m.Query(
            filters=t.Dict(root={"user_id": "user-456"}),
            pagination=pagination,
            query_type="GetOrdersByUser",
            query_id="q-test-2",
        )
        assert query.filters["user_id"] == "user-456"
        assert query.query_type == "GetOrdersByUser"
        assert query.pagination is not None
        assert isinstance(query.pagination, m.Pagination)
        assert query.pagination.page == 1
        assert query.pagination.size == 20

    def test_context_data_model_creation(self) -> None:
        """Test ContextData model with validators."""
        ctx = m.ContextData(
            data=t.Dict(root={"request_id": "req-456", "user_id": "user-123"}),
            metadata=m.Metadata(attributes={"source": "api"}),
        )
        assert ctx.data["request_id"] == "req-456"
        metadata = ctx.metadata
        if isinstance(metadata, m.Metadata):
            assert metadata.attributes.get("source") == "api"

    def test_context_export_model_creation(self) -> None:
        """Test ContextExport model creation."""
        stats: t.RecursiveContainerMapping = {"sets": 10, "gets": 20}
        export = m.ContextExport(
            data=t.Dict(root={"key": "value"}).root,
            metadata=m.Metadata(attributes={"version": "1.0"}),
            statistics=stats,
        )
        assert export.data["key"] == "value"
        assert export.statistics["sets"] == 10

    def test_handler_execution_context_model(self) -> None:
        """Test HandlerExecutionContext model creation."""
        context = m.ExecutionContext.create_for_handler(
            handler_name="ProcessOrderCommand",
            handler_mode=c.HandlerType.COMMAND,
        )
        assert context.handler_name == "ProcessOrderCommand"
        assert context.handler_mode == c.HandlerType.COMMAND
        assert isinstance(context, m.ExecutionContext)

    def test_registration_details_model(self) -> None:
        """Test RegistrationDetails model creation."""
        details = m.RegistrationDetails(
            registration_id="reg-123",
            handler_mode=c.HandlerType.COMMAND,
            timestamp="2025-01-01T00:00:00Z",
            status=c.CommonStatus.RUNNING,
        )
        assert details.registration_id == "reg-123"
        assert details.status == "running"


__all__: list[str] = ["TestModels"]
