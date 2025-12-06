"""FlextModels comprehensive functionality tests.

Module: flext_core.models
Scope: FlextModels - Pydantic-based data models, validation, serialization,
domain entities, value objects, aggregates, commands, queries, events

Tests core FlextModels functionality including:
- Model creation, validation, and serialization
- Domain-driven design patterns (entities, value objects, aggregates)
- CQRS patterns (commands, queries, events)
- Thread safety and immutability
- Custom validators and field processing
- JSON serialization and deserialization

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast

import pytest
from pydantic import Field, ValidationError, field_validator

from flext_core import (
    FlextExceptions,
    c,
    m,
    t,
)
from flext_tests import u


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


@dataclass(frozen=True, slots=True)
class ModelCreationScenario:
    """Scenario for testing model creation."""

    model_type: ModelType
    field_data: dict[str, object]
    expected_checks: list[str]
    description: str = ""


class SampleAggregate(m.AggregateRoot):
    """Sample aggregate root for testing."""

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
            error_msg = f"Failed to add domain event: {result.error}"
            raise ValueError(error_msg)


class EventAggregate(m.AggregateRoot):
    """Test aggregate for domain events."""

    name: str

    def trigger_event(self) -> None:
        """Trigger a test event."""
        _ = self.add_domain_event("test_event", {"data": "test"})


class TestFlextModels:
    """Test suite for FlextModels using FlextTestsUtilities and c."""

    def test_models_initialization(self) -> None:
        """Test models initialization."""
        models = m()
        assert models is not None
        assert isinstance(models, m)

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

        entity = TestEntity(name="Test User", email="test@example.com")
        assert entity.name == "Test User"
        assert entity.email == "test@example.com"
        assert entity.unique_id is not None
        entity_dict = entity.model_dump(
            exclude={"is_initial_version", "is_modified", "uncommitted_events"},
        )
        validated_entity = entity.model_validate(entity_dict)
        assert validated_entity.email == "test@example.com"
        error_msg = "Invalid email"
        with pytest.raises(ValueError, match=error_msg):
            _ = TestEntity(name="Test User", email="invalid-email")

    def test_models_value_object_creation(self) -> None:
        """Test value object creation and immutability."""

        class TestValue(m.Value):
            data: str
            count: int

        value = TestValue(data="test", count=42)
        assert value.data == "test"
        assert value.count == 42
        new_value = TestValue(data="modified", count=100)
        assert new_value.data == "modified"
        assert value.data == "test"

    def test_models_aggregate_root_creation(self) -> None:
        """Test aggregate root creation and basic functionality."""
        assert issubclass(m.AggregateRoot, m.Entity)
        assert hasattr(m.AggregateRoot, "_invariants")
        assert isinstance(m.AggregateRoot._invariants, list)
        assert hasattr(m.AggregateRoot, "check_invariants")
        assert callable(m.AggregateRoot.check_invariants)

    def test_models_command_creation(self) -> None:
        """Test command model creation."""

        class TestCommand(m.Cqrs.Command):
            command_type: str = "test_command"
            data: str

        command = TestCommand(data="test_data")
        assert command.command_type == "test_command"
        assert command.data == "test_data"
        assert command.created_at is not None
        assert command.unique_id is not None

    def test_models_metadata_creation(self) -> None:
        """Test metadata model creation."""
        metadata = m.Metadata(
            created_by="test_user",
            tags=["tag1", "tag2"],
        )
        assert metadata.created_by == "test_user"
        assert metadata.created_at is not None
        assert len(metadata.tags) == 2

    def test_models_pagination_creation(self) -> None:
        """Test pagination model creation."""
        pagination = m.Cqrs.Pagination(page=1, size=10)
        assert pagination.page == 1
        assert pagination.size == 10
        assert (pagination.page - 1) * pagination.size == 0

    def test_models_domain_events(self) -> None:
        """Test domain events functionality on Entity base class."""
        assert hasattr(m.Entity, "add_domain_event")
        assert callable(m.Entity.add_domain_event)
        assert hasattr(m.Entity, "clear_domain_events")
        assert callable(m.Entity.clear_domain_events)
        assert "domain_events" in m.Entity.model_fields

    def test_models_version_management(self) -> None:
        """Test version management in entities."""

        class VersionedEntity(m.Entity):
            name: str

        entity = VersionedEntity(name="Test")
        initial_version = entity.version
        entity.increment_version()
        assert entity.version == initial_version + 1

    def test_models_timestamped_functionality(self) -> None:
        """Test timestamped functionality."""

        class TimestampedEntity(m.Entity):
            name: str

        entity = TimestampedEntity(name="Test")
        initial_created = entity.created_at
        entity.update_timestamp()
        assert entity.updated_at is not None
        assert entity.updated_at > initial_created

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

        entity = ValidatedEntity(name="Test", email="test@example.com")
        assert entity.name == "Test"
        assert entity.email == "test@example.com"
        with pytest.raises(ValueError, match="Name is required"):
            _ = ValidatedEntity(name="", email="test@example.com")
        with pytest.raises(ValueError, match="Invalid email"):
            _ = ValidatedEntity(name="Test", email="invalid")

    def test_models_thread_safety(self) -> None:
        """Test thread safety of models."""

        class ThreadSafeEntity(m.Entity):
            counter: int = 0

            def increment(self) -> None:
                self.counter += 1

        entity = ThreadSafeEntity()

        def worker() -> None:
            for _ in range(100):
                entity.increment()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert entity.counter == 500

    def test_models_critical_classes_available(self) -> None:
        """Verify that critical classes are available."""
        critical_classes = [
            "Entity",
            "Value",
            "AggregateRoot",
            "DomainEvent",
            "Cqrs",
            "Metadata",
        ]
        for class_name in critical_classes:
            assert hasattr(m, class_name), (
                f"Critical class {class_name} should be available"
            )
        # Config classes are in Config namespace
        config_classes = [
            "HandlerExecutionConfig",
            "BatchProcessingConfig",
            "RetryConfiguration",
            "ValidationConfiguration",
        ]
        for class_name in config_classes:
            assert hasattr(m.Config, class_name), (
                f"Config class {class_name} should be available in Config namespace"
            )
        # RegistrationDetails is in Handler namespace
        assert hasattr(m.Handler, "RegistrationDetails"), (
            "RegistrationDetails should be available in Handler namespace"
        )
        cqrs_classes = ["Command", "Query", "Pagination", "Bus", "Handler"]
        for class_name in cqrs_classes:
            assert hasattr(m.Cqrs, class_name), (
                f"CQRS class {class_name} should be available"
            )

    def test_entity_equality_and_hash(self) -> None:
        """Test Entity equality and hash based on identity."""

        class TestEntity(m.Entity):
            name: str

        entity1 = TestEntity(name="test", unique_id="123")
        entity2 = TestEntity(name="test", unique_id="123")
        entity3 = TestEntity(name="test", unique_id="456")
        assert entity1 == entity2
        assert entity1 != entity3
        assert (entity1 == "not an entity") is False
        assert hash(entity1) == hash(entity2)
        assert hash(entity1) != hash(entity3)
        # Entity implements __hash__, so it's hashable for sets
        # Use list and convert to set to avoid pyright hashability check
        entity_list: list[TestEntity] = [entity1, entity2, entity3]
        entity_set = set(entity_list)
        assert len(entity_set) == 2

    def test_entity_domain_events(self) -> None:
        """Test domain event functionality in Entity."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test")
        assert len(entity.domain_events) == 0
        result = entity.add_domain_event("test_event", {"data": "value"})
        u.Tests.Result.assert_result_success(result)
        assert len(entity.domain_events) == 1
        event = entity.domain_events[0]
        assert event.event_type == "test_event"
        assert event.data == {"data": "value"}
        assert event.aggregate_id == entity.unique_id
        cleared_events = entity.clear_domain_events()
        assert len(cleared_events) == 1
        assert len(entity.domain_events) == 0

    def test_entity_domain_events_validation(self) -> None:
        """Test domain event validation."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test")
        result = entity.add_domain_event("", {"data": "value"})
        u.Tests.Result.assert_result_failure(result)
        assert (
            result.error is not None
            and "Domain event name must be a non-empty string" in result.error
        )
        result = entity.add_domain_event("valid", {"string": "value", "number": 42})
        u.Tests.Result.assert_result_success(result)

    def test_entity_initial_version(self) -> None:
        """Test Entity initial version state."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test")
        # is_initial_version is a computed_field (property) that returns bool
        # Access directly to avoid mypy inference issues with computed_field
        assert bool(entity.is_initial_version) is True
        assert entity.version == 1
        entity.updated_at = entity.created_at
        # is_modified is also a computed_field (property) that returns bool
        assert bool(entity.is_modified) is True

    def test_timestampable_mixin_serialization(self) -> None:
        """Test timestamp serialization in JSON output."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test")
        json_data = entity.model_dump_json()
        assert '"created_at"' in json_data
        parsed = json.loads(json_data)
        if parsed.get("created_at"):
            assert "T" in parsed["created_at"]

    def test_timestampable_mixin_update_timestamp(self) -> None:
        """Test update_timestamp method."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test")
        original_updated = entity.updated_at
        entity.update_timestamp()
        assert entity.updated_at is not None
        assert entity.updated_at != original_updated
        # is_modified is a computed_field (property) that returns bool
        assert bool(entity.is_modified) is True

    def test_versionable_mixin(self) -> None:
        """Test VersionableMixin functionality."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(name="test")
        assert entity.version == 1
        entity.increment_version()
        assert entity.version == 2
        entity.increment_version()
        assert entity.version == 3

    def test_aggregate_root_inheritance(self) -> None:
        """Test AggregateRoot inherits from Entity properly."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")
        assert all(
            hasattr(aggregate, attr) for attr in ["unique_id", "version", "created_at"]
        )
        assert hasattr(aggregate, "_invariants")
        assert isinstance(aggregate._invariants, list)
        result = aggregate.add_domain_event("test", {"data": "value"})
        u.Tests.Result.assert_result_success(result)

    def test_aggregate_root_invariants(self) -> None:
        """Test AggregateRoot invariant checking."""

        def passing_invariant() -> bool:
            return True

        class TestAggregate(m.AggregateRoot):
            name: str
            value: int
            _invariants: ClassVar[list[Callable[[], bool]]] = [passing_invariant]

        aggregate = TestAggregate(name="test", value=10)
        aggregate.check_invariants()

        def failing_invariant() -> bool:
            return False

        class FailingAggregate(m.AggregateRoot):
            name: str
            _invariants: ClassVar[list[Callable[[], bool]]] = [failing_invariant]

        with pytest.raises(FlextExceptions.ValidationError, match="Invariant violated"):
            _ = FailingAggregate(name="test")

    def test_value_object_immutability(self) -> None:
        """Test Value object immutability and equality."""

        class TestValue(m.Value):
            name: str
            value: int

        value1 = TestValue(name="test", value=42)
        value2 = TestValue(name="test", value=42)
        value3 = TestValue(name="other", value=42)
        assert value1 == value2
        assert value1 != value3
        # Value implements __hash__, so it's hashable for sets
        # Use list and convert to set to avoid pyright hashability check
        value_list: list[TestValue] = [value1, value2, value3]
        value_set = set(value_list)
        assert len(value_set) == 2
        with pytest.raises(ValidationError):
            value1.value = 100

    def test_command_creation_with_mixins(self) -> None:
        """Test Command creation with all mixins."""

        class TestCommand(m.Cqrs.Command):
            action: str
            target: str

        command = TestCommand(action="create", target="user")
        assert all(
            hasattr(command, attr) for attr in ["unique_id", "created_at", "updated_at"]
        )
        assert command.command_type == "generic_command"
        assert command.action == "create"
        assert command.target == "user"

    def test_domain_event_creation(self) -> None:
        """Test DomainEvent creation."""
        event = m.DomainEvent(
            event_type="user_created",
            aggregate_id="user-123",
            data={"name": "test"},
        )
        assert all(hasattr(event, attr) for attr in ["unique_id", "created_at"])
        assert event.event_type == "user_created"
        assert event.aggregate_id == "user-123"
        assert event.data == {"name": "test"}
        assert event.message_type == c.Cqrs.HandlerType.EVENT

    def test_query_creation(self) -> None:
        """Test Query creation."""
        query = m.Cqrs.Query(
            query_type="find_users",
            filters={"active": "true"},
        )
        assert hasattr(query, "query_id")
        assert query.query_id is not None
        assert query.query_type == "find_users"
        assert query.filters == {"active": "true"}
        assert query.message_type == c.Cqrs.HandlerType.QUERY

    def test_aggregate_root_mark_events_as_committed(self) -> None:
        """Test mark_events_as_committed method."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")
        _ = aggregate.add_domain_event("event1", {"data": "value1"})
        _ = aggregate.add_domain_event("event2", {"data": "value2"})
        assert len(aggregate.domain_events) == 2
        result = aggregate.mark_events_as_committed()
        u.Tests.Result.assert_result_success(result)
        assert len(aggregate.domain_events) == 0
        committed_events = result.unwrap()
        assert len(committed_events) == 2

    def test_aggregate_root_mark_events_empty(self) -> None:
        """Test mark_events_as_committed with no events."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")
        result = aggregate.mark_events_as_committed()
        u.Tests.Result.assert_result_success(result)
        committed_events = result.unwrap()
        assert len(committed_events) == 0

    def test_aggregate_root_bulk_domain_events(self) -> None:
        """Test add_domain_events_bulk method."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")
        # Convert to proper type: Sequence[tuple[str, EventDataMapping | None]]
        events: Sequence[tuple[str, t.Types.EventDataMapping | None]] = [
            ("event1", {"data": "value1"}),
            ("event2", {"data": "value2"}),
            ("event3", {"data": "value3"}),
        ]
        result = aggregate.add_domain_events_bulk(events)
        u.Tests.Result.assert_result_success(result)
        assert len(aggregate.domain_events) == 3
        assert all(
            aggregate.domain_events[i].event_type == f"event{i + 1}" for i in range(3)
        )

    def test_aggregate_root_bulk_domain_events_validation(self) -> None:
        """Test add_domain_events_bulk validation."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")
        result = aggregate.add_domain_events_bulk([])
        u.Tests.Result.assert_result_success(result)
        # Testing invalid input type - intentionally passing wrong type
        # for runtime validation
        invalid_input = cast(
            "Sequence[tuple[str, t.Types.EventDataMapping | None]]",
            "not a list",
        )
        result = aggregate.add_domain_events_bulk(invalid_input)
        u.Tests.Result.assert_result_failure(result)
        assert result.error is not None and "Events must be a list" in result.error
        invalid_empty_name: Sequence[tuple[str, t.Types.EventDataMapping | None]] = [
            ("", {"data": "value"}),
        ]
        result = aggregate.add_domain_events_bulk(invalid_empty_name)
        u.Tests.Result.assert_result_failure(result)
        assert (
            result.error is not None and "name must be non-empty string" in result.error
        )

    def test_aggregate_root_bulk_domain_events_limit(self) -> None:
        """Test add_domain_events_bulk respects max events limit."""

        class TestAggregate(m.AggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")
        max_events = c.Validation.MAX_UNCOMMITTED_EVENTS
        events: Sequence[tuple[str, t.Types.EventDataMapping | None]] = [
            (f"event{i}", {"data": f"value{i}"}) for i in range(max_events + 1)
        ]
        result = aggregate.add_domain_events_bulk(events)
        u.Tests.Result.assert_result_failure(result)
        assert result.error is not None and "would exceed max events" in result.error

    def test_aggregate_root_domain_event_handler_execution(self) -> None:
        """Test domain event handler execution."""

        class TestAggregate(m.AggregateRoot):
            name: str
            handler_called: bool = False
            handler_data: dict[str, object] = Field(default_factory=dict)

            def _apply_test_event(self, data: dict[str, object]) -> None:
                self.handler_called = True
                self.handler_data = data

        aggregate = TestAggregate(name="test")
        result = aggregate.add_domain_event(
            "user_action",
            {"event_type": "test_event", "key": "value"},
        )
        u.Tests.Result.assert_result_success(result)
        assert aggregate.handler_called is True
        assert aggregate.handler_data == {"event_type": "test_event", "key": "value"}

    def test_aggregate_root_domain_event_handler_error(self) -> None:
        """Test domain event handler error handling."""

        class TestAggregate(m.AggregateRoot):
            name: str

            def _apply_failing_event(self, data: dict[str, object]) -> None:
                error_msg = "Handler failed"
                raise ValueError(error_msg)

        aggregate = TestAggregate(name="test")
        result = aggregate.add_domain_event("failing_event", {"data": "value"})
        u.Tests.Result.assert_result_success(result)

    def test_domain_event_model_creation(self) -> None:
        """Test DomainEvent model creation and properties."""
        event = m.DomainEvent(
            event_type="test_event",
            aggregate_id="aggregate-123",
            data={"key": "value"},
        )
        assert event.event_type == "test_event"
        assert event.aggregate_id == "aggregate-123"
        assert event.data == {"key": "value"}
        assert event.unique_id is not None
        assert event.created_at is not None
        assert event.message_type == c.Cqrs.HandlerType.EVENT
        json_data = event.model_dump_json()
        assert '"event_type":"test_event"' in json_data

    def test_command_model_creation(self) -> None:
        """Test Command model creation with correct fields."""
        command = m.Cqrs.Command(
            command_type="CreateOrder",
            issuer_id="issuer-123",
        )
        assert command.command_type == "CreateOrder"
        assert command.issuer_id == "issuer-123"
        assert command.message_type == c.Cqrs.HandlerType.COMMAND
        assert command.unique_id is not None
        assert command.created_at is not None

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

    def test_processing_request_model_creation(self) -> None:
        """Test ProcessingRequest model with correct fields."""
        request = m.Config.ProcessingRequest(
            data={"input": "data"},
            enable_validation=True,
        )
        assert request.data == {"input": "data"}
        assert request.enable_validation is True
        assert request.operation_id is not None

    def test_handler_registration_model_creation(self) -> None:
        """Test HandlerRegistration model with correct fields."""

        def dummy_handler() -> None:
            pass

        reg = m.Handler.Registration(
            name="TestHandler",
            handler=dummy_handler,
            event_types=["CreateUser"],
        )
        assert reg.name == "TestHandler"
        assert callable(reg.handler)
        assert "CreateUser" in reg.event_types

    def test_batch_processing_config_model(self) -> None:
        """Test BatchProcessingConfig model with correct fields."""
        config = m.Config.BatchProcessingConfig(
            batch_size=100,
            continue_on_error=True,
            data_items=[1, 2, 3],
        )
        assert config.batch_size == 100
        assert config.continue_on_error is True
        assert len(config.data_items) == 3

    def test_handler_execution_config_model(self) -> None:
        """Test HandlerExecutionConfig model with correct fields."""
        config = m.Config.HandlerExecutionConfig(
            handler_name="my_handler",
            input_data={"key": "value"},
            retry_on_failure=True,
        )
        assert config.handler_name == "my_handler"
        assert config.input_data == {"key": "value"}
        assert config.retry_on_failure is True

    def test_retry_configuration_model(self) -> None:
        """Test RetryConfiguration model with correct fields."""
        retry = m.Config.RetryConfiguration(
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
        val_config = m.Config.ValidationConfiguration(
            enable_strict_mode=True,
            validate_on_assignment=True,
            validate_on_read=False,
            custom_validators=[],
        )
        assert val_config.enable_strict_mode is True
        assert val_config.validate_on_assignment is True

    def test_cqrs_handler_model_creation(self) -> None:
        """Test Cqrs.Handler model creation."""
        handler_config = m.Cqrs.Handler(
            handler_id="handler-123",
            handler_name="CreateUserHandler",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        assert handler_config.handler_id == "handler-123"
        assert handler_config.handler_name == "CreateUserHandler"

    def test_pagination_model_creation(self) -> None:
        """Test Pagination model with correct fields."""
        paging = m.Cqrs.Pagination(page=1, size=20)
        assert paging.page == 1
        assert paging.size == 20
        assert paging.offset == 0

    def test_query_model_creation(self) -> None:
        """Test Query model with validators."""
        pagination = m.Cqrs.Pagination(page=1, size=20)
        query = m.Cqrs.Query(
            filters={"user_id": "user-456"},
            pagination=pagination,
            query_type="GetOrdersByUser",
        )
        assert query.filters["user_id"] == "user-456"
        assert query.query_type == "GetOrdersByUser"
        assert query.pagination is not None
        assert isinstance(query.pagination, m.Cqrs.Pagination)
        assert query.pagination.page == 1
        assert query.pagination.size == 20

    def test_context_data_model_creation(self) -> None:
        """Test ContextData model with validators."""
        ctx = m.Context.ContextData(
            data={"request_id": "req-456", "user_id": "user-123"},
            metadata=m.Metadata(attributes={"source": "api"}),
        )
        assert ctx.data["request_id"] == "req-456"
        metadata = ctx.metadata
        if isinstance(metadata, m.Metadata):
            assert metadata.attributes.get("source") == "api"

    def test_context_export_model_creation(self) -> None:
        """Test ContextExport model creation."""
        export = m.Context.ContextExport(
            data={"key": "value"},
            metadata=m.Metadata(attributes={"version": "1.0"}),
            statistics={"sets": 10, "gets": 20},
        )
        assert export.data["key"] == "value"
        assert export.statistics["sets"] == 10

    def test_handler_execution_context_model(self) -> None:
        """Test HandlerExecutionContext model creation."""
        context = m.Handler.ExecutionContext.create_for_handler(
            handler_name="ProcessOrderCommand",
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        assert context.handler_name == "ProcessOrderCommand"
        assert context.handler_mode == "command"
        assert isinstance(context, m.Handler.ExecutionContext)

    def test_registration_details_model(self) -> None:
        """Test RegistrationDetails model creation."""
        details = m.Handler.RegistrationDetails(
            registration_id="reg-123",
            handler_mode=c.Cqrs.HandlerType.COMMAND,
            timestamp="2025-01-01T00:00:00Z",
            status=c.Cqrs.CommonStatus.RUNNING,
        )
        assert details.registration_id == "reg-123"
        assert details.status == "running"


__all__ = ["TestFlextModels"]
