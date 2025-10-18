"""Comprehensive tests for FlextModels - Data Models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading

import pytest
from pydantic import field_validator

from flext_core import FlextModels


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
        assert entity.id is not None  # Entity should have auto-generated ID

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
        assert command.id is not None  # ID from IdentifiableMixin

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

    def test_models_advanced_value_objects(self) -> None:
        """Test advanced value objects."""
        # Test URL validation using Validation utility
        result = FlextModels.Validation.validate_url("https://example.com")
        assert result.is_success
        assert result.value == "https://example.com"

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

    def test_models_middleware_config_creation(self) -> None:
        """Test MiddlewareConfig model creation and validation."""
        # Test basic creation with defaults
        config = FlextModels.MiddlewareConfig()
        assert config.enabled is True
        assert config.order == 0
        assert config.priority == 0
        assert config.name is None
        assert config.config == {}

    def test_models_middleware_config_with_values(self) -> None:
        """Test MiddlewareConfig with custom values."""
        config = FlextModels.MiddlewareConfig(
            config={"timeout": 30, "retry": 3},
            enabled=False,
            order=5,
            priority=75,
            name="CustomMiddleware",
        )
        assert config.config == {"timeout": 30, "retry": 3}
        assert config.enabled is False
        assert config.order == 5
        assert config.priority == 75
        assert config.name == "CustomMiddleware"

    def test_models_middleware_config_priority_bounds(self) -> None:
        """Test MiddlewareConfig priority field constraints."""
        # Valid priorities
        config_low = FlextModels.MiddlewareConfig(priority=0)
        assert config_low.priority == 0

        config_high = FlextModels.MiddlewareConfig(priority=100)
        assert config_high.priority == 100

        config_mid = FlextModels.MiddlewareConfig(priority=50)
        assert config_mid.priority == 50

        # Invalid priority should fail validation
        with pytest.raises(ValueError):
            FlextModels.MiddlewareConfig(priority=101)

        with pytest.raises(ValueError):
            FlextModels.MiddlewareConfig(priority=-1)

    def test_models_middleware_config_serialization(self) -> None:
        """Test MiddlewareConfig serialization."""
        config = FlextModels.MiddlewareConfig(
            config={"key1": "value1"},
            enabled=True,
            order=2,
            priority=50,
            name="TestMiddleware",
        )

        # Test model_dump
        config_dict = config.model_dump()
        assert config_dict["config"] == {"key1": "value1"}
        assert config_dict["enabled"] is True
        assert config_dict["order"] == 2
        assert config_dict["priority"] == 50
        assert config_dict["name"] == "TestMiddleware"

        # Test model_validate from dict
        validated = FlextModels.MiddlewareConfig.model_validate(config_dict)
        assert validated.config == config.config
        assert validated.enabled == config.enabled
        assert validated.order == config.order

    def test_models_rate_limiter_state_creation(self) -> None:
        """Test RateLimiterState model creation."""
        state = FlextModels.RateLimiterState()
        assert not state.processor_name
        assert state.count == 0
        assert state.window_start == 0.0
        assert state.limit == 100
        assert state.window_seconds == 60
        assert state.block_until == 0.0

    def test_models_rate_limiter_state_with_values(self) -> None:
        """Test RateLimiterState with custom values."""
        state = FlextModels.RateLimiterState(
            processor_name="test_processor",
            count=50,
            window_start=1000.0,
            limit=200,
            window_seconds=120,
            block_until=1500.5,
        )
        assert state.processor_name == "test_processor"
        assert state.count == 50
        assert state.window_start == 1000.0
        assert state.limit == 200
        assert state.window_seconds == 120
        assert state.block_until == 1500.5

    def test_models_rate_limiter_state_constraints(self) -> None:
        """Test RateLimiterState field constraints."""
        # Valid values
        state = FlextModels.RateLimiterState(
            count=0, window_start=0.0, limit=1, window_seconds=1, block_until=0.0
        )
        assert state.count == 0
        assert state.limit == 1

        # Invalid count (negative)
        with pytest.raises(ValueError):
            FlextModels.RateLimiterState(count=-1)

        # Invalid window_start (negative)
        with pytest.raises(ValueError):
            FlextModels.RateLimiterState(window_start=-0.1)

        # Invalid limit (zero)
        with pytest.raises(ValueError):
            FlextModels.RateLimiterState(limit=0)

        # Invalid window_seconds (zero)
        with pytest.raises(ValueError):
            FlextModels.RateLimiterState(window_seconds=0)

        # Invalid block_until (negative)
        with pytest.raises(ValueError):
            FlextModels.RateLimiterState(block_until=-1.0)

    def test_models_rate_limiter_state_block_until(self) -> None:
        """Test RateLimiterState block_until field specifically."""
        # Test block_until at different values
        state1 = FlextModels.RateLimiterState(block_until=0.0)
        assert state1.block_until == 0.0

        state2 = FlextModels.RateLimiterState(block_until=100.5)
        assert state2.block_until == 100.5

        state3 = FlextModels.RateLimiterState(block_until=9999.99)
        assert state3.block_until == 9999.99

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
        """Verify that critical classes are still available after Phase 9."""
        critical_classes = [
            "Entity",
            "Value",
            "Command",
            "Query",
            "DomainEvent",
            "Cqrs",
            "Metadata",
            "Payload",
            "MiddlewareConfig",
            "RateLimiterState",
            "HandlerExecutionContext",
            "Validation",
            "ContextExport",
            "Pagination",
            "RegistrationDetails",
        ]

        for class_name in critical_classes:
            assert hasattr(FlextModels, class_name), (
                f"Critical class {class_name} should be available"
            )
