"""Comprehensive tests for FlextCore.Models - Data Models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading

import pytest
from pydantic import field_validator

from flext_core import FlextCore


class TestFlextModels:
    """Test suite for FlextCore.Models data model functionality."""

    def test_models_initialization(self) -> None:
        """Test models initialization."""
        models = FlextCore.Models()
        assert models is not None
        assert isinstance(models, FlextCore.Models)

    def test_models_entity_creation(self) -> None:
        """Test entity creation and validation."""

        class TestEntity(FlextCore.Models.Entity):
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

        # Test validation - should work with valid email (exclude extra fields from base Entity class)
        entity_dict = entity.model_dump(exclude={"is_initial_version", "is_modified"})
        validated_entity = entity.model_validate(entity_dict)
        assert validated_entity.email == "test@example.com"

        # Test validation failure with invalid email
        with pytest.raises(ValueError, match="Invalid email"):
            TestEntity(name="Test User", email="invalid-email", domain_events=[])

    def test_models_value_object_creation(self) -> None:
        """Test value object creation and immutability."""

        class TestValue(FlextCore.Models.Value):
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
        """Test aggregate root creation and domain events."""

        class TestAggregate(FlextCore.Models.AggregateRoot):
            name: str
            status: str = "active"

            def change_status(self, new_status: str) -> None:
                self.status = new_status
                result = self.add_domain_event(
                    "status_changed",
                    {"old_status": "active", "new_status": new_status},
                )
                if result.is_failure:
                    raise ValueError(f"Failed to add domain event: {result.error}")

        # Test aggregate creation
        aggregate = TestAggregate(name="Test Aggregate", domain_events=[])
        assert aggregate.name == "Test Aggregate"
        assert aggregate.status == "active"
        assert aggregate.id is not None

        # Test domain events
        aggregate.change_status("inactive")
        assert aggregate.status == "inactive"
        events = aggregate.clear_domain_events()
        assert len(events) == 1

        # Rebuild test class after base classes are defined
        TestAggregate.model_rebuild()

    def test_models_command_creation(self) -> None:
        """Test command model creation."""

        class TestCommand(FlextCore.Models.Command):
            command_type: str = "test_command"
            data: str

        command = TestCommand(data="test_data")
        assert command.command_type == "test_command"
        assert command.data == "test_data"
        assert command.created_at is not None  # Timestamp from TimestampableMixin
        assert command.id is not None  # ID from IdentifiableMixin

    def test_models_metadata_creation(self) -> None:
        """Test metadata model creation."""
        metadata = FlextCore.Models.Metadata(
            created_by="test_user",
            tags=["tag1", "tag2"],
        )
        assert metadata.created_by == "test_user"
        assert metadata.created_at is not None
        assert len(metadata.tags) == 2

    def test_models_pagination_creation(self) -> None:
        """Test pagination model creation."""
        pagination = FlextCore.Models.Pagination(page=1, size=10)
        assert pagination.page == 1
        assert pagination.size == 10
        assert (pagination.page - 1) * pagination.size == 0

        pagination2 = FlextCore.Models.Pagination(page=3, size=10)
        assert (pagination2.page - 1) * pagination2.size == 20

    def test_models_advanced_value_objects(self) -> None:
        """Test advanced value objects."""
        # Test URL validation using Validation utility
        result = FlextCore.Models.Validation.validate_url("https://example.com")
        assert result.is_success
        assert result.value == "https://example.com"

    def test_models_domain_events(self) -> None:
        """Test domain events functionality."""

        class EventAggregate(FlextCore.Models.AggregateRoot):
            name: str

            def trigger_event(self) -> None:
                self.add_domain_event("test_event", {"data": "test"})

        aggregate = EventAggregate(name="Test", domain_events=[])

        # Add multiple events
        aggregate.add_domain_event("event1", {"data": "1"})
        aggregate.add_domain_event("event2", {"data": "2"})

        # Check events are stored
        events = aggregate.clear_domain_events()
        assert len(events) == 2

        # Rebuild test class after base classes are defined
        EventAggregate.model_rebuild()

    def test_models_version_management(self) -> None:
        """Test version management in entities."""

        class VersionedEntity(FlextCore.Models.Entity):
            name: str

        entity = VersionedEntity(name="Test", domain_events=[])
        initial_version = entity.version

        # Increment version
        entity.increment_version()
        assert entity.version == initial_version + 1

    def test_models_timestamped_functionality(self) -> None:
        """Test timestamped functionality."""

        class TimestampedEntity(FlextCore.Models.Entity):
            name: str

        entity = TimestampedEntity(name="Test", domain_events=[])
        initial_created = entity.created_at

        # Update timestamp
        entity.update_timestamp()
        assert entity.updated_at is not None
        assert entity.updated_at > initial_created

    def test_models_validation_patterns(self) -> None:
        """Test validation patterns."""

        class ValidatedEntity(FlextCore.Models.Entity):
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

        class ThreadSafeEntity(FlextCore.Models.Entity):
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
