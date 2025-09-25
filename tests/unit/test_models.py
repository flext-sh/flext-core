"""Comprehensive tests for FlextModels - Data Models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time

import pytest

from flext_core import FlextModels, FlextResult


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

            def validate(self) -> FlextResult[None]:
                if "@" not in self.email:
                    return FlextResult[None].fail("Invalid email")
                return FlextResult[None].ok(None)

        # Test entity creation
        entity = TestEntity(name="Test User", email="test@example.com")
        assert entity.name == "Test User"
        assert entity.email == "test@example.com"
        assert entity.id is not None  # Entity should have auto-generated ID

        # Test validation
        result = entity.validate()
        assert result.is_success

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
        # This should raise an error if we try to modify
        with pytest.raises(Exception) as exc_info:
            value.data = "modified"  # type: ignore[assignment]

        # Verify the error message contains expected keywords
        error_msg = str(exc_info.value).lower()
        assert "immutable" in error_msg or "read-only" in error_msg

    def test_models_aggregate_root_creation(self) -> None:
        """Test aggregate root creation and domain events."""

        class TestAggregate(FlextModels.AggregateRoot):
            name: str
            status: str = "active"

            def change_status(self, new_status: str) -> None:
                self.status = new_status
                self.add_domain_event(
                    "status_changed", {"old_status": "active", "new_status": new_status}
                )

        # Test aggregate creation
        aggregate = TestAggregate(name="Test Aggregate")
        assert aggregate.name == "Test Aggregate"
        assert aggregate.status == "active"
        assert aggregate.id is not None

        # Test domain events
        aggregate.change_status("inactive")
        assert aggregate.status == "inactive"
        events = aggregate.clear_domain_events()
        assert len(events) == 1

    def test_models_built_in_models(self) -> None:
        """Test built-in models from FlextModels."""
        # Test User model
        user = FlextModels.User(username="testuser", email="test@example.com")
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.id is not None
        assert user.roles == []  # Default empty list

        # Test EmailAddress value object
        email = FlextModels.EmailAddress(address="test@example.com")
        assert email.address == "test@example.com"

        # Test Host value object
        host = FlextModels.Host(hostname="example.com")
        assert host.hostname == "example.com"

    def test_models_command_creation(self) -> None:
        """Test command model creation."""

        class TestCommand(FlextModels.Command):
            command_type: str = "test_command"
            data: str

        command = TestCommand(data="test_data")
        assert command.command_type == "test_command"
        assert command.data == "test_data"
        assert command.issued_at is not None
        assert command.command_id is not None

    def test_models_payload_creation(self) -> None:
        """Test payload model creation."""
        payload = FlextModels.Payload(data="test_data")
        assert payload.data == "test_data"
        assert payload.created_at is not None
        assert payload.message_id is not None

        # Test expiration functionality
        assert not payload.is_expired

    def test_models_pagination_creation(self) -> None:
        """Test pagination model creation."""
        pagination = FlextModels.Pagination(page=1, size=10)
        assert pagination.page == 1
        assert pagination.size == 10
        assert pagination.offset == 0

        pagination2 = FlextModels.Pagination(page=3, size=10)
        assert pagination2.offset == 20

    def test_models_value_object_validation(self) -> None:
        """Test value object validation."""
        # Test EmailAddress validation
        with pytest.raises(Exception) as exc_info:
            email = FlextModels.EmailAddress(address="invalid-email")
        
        # Verify the error message contains expected keywords
        error_msg = str(exc_info.value).lower()
        assert "email" in error_msg or "validation" in error_msg

        # Test valid email
        email = FlextModels.EmailAddress(address="valid@example.com")
        assert email.address == "valid@example.com"

        # Test Host validation
        host = FlextModels.Host(hostname="valid-hostname")
        assert host.hostname == "valid-hostname"

    def test_models_advanced_value_objects(self) -> None:
        """Test advanced value objects."""
        # Test Coordinates
        coords = FlextModels.Coordinates(latitude=40.7128, longitude=-74.0060)
        assert coords.latitude == 40.7128
        assert coords.longitude == -74.0060

        # Test Money
        money = FlextModels.Money(amount=100.50, currency="USD")
        assert money.amount == 100.50
        assert money.currency == "USD"

        # Test PhoneNumber
        phone = FlextModels.PhoneNumber(number="+1234567890")
        assert phone.number == "+1234567890"

        # Test Url
        url = FlextModels.Url(url="https://example.com")
        assert url.url == "https://example.com"

    def test_models_project_entity(self) -> None:
        """Test Project entity."""
        project = FlextModels.Project(
            name="Test Project",
            organization_id="org-123",
            repository_path="/path/to/repo",
        )
        assert project.name == "Test Project"
        assert project.organization_id == "org-123"
        assert project.repository_path == "/path/to/repo"
        assert project.id is not None

    def test_models_workspace_aggregate(self) -> None:
        """Test WorkspaceInfo aggregate."""
        workspace = FlextModels.WorkspaceInfo(
            workspace_id="ws-123", name="Test Workspace", root_path="/workspace/root"
        )
        assert workspace.workspace_id == "ws-123"
        assert workspace.name == "Test Workspace"
        assert workspace.root_path == "/workspace/root"
        assert workspace.id is not None

    def test_models_domain_events(self) -> None:
        """Test domain events functionality."""

        class EventAggregate(FlextModels.AggregateRoot):
            name: str

            def trigger_event(self) -> None:
                self.add_domain_event("test_event", {"data": "test"})

        aggregate = EventAggregate(name="Test")

        # Add multiple events
        aggregate.add_domain_event("event1", {"data": "1"})
        aggregate.add_domain_event("event2", {"data": "2"})

        # Check events are stored
        events = aggregate.clear_domain_events()
        assert len(events) == 2

    def test_models_version_management(self) -> None:
        """Test version management in entities."""

        class VersionedEntity(FlextModels.Entity):
            name: str

        entity = VersionedEntity(name="Test")
        initial_version = entity.version

        # Increment version
        entity.increment_version()
        assert entity.version == initial_version + 1

    def test_models_timestamped_functionality(self) -> None:
        """Test timestamped functionality."""

        class TimestampedEntity(FlextModels.Entity):
            name: str

        entity = TimestampedEntity(name="Test")
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

            def validate(self) -> FlextResult[None]:
                if not self.name:
                    return FlextResult[None].fail("Name is required")
                if "@" not in self.email:
                    return FlextResult[None].fail("Invalid email")
                return FlextResult[None].ok(None)

        # Test valid entity
        entity = ValidatedEntity(name="Test", email="test@example.com")
        result = entity.validate()
        assert result.is_success

        # Test invalid entity
        entity_invalid = ValidatedEntity(name="", email="invalid")
        result_invalid = entity_invalid.validate()
        assert result_invalid.is_failure

    def test_models_serialization(self) -> None:
        """Test model serialization."""
        entity = FlextModels.User(username="test", email="test@example.com")

        # Test to_dict
        data = entity.model_dump()
        assert "username" in data
        assert "email" in data
        assert "id" in data

        # Test JSON serialization
        json_data = entity.model_dump_json()
        assert isinstance(json_data, str)
        assert "test" in json_data

    def test_models_thread_safety(self) -> None:
        """Test thread safety of models."""

        class ThreadSafeEntity(FlextModels.Entity):
            counter: int = 0

            def increment(self) -> None:
                self.counter += 1

        entity = ThreadSafeEntity(name="Test")

        def worker() -> None:
            for _ in range(100):
                entity.increment()

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check final counter
        assert entity.counter == 500

    def test_models_performance(self) -> None:
        """Test model creation performance."""
        start_time = time.time()

        # Create many entities
        entities = []
        for i in range(1000):
            entity = FlextModels.User(username=f"user{i}", email=f"user{i}@example.com")
            entities.append(entity)

        end_time = time.time()
        creation_time = end_time - start_time

        # Should be fast (less than 1 second for 1000 entities)
        assert creation_time < 1.0
        assert len(entities) == 1000
