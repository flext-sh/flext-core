"""FLEXT Core Tests.

Copyright (c) 2025 Flext. All rights reserved.
SPDX-License-Identifier: MIT

Test suite for FLEXT Core framework.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from flext_core.domain.mixins import StatusMixin, TimestampMixin
from flext_core.domain.types import EntityStatus


class TestTimestampMixin:
    """Test TimestampMixin functionality."""

    def test_timestamp_mixin_protocol(self) -> None:
        """Test TimestampMixin protocol definition."""

        # Create a real entity that implements the protocol
        class TestEntity:
            def __init__(self) -> None:
                self.created_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)

            def mark_updated(self) -> None:
                self.updated_at = datetime.now(UTC)

            def is_recently_created(self, seconds: int = 300) -> bool:
                return datetime.now(UTC) - self.created_at < timedelta(seconds=seconds)

        entity = TestEntity()

        # Should be recognized as implementing the protocol
        assert isinstance(entity, TimestampMixin)

    def test_mark_updated_functionality(self) -> None:
        """Test mark_updated method functionality."""

        class TestEntity:
            def __init__(self) -> None:
                self.created_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)

            def mark_updated(self) -> None:
                """Mark entity as updated with current timestamp."""
                self.updated_at = datetime.now(UTC)

        entity = TestEntity()
        old_updated_at = entity.updated_at

        # Wait a small amount and update
        entity.mark_updated()

        # Updated timestamp should be different (newer)
        assert entity.updated_at >= old_updated_at

    def test_is_recently_created_true(self) -> None:
        """Test is_recently_created returns True for recent entities."""

        class TestEntity:
            def __init__(self) -> None:
                self.created_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)

            def is_recently_created(self, seconds: int = 300) -> bool:
                """Check if entity was created within the specified seconds."""
                return datetime.now(UTC) - self.created_at < timedelta(seconds=seconds)

        entity = TestEntity()

        # Should be recently created (within default 300 seconds)
        assert entity.is_recently_created() is True

        # Should be recently created within 10 seconds
        assert entity.is_recently_created(10) is True

    def test_is_recently_created_false(self) -> None:
        """Test is_recently_created returns False for old entities."""

        class TestEntity:
            def __init__(self) -> None:
                # Set created_at to 1 hour ago
                self.created_at = datetime.now(UTC) - timedelta(hours=1)
                self.updated_at = datetime.now(UTC)

            def is_recently_created(self, seconds: int = 300) -> bool:
                """Check if entity was created within the specified seconds."""
                return datetime.now(UTC) - self.created_at < timedelta(seconds=seconds)

        entity = TestEntity()

        # Should not be recently created (1 hour > 300 seconds)
        assert entity.is_recently_created() is False

        # Should definitely not be recently created within 10 seconds
        assert entity.is_recently_created(10) is False

    def test_is_recently_created_custom_seconds(self) -> None:
        """Test is_recently_created with custom seconds parameter."""

        class TestEntity:
            def __init__(self) -> None:
                # Set created_at to 100 seconds ago
                self.created_at = datetime.now(UTC) - timedelta(seconds=100)
                self.updated_at = datetime.now(UTC)

            def is_recently_created(self, seconds: int = 300) -> bool:
                """Check if entity was created within the specified seconds."""
                return datetime.now(UTC) - self.created_at < timedelta(seconds=seconds)

        entity = TestEntity()

        # Should be recently created within 200 seconds
        assert entity.is_recently_created(200) is True

        # Should not be recently created within 50 seconds
        assert entity.is_recently_created(50) is False


class TestStatusMixin:
    """Test StatusMixin functionality."""

    def test_status_mixin_protocol(self) -> None:
        """Test StatusMixin protocol definition."""

        # Create a real entity that implements the protocol
        class TestEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.ACTIVE

            def is_active(self) -> bool:
                return self.status.value == "active"

            def is_inactive(self) -> bool:
                return self.status.value == "inactive"

            def activate(self) -> None:
                self.status = EntityStatus.ACTIVE

            def deactivate(self) -> None:
                self.status = EntityStatus.INACTIVE

        entity = TestEntity()

        # Should be recognized as implementing the protocol
        assert isinstance(entity, StatusMixin)

    def test_is_active_true(self) -> None:
        """Test is_active returns True for active entities."""

        class TestEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.ACTIVE

            def is_active(self) -> bool:
                """Check if entity is active."""
                return self.status.value == "active"

        entity = TestEntity()
        assert entity.is_active() is True

    def test_is_active_false(self) -> None:
        """Test is_active returns False for inactive entities."""

        class TestEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.INACTIVE

            def is_active(self) -> bool:
                """Check if entity is active."""
                return self.status.value == "active"

        entity = TestEntity()
        assert entity.is_active() is False

    def test_is_inactive_true(self) -> None:
        """Test is_inactive returns True for inactive entities."""

        class TestEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.INACTIVE

            def is_inactive(self) -> bool:
                """Check if entity is inactive."""
                return self.status.value == "inactive"

        entity = TestEntity()
        assert entity.is_inactive() is True

    def test_is_inactive_false(self) -> None:
        """Test is_inactive returns False for active entities."""

        class TestEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.ACTIVE

            def is_inactive(self) -> bool:
                """Check if entity is inactive."""
                return self.status.value == "inactive"

        entity = TestEntity()
        assert entity.is_inactive() is False

    def test_activate_method(self) -> None:
        """Test activate method changes status to active."""

        class TestEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.INACTIVE

            def activate(self) -> None:
                """Activate the entity."""
                self.status = EntityStatus.ACTIVE

        entity = TestEntity()
        assert entity.status.value == EntityStatus.INACTIVE.value

        entity.activate()
        assert entity.status.value == EntityStatus.ACTIVE.value

    def test_deactivate_method(self) -> None:
        """Test deactivate method changes status to inactive."""

        class TestEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.ACTIVE

            def deactivate(self) -> None:
                """Deactivate the entity."""
                self.status = EntityStatus.INACTIVE

        entity = TestEntity()
        assert entity.status.value == EntityStatus.ACTIVE.value

        entity.deactivate()
        assert entity.status.value == EntityStatus.INACTIVE.value


class TestMixinIntegration:
    """Test mixin integration and combined usage."""

    def test_combined_mixins(self) -> None:
        """Test entity implementing both mixins."""

        class TestEntity:
            def __init__(self) -> None:
                self.created_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)
                self.status = EntityStatus.ACTIVE

            # TimestampMixin methods
            def mark_updated(self) -> None:
                self.updated_at = datetime.now(UTC)

            def is_recently_created(self, seconds: int = 300) -> bool:
                return datetime.now(UTC) - self.created_at < timedelta(seconds=seconds)

            # StatusMixin methods
            def is_active(self) -> bool:
                return self.status.value == "active"

            def is_inactive(self) -> bool:
                return self.status.value == "inactive"

            def activate(self) -> None:
                self.status = EntityStatus.ACTIVE

            def deactivate(self) -> None:
                self.status = EntityStatus.INACTIVE

        entity = TestEntity()

        # Should implement both protocols
        assert isinstance(entity, TimestampMixin)
        assert isinstance(entity, StatusMixin)

        # Should have both functionalities
        assert entity.is_active() is True
        assert entity.is_recently_created() is True

        # Test status change
        entity.deactivate()
        assert entity.is_inactive() is True

        # Test timestamp update
        old_updated = entity.updated_at
        entity.mark_updated()
        assert entity.updated_at >= old_updated

    def test_mixin_protocol_checking(self) -> None:
        """Test protocol checking works correctly."""

        # Entity with full timestamp functionality
        class TimestampEntity:
            def __init__(self) -> None:
                self.created_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)

            def mark_updated(self) -> None:
                self.updated_at = datetime.now(UTC)

            def is_recently_created(self, seconds: int = 300) -> bool:
                return datetime.now(UTC) - self.created_at < timedelta(seconds=seconds)

        # Entity with full status functionality
        class StatusEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.ACTIVE

            def is_active(self) -> bool:
                return self.status.value == "active"

            def is_inactive(self) -> bool:
                return self.status.value == "inactive"

            def activate(self) -> None:
                self.status = EntityStatus.ACTIVE

            def deactivate(self) -> None:
                self.status = EntityStatus.INACTIVE

        timestamp_entity = TimestampEntity()
        status_entity = StatusEntity()

        # Should implement respective protocols
        assert isinstance(timestamp_entity, TimestampMixin)
        assert not isinstance(timestamp_entity, StatusMixin)

        assert isinstance(status_entity, StatusMixin)
        assert not isinstance(status_entity, TimestampMixin)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_entity_status_enum_values(self) -> None:
        """Test EntityStatus enum has expected values."""
        assert EntityStatus.ACTIVE.value == "active"
        assert EntityStatus.INACTIVE.value == "inactive"

    def test_timestamp_timezone_handling(self) -> None:
        """Test timestamp handling with UTC timezone."""

        class TestEntity:
            def __init__(self) -> None:
                self.created_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)

            def mark_updated(self) -> None:
                self.updated_at = datetime.now(UTC)

        entity = TestEntity()

        # Should use UTC timezone
        assert entity.created_at.tzinfo == UTC
        assert entity.updated_at.tzinfo == UTC

        entity.mark_updated()
        assert entity.updated_at.tzinfo == UTC

    def test_status_change_cycle(self) -> None:
        """Test status can be changed multiple times."""

        class TestEntity:
            def __init__(self) -> None:
                self.status = EntityStatus.ACTIVE

            def activate(self) -> None:
                self.status = EntityStatus.ACTIVE

            def deactivate(self) -> None:
                self.status = EntityStatus.INACTIVE

            def is_active(self) -> bool:
                return self.status.value == "active"

            def is_inactive(self) -> bool:
                return self.status.value == "inactive"

        entity = TestEntity()

        # Test multiple status changes
        assert entity.is_active() is True

        entity.deactivate()
        assert entity.is_inactive() is True

        entity.activate()
        assert entity.is_active() is True

        entity.deactivate()
        assert entity.is_inactive() is True
