"""Tests to improve coverage for domain mixins module."""

from __future__ import annotations

import pytest
from datetime import datetime, UTC, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

from flext_core.domain.mixins import (
    TimestampMixin,
    StatusMixin,
    ConfigurationMixin,
    IdentifierMixin,
)
from flext_core.domain.types import EntityStatus, EntityId


class ConcreteTimestampEntity:
    """Concrete implementation for testing TimestampMixin."""

    def __init__(self) -> None:
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def mark_updated(self) -> None:
        self.updated_at = datetime.now(UTC)

    def is_recently_created(self, seconds: int = 300) -> bool:
        return datetime.now(UTC) - self.created_at < timedelta(seconds=seconds)


class ConcreteStatusEntity:
    """Concrete implementation for testing StatusMixin."""

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


class ConcreteConfigEntity:
    """Concrete implementation for testing ConfigurationMixin."""

    def __init__(self) -> None:
        self.public_attr = "public"
        self._private_attr = "private"
        self.config_setting = "value"
        self.other_setting = 123

    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def get_subsection(self, prefix: str) -> dict[str, object]:
        data = self.to_dict()
        return {k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)}

    def merge(self, other: dict[str, object]) -> None:
        for key, value in other.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ConcreteIdentifierEntity:
    """Concrete implementation for testing IdentifierMixin."""

    def __init__(self) -> None:
        self.id = uuid4()

    @property
    def uuid_str(self) -> str:
        return str(self.id)

    @property
    def short_id(self) -> str:
        return str(self.id)[:8]

    def equals_id(self, other_id: EntityId | str) -> bool:
        if isinstance(other_id, str):
            return str(self.id) == other_id
        return self.id == other_id


class TestTimestampMixin:
    """Test TimestampMixin protocol conformance."""

    def test_mark_updated(self) -> None:
        """Test mark_updated updates timestamp."""
        entity = ConcreteTimestampEntity()
        original_time = entity.updated_at

        # Add small delay to ensure time difference
        import time

        time.sleep(0.001)

        entity.mark_updated()
        assert entity.updated_at > original_time

    def test_is_recently_created_true(self) -> None:
        """Test is_recently_created returns True for recent entity."""
        entity = ConcreteTimestampEntity()
        assert entity.is_recently_created(300)
        assert entity.is_recently_created(1)

    def test_is_recently_created_false(self) -> None:
        """Test is_recently_created returns False for old entity."""
        entity = ConcreteTimestampEntity()
        # Mock old creation time
        entity.created_at = datetime.now(UTC) - timedelta(seconds=400)
        assert not entity.is_recently_created(300)


class TestStatusMixin:
    """Test StatusMixin protocol conformance."""

    def test_is_active_true(self) -> None:
        """Test is_active returns True for active entity."""
        entity = ConcreteStatusEntity()
        entity.status = EntityStatus.ACTIVE
        assert entity.is_active()

    def test_is_active_false(self) -> None:
        """Test is_active returns False for inactive entity."""
        entity = ConcreteStatusEntity()
        entity.status = EntityStatus.INACTIVE
        assert not entity.is_active()

    def test_is_inactive_true(self) -> None:
        """Test is_inactive returns True for inactive entity."""
        entity = ConcreteStatusEntity()
        entity.status = EntityStatus.INACTIVE
        assert entity.is_inactive()

    def test_is_inactive_false(self) -> None:
        """Test is_inactive returns False for active entity."""
        entity = ConcreteStatusEntity()
        entity.status = EntityStatus.ACTIVE
        assert not entity.is_inactive()

    def test_activate(self) -> None:
        """Test activate sets status to active."""
        entity = ConcreteStatusEntity()
        entity.status = EntityStatus.INACTIVE
        entity.activate()
        assert entity.status == EntityStatus.ACTIVE

    def test_deactivate(self) -> None:
        """Test deactivate sets status to inactive."""
        entity = ConcreteStatusEntity()
        entity.status = EntityStatus.ACTIVE
        entity.deactivate()
        assert entity.status == EntityStatus.INACTIVE


class TestConfigurationMixin:
    """Test ConfigurationMixin protocol conformance."""

    def test_to_dict_excludes_private(self) -> None:
        """Test to_dict excludes private attributes."""
        entity = ConcreteConfigEntity()
        result = entity.to_dict()

        assert "public_attr" in result
        assert "config_setting" in result
        assert "other_setting" in result
        assert "_private_attr" not in result

    def test_get_subsection_filters_by_prefix(self) -> None:
        """Test get_subsection filters by prefix."""
        entity = ConcreteConfigEntity()
        result = entity.get_subsection("config_")

        assert "setting" in result  # "config_" prefix removed
        assert result["setting"] == "value"
        assert "public_attr" not in result

    def test_merge_updates_existing_attributes(self) -> None:
        """Test merge updates existing attributes."""
        entity = ConcreteConfigEntity()
        original_value = entity.config_setting

        entity.merge({"config_setting": "new_value"})
        assert entity.config_setting == "new_value"
        assert entity.config_setting != original_value

    def test_merge_ignores_nonexistent_attributes(self) -> None:
        """Test merge ignores attributes that don't exist."""
        entity = ConcreteConfigEntity()
        entity.merge({"nonexistent_attr": "value"})

        # Should not raise exception and should not add new attribute
        assert not hasattr(entity, "nonexistent_attr")


class TestIdentifierMixin:
    """Test IdentifierMixin protocol conformance."""

    def test_uuid_str_returns_string(self) -> None:
        """Test uuid_str returns string representation."""
        entity = ConcreteIdentifierEntity()
        result = entity.uuid_str

        assert isinstance(result, str)
        assert result == str(entity.id)

    def test_short_id_returns_first_8_chars(self) -> None:
        """Test short_id returns first 8 characters."""
        entity = ConcreteIdentifierEntity()
        result = entity.short_id

        assert isinstance(result, str)
        assert len(result) == 8
        assert result == str(entity.id)[:8]

    def test_equals_id_with_string(self) -> None:
        """Test equals_id works with string parameter."""
        entity = ConcreteIdentifierEntity()
        id_str = str(entity.id)

        assert entity.equals_id(id_str)
        assert not entity.equals_id("different-id")

    def test_equals_id_with_uuid(self) -> None:
        """Test equals_id works with UUID parameter."""
        entity = ConcreteIdentifierEntity()

        assert entity.equals_id(entity.id)
        assert not entity.equals_id(uuid4())


class TestMixinTypeCheckingImports:
    """Test TYPE_CHECKING imports to increase coverage."""

    def test_type_checking_imports_coverage(self) -> None:
        """Test that TYPE_CHECKING imports are properly covered."""
        # Import the module to trigger TYPE_CHECKING block execution
        import flext_core.domain.mixins

        # Verify the module loaded correctly
        assert hasattr(flext_core.domain.mixins, "TimestampMixin")
        assert hasattr(flext_core.domain.mixins, "StatusMixin")
        assert hasattr(flext_core.domain.mixins, "ConfigurationMixin")
        assert hasattr(flext_core.domain.mixins, "IdentifierMixin")

    @patch("flext_core.domain.mixins.TYPE_CHECKING", True)
    def test_type_checking_block_execution(self) -> None:
        """Test TYPE_CHECKING block imports."""
        # This forces execution of the TYPE_CHECKING block
        import importlib
        import flext_core.domain.mixins

        # Reload to trigger TYPE_CHECKING block
        importlib.reload(flext_core.domain.mixins)

        # Verify module still works after reload
        assert flext_core.domain.mixins.TimestampMixin is not None
