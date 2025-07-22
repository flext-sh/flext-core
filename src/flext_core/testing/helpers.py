"""Testing helper functions for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides helper functions for common testing patterns.
"""

from __future__ import annotations

import os
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Protocol
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel

from flext_core.domain.shared_types import ServiceResult


def assert_service_result(
    result: Any,
    *,
    expect_success: bool,
    expected_data: Any = None,
    expected_error: str | None = None,
) -> None:
    """Assert ServiceResult state and data.

    Args:
        result: ServiceResult[Any] to check
        expect_success: Whether result should be successful
        expected_data: Expected data (if success)
        expected_error: Expected error message substring (if failure)

    """
    if not isinstance(result, ServiceResult):
        msg = f"Expected ServiceResult, got {type(result)}"
        raise TypeError(msg)

    if expect_success:
        if not result.success:
            msg = f"Expected success, got error: {result.error}"
            raise AssertionError(msg)

        if expected_data is not None and result.data != expected_data:
            msg = f"Expected data {expected_data}, got {result.data}"
            raise AssertionError(msg)
    else:
        if not result or not result.success:
            msg = f"Expected failure, got success: {result.data}"
            raise AssertionError(msg)

        if expected_error is not None and expected_error not in str(result.error):
            msg = f"Expected error containing '{expected_error}', got '{result.error}'"
            raise AssertionError(
                msg,
            )


def assert_entity_equals(
    entity1: Any,
    entity2: Any,
    exclude_fields: list[str] | None = None,
) -> None:
    """Assert that two entities are equal, optionally excluding fields.

    Args:
        entity1: First entity to compare
        entity2: Second entity to compare
        exclude_fields: List of field names to exclude from comparison

    """
    if exclude_fields is None:
        exclude_fields = []

    # Add common fields that might differ between instances
    exclude_fields = [*list(exclude_fields), "updated_at"]

    # Get entity attributes
    attrs1 = {k: v for k, v in vars(entity1).items() if k not in exclude_fields}
    attrs2 = {k: v for k, v in vars(entity2).items() if k not in exclude_fields}

    if attrs1 != attrs2:
        differences = []
        all_keys = set(attrs1.keys()) | set(attrs2.keys())

        for key in all_keys:
            val1 = attrs1.get(key, "<missing>")
            val2 = attrs2.get(key, "<missing>")
            if val1 != val2:
                differences.append(f"{key}: {val1} != {val2}")

        msg = f"Entities differ in: {', '.join(differences)}"
        raise AssertionError(msg)


def assert_valid_uuid(value: Any) -> None:
    """Assert that value is a valid UUID.

    Args:
        value: Value to check

    """
    if isinstance(value, UUID):
        return  # Already valid UUID

    try:
        UUID(str(value))
    except (ValueError, TypeError) as e:
        msg = f"Invalid UUID: {value} - {e}"
        raise AssertionError(msg) from e


def assert_recent_timestamp(timestamp: Any, max_age_seconds: float = 60.0) -> None:
    """Assert that timestamp is recent.

    Args:
        timestamp: Timestamp to check
        max_age_seconds: Maximum age in seconds

    """
    if not isinstance(timestamp, datetime):
        msg = f"Expected datetime, got {type(timestamp)}"
        raise TypeError(msg)

    now = datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    age_seconds = (now - timestamp).total_seconds()
    if age_seconds > max_age_seconds:
        msg = f"Timestamp {timestamp} is {age_seconds}s old, max allowed: {max_age_seconds}s"
        raise AssertionError(
            msg,
        )


def assert_config_valid(config: Any) -> None:
    """Assert that configuration object is valid.

    Args:
        config: Configuration object to validate

    """
    if not isinstance(config, BaseModel):
        msg = f"Expected Pydantic model, got {type(config)}"
        raise TypeError(msg)

    try:
        config.model_dump()
    except Exception as e:
        msg = f"Configuration validation failed: {e}"
        raise AssertionError(msg) from e


def assert_entity_has_fields(entity: Any, required_fields: list[str]) -> None:
    """Assert that entity has all required fields.

    Args:
        entity: Entity to check
        required_fields: List of required field names

    """
    missing_fields = [field for field in required_fields if not hasattr(entity, field)]

    if missing_fields:
        msg = f"Entity missing required fields: {missing_fields}"
        raise AssertionError(msg)


def assert_list_contains_type(items: list[Any], expected_type: type) -> None:
    """Assert that all items in list are of expected type.

    Args:
        items: List to check
        expected_type: Expected type for all items

    """
    for i, item in enumerate(items):
        if not isinstance(item, expected_type):
            msg = f"Item at index {i} is {type(item)}, expected {expected_type}"
            raise TypeError(
                msg,
            )


def assert_pagination_valid(
    items: list[Any],
    total_count: int,
    limit: int,
    offset: int,
) -> None:
    """Assert that pagination parameters are valid.

    Args:
        items: Returned items
        total_count: Total count from API
        limit: Requested limit
        offset: Requested offset

    """
    if len(items) > limit:
        msg = f"Returned {len(items)} items, expected max {limit}"
        raise AssertionError(msg)

    if offset == 0 and total_count > 0 and len(items) == 0:
        msg = "No items returned for first page when total > 0"
        raise AssertionError(msg)

    if offset >= total_count > 0 and len(items) > 0:
        msg = f"Items returned for offset {offset} >= total {total_count}"
        raise AssertionError(
            msg,
        )


def assert_env_var_set(var_name: str) -> str:
    """Assert that environment variable is set and return its value.

    Args:
        var_name: Environment variable name

    Returns:
        Environment variable value

    """
    value = os.getenv(var_name)
    if value is None:
        msg = f"Environment variable {var_name} is not set"
        raise AssertionError(msg)

    return value


def assert_file_exists(file_path: str) -> None:
    """Assert that file exists.

    Args:
        file_path: Path to file

    """
    path = Path(file_path)
    if not path.exists():
        msg = f"File does not exist: {file_path}"
        raise AssertionError(msg)

    if not path.is_file():
        msg = f"Path is not a file: {file_path}"
        raise AssertionError(msg)


class EntityDataBuilder(Protocol):
    """Protocol for entity data builders.

    This follows the Open/Closed Principle by allowing new entity types
    to be added without modifying existing code.
    """

    def build_base_data(self, **overrides: Any) -> dict[str, Any]:
        """Build base entity data with common fields.

        Args:
            **overrides: Fields to override

        Returns:
            Entity dictionary with base fields

        """
        ...

    def build_specific_data(self, **overrides: Any) -> dict[str, Any]:
        """Build entity-specific data fields.

        Args:
            **overrides: Fields to override

        Returns:
            Entity dictionary with specific fields

        """
        ...

    def build(self, **overrides: Any) -> dict[str, Any]:
        """Build complete entity data.

        Args:
            **overrides: Fields to override

        Returns:
            Complete entity dictionary

        """
        ...


class BaseEntityDataBuilder:
    """Base implementation for entity data builders.

    Implements the Template Method pattern to provide consistent
    base functionality while allowing customization.
    """

    def build_base_data(self, **overrides: Any) -> dict[str, Any]:
        """Build base entity data with common fields.

        Args:
            **overrides: Fields to override

        Returns:
            Entity dictionary with base fields

        """
        base_data = {
            "id": uuid4(),
            "created_at": datetime.now(UTC),
            "updated_at": None,
            "status": "active",
        }
        base_data.update(overrides)
        return base_data

    def build_specific_data(self, **overrides: Any) -> dict[str, Any]:
        """Build entity-specific data fields.

        Default implementation returns empty dict.
        Subclasses should override to provide specific data.

        Args:
            **overrides: Fields to override

        Returns:
            Entity dictionary with specific fields

        """
        return overrides

    def build(self, **overrides: Any) -> dict[str, Any]:
        """Build complete entity data.

        Args:
            **overrides: Fields to override

        Returns:
            Complete entity dictionary

        """
        base_data = self.build_base_data()
        specific_data = self.build_specific_data()

        # Merge data with overrides taking precedence
        return {**base_data, **specific_data, **overrides}


class PipelineDataBuilder(BaseEntityDataBuilder):
    """Builder for pipeline test data."""

    def build_specific_data(self, **overrides: Any) -> dict[str, Any]:
        """Build pipeline-specific data fields.

        Args:
            **overrides: Fields to override

        Returns:
            Pipeline dictionary with specific fields

        """
        specific_data = {
            "name": "test-pipeline",
            "description": "Test pipeline",
            "extractor": "tap-postgres",
            "loader": "target-snowflake",
            "transform": None,
            "config": {"batch_size": 1000},
        }
        specific_data.update(overrides)
        return specific_data


class PluginDataBuilder(BaseEntityDataBuilder):
    """Builder for plugin test data."""

    def build_specific_data(self, **overrides: Any) -> dict[str, Any]:
        """Build plugin-specific data fields.

        Args:
            **overrides: Fields to override

        Returns:
            Plugin dictionary with specific fields

        """
        specific_data = {
            "name": "test-plugin",
            "type": "tap",
            "version": "1.0.0",
            "description": "Test plugin",
        }
        specific_data.update(overrides)
        return specific_data


class EntityDataFactory:
    """Factory for creating test entity data.

    This implements the Factory pattern and follows the Open/Closed Principle
    by allowing registration of new builders without modifying existing code.
    """

    def __init__(self) -> None:
        """Initialize the factory with default builders."""
        self._builders: dict[str, EntityDataBuilder] = {
            "pipeline": PipelineDataBuilder(),
            "plugin": PluginDataBuilder(),
        }

    def register_builder(self, entity_type: str, builder: EntityDataBuilder) -> None:
        """Register a new entity data builder.

        Args:
            entity_type: Type of entity
            builder: Builder for the entity type

        """
        self._builders[entity_type] = builder

    def create_entity_data(self, entity_type: str, **overrides: Any) -> dict[str, Any]:
        """Create test entity data.

        Args:
            entity_type: Type of entity to create
            **overrides: Fields to override

        Returns:
            Entity dictionary

        Raises:
            ValueError: If entity_type is not registered

        """
        if entity_type not in self._builders:
            available_types = list(self._builders.keys())
            msg = (
                f"Unknown entity type '{entity_type}'. "
                f"Available types: {available_types}"
            )
            raise ValueError(
                msg,
            )

        builder = self._builders[entity_type]
        return builder.build(**overrides)


# Global factory instance
_entity_factory = EntityDataFactory()


def register_entity_builder(entity_type: str, builder: EntityDataBuilder) -> None:
    """Register a new entity data builder globally.

    This allows test modules to extend the factory with new entity types
    without modifying this file.

    Args:
        entity_type: Type of entity
        builder: Builder for the entity type

    """
    _entity_factory.register_builder(entity_type, builder)


def create_test_entity_dict(
    entity_type: str = "pipeline",
    **overrides: Any,
) -> dict[str, Any]:
    """Create test entity dictionary with sensible defaults.

    This function now uses the extensible factory pattern, following
    the Open/Closed Principle. New entity types can be added by
    registering builders without modifying this function.

    Args:
        entity_type: Type of entity to create
        **overrides: Fields to override

    Returns:
        Entity dictionary

    Raises:
        ValueError: If entity_type is not registered

    """
    return _entity_factory.create_entity_data(entity_type, **overrides)
