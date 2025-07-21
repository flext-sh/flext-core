"""Mock objects for FLEXT testing.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides mock objects and utilities for testing FLEXT components.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import Mock
from uuid import UUID
from uuid import uuid4

from flext_core.domain.shared_types import ServiceResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class MockRepository:
    """Mock repository for testing."""

    def __init__(self) -> None:
        """Initialize mock repository."""
        self._data: dict[UUID, Any] = {}
        self._call_history: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record_call(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Record method call for verification."""
        self._call_history.append((method_name, args, kwargs))

    def get_call_history(self) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
        """Get history of method calls."""
        return self._call_history.copy()

    def clear_call_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()

    async def save(self, entity: Any) -> ServiceResult[Any]:
        """Mock save method."""
        self._record_call("save", entity)

        # Generate ID if not present
        if not hasattr(entity, "id") or entity.id is None:
            entity.id = uuid4()

        self._data[entity.id] = entity
        return ServiceResult.ok(entity)

    async def find_by_id(self, entity_id: UUID) -> ServiceResult[Any]:
        """Mock find by ID method."""
        self._record_call("find_by_id", entity_id)

        if entity_id in self._data:
            return ServiceResult.ok(self._data[entity_id])
        return ServiceResult.fail(f"Entity with ID {entity_id} not found")

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> ServiceResult[Sequence[Any]]:
        """Mock list method."""
        self._record_call("list", limit, offset)

        all_entities = list(self._data.values())
        paginated = all_entities[offset : offset + limit]
        return ServiceResult.ok(paginated)

    async def delete(self, entity_id: UUID) -> ServiceResult[bool]:
        """Mock delete method."""
        self._record_call("delete", entity_id)

        if entity_id in self._data:
            del self._data[entity_id]
            deleted = True
            return ServiceResult.ok(deleted)
        return ServiceResult.fail(f"Entity with ID {entity_id} not found")

    async def count(self) -> ServiceResult[int]:
        """Mock count method."""
        self._record_call("count")
        return ServiceResult.ok(len(self._data))

    def clear_data(self) -> None:
        """Clear all stored data."""
        self._data.clear()

    def add_mock_data(self, entity_id: UUID, entity: Any) -> None:
        """Add mock data directly."""
        self._data[entity_id] = entity


class MockLogger:
    """Mock logger for testing."""

    def __init__(self) -> None:
        """Initialize mock logger."""
        self.logs: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []

    def _log(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        """Record log message."""
        self.logs.append((level, message, args, kwargs))

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Mock debug log."""
        self._log("DEBUG", message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Mock info log."""
        self._log("INFO", message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Mock warning log."""
        self._log("WARNING", message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Mock error log."""
        self._log("ERROR", message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Mock exception log."""
        self._log("ERROR", message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Mock critical log."""
        self._log("CRITICAL", message, *args, **kwargs)

    def bind(self, **_kwargs: Any) -> MockLogger:
        """Mock bind method."""
        # Return self for simplicity in tests
        return self

    def get_logs(
        self,
        level: str | None = None,
    ) -> list[tuple[str, str, tuple[Any, ...], dict[str, Any]]]:
        """Get logged messages, optionally filtered by level."""
        if level is None:
            return self.logs.copy()
        return [log for log in self.logs if log[0] == level]

    def clear_logs(self) -> None:
        """Clear all logged messages."""
        self.logs.clear()

    def assert_logged(self, level: str, message_contains: str) -> None:
        """Assert that a message was logged at specified level."""
        matching_logs = [
            log for log in self.logs if log[0] == level and message_contains in log[1]
        ]
        if not matching_logs:
            available_logs = [f"{log[0]}: {log[1]}" for log in self.logs]
            msg = (
                f"No {level} log containing '{message_contains}' found. "
                f"Available logs: {available_logs}"
            )
            raise AssertionError(
                msg,
            )


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize mock config with provided values."""
        # Set default values
        self.project_name = kwargs.get("project_name", "test-project")
        self.project_version = kwargs.get("project_version", "1.0.0")
        self.environment = kwargs.get("environment", "test")
        self.debug = kwargs.get("debug", True)
        self.log_level = kwargs.get("log_level", "DEBUG")

        # Set any additional values
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        """Mock model_dump method."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def model_dump_json(self, **kwargs: Any) -> str:
        """Mock model_dump_json method."""
        return json.dumps(self.model_dump(**kwargs))

    def is_production(self) -> bool:
        """Check if running in production."""
        return str(self.environment) == "production"

    def is_development(self) -> bool:
        """Check if running in development."""
        return str(self.environment) == "development"


class MockService:
    """Mock service for testing application layer."""

    def __init__(self, repository: MockRepository | None = None) -> None:
        """Initialize mock service."""
        self.repository = repository or MockRepository()
        self.call_history: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record_call(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Record method call for verification."""
        self.call_history.append((method_name, args, kwargs))

    async def create_entity(self, data: dict[str, Any]) -> ServiceResult[Any]:
        """Mock create entity method."""
        self._record_call("create_entity", data)

        # Create mock entity with ID
        entity = Mock()
        entity.id = uuid4()
        for key, value in data.items():
            setattr(entity, key, value)

        return await self.repository.save(entity)

    async def get_entity(self, entity_id: UUID) -> ServiceResult[Any]:
        """Mock get entity method."""
        self._record_call("get_entity", entity_id)
        return await self.repository.find_by_id(entity_id)

    async def list_entities(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> ServiceResult[Sequence[Any]]:
        """Mock list entities method."""
        self._record_call("list_entities", limit, offset)
        return await self.repository.list(limit, offset)

    async def delete_entity(self, entity_id: UUID) -> ServiceResult[bool]:
        """Mock delete entity method."""
        self._record_call("delete_entity", entity_id)
        return await self.repository.delete(entity_id)

    def get_call_history(self) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
        """Get service call history."""
        return self.call_history.copy()

    def clear_call_history(self) -> None:
        """Clear service call history."""
        self.call_history.clear()
