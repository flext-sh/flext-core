"""Test fixtures for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides common test fixtures that can be shared across
all FLEXT projects to eliminate duplication in conftest.py files.
"""

from __future__ import annotations

import os
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Generator
    from uuid import UUID

# Pytest import handled conditionally
if TYPE_CHECKING:
    import pytest
else:
    try:
        import pytest
    except ImportError:
        pytest = None  # type: ignore[assignment]


# ==============================================================================
# CENTRALIZED TEST ENVIRONMENT SETUP - ELIMINATES CONFTEST.PY DUPLICATION
# ==============================================================================


def setup_flext_test_environment() -> None:
    """Set up standard FLEXT test environment variables.

    This function should be called in each project's conftest.py to ensure
    consistent test environment setup across all FLEXT projects.

    Eliminates duplication across 19+ conftest.py files.
    """
    os.environ["FLEXT_ENV"] = "testing"
    os.environ["FLEXT_DEBUG"] = "true"
    os.environ["FLEXT_ENVIRONMENT"] = "test"


def get_test_environment_fixture() -> Any:
    """Get the standard test environment fixture.

    Returns a pytest fixture that automatically sets test environment
    for all tests except those explicitly testing default configurations.

    Usage in conftest.py:
    ```python
    from flext_core.testing.fixtures import get_test_environment_fixture

    set_test_environment = get_test_environment_fixture()
    ```
    """

    # pytest is imported conditionally above and required for this function
    @pytest.fixture(autouse=True)
    def set_test_environment(
        request: pytest.FixtureRequest,
    ) -> Generator[None]:
        """Automatically set environment to 'test' for all tests except those testing real defaults.

        This fixture ensures tests run in test environment unless explicitly testing defaults.
        """
        # Skip setting test environment for tests that explicitly test defaults
        test_name = request.node.name
        parent_name = request.node.parent.name.lower() if request.node.parent else ""

        # Tests that intentionally test real defaults without interference
        skip_tests = [
            ("test_base_settings_creation", "comprehensive"),
            ("test_base_settings_defaults", ""),  # Any parent name for this test
            ("test_default_configuration", ""),  # Tests development defaults
            ("test_load_from_file", ""),  # Tests file-based configuration loading
        ]

        for test_pattern, parent_pattern in skip_tests:
            if test_pattern in test_name and (
                not parent_pattern or parent_pattern in parent_name
            ):
                yield
                return

        # Store original value to restore after test
        original_env = os.environ.get("FLEXT_ENVIRONMENT")

        # Set test environment (using FLEXT_ prefix as defined in ConfigDefaults)
        os.environ["FLEXT_ENVIRONMENT"] = "test"

        # Yield control back to test
        yield

        # Restore original environment after test
        if original_env is not None:
            os.environ["FLEXT_ENVIRONMENT"] = original_env
        else:
            os.environ.pop("FLEXT_ENVIRONMENT", None)

    return set_test_environment


def get_project_root_fixture() -> Any:
    """Get the standard project root fixture.

    Returns a pytest fixture that finds the project root by looking for pyproject.toml.

    Usage in conftest.py:
    ```python
    from flext_core.testing.fixtures import get_project_root_fixture

    project_root = get_project_root_fixture()
    ```
    """

    # pytest is imported conditionally above and required for this function
    @pytest.fixture(scope="session")
    def project_root() -> Path:
        """Get project root directory."""
        # Navigate up from any project's test directory to find root
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return Path.cwd()

    return project_root


class TestFixtures:
    """Common test data fixtures."""

    @staticmethod
    def valid_uuid() -> UUID:
        """Generate a valid UUID for testing."""
        return uuid4()

    @staticmethod
    def current_timestamp() -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(UTC)

    @staticmethod
    def test_project_name() -> str:
        """Get test project name."""
        return "test-project"

    @staticmethod
    def test_version() -> str:
        """Get test version."""
        return "1.0.0"

    @staticmethod
    def test_config_data() -> dict[str, Any]:
        """Get test configuration data."""
        return {
            "project_name": TestFixtures.test_project_name(),
            "project_version": TestFixtures.test_version(),
            "environment": "test",
            "debug": True,
            "log_level": "DEBUG",
        }

    @staticmethod
    def test_entity_data() -> dict[str, Any]:
        """Get test entity data."""
        return {
            "id": TestFixtures.valid_uuid(),
            "created_at": TestFixtures.current_timestamp(),
            "updated_at": None,
            "status": "active",
        }

    @staticmethod
    def test_pipeline_data() -> dict[str, Any]:
        """Get test pipeline data."""
        base_data = TestFixtures.test_entity_data()
        return {
            **base_data,
            "name": "test-pipeline",
            "description": "Test pipeline for unit testing",
            "extractor": "tap-postgres",
            "loader": "target-snowflake",
            "transform": None,
            "config": {"batch_size": 1000},
            "schedule": "0 0 * * *",
        }

    @staticmethod
    def test_plugin_data() -> dict[str, Any]:
        """Get test plugin data."""
        base_data = TestFixtures.test_entity_data()
        return {
            **base_data,
            "name": "test-plugin",
            "type": "tap",
            "version": TestFixtures.test_version(),
            "description": "Test plugin for unit testing",
            "config_schema": {"type": "object", "properties": {}},
        }


class MemoryFixtures:
    """In-memory test fixtures for repositories."""

    def __init__(self) -> None:
        """Initialize memory fixtures."""
        self._data: dict[str, dict[UUID, Any]] = {}

    def get_repository_data(self, repository_type: str) -> dict[UUID, Any]:
        """Get repository data for specific type."""
        if repository_type not in self._data:
            self._data[repository_type] = {}
        return self._data[repository_type]

    def clear_all(self) -> None:
        """Clear all repository data."""
        self._data.clear()

    def clear_repository(self, repository_type: str) -> None:
        """Clear specific repository data."""
        if repository_type in self._data:
            self._data[repository_type].clear()

    def add_test_pipeline(self) -> dict[str, Any]:
        """Add test pipeline to memory and return it."""
        pipeline_data = TestFixtures.test_pipeline_data()
        pipelines = self.get_repository_data("pipeline")
        pipelines[pipeline_data["id"]] = pipeline_data
        return pipeline_data

    def add_test_plugin(self) -> dict[str, Any]:
        """Add test plugin to memory and return it."""
        plugin_data = TestFixtures.test_plugin_data()
        plugins = self.get_repository_data("plugin")
        plugins[plugin_data["id"]] = plugin_data
        return plugin_data


class DatabaseFixtures:
    """Database test fixtures."""

    @staticmethod
    def get_test_database_url() -> str:
        """Get test database URL."""
        return "sqlite:///./test.db"

    @staticmethod
    def get_async_test_database_url() -> str:
        """Get async test database URL."""
        return "sqlite+aiosqlite:///./test.db"

    @staticmethod
    def get_postgres_test_url() -> str:
        """Get PostgreSQL test database URL."""
        return "postgresql://test_user:test_pass@localhost:5432/test_db"

    @staticmethod
    def get_async_postgres_test_url() -> str:
        """Get async PostgreSQL test database URL."""
        return "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db"

    @staticmethod
    def get_redis_test_url() -> str:
        """Get Redis test URL."""
        return "redis://localhost:6379/1"  # Use database 1 for testing

    @classmethod
    def create_test_config(cls) -> dict[str, Any]:
        """Create test configuration with database URLs."""
        return {
            **TestFixtures.test_config_data(),
            "database_url": cls.get_test_database_url(),
            "database_async_url": cls.get_async_test_database_url(),
            "redis_url": cls.get_redis_test_url(),
        }
