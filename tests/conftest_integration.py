"""Pytest configuration and shared fixtures for FLEXT Core tests.

Modern pytest fixtures following best practices for enterprise testing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from flext_core.config import FlextSettings
from flext_core.constants import FlextEnvironment, FlextLogLevel
from flext_core.container import FlextContainer, get_flext_container

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from flext_core import TEntityId


@pytest.fixture
def clean_container() -> Generator[FlextContainer]:
    """Provide a clean FlextContainer for each test.

    Automatically cleans up after test completion.
    """
    container = FlextContainer()
    yield container
    # Cleanup is handled by container.clear() in teardown
    container.clear()


@pytest.fixture
def global_container_reset() -> Generator[None]:
    """Reset global container before and after each test."""
    # Reset before test
    container = get_flext_container()
    container.clear()
    yield
    # Reset after test
    container.clear()


@pytest.fixture
def sample_settings() -> FlextSettings:
    """Provide sample FlextSettings for testing."""

    class TestSettings(FlextSettings):
        """Test-specific settings with required fields."""

        environment: FlextEnvironment = FlextEnvironment.TESTING
        log_level: FlextLogLevel = FlextLogLevel.DEBUG
        debug: bool = True

    return TestSettings()


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for configuration files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_entity_ids() -> list[TEntityId]:
    """Provide sample FlextEntityId instances for testing."""
    return [
        "user-123",
        "user-456",
        "order-789",
        "product-abc",
    ]


@pytest.fixture
def mock_database() -> MagicMock:
    """Provide mock database service for DI testing."""
    mock_db = MagicMock()
    mock_db.get_user.return_value = {"id": "user-123", "name": "Test User"}
    mock_db.save_user.return_value = True
    return mock_db


@pytest.fixture
def mock_logger() -> MagicMock:
    """Provide mock logger service for DI testing."""
    mock_logger = MagicMock()
    mock_logger.info.return_value = None
    mock_logger.error.return_value = None
    mock_logger.debug.return_value = None
    return mock_logger


@pytest.fixture(params=["user-123", "order-456", "product-789"])
def entity_id_samples(request: pytest.FixtureRequest) -> TEntityId:
    """Parametrized fixture for different entity IDs."""
    return str(request.param)


@pytest.fixture(params=["development", "testing", "staging", "production"])
def environment_samples(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for different environments."""
    return str(request.param)


@pytest.fixture(params=["DEBUG", "INFO", "WARNING", "ERROR"])
def log_level_samples(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for different log levels."""
    return str(request.param)


class MockService:
    """Mock service class for testing dependency injection."""

    def __init__(self, name: str, database: object = None) -> None:
        """Initialize mock service with name and optional database."""
        self.name = name
        self.database = database
        self.call_count = 0

    def process(self, data: str) -> str:
        """Mock processing method."""
        self.call_count += 1
        return f"Processed {data} by {self.name} (call #{self.call_count})"


@pytest.fixture
def mock_service_factory() -> object:
    """Provide factory function for creating mock services."""

    def create_service(name: str) -> MockService:
        return MockService(name)

    return create_service


# Integration-specific pytest configuration
# Note: Main configuration is in tests/conftest.py


# Auto-use fixtures for common setup
@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-reset singletons between tests to prevent test interference."""
    # Clear any FLEXT environment variables that might interfere
    for key in list(os.environ.keys()):
        if key.startswith("FLEXT_"):
            monkeypatch.delenv(key, raising=False)

    # Reset global state completed automatically
