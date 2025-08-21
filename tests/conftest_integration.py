"""Pytest configuration and shared fixtures for FLEXT Core tests.

Modern pytest fixtures following best practices for enterprise testing.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from flext_core import (
    FlextEnvironment,
    FlextLogLevel,
    FlextResult,
    FlextSettings,
    TEntityId,
    get_flext_container,
)

# NOTE: clean_container fixture removed to eliminate duplication with main conftest.py
# Integration tests should import it from the main conftest.py


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


class FunctionalDatabaseService:
    """Functional database service for DI testing - real implementation."""

    def __init__(self) -> None:
        """Initialize with test data."""
        self._users = {"user-123": {"id": "user-123", "name": "Test User"}}
        self._healthy = True

    def get_user(self, user_id: str) -> FlextResult[dict[str, object]]:
        """Get user by ID - functional implementation."""
        if not self._healthy:
            return FlextResult[dict[str, object]].fail("Database unavailable")
        if user_id in self._users:
            return FlextResult[dict[str, object]].ok(self._users[user_id])
        return FlextResult[dict[str, object]].fail(f"User {user_id} not found")

    def save_user(self, user_data: dict[str, object]) -> FlextResult[bool]:
        """Save user data - functional implementation."""
        if not self._healthy:
            return FlextResult[bool].fail("Database unavailable")
        user_id = str(user_data.get("id", ""))
        if user_id:
            self._users[user_id] = user_data
            return FlextResult[bool].ok(True)
        return FlextResult[bool].fail("User ID required")

    def is_healthy(self) -> bool:
        """Check database health."""
        return self._healthy


class FunctionalLoggerService:
    """Functional logger service for DI testing - real implementation."""

    def __init__(self) -> None:
        """Initialize with log storage."""
        self.logs: list[dict[str, str]] = []

    def info(self, message: str) -> None:
        """Log info message - functional implementation."""
        self.logs.append({"level": "info", "message": message})

    def error(self, message: str) -> None:
        """Log error message - functional implementation."""
        self.logs.append({"level": "error", "message": message})

    def debug(self, message: str) -> None:
        """Log debug message - functional implementation."""
        self.logs.append({"level": "debug", "message": message})

    def get_logs(self, level: str | None = None) -> list[dict[str, str]]:
        """Get all logs or filtered by level."""
        if level is None:
            return self.logs.copy()
        return [log for log in self.logs if log["level"] == level]


@pytest.fixture
def functional_database() -> FunctionalDatabaseService:
    """Provide functional database service for DI testing."""
    return FunctionalDatabaseService()


@pytest.fixture
def functional_logger() -> FunctionalLoggerService:
    """Provide functional logger service for DI testing."""
    return FunctionalLoggerService()


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


class FunctionalTestService:
    """Functional service class for testing dependency injection - real implementation."""

    def __init__(
        self, name: str, database: FunctionalDatabaseService | None = None
    ) -> None:
        """Initialize functional service with name and optional database."""
        self.name = name
        self.database = database
        self.call_count = 0
        self.processed_data: list[str] = []

    def process(self, data: str) -> FlextResult[str]:
        """Functional processing method - validates real behavior."""
        if not data or data.strip() == "":
            return FlextResult[str].fail("Data cannot be empty")

        self.call_count += 1
        result = f"Processed {data} by {self.name} (call #{self.call_count})"
        self.processed_data.append(data)
        return FlextResult[str].ok(result)

    def get_processed_count(self) -> int:
        """Get count of processed items."""
        return len(self.processed_data)

    def is_healthy(self) -> bool:
        """Check service health."""
        return True


@pytest.fixture
def functional_service_factory() -> object:
    """Provide factory function for creating functional services."""

    def create_service(
        name: str, database: FunctionalDatabaseService | None = None
    ) -> FunctionalTestService:
        return FunctionalTestService(name, database)

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
