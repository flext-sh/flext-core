# ruff: noqa: PLC0415, ANN401
"""Advanced pytest fixtures using the full pytest ecosystem.

This module provides comprehensive fixtures leveraging:
- pytest-asyncio for async testing
- pytest-benchmark for performance testing
- pytest-randomly for randomized testing
- pytest-timeout for timeout handling
- pytest-xdist for parallel execution
- pytest-mock for mocking
- pytest-clarity for better assertions

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any, Self
from unittest.mock import Mock

import pytest
import pytest_asyncio

from flext_core import FlextResult, get_logger
from flext_core.container import FlextContainer
from flext_core.loggings import FlextLogger
from tests.support.factories import (
    BaseTestEntity,
    ConfigurationFactory,
    FlextResultFactory,
    TestEntityFactory,
    UserDataFactory,
    create_batch,
)

# =============================================================================
# CORE FIXTURES - Used across all tests
# =============================================================================


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Configure event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def logger() -> FlextLogger:
    """Provide configured logger for testing."""
    return get_logger("test_logger")


@pytest.fixture(scope="session")
def temp_directory() -> Generator[Path]:
    """Provide temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def container() -> Generator[FlextContainer]:
    """Provide FlextContainer for dependency injection testing."""
    container = FlextContainer()

    # Register common test services
    container.register("logger", get_logger("test"))
    container.register("config", ConfigurationFactory())

    yield container

    # Cleanup
    container.clear()


# =============================================================================
# FACTORY FIXTURES - Generate test data
# =============================================================================


@pytest.fixture
def user_data() -> dict[str, Any]:
    """Generate single user data."""
    return UserDataFactory()


@pytest.fixture
def user_batch(request: pytest.FixtureRequest) -> list[dict[str, Any]]:
    """Generate batch of users with configurable size.

    Usage:
        @pytest.mark.parametrize("user_batch", [10], indirect=True)
        def test_with_users(user_batch):
            assert len(user_batch) == 10
    """
    count = getattr(request, "param", 5)
    return create_batch.create_user_batch(count)


@pytest.fixture
def admin_user() -> dict[str, Any]:
    """Generate admin user data."""
    return UserDataFactory(is_admin=True)


@pytest.fixture
def inactive_user() -> dict[str, Any]:
    """Generate inactive user data."""
    return UserDataFactory(is_inactive=True)


@pytest.fixture
def test_entity() -> BaseTestEntity:
    """Generate test entity."""
    return TestEntityFactory()


@pytest.fixture
def entity_batch(request: pytest.FixtureRequest) -> list[BaseTestEntity]:
    """Generate batch of entities with configurable size."""
    count = getattr(request, "param", 3)
    return create_batch.create_entity_batch(count)


@pytest.fixture
def high_priority_entity() -> BaseTestEntity:
    """Generate high priority entity."""
    return TestEntityFactory(high_priority=True)


# =============================================================================
# RESULT FIXTURES - For FlextResult testing
# =============================================================================


@pytest.fixture
def success_result() -> FlextResult[str]:
    """Generate successful FlextResult."""
    return FlextResultFactory.create_success("test_success_data")


@pytest.fixture
def failure_result() -> FlextResult[Any]:
    """Generate failed FlextResult."""
    return FlextResultFactory.create_failure(
        "Test failure message",
        "TEST_FAILURE",
    )


@pytest.fixture
def validation_failure_result() -> FlextResult[Any]:
    """Generate validation failure FlextResult."""
    return FlextResultFactory.create_validation_failure("test_field")


@pytest.fixture
def mixed_results() -> list[FlextResult[Any]]:
    """Generate mixed success/failure results for batch testing."""
    return create_batch.create_mixed_results(success_count=3, failure_count=2)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Generate test configuration."""
    return ConfigurationFactory(development=True)


@pytest.fixture
def production_config() -> dict[str, Any]:
    """Generate production-like configuration."""
    return ConfigurationFactory(production=True)


@pytest.fixture
def config_batch() -> list[dict[str, Any]]:
    """Generate batch of configurations."""
    return create_batch.create_config_batch(3)


# =============================================================================
# MOCK FIXTURES - For testing with mocks
# =============================================================================


@pytest.fixture
def mock_service(mocker: object) -> Mock:
    """Provide mock service for testing."""
    mock = mocker.Mock()
    mock.get_data.return_value = FlextResultFactory.create_success("mock_data")
    mock.process.return_value = FlextResultFactory.create_success("processed")
    mock.health_check.return_value = True
    return mock


@pytest.fixture
def mock_database(mocker: object) -> Mock:
    """Provide mock database for testing."""
    mock = mocker.Mock()
    mock.connect.return_value = True
    mock.execute.return_value = FlextResultFactory.create_success([])
    mock.close.return_value = None
    return mock


@pytest.fixture
def mock_logger(mocker: object) -> Mock:
    """Provide mock logger for testing."""
    return mocker.patch("flext_core.loggings.get_logger")


# =============================================================================
# ASYNC FIXTURES - For asyncio testing
# =============================================================================


@pytest_asyncio.fixture  # type: ignore[misc]
async def async_service() -> AsyncGenerator[Any]:
    """Provide async service for testing."""

    class AsyncTestService:
        async def async_process(self, data: object) -> FlextResult[Any]:
            """Simulate async processing."""
            await asyncio.sleep(0.01)  # Simulate async work
            return FlextResultFactory.create_success(f"processed_{data}")

        async def async_batch_process(self, items: list[Any]) -> list[FlextResult[Any]]:
            """Simulate async batch processing."""
            results = []
            for item in items:
                result = await self.async_process(item)
                results.append(result)
            return results

    service = AsyncTestService()
    yield service


@pytest_asyncio.fixture  # type: ignore[misc]
async def async_context_manager() -> AsyncGenerator[Any]:
    """Provide async context manager for testing."""

    class AsyncContextManager:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False

        async def __aenter__(self) -> Self:
            self.entered = True
            await asyncio.sleep(0.001)  # Simulate async setup
            return self

        async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
            self.exited = True
            await asyncio.sleep(0.001)  # Simulate async cleanup

    yield AsyncContextManager


# =============================================================================
# PERFORMANCE FIXTURES - For benchmarking
# =============================================================================


@pytest.fixture
def benchmark_data() -> dict[str, Any]:
    """Generate data for benchmarking tests."""
    return {
        "small_dataset": create_batch.create_user_batch(10),
        "medium_dataset": create_batch.create_user_batch(100),
        "large_dataset": create_batch.create_user_batch(1000),
        "entities": create_batch.create_entity_batch(50),
    }


@pytest.fixture
def performance_thresholds() -> dict[str, float]:
    """Define performance thresholds for benchmark tests."""
    return {
        "max_time": 1.0,  # seconds
        "max_memory": 100 * 1024 * 1024,  # 100MB
        "min_ops_per_sec": 1000,
    }


# =============================================================================
# FILE SYSTEM FIXTURES
# =============================================================================


@pytest.fixture
def test_files(temp_directory: Path) -> dict[str, Path]:
    """Create test files in temporary directory."""
    files = {}

    # JSON config file
    config_file = temp_directory / "test_config.json"
    config_file.write_text('{"debug": true, "version": "1.0.0"}')
    files["config"] = config_file

    # Log file
    log_file = temp_directory / "test.log"
    log_file.write_text("INFO: Test log entry\nERROR: Test error\n")
    files["log"] = log_file

    # CSV data file
    csv_file = temp_directory / "test_data.csv"
    csv_file.write_text("name,age,active\nTest User,25,true\nOther User,30,false\n")
    files["csv"] = csv_file

    # Empty file
    empty_file = temp_directory / "empty.txt"
    empty_file.touch()
    files["empty"] = empty_file

    return files


# =============================================================================
# ERROR HANDLING FIXTURES
# =============================================================================


@pytest.fixture
def error_scenarios() -> dict[str, Exception]:
    """Provide various error scenarios for testing."""
    return {
        "value_error": ValueError("Invalid value provided"),
        "type_error": TypeError("Wrong type provided"),
        "key_error": KeyError("missing_key"),
        "runtime_error": RuntimeError("Runtime failure"),
        "custom_error": Exception("Custom test error"),
    }


@pytest.fixture
def timeout_scenarios() -> dict[str, float]:
    """Provide timeout scenarios for testing."""
    return {
        "fast": 0.1,  # 100ms
        "normal": 1.0,  # 1 second
        "slow": 5.0,  # 5 seconds
        "very_slow": 30.0,  # 30 seconds
    }


# =============================================================================
# PARAMETRIZED FIXTURES - For comprehensive testing
# =============================================================================


@pytest.fixture(params=[1, 5, 10, 50])
def batch_sizes(request: pytest.FixtureRequest) -> int:
    """Parametrized fixture for different batch sizes."""
    return request.param


@pytest.fixture(params=["string", "integer", "boolean", "float"])
def field_types(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for different field types."""
    return request.param


@pytest.fixture(params=[True, False])
def boolean_flags(request: pytest.FixtureRequest) -> bool:
    """Parametrized fixture for boolean flags."""
    return request.param


# =============================================================================
# INTEGRATION FIXTURES - For complex scenarios
# =============================================================================


@pytest.fixture
def integration_scenario() -> dict[str, Any]:
    """Provide complete integration testing scenario."""
    return {
        "users": create_batch.create_user_batch(10),
        "configs": create_batch.create_config_batch(3),
        "entities": create_batch.create_entity_batch(5),
        "results": create_batch.create_mixed_results(3, 2),
        "settings": {
            "parallel": True,
            "timeout": 30,
            "retries": 3,
            "cache_enabled": True,
        },
    }


# =============================================================================
# CLEANUP FIXTURES - Auto cleanup resources
# =============================================================================


@pytest.fixture(autouse=True)
def auto_cleanup() -> Generator[None]:
    """Automatically cleanup test resources after each test."""
    # Setup - runs before each test
    yield

    # Cleanup - runs after each test
    # Force garbage collection to clean up test objects
    import gc

    gc.collect()


# =============================================================================
# CONDITION FIXTURES - For conditional testing
# =============================================================================


@pytest.fixture
def skip_conditions() -> dict[str, bool]:
    """Provide conditions for conditional test skipping."""
    import os
    import sys

    return {
        "is_windows": sys.platform == "win32",
        "is_linux": sys.platform == "linux",
        "is_mac": sys.platform == "darwin",
        "has_docker": bool(os.environ.get("DOCKER_AVAILABLE")),
        "is_ci": bool(os.environ.get("CI")),
        "python_gt_39": sys.version_info >= (3, 9),
        "python_gt_311": sys.version_info >= (3, 11),
    }
