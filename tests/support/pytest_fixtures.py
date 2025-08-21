"""Unified pytest fixtures for flext-core tests.

Comprehensive fixture library using pytest ecosystem plugins:
- pytest-asyncio for async testing
- pytest-benchmark for performance testing
- pytest-mock for mocking
- pytest-timeout for test timeouts
- pytest-randomly for randomized testing
- factory_boy for data generation

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import math
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from flext_core import (
    FlextContainer,
    FlextEntityId,
    FlextResult,
    FlextTimestamp,
    get_flext_container,
)
from flext_core.config import FlextCoreConfig
from flext_core.constants import FlextFieldType

from .factory_boy_factories import (
    BatchFactories,
    ConfigFactory,
    EdgeCaseGenerators,
    FlextResultFactory,
    UserFactory,
)

# ============================================================================
# CORE FIXTURES - Basic infrastructure for all tests
# ============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop]:
    """Create event loop for entire test session - pytest-asyncio integration."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Provide temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Provide temporary file for file-based tests."""
    temp_file = temp_dir / "test_file.json"
    temp_file.write_text('{"test": "data"}')
    return temp_file


# ============================================================================
# FACTORY FIXTURES - Data generation with factory_boy
# ============================================================================


@pytest.fixture
def user_factory() -> type[UserFactory]:
    """Provide UserFactory for creating test users."""
    return UserFactory


@pytest.fixture
def config_factory() -> type[ConfigFactory]:
    """Provide ConfigFactory for creating test configurations."""
    return ConfigFactory


@pytest.fixture
def batch_factories() -> type[BatchFactories]:
    """Provide BatchFactories for bulk data creation."""
    return BatchFactories


@pytest.fixture
def edge_cases() -> type[EdgeCaseGenerators]:
    """Provide EdgeCaseGenerators for boundary testing."""
    return EdgeCaseGenerators


@pytest.fixture
def test_user():
    """Provide single test user instance."""
    return UserFactory()


@pytest.fixture
def test_users():
    """Provide batch of test users."""
    return UserFactory.create_batch(5)


@pytest.fixture
def test_config():
    """Provide test configuration instance."""
    return ConfigFactory()


# ============================================================================
# FLEXT CORE FIXTURES - Core framework components
# ============================================================================


@pytest.fixture
def flext_container() -> FlextContainer:
    """Provide clean FlextContainer instance for each test."""
    return get_flext_container()


@pytest.fixture
def flext_config() -> FlextCoreConfig:
    """Provide FlextCoreConfig with test defaults."""
    return FlextCoreConfig(
        app_name="test_app",
        environment="test",
        debug=True,
        log_level="DEBUG",
    )


@pytest.fixture
def entity_id() -> FlextEntityId:
    """Provide test entity ID."""
    return FlextEntityId("test_entity_123")


@pytest.fixture
def timestamp() -> FlextTimestamp:
    """Provide test timestamp."""
    from datetime import UTC, datetime

    return FlextTimestamp(datetime.now(UTC))


# ============================================================================
# RESULT FIXTURES - FlextResult testing patterns
# ============================================================================


@pytest.fixture
def success_result() -> FlextResult[str]:
    """Provide successful FlextResult."""
    return FlextResultFactory.success("test_data")


@pytest.fixture
def failure_result() -> FlextResult[str]:
    """Provide failed FlextResult."""
    return FlextResultFactory.is_failure("test_error")


@pytest.fixture
def result_factory() -> type[FlextResultFactory]:
    """Provide FlextResultFactory for custom results."""
    return FlextResultFactory


# ============================================================================
# MOCK FIXTURES - Mocking with pytest-mock integration
# ============================================================================


@pytest.fixture
def mock_service() -> MagicMock:
    """Provide mock service for dependency injection testing."""
    mock = MagicMock()
    mock.name = "test_service"
    mock.version = "1.0.0"
    mock.healthy = True
    mock.get_status.return_value = {"status": "ok"}
    return mock


@pytest.fixture
def async_mock_service() -> AsyncMock:
    """Provide async mock service for async testing."""
    mock = AsyncMock()
    mock.name = "async_test_service"
    mock.version = "1.0.0"
    mock.healthy = True
    mock.get_status.return_value = {"status": "ok"}
    mock.async_operation.return_value = "async_result"
    return mock


@pytest.fixture
def mock_database() -> MagicMock:
    """Provide mock database for repository testing."""
    mock = MagicMock()
    mock.connected = True
    mock.execute.return_value = {"rows_affected": 1}
    mock.fetch_one.return_value = {"id": 1, "name": "test"}
    mock.fetch_all.return_value = [{"id": 1, "name": "test"}]
    return mock


@pytest.fixture
async def async_mock_database() -> AsyncMock:
    """Provide async mock database for async repository testing."""
    mock = AsyncMock()
    mock.connected = True
    mock.execute.return_value = {"rows_affected": 1}
    mock.fetch_one.return_value = {"id": 1, "name": "test"}
    mock.fetch_all.return_value = [{"id": 1, "name": "test"}]
    return mock


# ============================================================================
# PERFORMANCE FIXTURES - pytest-benchmark integration
# ============================================================================


@pytest.fixture
def benchmark_config() -> dict[str, Any]:
    """Provide benchmark configuration for performance tests."""
    return {
        "min_rounds": 5,
        "max_time": 1.0,
        "min_time": 0.000005,
        "timer": "time.perf_counter",
        "disable_gc": True,
        "warmup": False,
    }


@pytest.fixture
def performance_data() -> dict[str, Any]:
    """Provide performance test data sets."""
    return {
        "small": UserFactory.create_batch(10),
        "medium": UserFactory.create_batch(100),
        "large": UserFactory.create_batch(1000),
    }


# ============================================================================
# VALIDATION FIXTURES - Field and schema testing
# ============================================================================


@pytest.fixture
def field_types() -> list[FlextFieldType]:
    """Provide all FlextFieldType values for comprehensive testing."""
    return list(FlextFieldType)


@pytest.fixture
def field_test_matrix() -> list[dict[str, Any]]:
    """Provide comprehensive field testing matrix."""
    return [
        {
            "field_type": FlextFieldType.STRING,
            "valid_values": ["test", "hello world", "123"],
            "invalid_values": [123, [], {}, None],
        },
        {
            "field_type": FlextFieldType.INTEGER,
            "valid_values": [0, 1, -1, 999999],
            "invalid_values": ["123", math.pi, [], None],
        },
        {
            "field_type": FlextFieldType.BOOLEAN,
            "valid_values": [True, False],
            "invalid_values": ["true", "false", 1, 0, [], None],
        },
        {
            "field_type": FlextFieldType.FLOAT,
            "valid_values": [0.0, math.pi, -2.5, 1e10],
            "invalid_values": ["3.14", [], {}, None],
        },
    ]


@pytest.fixture
def edge_case_values() -> dict[str, list[Any]]:
    """Provide edge case values for boundary testing."""
    return EdgeCaseGenerators().empty_values()


# ============================================================================
# ASYNC FIXTURES - pytest-asyncio integration
# ============================================================================


@pytest.fixture
async def async_container() -> AsyncGenerator[FlextContainer]:
    """Provide async FlextContainer for async testing."""
    return get_flext_container()
    # Cleanup if needed


@pytest.fixture
async def async_service_mock() -> AsyncGenerator[AsyncMock]:
    """Provide async service mock that cleans up properly."""
    mock = AsyncMock()
    mock.name = "async_service"
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    mock.healthy = True

    yield mock

    # Cleanup
    if mock.stop.called:
        await mock.stop()


# ============================================================================
# INTEGRATION FIXTURES - Multi-component testing
# ============================================================================


@pytest.fixture
def integrated_system(
    flext_container: FlextContainer, flext_config: FlextCoreConfig
) -> dict[str, Any]:
    """Provide integrated system for end-to-end testing."""
    return {
        "container": flext_container,
        "config": flext_config,
        "users": UserFactory.create_batch(3),
        "services": ["auth", "cache", "database"],
    }


@pytest.fixture
def test_scenario_basic() -> dict[str, Any]:
    """Provide basic test scenario data."""
    return {
        "name": "basic_scenario",
        "users": UserFactory.create_batch(2),
        "config": ConfigFactory(),
        "expected_results": {
            "success": True,
            "user_count": 2,
        },
    }


@pytest.fixture
def test_scenario_complex() -> dict[str, Any]:
    """Provide complex test scenario data."""
    return {
        "name": "complex_scenario",
        "users": UserFactory.create_batch(10),
        "config": ConfigFactory(debug=False),
        "batch_data": BatchFactories.create_field_matrix(),
        "expected_results": {
            "success": True,
            "user_count": 10,
            "field_count": 10,
        },
    }


# ============================================================================
# ERROR TESTING FIXTURES - Exception and error handling
# ============================================================================


@pytest.fixture
def error_scenarios() -> list[dict[str, Any]]:
    """Provide error scenarios for exception testing."""
    return [
        {
            "name": "null_pointer",
            "exception": ValueError,
            "message": "Value cannot be None",
        },
        {
            "name": "invalid_type",
            "exception": TypeError,
            "message": "Expected string, got int",
        },
        {
            "name": "out_of_range",
            "exception": IndexError,
            "message": "Index out of range",
        },
    ]


@pytest.fixture
def timeout_config() -> dict[str, Any]:
    """Provide timeout configuration for pytest-timeout."""
    return {
        "timeout": 30,  # 30 seconds default
        "timeout_method": "thread",
    }


# ============================================================================
# PARAMETRIZE HELPERS - Common parametrization patterns
# ============================================================================

# Parametrize decorators for common test patterns
field_type_params = pytest.mark.parametrize(
    "field_type",
    [
        FlextFieldType.STRING,
        FlextFieldType.INTEGER,
        FlextFieldType.BOOLEAN,
        FlextFieldType.FLOAT,
    ],
    ids=["string", "integer", "boolean", "float"],
)

result_state_params = pytest.mark.parametrize(
    ("is_success", "expected"),
    [(True, "success"), (False, "failure")],
    ids=["success_case", "failure_case"],
)

batch_size_params = pytest.mark.parametrize(
    "batch_size",
    [1, 10, 100],
    ids=["single", "small_batch", "large_batch"],
)


# ============================================================================
# UTILITY FIXTURES - Helper functions and utilities
# ============================================================================


@pytest.fixture
def assert_helpers() -> dict[str, Any]:
    """Provide assertion helper functions."""

    def assert_success(result: Any) -> None:
        assert result.success

    def assert_failure(result: Any) -> None:
        assert not result.success

    def assert_value_equals(result: Any, expected: Any) -> None:
        assert result.value == expected

    def assert_error_contains(result: Any, text: str) -> None:
        assert text in result.error

    return {
        "assert_success": assert_success,
        "assert_failure": assert_failure,
        "assert_value_equals": assert_value_equals,
        "assert_error_contains": assert_error_contains,
    }


@pytest.fixture
def test_markers() -> dict[str, Any]:
    """Provide test markers and categories."""
    return {
        "unit": pytest.mark.unit,
        "integration": pytest.mark.integration,
        "e2e": pytest.mark.e2e,
        "performance": pytest.mark.benchmark,
        "slow": pytest.mark.slow,
        "async": pytest.mark.asyncio,
    }


# ============================================================================
# EXPORT PUBLIC FIXTURES
# ============================================================================

__all__ = [
    # Utility fixtures
    "assert_helpers",
    # Async fixtures
    "async_container",
    "async_mock_database",
    "async_mock_service",
    "async_service_mock",
    "batch_factories",
    "batch_size_params",
    # Performance fixtures
    "benchmark_config",
    "config_factory",
    "edge_case_values",
    "edge_cases",
    "entity_id",
    # Error fixtures
    "error_scenarios",
    # Core fixtures
    "event_loop",
    "failure_result",
    "field_test_matrix",
    # Parametrize helpers
    "field_type_params",
    # Validation fixtures
    "field_types",
    "flext_config",
    # FlextCore fixtures
    "flext_container",
    # Integration fixtures
    "integrated_system",
    "mock_database",
    # Mock fixtures
    "mock_service",
    "performance_data",
    "result_factory",
    "result_state_params",
    # Result fixtures
    "success_result",
    "temp_dir",
    "temp_file",
    "test_config",
    "test_markers",
    "test_scenario_basic",
    "test_scenario_complex",
    "test_user",
    "test_users",
    "timeout_config",
    "timestamp",
    # Factory fixtures
    "user_factory",
]
