"""Comprehensive test configuration for flext-core with advanced pytest features.

Provides centralized fixtures, test utilities, and configuration for all flext-core tests
using the consolidated tests/support/ infrastructure for maximum testing efficiency.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flext_core import FlextConfig, FlextContainer, FlextCore, FlextLogger, FlextTypes
from flext_tests import FlextTestsAsyncs


# Core Fixtures
@pytest.fixture
def test_scenario() -> FlextTypes.Core.Headers:
    """Basic test scenario fixture."""
    return {"status": "test", "environment": "test"}


@pytest.fixture
def clean_container() -> Generator[FlextContainer]:
    """Provide isolated FlextContainer for dependency injection testing.

    Enterprise-grade DI container fixture ensuring complete test isolation.
    Each test receives a fresh container with no service registrations.

    Yields:
      FlextContainer instance with proper cleanup

    """
    container = FlextContainer()
    yield container
    # Cleanup: Clear the container after test
    container.clear()


@pytest.fixture
def sample_data() -> FlextTypes.Core.Dict:
    """Provide deterministic sample data for tests.

    Enterprise-grade test data factory providing consistent, typed sample data
    for testing data transformation and validation patterns across the ecosystem.

    Returns:
      Dict containing various data types for comprehensive testing

    """
    return {
        "string": "test_string",
        "integer": 42,
        "float": math.pi,
        "boolean": True,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "none": None,
        "timestamp": datetime.now(UTC).isoformat(),
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
    }


@pytest.fixture
def test_user_data() -> dict[str, str | int | bool | FlextTypes.Core.StringList | None]:
    """Provide consistent user data for domain testing.

    User data factory aligned with shared domain patterns
    for testing DDD entities and value objects.

    Returns:
      Dict containing user data for testing

    """
    return {
        "id": "user-12345",
        "name": "Test User",
        "email": "test.user@flext.example.com",
        "age": 30,
        "is_active": True,
        "created_at": datetime.now(UTC).isoformat(),
        "roles": ["user", "tester"],
    }


@pytest.fixture
def temp_directory() -> Generator[Path]:
    """Provide temporary directory for file-based testing.

    Temporary directory fixture for testing file operations,
    configuration loading, and data persistence patterns.

    Yields:
      Path to temporary directory with automatic cleanup

    """
    temp_dir = Path(tempfile.mkdtemp(prefix="flext_test_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def configured_container(
    clean_container: FlextContainer,
    mock_external_service: object,
) -> FlextContainer:
    """Provide pre-configured container for integration testing.

    Container factory with common service registrations for testing
    service integration patterns and dependency resolution.

    Args:
      clean_container: Fresh container instance
      mock_external_service: External service for functional testing

    Returns:
      FlextContainer with standard test services registered

    """
    clean_container.register("external_service", mock_external_service)
    clean_container.register("config", {"test_mode": True})
    clean_container.register("logger", "test_logger")
    return clean_container


@pytest.fixture
def error_context() -> dict[str, str | None]:
    """Provide structured error context for testing.

    Error context factory for testing FlextResult error handling,
    logging, and observability patterns.

    Returns:
      Dict containing error context fields for testing

    """
    return {
        "error_code": "TEST_ERROR_001",
        "severity": "medium",
        "component": "test_module",
        "user_id": "test_user_123",
        "request_id": "test-request-456",
        "timestamp": datetime.now(UTC).isoformat(),
        "stack_trace": None,  # Populated by actual error handling
    }


@pytest.fixture
def performance_threshold() -> dict[str, float]:
    """Provide performance thresholds for testing.

    Performance threshold configuration for validating
    that core operations meet enterprise performance standards.

    Returns:
      Dict containing performance thresholds in seconds

    """
    return {
        "result_creation": 0.001,  # 1ms for FlextResult creation
        "container_registration": 0.005,  # 5ms for service registration
        "container_retrieval": 0.001,  # 1ms for service retrieval
        "validation": 0.01,  # 10ms for validation operations
        "serialization": 0.05,  # 50ms for serialization
    }


@pytest.fixture
def benchmark_data() -> FlextTypes.Core.Dict:
    """Provide standardized data for performance testing.

    Benchmark data factory for testing performance characteristics
    of core patterns and algorithms.

    Returns:
      Dict containing benchmark data sets

    """
    return {
        "small_dataset": list(range(100)),
        "medium_dataset": list(range(1000)),
        "large_dataset": list(range(10000)),
        "complex_dict": {f"key_{i}": f"value_{i}" for i in range(1000)},
        "nested_structure": {
            "level1": {"level2": {"level3": {"data": list(range(100))}}},
        },
    }


# Factory Fixtures - COMMENTED OUT: Use FlextTestsMatchers directly instead of aliases
# @pytest.fixture
# def user_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def advanced_user_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def config_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers


# COMMENTED OUT: Use FlextTestsMatchers directly instead of aliases
# @pytest.fixture
# def advanced_config_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def result_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def service_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def payload_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers


# -----------------------------------------------------------------------------
# Test isolation for FlextCore singleton state
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_flext_core_state() -> None:
    """Clear FlextCore specialized configs before each test.

    Prevents cross-test leakage of database/security/logging configs that could
    make property-default tests flaky or order-dependent.
    """
    # FlextCore was simplified - no longer has _specialized_configs
    # Just reset the singleton instance
    FlextCore.reset_instance()


@pytest.fixture
def logging_test_env() -> Generator[None]:
    """Set up test environment for logging tests.

    Ensures logging tests get consistent log level by temporarily overriding
    the FLEXT_LOG_LEVEL environment variable to WARNING as expected by tests.
    """
    # Save original value
    original_log_level = os.environ.get("FLEXT_LOG_LEVEL")

    try:
        # Clear both config and logger singleton states
        FlextConfig.clear_global_instance()

        # Reset logger singleton state
        FlextLogger._configured = False
        FlextLogger._instances.clear()

        # Set the expected log level for logging tests
        os.environ["FLEXT_LOG_LEVEL"] = "WARNING"
        yield
    finally:
        # Clear both singleton states again and restore original value
        FlextConfig.clear_global_instance()
        FlextLogger._configured = False
        FlextLogger._instances.clear()

        if original_log_level is not None:
            os.environ["FLEXT_LOG_LEVEL"] = original_log_level
        elif "FLEXT_LOG_LEVEL" in os.environ:
            del os.environ["FLEXT_LOG_LEVEL"]


# Performance Testing Fixtures
# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# COMMENTED OUT: # def benchmark_utils() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# COMMENTED OUT: # def memory_profiler() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# COMMENTED OUT: # def performance_profiler() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# HTTP Testing Fixtures
# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# COMMENTED OUT: # def http_test_utils() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# COMMENTED OUT: # def api_test_client() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
@pytest.fixture
def async_test_utils() -> FlextTestsAsyncs:
    """Provide async test utilities."""
    return FlextTestsAsyncs()


#
# @pytest.fixture
# COMMENTED OUT: # def test_builders() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# COMMENTED OUT: # def flext_matchers() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# Shared test configuration
@pytest.fixture(autouse=True)
def setup_test_environment() -> None:
    """Automatically set up test environment for all tests."""
    # object global test setup can go here
    return
    # object global test teardown can go here


# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# COMMENTED OUT: # def user_repository() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# def email_service() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# def audit_service() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# def failing_repository() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# def real_services() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# Mark configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest marks."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "core: Core framework tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "asyncio: Async tests")
