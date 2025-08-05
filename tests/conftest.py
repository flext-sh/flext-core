"""Pytest configuration for FLEXT Core test suite.

Enterprise-grade pytest configuration providing comprehensive shared fixtures,
markers, and test infrastructure aligned with modern pytest best practices.
Supports unit, integration, and e2e testing with proper isolation and type safety.

Architecture:
    Testing Infrastructure → Fixture Management → Test Isolation → Quality Gates

    This configuration module establishes:
    - Comprehensive test markers for organized test execution
    - Isolated fixtures preventing cross-test contamination
    - Modern pytest patterns following AAA (Arrange-Act-Assert) principles
    - Type-safe fixture definitions with proper cleanup
    - Enterprise-grade test data factories and builders

Quality Standards:
    - Test Isolation: Each test runs with clean state
    - Type Safety: All fixtures properly typed
    - Performance: Fast fixture setup/teardown
    - Reliability: Deterministic test behavior
"""

import math
import os
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import structlog

from flext_core.container import FlextContainer
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult


# Pytest configuration for test markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers for organized test execution."""
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (deselect with '-m \"not unit\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "pep8: marks tests for PEP8 compliance validation",
    )
    config.addinivalue_line(
        "markers",
        "core: marks tests for core framework functionality",
    )
    config.addinivalue_line(
        "markers",
        "architecture: marks tests for architectural patterns",
    )
    config.addinivalue_line(
        "markers",
        "ddd: marks tests for domain-driven design",
    )
    config.addinivalue_line(
        "markers",
        "happy_path: marks tests for successful execution paths",
    )
    config.addinivalue_line(
        "markers",
        "error_path: marks tests for error handling and failure scenarios",
    )
    config.addinivalue_line(
        "markers",
        "boundary: marks tests for boundary conditions and edge cases",
    )
    config.addinivalue_line(
        "markers",
        "regression: marks tests for preventing regression bugs",
    )


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables that might interfere with tests."""
    # Clear any FLEXT environment variables that might interfere
    for key in list(os.environ.keys()):
        if key.startswith("FLEXT_"):
            monkeypatch.delenv(key, raising=False)


# ============================================================================
# Test Data Factories and Builders
# ============================================================================


@pytest.fixture
def sample_data() -> dict[
    str, str | int | float | bool | list[int] | dict[str, str] | None
]:
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
def sample_metadata() -> dict[str, str | float | list[str]]:
    """Provide sample metadata for comprehensive testing.

    Metadata factory for testing message payloads, audit trails,
    and correlation patterns throughout the FLEXT ecosystem.

    Returns:
        Dict containing metadata fields for testing
    """
    return {
        "source": "test_suite",
        "timestamp": datetime.now(UTC).timestamp(),
        "version": "1.0.0",
        "tags": ["unit", "test", "flext-core"],
        "correlation_id": "test-correlation-12345",
        "trace_id": "test-trace-67890",
        "environment": "test",
    }


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
def test_user_data() -> dict[str, str | int | bool | list[str]]:
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
def mock_external_service() -> Generator[MagicMock]:
    """Provide isolated mock for external service testing.

    Mock factory for testing service integration patterns
    without external dependencies.

    Yields:
        MagicMock configured for external service simulation
    """
    mock = MagicMock()
    mock.is_healthy.return_value = True
    mock.get_data.return_value = {"status": "success", "data": "test_data"}
    mock.process.return_value = FlextResult.ok("processed")

    yield mock

    # Cleanup
    mock.reset_mock()


# ============================================================================
# Core Component Fixtures
# ============================================================================


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

    # Explicit cleanup to prevent cross-test contamination
    container.clear()


@pytest.fixture
def configured_container(
    clean_container: FlextContainer,
    mock_external_service: MagicMock,
) -> FlextContainer:
    """Provide pre-configured container for integration testing.

    Container factory with common service registrations for testing
    service integration patterns and dependency resolution.

    Args:
        clean_container: Fresh container instance
        mock_external_service: Mock external service

    Returns:
        FlextContainer with standard test services registered
    """
    # Register common test services
    clean_container.register("external_service", mock_external_service)
    clean_container.register("config", {"test_mode": True})
    clean_container.register("logger", "test_logger")

    return clean_container


@pytest.fixture
def temp_directory() -> Generator[Path]:
    """Provide temporary directory for file-based testing.

    Temporary directory fixture for testing file operations,
    configuration loading, and data persistence patterns.

    Yields:
        Path to temporary directory with automatic cleanup
    """
    import shutil
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix="flext_test_"))

    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# ============================================================================
# State Management and Cleanup
# ============================================================================


@pytest.fixture(autouse=True)
def clean_logging_state() -> Generator[None]:
    """Reset logging configuration for test isolation.

    Automatic fixture ensuring logging state is clean between tests.
    Prevents log pollution and ensures deterministic test behavior.

    Yields:
        None (setup/teardown only)
    """
    # Setup: Clear initial state
    FlextLoggerFactory.clear_loggers()
    FlextLoggerFactory.clear_log_store()

    yield

    # Teardown: Reset to clean state
    FlextLoggerFactory.clear_loggers()
    FlextLoggerFactory.clear_log_store()

    # Clear structlog caches if they exist
    try:
        if (
            hasattr(structlog, "_CONFIG")
            and hasattr(structlog._CONFIG, "logger_factory")
            and hasattr(structlog._CONFIG.logger_factory, "_cache")
        ):
            structlog._CONFIG.logger_factory._cache.clear()
    except AttributeError:
        # structlog configuration might not be set up yet
        pass


# ============================================================================
# Performance and Benchmarking
# ============================================================================


@pytest.fixture
def benchmark_data() -> dict[
    str,
    list[int] | dict[str, str] | dict[str, dict[str, dict[str, dict[str, list[int]]]]],
]:
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
            "level1": {"level2": {"level3": {"data": list(range(100))}}}
        },
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
