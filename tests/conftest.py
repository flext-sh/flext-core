"""Comprehensive test configuration for flext-core with advanced pytest features.

Provides centralized fixtures, test utilities, and configuration for all flext-core tests
using the consolidated tests/support/ infrastructure for maximum testing efficiency.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flext_core import FlextContainer
from tests.support import (
    APITestClient,
    AsyncTestUtils,
    BenchmarkUtils,
    ConfigFactory,
    FailingUserRepository,
    FlextMatchers,
    FlextResultFactory,
    HTTPTestUtils,
    InMemoryUserRepository,
    MemoryProfiler,
    PayloadDataFactory,
    PerformanceProfiler,
    RealAuditService,
    RealEmailService,
    ServiceDataFactory,
    SimpleConfigurationFactory,
    TestBuilders,
    UserDataFactory,
    UserFactory,
)


# Core Fixtures
@pytest.fixture
def test_scenario() -> dict[str, str]:
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
def sample_data() -> dict[str, object]:
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
def test_user_data() -> dict[str, str | int | bool | list[str] | None]:
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
def benchmark_data() -> dict[str, object]:
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


# Factory Fixtures - using consolidated tests/support
@pytest.fixture
def user_factory() -> type[UserDataFactory]:
    """User data factory fixture (simple dict-based)."""
    return UserDataFactory


@pytest.fixture
def advanced_user_factory() -> type[UserFactory]:
    """Advanced user factory fixture (factory-boy with Pydantic)."""
    return UserFactory


@pytest.fixture
def config_factory() -> type[SimpleConfigurationFactory]:
    """Configuration factory fixture."""
    return SimpleConfigurationFactory


@pytest.fixture
def advanced_config_factory() -> type[ConfigFactory]:
    """Advanced configuration factory fixture (factory-boy)."""
    return ConfigFactory


@pytest.fixture
def result_factory() -> type[FlextResultFactory]:
    """FlextResult factory fixture."""
    return FlextResultFactory


@pytest.fixture
def service_factory() -> type[ServiceDataFactory]:
    """Service data factory fixture."""
    return ServiceDataFactory


@pytest.fixture
def payload_factory() -> type[PayloadDataFactory]:
    """Payload factory fixture."""
    return PayloadDataFactory


# Performance Testing Fixtures
@pytest.fixture
def benchmark_utils() -> BenchmarkUtils:
    """Benchmark utilities for performance tests."""
    return BenchmarkUtils()


@pytest.fixture
def memory_profiler() -> MemoryProfiler:
    """Memory profiler for memory usage tests."""
    return MemoryProfiler()


@pytest.fixture
def performance_profiler() -> PerformanceProfiler:
    """Performance profiler for comprehensive profiling."""
    return PerformanceProfiler()


# HTTP Testing Fixtures
@pytest.fixture
def http_test_utils() -> HTTPTestUtils:
    """HTTP testing utilities."""
    return HTTPTestUtils()


@pytest.fixture
def api_test_client() -> APITestClient:
    """API test client for HTTP testing."""
    return APITestClient()


# Async Testing Fixtures
@pytest.fixture
def async_test_utils() -> AsyncTestUtils:
    """Async testing utilities."""
    return AsyncTestUtils()


# Builder Fixtures
@pytest.fixture
def test_builders() -> TestBuilders:
    """Test builders for complex object creation."""
    return TestBuilders()


# Matcher Fixtures
@pytest.fixture
def flext_matchers() -> FlextMatchers:
    """Advanced assertion matchers."""
    return FlextMatchers()


# Shared test configuration
@pytest.fixture(autouse=True)
def setup_test_environment() -> None:
    """Automatically set up test environment for all tests."""
    # object global test setup can go here
    return
    # object global test teardown can go here


# =============================================================================
# FUNCTIONAL TESTING FIXTURES - Real implementations without mocks
# =============================================================================


@pytest.fixture
def user_repository() -> InMemoryUserRepository:
    """Provide clean in-memory user repository for functional testing."""
    return InMemoryUserRepository()


@pytest.fixture
def email_service() -> RealEmailService:
    """Provide real email service for functional testing."""
    return RealEmailService()


@pytest.fixture
def audit_service() -> RealAuditService:
    """Provide real audit service for functional testing."""
    return RealAuditService()


@pytest.fixture
def failing_repository() -> FailingUserRepository:
    """Provide failing repository for error scenario testing."""
    return FailingUserRepository()


@pytest.fixture
def real_services(
    user_repository: InMemoryUserRepository,
    email_service: RealEmailService,
    audit_service: RealAuditService,
) -> dict[str, InMemoryUserRepository | RealEmailService | RealAuditService]:
    """Provide complete set of real services for integration testing."""
    return {
        "user_repository": user_repository,
        "email_service": email_service,
        "audit_service": audit_service,
    }


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


# Mark configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest marks."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "core: Core framework tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "asyncio: Async tests")
