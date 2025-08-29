"""Comprehensive test configuration for flext-core with advanced pytest features.

Provides centralized fixtures, test utilities, and configuration for all flext-core tests
using the consolidated tests/support/ infrastructure for maximum testing efficiency.
"""

from __future__ import annotations

import pytest

from .support import (
    APITestClient,
    AsyncTestUtils,
    BenchmarkUtils,
    ConfigFactory,
    FlextMatchers,
    FlextResultFactory,
    HTTPTestUtils,
    MemoryProfiler,
    PayloadDataFactory,
    PerformanceProfiler,
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


# Mark configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest marks."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "core: Core framework tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "asyncio: Async tests")
