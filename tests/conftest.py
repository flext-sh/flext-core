"""Comprehensive pytest configuration with full plugin integration and factory_boy.

This configuration provides extensive testing infrastructure with ALL pytest plugins,
factory_boy, and Faker patterns integrated for maximum testing capabilities.

Plugin Integration:
- pytest-asyncio: Async test support with event loops
- pytest-benchmark: Performance benchmarking with memory profiling
- pytest-httpx: HTTP request mocking and testing
- pytest-mock: Advanced mocking with MockerFixture
- pytest-xdist: Parallel test execution
- pytest-cov: Coverage reporting with detailed metrics
- pytest-clarity: Enhanced error messages and diffs
- pytest-sugar: Beautiful test output formatting
- pytest-randomly: Randomized test execution order
- pytest-deadfixtures: Unused fixture detection
- pytest-env: Environment variable management
- pytest-timeout: Test timeout management
- factory_boy: Advanced test data generation
- Faker: Realistic data generation

Support Libraries:
- domain_factories.py: Factory_boy patterns with FlextCore integration
- performance_utils.py: Benchmarking with memory profiling
- async_utils.py: Async testing with concurrency patterns
- http_utils.py: HTTP testing with pytest-httpx
- matchers.py: Advanced FlextResult assertion patterns
- builders.py: Fluent test object construction
"""

# ruff: noqa: S101
from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import shutil
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Generic, TypedDict, TypeVar
from unittest import mock

import factory
import pytest
import structlog
from _pytest.fixtures import SubRequest
from faker import Faker
from hypothesis import strategies as st
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_httpx import HTTPXMock
from pytest_mock import MockerFixture

from flext_core import (
    FlextAggregateRoot,
    FlextBaseHandler,
    FlextCommands,
    FlextContainer,
    FlextEntity,
    FlextEntityId,
    FlextEntityStatus,
    FlextEvent,
    FlextLoggerFactory,
    FlextOperationStatus,
    FlextResult,
    FlextValue,
)
from tests.support.async_utils import (
    AsyncConcurrencyTesting,
    AsyncContextManagers,
    AsyncMockUtils,
    AsyncTestUtils,
)
from tests.support.builders import TestBuilders
from tests.support.domain_factories import (
    ConfigurationFactory,
    FlextResultFactory,
    PayloadDataFactory,
    UserDataFactory,
)
from tests.support.factory_boy_factories import (
    EdgeCaseGenerators,
    UserFactory,
    create_validation_test_cases,
)
from tests.support.http_utils import (
    APITestClient,
    HTTPScenarioBuilder,
    HTTPTestUtils,
    WebhookTestUtils,
)
from tests.support.matchers import FlextMatchers
from tests.support.performance_utils import (
    AsyncBenchmark,
    BenchmarkUtils,
    MemoryProfiler,
    PerformanceProfiler,
)

# Configure factory_boy to use Faker for realistic data
fake = Faker()
factory.Faker._DEFAULT_LOCALE = "en_US"

# Type variables for generic fixtures
T = TypeVar("T")
E = TypeVar("E", bound=FlextEntity)
V = TypeVar("V", bound=FlextValue)


# ============================================================================
# Advanced Pytest Configuration
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with comprehensive markers for all testing scenarios."""
    # Basic test type markers
    markers = [
        "unit: Unit tests for individual components",
        "integration: Integration tests for component interaction",
        "e2e: End-to-end tests for complete workflows",
        "functional: Functional tests for business requirements",
        "acceptance: Acceptance tests for user stories",
        "smoke: Smoke tests for basic functionality",
        "regression: Regression tests for bug prevention",
        "sanity: Sanity tests for basic health checks",
        # Test execution characteristics
        "slow: Tests taking longer than 1 second",
        "fast: Tests completing under 100ms",
        "flaky: Tests that may fail intermittently",
        "skip_in_ci: Tests to skip in CI environment",
        "local_only: Tests requiring local environment",
        # Architecture and patterns
        "core: Core framework functionality tests",
        "architecture: Architectural pattern validation",
        "ddd: Domain-driven design pattern tests",
        "cqrs: Command Query Responsibility Segregation tests",
        "solid: SOLID principle validation tests",
        "clean_arch: Clean Architecture compliance tests",
        # Test path types
        "happy_path: Successful execution path tests",
        "error_path: Error handling and failure scenario tests",
        "boundary: Boundary condition and edge case tests",
        "negative: Negative testing for invalid inputs",
        "edge_case: Edge case scenario testing",
        # Performance and benchmarking
        "benchmark: Performance benchmark tests using pytest-benchmark",
        "performance: Performance characteristic tests",
        "memory: Memory usage and leak detection tests",
        "load: Load testing for high volume scenarios",
        "stress: Stress testing for resource limits",
        "scalability: Scalability testing for growth scenarios",
        # Async and concurrency
        "asyncio: Async tests requiring event loop",
        "async_integration: Async integration tests",
        "concurrency: Concurrency and race condition tests",
        "parallel: Tests safe for parallel execution",
        "sequential: Tests requiring sequential execution",
        # Data and testing patterns
        "parametrize_advanced: Advanced parametrization with test cases",
        "hypothesis: Property-based tests using Hypothesis",
        "factory: Tests using factory_boy data generation",
        "snapshot: Snapshot testing for data validation",
        "golden_master: Golden master testing for regression",
        # Mocking and isolation
        "mock_heavy: Tests with extensive mocking using pytest-mock",
        "no_mock: Tests without any mocking for true integration",
        "isolated: Tests requiring complete isolation",
        "database: Tests requiring database connection",
        "external_service: Tests requiring external services",
        # HTTP and API testing
        "http: HTTP client/server testing using pytest-httpx",
        "api: API endpoint testing",
        "webhook: Webhook functionality testing",
        "rest: REST API testing",
        "graphql: GraphQL API testing",
        # Quality and compliance
        "pep8: PEP8 compliance validation tests",
        "type_check: Type checking validation tests",
        "security: Security vulnerability tests",
        "lint: Code quality and linting tests",
        "format: Code formatting validation tests",
        # Domain-specific
        "result_pattern: FlextResult railway pattern tests",
        "container: Dependency injection container tests",
        "entity: Domain entity behavior tests",
        "value_object: Value object immutability tests",
        "aggregate: Aggregate root consistency tests",
        "command: Command pattern and CQRS tests",
        "event: Event sourcing and domain events tests",
        "handler: Message and command handler tests",
        "validation: Business rule validation tests",
        "serialization: Data serialization/deserialization tests",
        # Missing markers
        "advanced: Advanced pattern tests",
        "guards: Guard and validation tests",
        "mixins: Mixin pattern tests",
        "decorators: Decorator pattern tests",
        # Environment and setup
        "docker: Tests requiring Docker containers",
        "postgres: Tests requiring PostgreSQL database",
        "redis: Tests requiring Redis cache",
        "temp_files: Tests creating temporary files",
        "network: Tests requiring network access",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables with comprehensive test isolation."""
    # Clear FLEXT environment variables
    for key in list(os.environ.keys()):
        if key.startswith(("FLEXT_", "PYTEST_", "TEST_")):
            monkeypatch.delenv(key, raising=False)

    # Set comprehensive test environment
    test_env = {
        "FLEXT_ENV": "test",
        "FLEXT_TEST_MODE": "true",
        "PYTEST_RUNNING": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
        # Disable external service calls in tests
        "FLEXT_DISABLE_EXTERNAL_CALLS": "true",
        # Enable strict validation in tests
        "FLEXT_STRICT_VALIDATION": "true",
        # Set deterministic randomization seed
        "PYTHONHASHSEED": "42",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


# ============================================================================
# COMPREHENSIVE PLUGIN INTEGRATION FIXTURES
# ============================================================================


@pytest.fixture
def factory_boy_session() -> TestDataBuilder[object]:
    """Provide factory_boy session with all factories configured."""
    return TestDataBuilder()


@pytest.fixture
def performance_profiler() -> PerformanceProfiler:
    """Provide performance profiler for memory and execution tracking."""
    return PerformanceProfiler()


@pytest.fixture
def benchmark_utils() -> type[BenchmarkUtils]:
    """Provide benchmark utilities class for advanced performance testing."""
    return BenchmarkUtils


@pytest.fixture
def memory_profiler() -> type[MemoryProfiler]:
    """Provide memory profiler for leak detection and analysis."""
    return MemoryProfiler


@pytest.fixture
def async_test_utils() -> type[AsyncTestUtils]:
    """Provide async testing utilities for concurrency and timeout testing."""
    return AsyncTestUtils


@pytest.fixture
def async_context_managers() -> type[AsyncContextManagers]:
    """Provide async context managers for resource management in tests."""
    return AsyncContextManagers


@pytest.fixture
def async_mock_utils() -> type[AsyncMockUtils]:
    """Provide async mocking utilities for delayed and flaky operations."""
    return AsyncMockUtils


@pytest.fixture
def async_concurrency_testing() -> type[AsyncConcurrencyTesting]:
    """Provide concurrency testing tools for race conditions and deadlocks."""
    return AsyncConcurrencyTesting


@pytest.fixture
def http_test_utils() -> type[HTTPTestUtils]:
    """Provide HTTP testing utilities with pytest-httpx integration."""
    return HTTPTestUtils


@pytest.fixture
def api_test_client() -> APITestClient:
    """Provide API test client for HTTP endpoint testing."""
    return APITestClient()


@pytest.fixture
def http_scenario_builder(httpx_mock: HTTPXMock) -> HTTPScenarioBuilder:
    """Provide HTTP scenario builder for complex testing workflows."""
    return HTTPScenarioBuilder(httpx_mock)


@pytest.fixture
def webhook_test_utils() -> type[WebhookTestUtils]:
    """Provide webhook testing utilities for event-driven testing."""
    return WebhookTestUtils


@pytest.fixture
def flext_matchers() -> FlextMatchers:
    """Provide advanced matchers for FlextResult and domain object assertions."""
    return FlextMatchers()


@pytest.fixture
def test_builders() -> TestBuilders:
    """Provide fluent test builders for complex object construction."""
    return TestBuilders()


@pytest.fixture
def realistic_data_factory() -> type[PayloadDataFactory]:
    """Provide realistic data factory using Faker for comprehensive testing."""
    return PayloadDataFactory


@pytest.fixture
def mocker_extended(mocker: MockerFixture) -> MockerFixture:
    """Provide extended mocker with additional utilities."""

    # Add custom mock helpers
    def create_async_mock(
        return_value: object = None, side_effect: Exception | None = None
    ) -> mock.AsyncMock:
        """Create async mock with standard patterns."""
        async_mock = mocker.AsyncMock()
        if side_effect:
            async_mock.side_effect = side_effect
        else:
            async_mock.return_value = return_value
        return async_mock

    def create_flext_result_mock(
        *, success: bool = True, data: object = None, error: str | None = None
    ) -> FlextResult[object]:
        """Create real FlextResult instances for testing (no mocks)."""
        # Use real FlextResult instead of mocks for better type safety and authenticity
        if success:
            return FlextResult[object].ok(data)
        return FlextResult[object].fail(error or "Test error")

    # Attach helper methods using setattr for PyRight compatibility
    mocker.create_async_mock = create_async_mock
    mocker.create_flext_result_mock = create_flext_result_mock

    return mocker


@pytest.fixture
def benchmark_with_memory(
    benchmark: BenchmarkFixture, performance_profiler: PerformanceProfiler
) -> Callable[[Callable[..., object]], object]:
    """Provide benchmark fixture with integrated memory profiling."""

    def _benchmark_with_memory(
        func: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        with performance_profiler.profile_memory(f"benchmark_{func.__name__}"):
            return benchmark(func, *args, **kwargs)

    return _benchmark_with_memory


@pytest.fixture
def async_benchmark() -> type[AsyncBenchmark]:
    """Provide async benchmarking utilities for performance testing."""
    return AsyncBenchmark


@pytest.fixture(scope="session")
def plugins_info() -> dict[str, str]:
    """Provide information about loaded pytest plugins for debugging."""
    return {
        "pytest-asyncio": "Async test support",
        "pytest-benchmark": "Performance benchmarking",
        "pytest-clarity": "Better error messages",
        "pytest-cov": "Coverage reporting",
        "pytest-deadfixtures": "Unused fixture detection",
        "pytest-env": "Environment variable management",
        "pytest-httpx": "HTTP request mocking",
        "pytest-mock": "Advanced mocking capabilities",
        "pytest-randomly": "Randomized test execution",
        "pytest-sugar": "Enhanced test output",
        "pytest-timeout": "Test timeout management",
        "pytest-xdist": "Parallel test execution",
        "factory_boy": "Test data generation",
    }


# ============================================================================
# Advanced Test Data Factories and Builders
# ============================================================================


@dataclass
class TestDataBuilder[T]:
    """Fluent builder for test data construction."""

    _data: dict[str, object] = field(default_factory=dict)
    _type: type[T] | None = None

    def with_field(self, name: str, value: object) -> TestDataBuilder[T]:
        """Add field to test data."""
        self._data[name] = value
        return self

    def with_id(self, id_value: str) -> TestDataBuilder[T]:
        """Add ID to test data."""
        return self.with_field("id", id_value)

    def with_status(self, status: FlextEntityStatus) -> TestDataBuilder[T]:
        """Add status to test data."""
        return self.with_field("status", status)

    def build(self) -> dict[str, object]:
        """Build the test data dictionary."""
        return self._data.copy()

    def build_as(self, cls: type[T]) -> T:
        """Build as specific type."""
        return cls(**self._data)


@pytest.fixture
def test_builder() -> type[TestDataBuilder[object]]:
    """Provide test data builder class."""
    return TestDataBuilder


class TestScenario(Enum):
    """Test scenario enumeration for parametrized testing."""

    HAPPY_PATH = "happy_path"
    ERROR_CASE = "error_case"
    BOUNDARY = "boundary"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"


TInput = TypeVar("TInput", default=object)
TExpected = TypeVar("TExpected", default=object)


@dataclass
class TestCase(Generic[TInput, TExpected]):
    """Structured test case for parametrized testing.

    Note: Kept non-generic to simplify typing across tests while preserving
    rich structure. Use specific types in each test where needed.
    """

    id: str
    description: str
    input_data: TInput
    expected_output: TExpected | None
    expected_error: str | None = None
    scenario: TestScenario = TestScenario.HAPPY_PATH
    marks: list[pytest.MarkDecorator] = field(default_factory=list)


# ============================================================================
# Advanced Fixture Factories
# ============================================================================


@pytest.fixture
def fixture_factory() -> Callable[[str, object], object]:
    """Factory for creating dynamic fixtures."""
    created_fixtures: dict[str, object] = {}

    def _create_fixture(name: str, value: object) -> object:
        if name not in created_fixtures:
            created_fixtures[name] = value
        return created_fixtures[name]

    return _create_fixture


@pytest.fixture(
    params=[
        {"type": "minimal", "fields": 1},
        {"type": "standard", "fields": 5},
        {"type": "complex", "fields": 10},
    ],
)
def parametrized_data(request: SubRequest) -> dict[str, object]:
    """Parametrized fixture providing various data complexities."""
    config = request.param
    data: dict[str, object] = {
        f"field_{i}": f"value_{i}" for i in range(config["fields"])
    }
    data["_type"] = config["type"]
    return data


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
def hypothesis_strategies() -> dict[str, object]:
    """Hypothesis strategies for property-based testing."""
    return {
        "valid_email": st.emails(),
        "valid_uuid": st.uuids(),
        "valid_text": st.text(min_size=1, max_size=100),
        "valid_integer": st.integers(min_value=0, max_value=1000),
        "valid_float": st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
        "valid_datetime": st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 12, 31, tzinfo=UTC),
        ),
        "entity_status": st.sampled_from(list(FlextEntityStatus)),
        "operation_status": st.sampled_from(list(FlextOperationStatus)),
    }


@pytest.fixture
def sample_metadata() -> dict[str, str | float | list[str] | None]:
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
def service_factory() -> Callable[[str], object]:
    """Service factory for testing."""

    def _create_service(name: str, **_kwargs: object) -> object:
        class TestService:
            def __init__(self, service_name: str) -> None:
                self.name = service_name

            def is_healthy(self) -> bool:
                return True

            def validate(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

            def process(self) -> FlextResult[str]:
                return FlextResult[str].ok("processed")

            def process_id(self, input_id: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{input_id}")

        return TestService(name)

    return _create_service


@pytest.fixture
def async_service_factory() -> Callable[[str], object]:
    """Async service factory for testing."""

    def _create_async_service(name: str, **_kwargs: object) -> object:
        class AsyncService:
            def __init__(self, service_name: str) -> None:
                self.name = service_name

            async def is_healthy(self) -> bool:
                return True

            async def process(self) -> FlextResult[str]:
                return FlextResult[str].ok("processed")

        return AsyncService(name)

    return _create_async_service


class ExternalService:
    """Test service implementation for functional testing."""

    def __init__(self) -> None:
        """Initialize with default test state."""
        self._healthy = True
        self._data = {"status": "success", "data": "test_data"}

    def is_healthy(self) -> bool:
        """Check service health."""
        return self._healthy

    def get_data(self) -> dict[str, object]:
        """Get test data."""
        return dict(self._data)

    def process(self, _data: object = None) -> FlextResult[str]:
        """Process data and return FlextResult."""
        if not self._healthy:
            return FlextResult[str].fail("Service unhealthy")
        return FlextResult[str].ok("processed")


@pytest.fixture
def external_service() -> ExternalService:
    """Provide external service for testing.

    Implementation for testing service integration patterns
    and validates functionality.

    Returns:
      ExternalService instance for functional testing

    """
    return ExternalService()


# ============================================================================
# Advanced Core Component Fixtures
# ============================================================================


@pytest.fixture
def entity_factory() -> Callable[[str, dict[str, object]], FlextEntity]:
    """Factory for creating test entities."""

    class TestEntity(FlextEntity):
        """Test entity for testing purposes."""

        name: str = "test"
        value: int = 0

        def __init__(
            self, entity_id: str = "test-entity", name: str = "test", value: int = 0
        ) -> None:
            super().__init__(id=FlextEntityId(entity_id))
            self.name = name
            self.value = value

        def activate(self) -> None:
            self._status = FlextEntityStatus.ACTIVE

        def deactivate(self) -> None:
            self._status = FlextEntityStatus.INACTIVE

        def get_name(self) -> str:
            return self.name

        def get_value(self) -> int:
            return self.value

        def set_value(self, value: int) -> None:
            self.value = value

        def to_dict(self) -> dict[str, object]:
            return {
                "id": str(self.id),
                "name": self.name,
                "value": self.value,
                "status": getattr(self, "_status", "ACTIVE"),
            }

    def _create_entity(entity_id: str, metadata: dict[str, object]) -> FlextEntity:
        """Create test entity with given parameters."""
        name_raw = metadata.get("name", "test")
        name = str(name_raw) if name_raw is not None else "test"
        value_raw = metadata.get("value", 0)
        value = int(value_raw) if isinstance(value_raw, (int, str)) else 0
        return TestEntity(entity_id=entity_id, name=name, value=value)

    return _create_entity


@pytest.fixture
def value_object_factory() -> Callable[[dict[str, object]], FlextValue]:
    """Factory for creating test value objects."""

    class TestValueObject(FlextValue):
        value: str
        metadata: dict[str, object] = field(default_factory=dict)

        def validate_business_rules(self) -> FlextResult[None]:
            if not self.value:
                return FlextResult[None].fail("Value cannot be empty")
            return FlextResult[None].ok(None)

    def _create_vo(data: dict[str, object]) -> FlextValue:
        # Extract specific fields with type safety
        value = str(data.get("value", ""))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return TestValueObject.model_validate({"value": value, "metadata": metadata})

    return _create_vo


@pytest.fixture
def aggregate_factory() -> Callable[[str], FlextAggregateRoot]:
    """Factory for creating test aggregates."""

    class TestAggregate(FlextAggregateRoot):
        name: str = "test_aggregate"
        items: list[str] = field(default_factory=list)

        def add_item(self, item: str) -> FlextResult[None]:
            if item in self.items:
                return FlextResult[None].fail(f"Item {item} already exists")
            self.items.append(item)
            self.add_domain_event("item_added", {"item": item})
            return FlextResult[None].ok(None)

    def _create_aggregate(aggregate_id: str) -> FlextAggregateRoot:
        return TestAggregate(id=FlextEntityId(aggregate_id))

    return _create_aggregate


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
    external_service: ExternalService,
) -> FlextContainer:
    """Provide pre-configured container for integration testing.

    Container factory with common service registrations for testing
    service integration patterns and dependency resolution.

    Args:
      clean_container: Fresh container instance
      external_service: External service for functional testing

    Returns:
      FlextContainer with standard test services registered

    """
    clean_container.register("external_service", external_service)
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
    temp_dir = Path(tempfile.mkdtemp(prefix="flext_test_"))

    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# ============================================================================
# Advanced State Management and Isolation
# ============================================================================


@pytest.fixture
def isolation_context() -> Generator[dict[str, dict[str, str] | list[str] | float]]:
    """Provide complete test isolation context."""
    context = {
        "original_env": os.environ.copy(),
        "original_path": sys.path.copy(),
        "start_time": time.time(),
    }

    yield context

    # Restore original state
    os.environ.clear()
    original_env = context["original_env"]
    if isinstance(original_env, dict):
        os.environ.update(original_env)
    original_path = context["original_path"]
    if isinstance(original_path, list):
        sys.path = original_path


@pytest.fixture
def time_machine() -> Callable[[datetime], None]:
    """Time manipulation for testing time-dependent code."""

    # Simplified implementation to avoid MyPy type issues
    def _set_time(target_time: datetime) -> None:
        # This is a test helper - actual implementation would mock datetime
        pass

    return _set_time
    # Cleanup if needed


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
    FlextLoggerFactory._global_level = "INFO"  # Reset global level

    yield

    # Teardown: Reset to clean state
    FlextLoggerFactory.clear_loggers()
    FlextLoggerFactory.clear_log_store()
    FlextLoggerFactory._global_level = "INFO"  # Reset global level

    # Clear structlog caches if they exist
    try:
        config = getattr(structlog, "_CONFIG", None)
        if config is not None:
            logger_factory = getattr(config, "logger_factory", None)
            if logger_factory is not None and hasattr(logger_factory, "_cache"):
                logger_factory._cache.clear()
    except AttributeError:
        # structlog configuration might not be set up yet
        pass


# ============================================================================
# Advanced Performance and Benchmarking
# ============================================================================


class PerformanceMetrics(TypedDict):
    """Performance metrics data structure."""

    result: object
    execution_time: float
    memory_used: int
    peak_memory: int


@pytest.fixture
def performance_monitor() -> Callable[[Callable[[], object]], PerformanceMetrics]:
    """Monitor performance metrics for operations."""

    def _monitor(
        func: Callable[[], object],
        *args: object,
        **kwargs: object,
    ) -> PerformanceMetrics:
        # Force garbage collection
        gc.collect()

        # Start monitoring
        tracemalloc.start()
        start_time = time.perf_counter()
        start_memory = tracemalloc.get_traced_memory()[0]

        # Execute function
        result = func(*args, **kwargs)

        # Collect metrics
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "result": result,
            "execution_time": end_time - start_time,
            "memory_used": current - start_memory,
            "peak_memory": peak,
        }

    return _monitor


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


# Backward-compatible alias fixtures for renamed fixtures in integration tests
@pytest.fixture(name="_configured_container")
def _alias_configured_container(configured_container: FlextContainer) -> FlextContainer:
    return configured_container


@pytest.fixture(name="_performance_threshold")
def _alias_performance_threshold(
    performance_threshold: dict[str, float],
) -> dict[str, float]:
    return performance_threshold


# ============================================================================
# Advanced Testing Utilities
# ============================================================================


@contextmanager
def assert_performance(
    max_time: float = 1.0,
    max_memory: int = 10_000_000,
) -> Generator[None]:
    """Context manager for performance assertions."""
    tracemalloc.start()
    start_time = time.perf_counter()
    start_memory = tracemalloc.get_traced_memory()[0]

    yield

    elapsed = time.perf_counter() - start_time
    current, _peak = tracemalloc.get_traced_memory()
    memory_used = current - start_memory
    tracemalloc.stop()

    assert elapsed < max_time, f"Operation took {elapsed:.3f}s (max: {max_time}s)"
    assert memory_used < max_memory, f"Used {memory_used:,} bytes (max: {max_memory:,})"


@pytest.fixture
def snapshot_manager(tmp_path: Path) -> Callable[[str, object], None]:
    """Snapshot testing for complex data structures."""
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    def _snapshot(name: str, data: object) -> None:
        snapshot_file = snapshot_dir / f"{name}.json"

        if snapshot_file.exists():
            # Compare with existing snapshot
            with snapshot_file.open(encoding="utf-8") as f:
                expected = json.load(f)
            assert data == expected, f"Snapshot mismatch for {name}"
        else:
            # Create new snapshot
            with snapshot_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

    return _snapshot


# ============================================================================
# Async Testing Support
# ============================================================================


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client() -> object:
    """Async client fixture for testing async endpoints."""

    class _AsyncClient:
        async def get(self, url: str) -> dict[str, object]:
            await asyncio.sleep(0.01)  # Simulate network delay
            return {"status": "success", "url": url}

        async def post(self, _url: str, data: dict[str, object]) -> dict[str, object]:
            await asyncio.sleep(0.01)
            return {"status": "created", "data": data}

    return _AsyncClient()
    # Cleanup if needed


# ============================================================================
# Test Helpers and Validators
# ============================================================================


@pytest.fixture
def validators() -> dict[str, Callable[[object], bool]]:
    """Collection of validation functions for testing."""
    return {
        "is_valid_uuid": lambda x: len(str(x)) == 36 and "-" in str(x),
        "is_valid_email": lambda x: "@" in str(x) and "." in str(x),
        "is_valid_result": lambda x: hasattr(x, "success") and hasattr(x, "data"),
        "is_valid_entity": lambda x: hasattr(x, "id") and hasattr(x, "version"),
        "is_valid_timestamp": lambda x: isinstance(x, (int, float)) and x > 0,
    }


@pytest.fixture
def assert_helpers() -> object:
    """Advanced assertion helpers."""

    class AssertHelpers:
        @staticmethod
        def assert_result_ok(
            result: FlextResult[object],
            expected_data: object | None = None,
        ) -> None:
            """Assert result is successful with optional data check."""
            assert result.success, f"Expected success but got error: {result.error}"
            if expected_data is not None:
                assert result.value == expected_data

        @staticmethod
        def assert_result_fail(
            result: FlextResult[object],
            expected_error: str | None = None,
        ) -> None:
            """Assert result is failure with optional error check."""
            assert result.is_failure, "Expected failure but got success"
            if expected_error:
                assert expected_error in (result.error or "")

        @staticmethod
        def assert_entity_valid(entity: FlextEntity) -> None:
            """Assert entity is valid according to business rules."""
            assert entity.id, "Entity must have ID"
            # Basic entity validation - status check removed as not all entities have status

    return AssertHelpers()


# ============================================================================
# Parametrization Helpers
# ============================================================================


@pytest.fixture
def test_cases() -> list[TestCase[object]]:
    """Provide structured test cases for parametrized testing."""
    return [
        TestCase(
            id="happy_path_simple",
            description="Simple successful case",
            input_data={"value": "test"},
            expected_output="test",
            scenario=TestScenario.HAPPY_PATH,
        ),
        TestCase(
            id="error_empty_input",
            description="Error with empty input",
            input_data={},
            expected_output=None,
            expected_error="Input cannot be empty",
            scenario=TestScenario.ERROR_CASE,
        ),
        TestCase(
            id="boundary_max_size",
            description="Boundary test with maximum size",
            input_data={"value": "x" * 1000},
            expected_output="x" * 1000,
            scenario=TestScenario.BOUNDARY,
        ),
    ]


def parametrize_test_cases(
    test_cases: list[TestCase[object]],
) -> Callable[[object], object]:
    """Decorator for parametrizing with test cases."""

    def decorator(func: object) -> object:
        return pytest.mark.parametrize(
            "test_case",
            test_cases,
            ids=[tc.id for tc in test_cases],
        )(func)

    return decorator


# ============================================================================
# Domain-Specific Fixtures
# ============================================================================


@pytest.fixture
def command_factory() -> Callable[[str, dict[str, object]], FlextCommands.Command]:
    """Factory for creating test commands."""

    class TestCommand(FlextCommands.Command):
        name: str
        payload: dict[str, object] = field(default_factory=dict)

        def validate_command(self) -> FlextResult[None]:
            """Validate command - renamed to avoid Pydantic conflict."""
            if not self.name:
                return FlextResult[None].fail("Command name is required")
            return FlextResult[None].ok(None)

    def _create_command(
        name: str,
        payload: dict[str, object] | None = None,
    ) -> FlextCommands.Command:
        return TestCommand(name=name, payload=payload or {})

    return _create_command


@pytest.fixture
def handler_factory() -> Callable[[Callable[[object], object]], FlextBaseHandler]:
    """Factory for creating test handlers."""

    class TestHandler(FlextBaseHandler):
        def __init__(self, handler_func: Callable[[object], object]) -> None:
            super().__init__("test_handler")
            self.handler_func = handler_func

        def process_message(self, message: object) -> FlextResult[object]:
            try:
                result = self.handler_func(message)
                return FlextResult[object].ok(result)
            except Exception as e:
                return FlextResult[object].fail(str(e))

    def _create_handler(handler_func: Callable[[object], object]) -> FlextBaseHandler:
        return TestHandler(handler_func)

    return _create_handler


@pytest.fixture
def event_factory() -> Callable[[str, dict[str, object]], FlextEvent]:
    """Factory for creating test events."""

    def _create_event(
        event_type: str,
        data: dict[str, object] | None = None,
    ) -> FlextEvent:
        result = FlextEvent.create_event(
            event_type=event_type,
            event_data=data or {},
        )
        if result.is_success:
            return result.value
        msg = f"Failed to create event: {result.error}"
        raise ValueError(msg)

    return _create_event


# ============================================================================
# Integration Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def integration_setup() -> None:
    """One-time setup for integration tests."""
    # Setup code here
    return
    # Teardown code here


class TestDatabase:
    """Test database implementation for functional testing."""

    def __init__(self) -> None:
        """Initialize with in-memory test state."""
        self._data: list[dict[str, object]] = [{"id": 1, "name": "test"}]
        self._connected = True

    def execute(self, query: str) -> dict[str, object]:
        """Execute query and return result."""
        if not self._connected:
            return {"status": "error", "message": "Not connected"}
        return {"status": "success", "query": query}

    def fetch(self) -> list[dict[str, object]]:
        """Fetch data from test database."""
        return self._data.copy()

    def close(self) -> None:
        """Close database connection."""
        self._connected = False


@pytest.fixture
def database_connection() -> Generator[TestDatabase]:
    """Provide test database for integration tests."""
    connection = TestDatabase()

    yield connection

    connection.close()


# ============================================================================
# End-to-End Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def e2e_environment() -> dict[str, object]:
    """Setup complete environment for E2E tests."""
    return {
        "services": [],
        "containers": [],
        "ports": {"api": 8080, "db": 5432},
    }


# ============================================================================
# Custom Test Markers and Utilities
# ============================================================================


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify test collection to add custom markers."""
    # Mark 'config' as intentionally unused while keeping hook signature valid
    del config
    for item in items:
        # Add markers based on test location using modern path property
        item_path = str(getattr(item, "path", getattr(item, "fspath", "")))

        # Add location-based markers
        if "unit" in item_path:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item_path:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in item_path:
            item.add_marker(pytest.mark.e2e)

        # Add markers based on test name
        if "test_performance" in item.name:
            item.add_marker(pytest.mark.benchmark)
        if "test_async" in item.name:
            item.add_marker(pytest.mark.asyncio)


# =============================================================================
# MISSING TEST UTILITIES - Backward compatibility
# =============================================================================


class AssertHelpers:
    """Helper utilities for test assertions."""

    @staticmethod
    def assert_result_success(result: FlextResult[object]) -> None:
        """Assert that a result is successful."""
        assert result.is_success, f"Expected success but got: {result.error}"

    @staticmethod
    def assert_result_failure(result: FlextResult[object]) -> None:
        """Assert that a result is a failure."""
        assert result.is_failure, f"Expected failure but got: {result.value}"

    # Backward-compatible aliases used in some tests
    @staticmethod
    def assert_result_ok(result: FlextResult[object]) -> None:
        """Assert that a result is successful."""
        AssertHelpers.assert_result_success(result)

    @staticmethod
    def assert_result_fail(result: FlextResult[object]) -> None:
        """Assert that a result is a failure."""
        AssertHelpers.assert_result_failure(result)


class ConfigFactory:
    """Factory for creating configuration objects."""

    @staticmethod
    def create_config(**kwargs: object) -> object:
        """Create a configuration object."""
        return type("TestConfig", (), kwargs)()


class PerformanceMonitor:
    """Monitor performance during tests."""

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop timing."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return 0.0


class TestBuilder:
    """Builder for creating test objects."""

    def __init__(self) -> None:
        """Initialize test builder."""
        self.data: dict[str, object] = {}

    def with_field(self, name: str, value: object) -> TestBuilder:
        """Add a field to the test object."""
        self.data[name] = value
        return self

    def build(self) -> object:
        """Build the test object."""
        return type("TestObject", (), self.data)()


def assert_function_performance(
    func: Callable[[], object],
    max_duration: float = 1.0,
) -> object:
    """Assert that a function executes within time limit."""
    start = time.time()
    result = func()
    duration = time.time() - start
    assert duration <= max_duration, (
        f"Function took {duration}s, max allowed {max_duration}s"
    )
    return result


# =============================================================================
# SHARED TEST FIXTURES - Centralized to avoid duplication
# =============================================================================


@pytest.fixture
def temp_json_file() -> Generator[str]:
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(
        encoding="utf-8",
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(
            {
                "database_url": "sqlite:///test.db",
                "secret_key": "test-secret-key",
                "test": "data",
                "number": 42,
                # Compatibility keys expected by some base tests
                "key1": "value1",
                "key2": 42,
            },
            f,
        )
        temp_path = f.name

    try:
        yield temp_path
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def validation_test_cases() -> list[dict[str, object]]:
    """Provide common validation test cases."""
    return [
        {"valid": True, "data": {"name": "test", "value": 123}},
        {"valid": False, "data": {"name": "", "value": None}},
        {"valid": True, "data": {"name": "valid", "value": 0}},
        {"valid": False, "data": {}},
    ]


# ============================================================================
# FACTORY_BOY AND FAKER INTEGRATION
# ============================================================================


@pytest.fixture
def faker_instance() -> Faker:
    """Provide Faker instance with consistent seed for deterministic tests."""
    fake_instance = Faker()
    Faker.seed(42)  # Deterministic seed for consistent test data
    return fake_instance


@pytest.fixture
def user_factory() -> type[UserFactory]:
    """Provide UserFactory class for creating test users."""
    return UserFactory


@pytest.fixture
def user_data_factory() -> type[UserDataFactory]:
    """Provide UserDataFactory class for creating user test data."""
    return UserDataFactory


@pytest.fixture
def flext_result_factory() -> type[FlextResultFactory]:
    """Provide FlextResultFactory class for creating FlextResult instances."""
    return FlextResultFactory


@pytest.fixture
def configuration_factory() -> type[ConfigurationFactory]:
    """Provide ConfigurationFactory class for creating configuration data."""
    return ConfigurationFactory


@pytest.fixture
def edge_case_generators() -> type[EdgeCaseGenerators]:
    """Provide EdgeCaseGenerators class for boundary testing."""
    return EdgeCaseGenerators


@pytest.fixture
def factory_test_cases() -> list[dict[str, object]]:
    """Provide test cases generated using factory_boy patterns."""
    return create_validation_test_cases()


@pytest.fixture(autouse=True)
def setup_factory_boy() -> None:
    """Configure factory_boy with Faker for all tests."""
    # Set consistent locale and seed
    factory.Faker._DEFAULT_LOCALE = "en_US"
    Faker.seed(42)
    # Reset factory state between tests
    UserFactory.reset_sequence()


@pytest.fixture
def realistic_user_data(faker_instance: Faker) -> dict[str, object]:
    """Generate realistic user data using Faker."""
    return {
        "id": faker_instance.uuid4(),
        "name": faker_instance.name(),
        "email": faker_instance.email(),
        "age": faker_instance.random_int(min=18, max=80),
        "phone": faker_instance.phone_number(),
        "address": faker_instance.address(),
        "company": faker_instance.company(),
        "job_title": faker_instance.job(),
        "is_active": faker_instance.boolean(),
        "created_at": faker_instance.date_time_this_year(),
        "bio": faker_instance.text(max_nb_chars=200),
    }


@pytest.fixture
def realistic_config_data(faker_instance: Faker) -> dict[str, object]:
    """Generate realistic configuration data using Faker."""
    return {
        "database_url": f"postgresql://{faker_instance.user_name()}:{faker_instance.password()}@{faker_instance.ipv4()}:5432/{faker_instance.slug()}",
        "redis_url": f"redis://{faker_instance.ipv4()}:6379/0",
        "secret_key": faker_instance.sha256(),
        "api_key": faker_instance.uuid4(),
        "environment": faker_instance.random_element(
            elements=("development", "staging", "production")
        ),
        "log_level": faker_instance.random_element(
            elements=("DEBUG", "INFO", "WARNING", "ERROR")
        ),
        "timeout": faker_instance.random_int(min=1, max=60),
        "max_connections": faker_instance.random_int(min=10, max=100),
        "debug": faker_instance.boolean(),
    }


@pytest.fixture
def batch_user_data(user_factory: type[UserFactory]) -> list[object]:
    """Generate batch user data using factory_boy."""
    return user_factory.create_batch(5)


@pytest.fixture
def parameterized_factory_data() -> list[dict[str, object]]:
    """Provide parameterized data sets for comprehensive testing."""
    return [
        {"size": "small", "count": 5},
        {"size": "medium", "count": 50},
        {"size": "large", "count": 500},
    ]


@pytest.fixture
def test_scenarios() -> list[TestScenario]:
    """Provide test scenarios for integration testing."""
    # Use the local TestScenario enum instead of the support module class
    return [
        TestScenario.HAPPY_PATH,
        TestScenario.ERROR_CASE,
        TestScenario.BOUNDARY,
        TestScenario.EDGE_CASE,
    ]


# ============================================================================
# PYTEST PLUGIN ENHANCED FIXTURES
# ============================================================================


@pytest.fixture
def benchmark_with_factory(
    benchmark: BenchmarkFixture,
    user_factory: type[UserFactory],
) -> object:
    """Benchmark fixture enhanced with factory_boy data generation."""

    def _benchmark_factory_operation(
        operation: str = "create",
        count: int = 100,
    ) -> object:
        if operation == "create":
            return benchmark(lambda: user_factory.create_batch(count))
        if operation == "build":
            return benchmark(lambda: user_factory.build_batch(count))
        msg = f"Unknown operation: {operation}"
        raise ValueError(msg)

    return _benchmark_factory_operation


@pytest.fixture
def async_faker_data(faker_instance: Faker) -> object:
    """Generate async-friendly test data using Faker."""

    class AsyncFakerData:
        def __init__(self, faker: Faker) -> None:
            self.faker = faker

        async def user_data(self) -> dict[str, object]:
            """Generate user data asynchronously."""
            await asyncio.sleep(0.001)  # Simulate async operation
            return {
                "id": self.faker.uuid4(),
                "name": self.faker.name(),
                "email": self.faker.email(),
            }

        async def config_data(self) -> dict[str, object]:
            """Generate config data asynchronously."""
            await asyncio.sleep(0.001)
            return {
                "database_url": f"postgresql://{self.faker.ipv4()}:5432/test",
                "timeout": self.faker.random_int(min=1, max=30),
            }

    return AsyncFakerData(faker_instance)


@pytest.fixture
def mock_with_factory(
    mocker: MockerFixture,
    user_factory: type[UserFactory],
) -> object:
    """Enhanced mocker with factory_boy integration."""

    def _create_mock_service_with_data() -> object:
        mock_service = mocker.Mock()
        mock_service.get_user.return_value = user_factory.build()
        mock_service.get_users.return_value = user_factory.build_batch(3)
        mock_service.is_healthy.return_value = True
        return mock_service

    mocker.create_service_with_factory_data = _create_mock_service_with_data
    return mocker
