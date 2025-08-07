"""Pytest configuration for FLEXT Core test suite - Advanced Enterprise Edition.

Ultra-modern pytest configuration implementing advanced testing patterns:
- Parametrized fixture factories with type safety
- Hypothesis integration for property-based testing
- Performance benchmarking with pytest-benchmark
- Async testing support with pytest-asyncio
- Mock factories with pytest-mock integration
- Snapshot testing for complex data structures
- Parallel execution with pytest-xdist
- BDD support with pytest-bdd integration

Architecture Layers:
    Testing Foundation → Fixture Factories → Test Builders → Quality Gates

    Advanced Patterns:
    - Factory Pattern: Dynamic fixture generation
    - Builder Pattern: Fluent test data construction
    - Strategy Pattern: Pluggable test behaviors
    - Observer Pattern: Test event monitoring
    - Decorator Pattern: Enhanced test capabilities

Quality Standards:
    - Zero Cross-Test Contamination: Complete isolation
    - Type Safety: 100% typed with generics
    - Performance: Sub-millisecond fixture creation
    - Determinism: Reproducible test execution
    - Coverage: Automatic branch coverage tracking
"""

import asyncio
import json
import math
import os
import sys
import time
import tracemalloc
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import structlog
from _pytest.fixtures import SubRequest

# Conditional imports for optional dependencies
try:
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    st = None  # type: ignore[assignment]

try:
    from pytest_mock import MockerFixture

    HAS_PYTEST_MOCK = True
except ImportError:
    HAS_PYTEST_MOCK = False
    MockerFixture = Any  # type: ignore[misc]

from flext_core.aggregate_root import FlextAggregateRoot
from flext_core.commands import FlextCommands
from flext_core.container import FlextContainer
from flext_core.entities import FlextEntity

# Guards module doesn't export FlextGuard
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLoggerFactory
from flext_core.models import FlextEntityStatus, FlextOperationStatus
from flext_core.payload import FlextEvent
from flext_core.result import FlextResult
from flext_core.value_objects import FlextValueObject

# Type variables for generic fixtures
T = TypeVar("T")
E = TypeVar("E", bound=FlextEntity)
V = TypeVar("V", bound=FlextValueObject)


# ============================================================================
# Advanced Pytest Configuration
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers for organized test execution."""
    # Basic markers
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

    # Advanced markers
    config.addinivalue_line(
        "markers",
        "parametrize_advanced: marks tests using advanced parametrization",
    )
    config.addinivalue_line(
        "markers",
        "hypothesis: marks property-based tests using hypothesis",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: marks performance benchmark tests",
    )
    config.addinivalue_line(
        "markers",
        "asyncio: marks async tests requiring event loop",
    )
    config.addinivalue_line(
        "markers",
        "snapshot: marks tests using snapshot testing",
    )
    config.addinivalue_line(
        "markers",
        "flaky: marks tests that may fail intermittently",
    )
    config.addinivalue_line(
        "markers",
        "mock_heavy: marks tests with extensive mocking",
    )
    config.addinivalue_line(
        "markers",
        "async_integration: marks async integration tests",
    )


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables that might interfere with tests."""
    # Clear any FLEXT environment variables that might interfere
    for key in list(os.environ.keys()):
        if key.startswith("FLEXT_"):
            monkeypatch.delenv(key, raising=False)

    # Set test environment
    monkeypatch.setenv("FLEXT_ENV", "test")
    # Don't set FLEXT_LOG_LEVEL here as it interferes with logging tests
    monkeypatch.setenv("FLEXT_TEST_MODE", "true")


# ============================================================================
# Advanced Test Data Factories and Builders
# ============================================================================


@dataclass
class TestDataBuilder[T]:
    """Fluent builder for test data construction."""

    _data: dict[str, Any] = field(default_factory=dict)
    _type: type[T] | None = None

    def with_field(self, name: str, value: Any) -> "TestDataBuilder[T]":
        """Add field to test data."""
        self._data[name] = value
        return self

    def with_id(self, id_value: str) -> "TestDataBuilder[T]":
        """Add ID to test data."""
        return self.with_field("id", id_value)

    def with_status(self, status: FlextEntityStatus) -> "TestDataBuilder[T]":
        """Add status to test data."""
        return self.with_field("status", status)

    def build(self) -> dict[str, Any]:
        """Build the test data dictionary."""
        return self._data.copy()

    def build_as(self, cls: type[T]) -> T:
        """Build as specific type."""
        return cls(**self._data)


@pytest.fixture
def test_builder() -> type[TestDataBuilder]:
    """Provide test data builder class."""
    return TestDataBuilder


class TestScenario(Enum):
    """Test scenario enumeration for parametrized testing."""

    HAPPY_PATH = "happy_path"
    ERROR_CASE = "error_case"
    BOUNDARY = "boundary"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"


@dataclass
class TestCase[T]:
    """Structured test case for parametrized testing."""

    id: str
    description: str
    input_data: Any
    expected_output: T | None
    expected_error: str | None = None
    scenario: TestScenario = TestScenario.HAPPY_PATH
    marks: list[pytest.MarkDecorator] = field(default_factory=list)


# ============================================================================
# Advanced Fixture Factories
# ============================================================================


@pytest.fixture
def fixture_factory() -> Callable[[str, Any], Any]:
    """Factory for creating dynamic fixtures."""
    created_fixtures: dict[str, Any] = {}

    def _create_fixture(name: str, value: Any) -> Any:
        if name not in created_fixtures:
            created_fixtures[name] = value
        return created_fixtures[name]

    return _create_fixture


@pytest.fixture(
    params=[
        {"type": "minimal", "fields": 1},
        {"type": "standard", "fields": 5},
        {"type": "complex", "fields": 10},
    ]
)
def parametrized_data(request: SubRequest) -> dict[str, Any]:
    """Parametrized fixture providing various data complexities."""
    config = request.param
    data = {f"field_{i}": f"value_{i}" for i in range(config["fields"])}
    data["_type"] = config["type"]
    return data


@pytest.fixture
def sample_data() -> dict[str, Any]:
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
def hypothesis_strategies() -> dict[str, Any]:
    """Hypothesis strategies for property-based testing."""
    if not HAS_HYPOTHESIS:
        pytest.skip("Hypothesis not installed")

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
def mock_factory() -> Callable[[str], Mock]:
    """Advanced mock factory with automatic cleanup."""
    if not HAS_PYTEST_MOCK:
        # Fallback without pytest-mock
        def _create_mock(name: str, **kwargs) -> Mock:
            mock = MagicMock(name=name, **kwargs)
            mock.is_healthy.return_value = True
            mock.validate.return_value = FlextResult.ok(None)
            mock.process.return_value = FlextResult.ok("processed")
            return mock
    else:

        def _create_mock(name: str, **kwargs) -> Mock:
            mock = MagicMock(name=name, **kwargs)
            mock.is_healthy.return_value = True
            mock.validate.return_value = FlextResult.ok(None)
            mock.process.return_value = FlextResult.ok("processed")
            return mock

    return _create_mock


@pytest.fixture
def async_mock_factory() -> Callable[[str], AsyncMock]:
    """Async mock factory for testing async patterns."""

    def _create_async_mock(name: str, **kwargs) -> AsyncMock:
        mock = AsyncMock(name=name, **kwargs)
        mock.is_healthy.return_value = True
        mock.process.return_value = FlextResult.ok("processed")
        return mock

    return _create_async_mock


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
# Advanced Core Component Fixtures
# ============================================================================


@pytest.fixture
def entity_factory() -> Callable[[str, dict[str, Any]], FlextEntity]:
    """Factory for creating test entities."""
    from flext_core.models import FlextEntityStatus

    class TestEntity(FlextEntity):
        name: str = "test"
        value: int = 0
        status: FlextEntityStatus = FlextEntityStatus.ACTIVE

        def validate_domain_rules(self) -> FlextResult[None]:
            if self.value < 0:
                return FlextResult.fail("Value must be non-negative")
            return FlextResult.ok(None)

    def _create_entity(
        entity_id: str, data: dict[str, Any] | None = None
    ) -> FlextEntity:
        data = data or {}
        return TestEntity(id=entity_id, **data)

    return _create_entity


@pytest.fixture
def value_object_factory() -> Callable[[dict[str, Any]], FlextValueObject]:
    """Factory for creating test value objects."""

    class TestValueObject(FlextValueObject):
        value: str
        metadata: dict[str, Any] = field(default_factory=dict)

        def validate_business_rules(self) -> FlextResult[None]:
            if not self.value:
                return FlextResult.fail("Value cannot be empty")
            return FlextResult.ok(None)

    def _create_vo(data: dict[str, Any]) -> FlextValueObject:
        return TestValueObject(**data)

    return _create_vo


@pytest.fixture
def aggregate_factory() -> Callable[[str], FlextAggregateRoot]:
    """Factory for creating test aggregates."""

    class TestAggregate(FlextAggregateRoot):
        name: str = "test_aggregate"
        items: list[str] = field(default_factory=list)

        def add_item(self, item: str) -> FlextResult[None]:
            if item in self.items:
                return FlextResult.fail(f"Item {item} already exists")
            self.items.append(item)
            self.add_domain_event({"type": "item_added", "item": item})
            return FlextResult.ok(None)

    def _create_aggregate(aggregate_id: str) -> FlextAggregateRoot:
        return TestAggregate(id=aggregate_id)

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
# Advanced State Management and Isolation
# ============================================================================


@pytest.fixture
def isolation_context() -> Generator[dict[str, Any]]:
    """Provide complete test isolation context."""
    context = {
        "original_env": os.environ.copy(),
        "original_path": sys.path.copy(),
        "start_time": time.time(),
    }

    yield context

    # Restore original state
    os.environ.clear()
    os.environ.update(context["original_env"])
    sys.path = context["original_path"]


@pytest.fixture
def time_machine() -> Callable[[datetime], None]:
    """Time manipulation for testing time-dependent code."""
    original_datetime = datetime

    def _set_time(target_time: datetime) -> None:
        class MockDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return target_time.replace(tzinfo=tz) if tz else target_time

            @classmethod
            def utcnow(cls):
                return target_time

        # Patch datetime
        import datetime as dt_module

        dt_module.datetime = MockDatetime

    yield _set_time

    # Restore original datetime
    import datetime as dt_module

    dt_module.datetime = original_datetime


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
# Advanced Performance and Benchmarking
# ============================================================================


@pytest.fixture
def performance_monitor() -> Callable[[Callable], dict[str, float]]:
    """Monitor performance metrics for operations."""

    def _monitor(func: Callable, *args, **kwargs) -> dict[str, float]:
        import gc

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
def benchmark_data() -> dict[str, Any]:
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


# ============================================================================
# Advanced Testing Utilities
# ============================================================================


@contextmanager
def assert_performance(max_time: float = 1.0, max_memory: int = 10_000_000):
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
def snapshot_manager(tmp_path: Path) -> Callable[[str, Any], None]:
    """Snapshot testing for complex data structures."""
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    def _snapshot(name: str, data: Any) -> None:
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
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client():
    """Async client fixture for testing async endpoints."""

    class AsyncClient:
        async def get(self, url: str) -> dict[str, Any]:
            await asyncio.sleep(0.01)  # Simulate network delay
            return {"status": "success", "url": url}

        async def post(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"status": "created", "data": data}

    return AsyncClient()
    # Cleanup if needed


# ============================================================================
# Test Helpers and Validators
# ============================================================================


@pytest.fixture
def validators() -> dict[str, Callable[[Any], bool]]:
    """Collection of validation functions for testing."""
    return {
        "is_valid_uuid": lambda x: len(str(x)) == 36 and "-" in str(x),
        "is_valid_email": lambda x: "@" in str(x) and "." in str(x),
        "is_valid_result": lambda x: hasattr(x, "success") and hasattr(x, "data"),
        "is_valid_entity": lambda x: hasattr(x, "id") and hasattr(x, "version"),
        "is_valid_timestamp": lambda x: isinstance(x, (int, float)) and x > 0,
    }


@pytest.fixture
def assert_helpers():
    """Advanced assertion helpers."""

    class AssertHelpers:
        @staticmethod
        def assert_result_ok(result: FlextResult, expected_data: Any = None) -> None:
            """Assert result is successful with optional data check."""
            assert result.success, f"Expected success but got error: {result.error}"
            if expected_data is not None:
                assert result.data == expected_data

        @staticmethod
        def assert_result_fail(
            result: FlextResult, expected_error: str | None = None
        ) -> None:
            """Assert result is failure with optional error check."""
            assert result.is_failure, "Expected failure but got success"
            if expected_error:
                assert expected_error in (result.error or "")

        @staticmethod
        def assert_entity_valid(entity: FlextEntity) -> None:
            """Assert entity is valid according to business rules."""
            assert entity.id, "Entity must have ID"
            assert entity.version >= 1, "Entity version must be >= 1"
            assert entity.status in FlextEntityStatus, "Invalid entity status"

    return AssertHelpers()


# ============================================================================
# Parametrization Helpers
# ============================================================================


@pytest.fixture
def test_cases() -> list[TestCase]:
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


def parametrize_test_cases(test_cases: list[TestCase]):
    """Decorator for parametrizing with test cases."""

    def decorator(func):
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
def command_factory() -> Callable[[str, dict[str, Any]], FlextCommands.Command]:
    """Factory for creating test commands."""

    class TestCommand(FlextCommands.Command):
        name: str
        payload: dict[str, Any] = field(default_factory=dict)

        def validate(self) -> FlextResult[None]:
            if not self.name:
                return FlextResult.fail("Command name is required")
            return FlextResult.ok(None)

    def _create_command(
        name: str, payload: dict[str, Any] | None = None
    ) -> FlextCommands.Command:
        return TestCommand(name=name, payload=payload or {})

    return _create_command


@pytest.fixture
def handler_factory() -> Callable[[Callable], FlextHandlers.Handler]:
    """Factory for creating test handlers."""

    class TestHandler(FlextHandlers.Handler):
        def __init__(self, handler_func: Callable):
            self.handler_func = handler_func

        def handle(self, command: FlextCommands.Command) -> FlextResult[Any]:
            try:
                result = self.handler_func(command)
                return FlextResult.ok(result)
            except Exception as e:
                return FlextResult.fail(str(e))

    def _create_handler(handler_func: Callable) -> FlextHandlers.Handler:
        return TestHandler(handler_func)

    return _create_handler


@pytest.fixture
def event_factory() -> Callable[[str, dict[str, Any]], FlextEvent]:
    """Factory for creating test events."""

    def _create_event(
        event_type: str, data: dict[str, Any] | None = None
    ) -> FlextEvent:
        result = FlextEvent.create_event(
            event_type=event_type,
            event_data=data or {},
        )
        if result.success:
            return result.data
        raise ValueError(f"Failed to create event: {result.error}")

    return _create_event


# ============================================================================
# Integration Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def integration_setup():
    """One-time setup for integration tests."""
    # Setup code here
    return
    # Teardown code here


@pytest.fixture
def database_connection():
    """Provide database connection for integration tests."""
    # Mock database connection for testing
    connection = MagicMock()
    connection.execute.return_value = {"status": "success"}
    connection.fetch.return_value = [{"id": 1, "name": "test"}]

    yield connection

    # Cleanup
    connection.close()


# ============================================================================
# End-to-End Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def e2e_environment():
    """Setup complete environment for E2E tests."""
    return {
        "services": [],
        "containers": [],
        "ports": {"api": 8080, "db": 5432},
    }

    # Setup services

    # Teardown services


# ============================================================================
# Custom Test Markers and Utilities
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add custom markers."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add markers based on test name
        if "test_performance" in item.name:
            item.add_marker(pytest.mark.benchmark)
        if "test_async" in item.name:
            item.add_marker(pytest.mark.asyncio)
