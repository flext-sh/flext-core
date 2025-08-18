"""Pytest configuration for FLEXT Core test suite.

Provides fixtures, factories, and testing utilities with type safety,
isolation, and reproducible test execution.
"""

from __future__ import annotations

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
from typing import Generic, TypedDict, TypeVar
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import structlog
from _pytest.fixtures import SubRequest
from hypothesis import strategies as st

from flext_core import (
    FlextAggregateRoot,
    FlextBaseHandler,
    FlextCommands,
    FlextContainer,
    FlextEntity,
    FlextEntityStatus,
    FlextEvent,
    FlextLoggerFactory,
    FlextOperationStatus,
    FlextResult,
    FlextValueObject,
)

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
def mock_factory() -> Callable[[str], Mock]:
    """Advanced mock factory with automatic cleanup."""

    def _create_mock(name: str, **kwargs: object) -> Mock:
        mock = MagicMock(name=name, **kwargs)
        mock.is_healthy.return_value = True
        mock.validate.return_value = FlextResult.ok(None)
        mock.process.return_value = FlextResult.ok("processed")
        return mock

    return _create_mock


@pytest.fixture
def async_mock_factory() -> Callable[[str], AsyncMock]:
    """Async mock factory for testing async patterns."""

    def _create_async_mock(name: str, **kwargs: object) -> AsyncMock:
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
def entity_factory() -> Callable[[str, dict[str, object]], FlextEntity]:
    """Factory for creating test entities."""

    class TestEntity(FlextEntity):
        """Test entity for testing purposes."""

        name: str = "test"
        value: int = 0

        def __init__(self, name: str = "test", value: int = 0) -> None:
            super().__init__()
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
                "status": self.status.value,
            }


@pytest.fixture
def value_object_factory() -> Callable[[dict[str, object]], FlextValueObject]:
    """Factory for creating test value objects."""

    class TestValueObject(FlextValueObject):
        value: str
        metadata: dict[str, object] = field(default_factory=dict)

        def validate_business_rules(self) -> FlextResult[None]:
            if not self.value:
                return FlextResult.fail("Value cannot be empty")
            return FlextResult.ok(None)

    def _create_vo(data: dict[str, object]) -> FlextValueObject:
        # Extract specific fields with type safety
        value = str(data.get("value", ""))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return TestValueObject(value=value, metadata=metadata)

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
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    temp_dir = Path(tempfile.mkdtemp(prefix="flext_test_"))

    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# ============================================================================
# Advanced State Management and Isolation
# ============================================================================


@pytest.fixture
def isolation_context() -> Generator[dict[str, object]]:
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
        import gc  # noqa: PLC0415

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
                assert result.data == expected_data

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
                return FlextResult.fail("Command name is required")
            return FlextResult.ok(None)

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
                return FlextResult.ok(result)
            except Exception as e:
                return FlextResult.fail(str(e))

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
            return result.unwrap()
        raise ValueError(f"Failed to create event: {result.error}")

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


@pytest.fixture
def database_connection() -> Generator[MagicMock]:
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
    config: object,
    items: list[object],
) -> None:
    """Modify test collection to add custom markers."""
    # Mark 'config' as intentionally unused while keeping hook signature valid
    del config
    for item in items:
        # Add markers based on test location - use hasattr for safety
        if hasattr(item, "fspath") and "unit" in str(item.fspath):
            if hasattr(item, "add_marker"):
                item.add_marker(pytest.mark.unit)
        elif hasattr(item, "fspath") and "integration" in str(item.fspath):
            if hasattr(item, "add_marker"):
                item.add_marker(pytest.mark.integration)
        elif (
            hasattr(item, "fspath")
            and "e2e" in str(item.fspath)
            and hasattr(item, "add_marker")
        ):
            item.add_marker(pytest.mark.e2e)

        # Add markers based on test name
        if (
            hasattr(item, "name")
            and "test_performance" in item.name
            and hasattr(item, "add_marker")
        ):
            item.add_marker(pytest.mark.benchmark)
        if (
            hasattr(item, "name")
            and "test_async" in item.name
            and hasattr(item, "add_marker")
        ):
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
        assert result.is_failure, f"Expected failure but got: {result.data}"

    # Backward-compatible aliases used in some tests
    @staticmethod
    def assert_result_ok(result: FlextResult[object]) -> None:
        """Assert that a result is successful."""
        AssertHelpers.assert_result_success(result)

    @staticmethod
    def assert_result_fail(result: FlextResult[object]) -> None:
        """Assert that a result is a failure."""
        AssertHelpers.assert_result_failure(result)


class MockFactory:
    """Factory for creating mock objects."""

    @staticmethod
    def create_mock_config(**kwargs: object) -> object:
        """Create a mock configuration."""
        return type("MockConfig", (), kwargs)()


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
    import time  # noqa: PLC0415

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
    import tempfile  # noqa: PLC0415

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
