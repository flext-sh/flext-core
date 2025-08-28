"""Unified fixture library for flext-core test suite.

Provides comprehensive test fixtures using pytest ecosystem:
- pytest-asyncio for async operations
- pytest-benchmark for performance testing
- pytest-mock for clean mocking
- pytest-httpx for HTTP client testing
- Simple data factories for realistic test data generation

Replaces multiple scattered fixture files with unified approach.
"""

# ruff: noqa: ARG001, ARG002
from __future__ import annotations

import asyncio
import contextlib
import datetime
import uuid
from collections.abc import AsyncGenerator, Callable

object

import httpx
import pytest
import pytest_mock
from pytest_httpx import HTTPXMock

from flext_core import FlextContainer, FlextEntity, FlextResult, FlextValue

# ============================================================================
# SIMPLE DATA FACTORIES - CLEAN IMPLEMENTATIONS
# ============================================================================


class SimpleDataFactory:
    """Simple data factory for test objects without complex dependencies."""

    @staticmethod
    def create_entity_data() -> dict[str, object]:
        """Create realistic entity data for testing."""
        return {
            "id": str(uuid.uuid4()),
            "name": f"test_entity_{uuid.uuid4().hex[:8]}",
            "description": "Test entity description for comprehensive testing",
            "active": True,
            "created_at": datetime.datetime.now(datetime.UTC),
            "updated_at": datetime.datetime.now(datetime.UTC),
        }

    @staticmethod
    def create_value_data() -> dict[str, object]:
        """Create realistic value data for testing."""
        test_value = f"test_value_{uuid.uuid4().hex[:8]}"
        return {
            "value": test_value,
            "normalized_value": test_value.lower().strip(),
            "value_type": "string",
            "metadata": {"source": "factory", "normalized": True},
            "is_valid": True,
            "validation_errors": [],
        }

    @staticmethod
    def create_success_result(
        value: object = "test_success_value",
    ) -> FlextResult[object]:
        """Create successful FlextResult."""
        return FlextResult[object].ok(value)

    @staticmethod
    def create_failure_result(error: str = "Test failure error") -> FlextResult[object]:
        """Create failed FlextResult."""
        return FlextResult[object].fail(error)

    @staticmethod
    def create_container_with_services() -> FlextContainer:
        """Create container with pre-registered services."""
        container = FlextContainer()

        # Register common test services
        container.register("test_service", lambda: "test_service_instance")
        container.register("config", lambda: {"test": True})

        return container


# ============================================================================
# CORE PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def default_timeout() -> float:
    """Provide default timeout for operations."""
    return 30.0  # Default timeout in seconds


@pytest.fixture
def test_timestamp() -> datetime.datetime:
    """Provide consistent test timestamp."""
    return datetime.datetime.now(datetime.UTC)


@pytest.fixture
def test_uuid() -> str:
    """Provide test UUID string."""
    return str(uuid.uuid4())


# ============================================================================
# ASYNC TESTING FIXTURES
# ============================================================================


@pytest.fixture
async def async_timeout() -> float:
    """Provide async operation timeout."""
    return 5.0


@pytest.fixture
async def async_context() -> AsyncGenerator[dict[str, object]]:
    """Provide async execution context with cleanup."""
    context: dict[str, object] = {
        "started_at": datetime.datetime.now(datetime.UTC),
        "tasks": [],
        "resources": [],
    }

    try:
        yield context
    finally:
        # Cleanup tasks
        for task in context.get("tasks", []):
            if not task.done():
                task.cancel()
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1.0)

        # Cleanup resources
        for resource in context.get("resources", []):
            if hasattr(resource, "close"):
                with contextlib.suppress(Exception):
                    await resource.close()


# ============================================================================
# HTTP TESTING FIXTURES
# ============================================================================


@pytest.fixture
def http_mock(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Provide HTTP mock with common responses."""
    # Setup common mock responses
    httpx_mock.add_response(
        method="GET",
        url="https://api.example.com/health",
        json={"status": "ok"},
        status_code=200,
    )

    httpx_mock.add_response(
        method="GET",
        url="https://api.example.com/error",
        json={"error": "Internal server error"},
        status_code=500,
    )

    return httpx_mock


@pytest.fixture
def http_client() -> httpx.Client:
    """Provide HTTP client for testing."""
    return httpx.Client(timeout=5.0)


@pytest.fixture
async def async_http_client() -> AsyncGenerator[httpx.AsyncClient]:
    """Provide async HTTP client for testing."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        yield client


# ============================================================================
# BENCHMARK FIXTURES
# ============================================================================


@pytest.fixture
def benchmark_rounds() -> int:
    """Provide benchmark rounds for performance testing."""
    return 10


@pytest.fixture
def performance_tracker(benchmark: object) -> Callable[[Callable[[], object]], object]:
    """Provide performance tracking utility."""

    def track(func: Callable[[], object]) -> object:
        """Track performance of a function."""
        # Simple benchmark without complex configuration
        return func()  # For now, just execute - can enhance later

    return track


# ============================================================================
# SIMPLE DATA FIXTURES
# ============================================================================


@pytest.fixture
def simple_factory() -> SimpleDataFactory:
    """Provide simple data factory for test data generation."""
    return SimpleDataFactory()


@pytest.fixture
def entity_data(simple_factory: SimpleDataFactory) -> dict[str, object]:
    """Provide entity data for testing."""
    return simple_factory.create_entity_data()


@pytest.fixture
def value_data(simple_factory: SimpleDataFactory) -> dict[str, object]:
    """Provide value data for testing."""
    return simple_factory.create_value_data()


@pytest.fixture
def success_result(simple_factory: SimpleDataFactory) -> FlextResult[str]:
    """Provide successful FlextResult for testing."""
    return simple_factory.create_success_result()


@pytest.fixture
def failure_result(simple_factory: SimpleDataFactory) -> FlextResult[str]:
    """Provide failed FlextResult for testing."""
    return simple_factory.create_failure_result()


@pytest.fixture
def test_container(simple_factory: SimpleDataFactory) -> FlextContainer:
    """Provide FlextContainer with test services."""
    return simple_factory.create_container_with_services()


# ============================================================================
# BATCH DATA FOR PERFORMANCE TESTING
# ============================================================================


@pytest.fixture
def entity_batch(simple_factory: SimpleDataFactory) -> list[dict[str, object]]:
    """Provide batch of entity data for performance testing."""
    return [simple_factory.create_entity_data() for _ in range(10)]


@pytest.fixture
def value_batch(simple_factory: SimpleDataFactory) -> list[dict[str, object]]:
    """Provide batch of value data for performance testing."""
    return [simple_factory.create_value_data() for _ in range(10)]


@pytest.fixture
def result_batch(simple_factory: SimpleDataFactory) -> list[FlextResult[str]]:
    """Provide batch of results for performance testing."""
    return [simple_factory.create_success_result(f"value_{i}") for i in range(5)]


# ============================================================================
# MOCK FIXTURES
# ============================================================================


@pytest.fixture
def mock_container(mocker: pytest_mock.MockerFixture) -> FlextContainer:
    """Provide mocked FlextContainer."""
    container = mocker.Mock(spec=FlextContainer)
    container.resolve = mocker.Mock()
    container.register = mocker.Mock()
    return container


@pytest.fixture
def mock_entity(
    mocker: pytest_mock.MockerFixture, entity_data: dict[str, object]
) -> FlextEntity:
    """Provide mocked FlextEntity."""
    entity = mocker.Mock(spec=FlextEntity)
    entity.id = entity_data["id"]
    entity.name = entity_data["name"]
    entity.description = entity_data["description"]
    entity.active = entity_data["active"]
    return entity


@pytest.fixture
def mock_value(
    mocker: pytest_mock.MockerFixture, value_data: dict[str, object]
) -> FlextValue:
    """Provide mocked FlextValue."""
    value = mocker.Mock(spec=FlextValue)
    value.value = value_data["value"]
    value.normalized_value = value_data["normalized_value"]
    value.is_valid = value_data["is_valid"]
    return value


# ============================================================================
# ERROR SIMULATION
# ============================================================================


@pytest.fixture
def error_simulator(mocker: pytest_mock.MockerFixture) -> Callable[[Exception], None]:
    """Provide error simulation utility."""

    def simulate(exception: Exception) -> None:
        """Simulate an error by mocking methods."""
        mocker.patch.object(
            FlextResult,
            "ok",
            side_effect=exception,
        )

    return simulate


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_test_entity(**kwargs: object) -> FlextEntity:
    """Create test entity with custom attributes."""
    factory = SimpleDataFactory()
    data = factory.create_entity_data()
    data.update(kwargs)

    # Create a simple entity-like object for testing
    class TestEntity(FlextEntity):
        """Test entity implementation."""

        def __init__(self, **data: object) -> None:
            super().__init__(**data)

    return TestEntity(**data)


def create_test_value(**kwargs: object) -> dict[str, object]:
    """Create test value data with custom attributes."""
    factory = SimpleDataFactory()
    data = factory.create_value_data()
    data.update(kwargs)
    return data


def create_test_result(
    *, success: bool = True, value: object = None, error: str | None = None
) -> FlextResult[object]:
    """Create test result with specified state."""
    if success:
        return FlextResult[object].ok(value or "test_value")
    return FlextResult[object].fail(error or "test_error")
