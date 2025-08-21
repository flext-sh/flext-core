"""Unified fixtures library with comprehensive pytest plugin integration.

This module consolidates all test fixtures and support utilities into a single,
comprehensive library using all available pytest plugins extensively:
- factory_boy: Advanced object creation with realistic data
- pytest-asyncio: Async testing utilities
- pytest-benchmark: Performance testing fixtures
- pytest-mock: Mocking utilities
- pytest-httpx: HTTP testing fixtures
- pytest-timeout: Timeout fixtures
- pytest-env: Environment management
- pytest-clarity: Enhanced assertion clarity

Provides centralized fixtures for:
- Domain objects with factory_boy
- Async operations with pytest-asyncio
- Performance testing with pytest-benchmark
- HTTP mocking with pytest-httpx
- Container management
- Error simulation
- Data builders
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import factory
import pytest
import structlog
from factory import Faker, LazyAttribute, LazyFunction, Trait
from factory.fuzzy import FuzzyChoice, FuzzyInteger, FuzzyText
from pytest_benchmark.fixture import BenchmarkFixture
from pytest_httpx import HTTPXMock
from pytest_mock import MockerFixture

from flext_core import (
    FlextContainer,
    FlextEntityId,
    FlextResult,
)

T = TypeVar("T")


@dataclass
class TestConfig:
    """Configuration for test execution with environment integration."""

    temp_dir: Path
    log_level: str = "DEBUG"
    async_timeout: float = 5.0
    benchmark_rounds: int = 10
    enable_profiling: bool = True
    mock_external_apis: bool = True


class FlextEntityFactory(factory.Factory):
    """Advanced factory_boy factory for FlextEntity with comprehensive traits."""

    class Meta:
        model = dict  # Will be converted to actual FlextEntity in post-generation

    # Base attributes with realistic data using Faker
    id = LazyFunction(lambda: str(FlextEntityId.generate()))
    created_at = LazyFunction(lambda: datetime.now(UTC))
    updated_at = LazyFunction(lambda: datetime.now(UTC))
    version = FuzzyInteger(1, 100)

    # Advanced traits for different scenarios
    class Params:
        # New entity trait
        new = Trait(
            created_at=LazyFunction(lambda: datetime.now(UTC)),
            updated_at=LazyFunction(lambda: datetime.now(UTC)),
            version=1,
        )

        # Mature entity trait
        mature = Trait(
            version=FuzzyInteger(50, 200),
            created_at=Faker("date_time_this_year", tzinfo=UTC),
        )

        # Recently updated trait
        recently_updated = Trait(
            updated_at=LazyFunction(lambda: datetime.now(UTC)),
            version=FuzzyInteger(10, 50),
        )


class FlextValueFactory(factory.Factory):
    """Factory for FlextValue objects with validation scenarios."""

    class Meta:
        model = dict

    # Common value object attributes
    value = FuzzyText(length=20)
    normalized_value = LazyAttribute(lambda obj: obj.value.lower().strip())

    class Params:
        # Valid value trait
        valid = Trait(
            value=Faker("word"),
        )

        # Invalid value trait
        invalid = Trait(
            value="",  # Empty value typically invalid
        )

        # Complex value trait
        complex = Trait(
            value=Faker("sentence", nb_words=5),
        )


class FlextResultFactory(factory.Factory):
    """Comprehensive FlextResult factory with all success/failure scenarios."""

    class Meta:
        model = dict

    # Result attributes
    success = True
    value = Faker("pydict", nb_elements=3)
    error = None
    error_code = None

    class Params:
        # Success scenarios
        successful = Trait(
            success=True,
            error=None,
            error_code=None,
        )

        # Failure scenarios
        failed = Trait(
            success=False,
            value=None,
            error=Faker("sentence"),
            error_code=FuzzyChoice([
                "VALIDATION_ERROR",
                "NOT_FOUND",
                "INTERNAL_ERROR",
                "PERMISSION_DENIED",
                "TIMEOUT",
            ]),
        )

        # Validation failure
        validation_failed = Trait(
            success=False,
            value=None,
            error="Validation failed",
            error_code="VALIDATION_ERROR",
        )

        # Not found failure
        not_found = Trait(
            success=False,
            value=None,
            error="Resource not found",
            error_code="NOT_FOUND",
        )


class FlextContainerFactory(factory.Factory):
    """Factory for FlextContainer with various service configurations."""

    class Meta:
        model = dict

    # Container configuration
    name = Faker("company")
    auto_wire = True
    lazy_loading = True

    class Params:
        # Minimal container
        minimal = Trait(
            auto_wire=False,
            lazy_loading=False,
        )

        # Production-like container
        production = Trait(
            auto_wire=True,
            lazy_loading=True,
        )


@pytest.fixture
def test_config(tmp_path: Path) -> TestConfig:
    """Provide comprehensive test configuration with temp directory."""
    return TestConfig(
        temp_dir=tmp_path,
        log_level=os.getenv("TEST_LOG_LEVEL", "DEBUG"),
        async_timeout=float(os.getenv("TEST_ASYNC_TIMEOUT", "5.0")),
        benchmark_rounds=int(os.getenv("TEST_BENCHMARK_ROUNDS", "10")),
        enable_profiling=os.getenv("TEST_ENABLE_PROFILING", "true").lower() == "true",
        mock_external_apis=os.getenv("TEST_MOCK_EXTERNAL", "true").lower() == "true",
    )


@pytest.fixture
def structured_logger() -> structlog.stdlib.BoundLogger:
    """Provide structured logger for tests with consistent configuration."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("test")


@pytest.fixture
def mock_container(mocker: MockerFixture) -> FlextContainer:
    """Provide mocked FlextContainer with common services."""
    container = mocker.Mock(spec=FlextContainer)
    container.resolve = mocker.Mock()
    container.register = mocker.Mock()
    container.create_scope = mocker.Mock()
    return container


@pytest.fixture
def entity_factory() -> type[FlextEntityFactory]:
    """Provide FlextEntity factory for test data generation."""
    return FlextEntityFactory


@pytest.fixture
def value_factory() -> type[FlextValueFactory]:
    """Provide FlextValue factory for test data generation."""
    return FlextValueFactory


@pytest.fixture
def result_factory() -> type[FlextResultFactory]:
    """Provide FlextResult factory for test data generation."""
    return FlextResultFactory


@pytest.fixture
def container_factory() -> type[FlextContainerFactory]:
    """Provide FlextContainer factory for test data generation."""
    return FlextContainerFactory


@pytest.fixture
async def async_context() -> AsyncGenerator[dict[str, Any]]:
    """Provide async execution context with timeout and resource management."""
    context = {
        "started_at": datetime.now(UTC),
        "tasks": [],
        "resources": [],
    }

    try:
        yield context
    finally:
        # Clean up any running tasks
        for task in context.get("tasks", []):
            if not task.done():
                task.cancel()
                with suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1.0)

        # Clean up resources
        for resource in context.get("resources", []):
            if hasattr(resource, "close"):
                await resource.close()


@pytest.fixture
def performance_tracker(benchmark: BenchmarkFixture) -> Callable[[str, Callable[[], T]], T]:
    """Provide performance tracking utility with benchmark integration."""

    def track_performance(name: str, func: Callable[[], T]) -> T:
        """Track performance of a function with benchmark."""
        return benchmark.pedantic(func, rounds=10, iterations=5)

    return track_performance


@pytest.fixture
def error_simulator(mocker: MockerFixture) -> Callable[[Exception], None]:
    """Provide error simulation utility for testing error scenarios."""

    def simulate_error(exception: Exception) -> None:
        """Simulate an error by patching relevant methods."""
        # This is a placeholder - specific implementations will patch
        # actual methods based on the exception type
        mocker.patch.object(
            FlextResult,
            "ok",
            side_effect=exception
        )

    return simulate_error


@pytest.fixture
def temp_file_manager(test_config: TestConfig) -> Generator[Callable[[str, str], Path]]:
    """Provide temporary file management with automatic cleanup."""
    files: list[Path] = []

    def create_temp_file(content: str, suffix: str = ".py") -> Path:
        """Create a temporary file with content."""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=test_config.temp_dir)
        file_path = Path(path)
        files.append(file_path)

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path

    try:
        yield create_temp_file
    finally:
        # Clean up all created files
        for file_path in files:
            if file_path.exists():
                file_path.unlink()


@pytest.fixture
def http_mock_client(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Provide HTTP mock client with common response patterns."""
    # Pre-configure common mock responses
    httpx_mock.add_response(
        method="GET",
        url="https://api.example.com/health",
        json={"status": "ok", "timestamp": datetime.now(UTC).isoformat()},
        status_code=200,
    )

    httpx_mock.add_response(
        method="GET",
        url="https://api.example.com/error",
        json={"error": "Internal server error", "code": "INTERNAL_ERROR"},
        status_code=500,
    )

    return httpx_mock


@contextmanager
def performance_context(enable_profiling: bool = True) -> Generator[dict[str, Any]]:
    """Context manager for performance monitoring with memory tracking."""
    import tracemalloc

    metrics = {
        "start_time": time.perf_counter(),
        "start_memory": 0,
        "end_time": 0,
        "end_memory": 0,
        "duration": 0,
        "memory_diff": 0,
    }

    if enable_profiling:
        tracemalloc.start()
        metrics["start_memory"] = tracemalloc.get_traced_memory()[0]

    try:
        yield metrics
    finally:
        metrics["end_time"] = time.perf_counter()
        metrics["duration"] = metrics["end_time"] - metrics["start_time"]

        if enable_profiling:
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            metrics["end_memory"] = current_memory
            metrics["memory_diff"] = current_memory - metrics["start_memory"]
            metrics["peak_memory"] = peak_memory
            tracemalloc.stop()


@asynccontextmanager
async def async_performance_context(enable_profiling: bool = True) -> AsyncGenerator[dict[str, Any]]:
    """Async context manager for performance monitoring."""
    with performance_context(enable_profiling) as metrics:
        yield metrics


# Factory integration functions for easy test data creation
def create_entity(**kwargs: Any) -> dict[str, Any]:
    """Create entity data using factory_boy with custom attributes."""
    return FlextEntityFactory.create(**kwargs)


def create_value(**kwargs: Any) -> dict[str, Any]:
    """Create value object data using factory_boy with custom attributes."""
    return FlextValueFactory.create(**kwargs)


def create_success_result(**kwargs: Any) -> dict[str, Any]:
    """Create successful FlextResult data using factory_boy."""
    return FlextResultFactory.create(successful=True, **kwargs)


def create_failed_result(**kwargs: Any) -> dict[str, Any]:
    """Create failed FlextResult data using factory_boy."""
    return FlextResultFactory.create(failed=True, **kwargs)


def create_container(**kwargs: Any) -> dict[str, Any]:
    """Create container data using factory_boy with custom attributes."""
    return FlextContainerFactory.create(**kwargs)


# Batch creation utilities
def create_entities(count: int, **kwargs: Any) -> list[dict[str, Any]]:
    """Create multiple entities using factory_boy batch creation."""
    return FlextEntityFactory.create_batch(count, **kwargs)


def create_values(count: int, **kwargs: Any) -> list[dict[str, Any]]:
    """Create multiple value objects using factory_boy batch creation."""
    return FlextValueFactory.create_batch(count, **kwargs)


def create_results(count: int, **kwargs: Any) -> list[dict[str, Any]]:
    """Create multiple results using factory_boy batch creation."""
    return FlextResultFactory.create_batch(count, **kwargs)
