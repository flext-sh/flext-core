"""Unified fixtures for flext-core tests using massive pytest ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import socket
import tempfile
import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import httpx
import pytest
import pytest_asyncio
from pydantic import Field
from pytest_mock import MockerFixture

from flext_core import (
    FlextCommands,
    FlextConfig,
    FlextContainer,
    FlextResult,
)
from flext_core.typings import FlextTypes

from .factories import (
    BaseTestEntity,
    BaseTestValueObject,
    FlextResultFactory,
    TestEntityFactory,
    TestValueObjectFactory,
)

# Configure logging for test fixtures
logger = logging.getLogger(__name__)


class BenchmarkFixture(Protocol):  # pragma: no cover - typing only
    """Benchmark fixture."""

    group: str

    def __call__(self, func: object, /, *args: object, **kwargs: object) -> object: ...

    def pedantic(self, func: object, /, *args: object, **kwargs: object) -> object: ...


# Simple placeholder factories for missing ones
class PerformanceDataFactory:
    """Simple performance data factory."""

    @staticmethod
    def create_large_payload(size_mb: int = 1) -> FlextTypes.Core.Dict:
        """Create large test payload."""
        return {"data": "x" * (size_mb * 1024), "size_mb": size_mb}

    @staticmethod
    def create_nested_structure(depth: int = 10) -> FlextTypes.Core.Dict:
        """Create nested data structure."""
        result: FlextTypes.Core.Dict = {"value": f"depth_{depth}"}
        current = result
        for i in range(depth - 1):
            current["nested"] = {"value": f"depth_{depth - i - 1}"}
            current = cast("FlextTypes.Core.Dict", current["nested"])
        return result


class ErrorSimulationFactory:
    """Simple error simulation factory."""

    @staticmethod
    def create_timeout_error() -> Exception:
        """Create timeout error."""
        return TimeoutError("Simulated timeout")

    @staticmethod
    def create_connection_error() -> Exception:
        """Create connection error."""
        return ConnectionError("Simulated connection error")

    @staticmethod
    def create_validation_error() -> Exception:
        """Create validation error."""
        return ValueError("Simulated validation error")

    @staticmethod
    def create_error_scenario(error_type: str) -> FlextTypes.Core.Dict:
        """Create error scenario dict."""
        error_map: dict[str, FlextTypes.Core.Dict] = {
            "ValidationError": {
                "type": "validation",
                "message": "Validation failed",
                "code": "VAL_001",
            },
            "ProcessingError": {
                "type": "processing",
                "message": "Processing failed",
                "code": "PROC_001",
            },
            "NetworkError": {
                "type": "network",
                "message": "Network error",
                "code": "NET_001",
            },
        }
        return error_map.get(
            error_type,
            {"type": "unknown", "message": "Unknown error", "code": "UNK_001"},
        )


class SequenceFactory:
    """Simple sequence factory."""

    @staticmethod
    def create_sequence(
        length: int = 10,
        prefix: str = "",
        count: int | None = None,
    ) -> FlextTypes.Core.StringList:
        """Create sequence with optional prefix."""
        actual_length = count if count is not None else length
        if prefix:
            return [f"{prefix}_{i}" for i in range(actual_length)]
        return [str(i) for i in range(actual_length)]

    @staticmethod
    def create_timeline_events(count: int = 10) -> list[FlextTypes.Core.Dict]:
        """Create timeline events."""
        return [
            {
                "id": f"event_{i}",
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "type": "test_event",
                "data": {"index": i, "description": f"Test event {i}"},
            }
            for i in range(count)
        ]


class FactoryRegistry:
    """Simple factory registry."""

    def __init__(self) -> None:
        self.factories: FlextTypes.Core.Dict = {}

    def register(self, name: str, factory: object) -> None:
        """Register a factory."""
        self.factories[name] = factory

    def get(self, name: str) -> object:
        """Get a factory."""
        return self.factories.get(name)


class FlextConfigFactory:
    """Simple config factory."""

    @staticmethod
    def create_test_config() -> FlextConfig:
        """Create test configuration."""
        return FlextConfig()

    @staticmethod
    def create_development_config() -> FlextConfig:
        """Create development configuration."""
        return FlextConfig()

    @staticmethod
    def create_production_config() -> FlextConfig:
        """Create production configuration."""
        return FlextConfig()


# Test command classes


class TestCommand(FlextCommands.Models.Command):
    """Test command for fixtures."""

    command_type: str = Field(default="test", description="Type of command")
    config: object = Field(default_factory=dict)


class BatchCommand(FlextCommands.Models.Command):
    """Batch command for fixtures."""

    command_type: str = Field(default="batch", description="Type of command")
    config: object = Field(default_factory=dict)


class ValidationCommand(FlextCommands.Models.Command):
    """Validation command for fixtures."""

    command_type: str = Field(default="validation", description="Type of command")
    config: object = Field(default_factory=dict)


class ProcessingCommand(FlextCommands.Models.Command):
    """Processing command for fixtures."""

    command_type: str = Field(default="processing", description="Type of command")
    config: object = Field(default_factory=dict)


class CommandFactory:
    """Simple command factory."""

    @staticmethod
    def create_test_command(config: object | None = None) -> TestCommand:
        """Create test command."""
        return TestCommand(config=config or {})

    @staticmethod
    def create_batch_command(config: object | None = None) -> BatchCommand:
        """Create batch command."""
        return BatchCommand(config=config or {})

    @staticmethod
    def create_validation_command(config: object | None = None) -> ValidationCommand:
        """Create validation command."""
        return ValidationCommand(config=config or {})

    @staticmethod
    def create_processing_command(config: object | None = None) -> ProcessingCommand:
        """Create processing command."""
        return ProcessingCommand(config=config or {})


# Core fixtures for basic testing
@pytest.fixture
def flext_config() -> FlextConfig:
    """Fixture providing configured FlextConfig instance."""
    return FlextConfig()


@pytest.fixture
def flext_container() -> FlextContainer:
    """Fixture providing configured FlextContainer instance."""
    return FlextContainer()


@pytest.fixture
def test_entity() -> BaseTestEntity:
    """Fixture providing test domain entity."""
    return TestEntityFactory.create()


@pytest.fixture
def test_value_object() -> BaseTestValueObject:
    """Fixture providing test value object."""
    return TestValueObjectFactory.create()


@pytest.fixture
def test_command() -> TestCommand:
    """Fixture providing test command."""
    return TestCommand(config={})


# Result fixtures
@pytest.fixture
def success_result() -> FlextResult[object]:
    """Fixture providing successful FlextResult."""
    return FlextResultFactory.create_success("test_success_data")


@pytest.fixture
def failure_result() -> FlextResult[object]:
    """Fixture providing failed FlextResult."""
    return FlextResultFactory.create_failure("test_failure_message")


@pytest.fixture
def result_chain() -> list[FlextResult[object]]:
    """Fixture providing chain of results for pipeline testing."""
    return [
        FlextResultFactory.create_success("step_1"),
        FlextResultFactory.create_success("step_2"),
        FlextResultFactory.create_failure("step_3_failed"),
        FlextResultFactory.create_success("step_4"),
    ]


# File system fixtures
@pytest.fixture
def temp_directory() -> Generator[Path]:
    """Fixture providing temporary directory that's automatically cleaned up."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_config_file(temp_directory: Path) -> Path:
    """Fixture providing temporary config file with test data."""
    config_file = temp_directory / "test_config.json"
    test_config = {
        "debug": True,
        "log_level": "DEBUG",
        "database_url": "sqlite:///test.db",
        "features": ["auth", "cache"],
    }
    config_file.write_text(json.dumps(test_config, indent=2))
    return config_file


@pytest.fixture
def config_builder() -> FlextConfig:
    """Fixture providing configuration builder for custom configs."""
    return FlextConfig()


# Performance testing fixtures
@pytest.fixture
def benchmark_config(benchmark: BenchmarkFixture) -> BenchmarkFixture:
    """Fixture providing configured benchmark for performance tests."""
    # Configure benchmark settings
    benchmark.group = "flext_core"

    # Note: BenchmarkFixture attributes are read-only in pytest-benchmark
    # Configuration should be done via pytest.ini or command line options
    return benchmark


@pytest.fixture
def performance_data() -> FlextTypes.Core.Dict:
    """Fixture providing performance test data."""
    return PerformanceDataFactory.create_large_payload(size_mb=1)


@pytest.fixture
def large_dataset() -> list[BaseTestEntity]:
    """Fixture providing large dataset for performance testing."""
    return TestEntityFactory.create_batch(size=1000)


@pytest.fixture
def nested_data() -> FlextTypes.Core.Dict:
    """Fixture providing deeply nested data structure."""
    return PerformanceDataFactory.create_nested_structure(depth=10)


# Async fixtures
@pytest_asyncio.fixture
async def async_container() -> AsyncGenerator[FlextContainer]:
    """Async fixture providing container with async services."""
    container = FlextContainer()

    # Register async services
    container.register("async_service", AsyncTestService())

    try:
        yield container
    finally:
        # Cleanup async resources
        shutdown_method = getattr(container, "shutdown", None)
        if shutdown_method and callable(shutdown_method):
            try:
                result = shutdown_method()
                # Check if result is awaitable using proper type checking
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                # Log cleanup errors but don't propagate them
                logger.debug("Container shutdown error: %s", e, exc_info=True)


@pytest_asyncio.fixture
async def async_executor() -> AsyncGenerator[object]:
    """Async fixture providing executor for async operations."""
    executor = AsyncExecutor()
    try:
        await executor.start()
        yield executor
    finally:
        await executor.stop()


class AsyncTestService:
    """Test service for async testing."""

    def __init__(self) -> None:
        self._executor = AsyncExecutor()

    async def process(self, data: object) -> FlextTypes.Core.Dict:
        """Process data asynchronously."""
        await asyncio.sleep(0.001)  # Simulate async work
        return {"processed": True, "original": data}

    async def validate(self, data: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
        """Validate data asynchronously."""
        await asyncio.sleep(0.001)  # Simulate async work
        has_required = "required_field" in data
        return {"valid": has_required}

    async def transform(self, data: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
        """Transform data asynchronously."""
        await asyncio.sleep(0.001)  # Simulate async work
        return {"transformed": True, "output": f"transformed_{data.get('input', '')}"}

    async def fail_operation(self) -> FlextResult[object]:
        """Simulate async failure."""
        await asyncio.sleep(0.001)
        return FlextResult[object].fail("async_operation_failed")


class AsyncExecutor:
    """Async executor for testing."""

    def __init__(self) -> None:
        self._running = False
        self._tasks: list[asyncio.Task[object]] = []

    async def start(self) -> None:
        """Start executor."""
        self._running = True

    async def stop(self) -> None:
        """Stop executor and cancel tasks."""
        self._running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def execute(self, coro: object) -> object:
        """Execute coroutine."""
        if not self._running:
            await self.start()

        # Ensure coro is a coroutine before creating task
        if not asyncio.iscoroutine(coro):
            msg = f"Expected coroutine, got {type(coro)}"
            raise TypeError(msg)

        task: asyncio.Task[object] = asyncio.create_task(coro)
        self._tasks.append(task)
        return await task

    async def execute_batch(self, coros: FlextTypes.Core.List) -> FlextTypes.Core.List:
        """Execute multiple coroutines in batch."""
        if not self._running:
            await self.start()

        tasks = []
        for coro in coros:
            if not asyncio.iscoroutine(coro):
                msg = f"Expected coroutine, got {type(coro)}"
                raise TypeError(msg)
            task = asyncio.create_task(coro)
            self._tasks.append(task)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return list(results)

    def cleanup(self) -> None:
        """Cleanup completed tasks."""
        self._tasks.clear()


# Error simulation fixtures
@pytest.fixture
def error_scenarios() -> list[FlextTypes.Core.Dict]:
    """Fixture providing various error scenarios."""
    return [
        ErrorSimulationFactory.create_error_scenario("ValidationError"),
        ErrorSimulationFactory.create_error_scenario("ProcessingError"),
        ErrorSimulationFactory.create_error_scenario("NetworkError"),
    ]


@pytest.fixture
def validation_error_result() -> FlextResult[object]:
    """Fixture providing validation error result."""
    error_scenario = ErrorSimulationFactory.create_error_scenario("ValidationError")
    return FlextResult[object].fail(str(error_scenario["message"]))


# Sequence and batch fixtures
@pytest.fixture
def item_sequence() -> FlextTypes.Core.StringList:
    """Fixture providing sequential items."""
    return SequenceFactory.create_sequence(prefix="test_item", count=20)


@pytest.fixture
def timeline_events() -> list[FlextTypes.Core.Dict]:
    """Fixture providing timeline events."""
    return SequenceFactory.create_timeline_events(count=10)


@pytest.fixture
def entity_batch() -> list[BaseTestEntity]:
    """Fixture providing batch of entities."""
    return TestEntityFactory.create_batch(size=50)


# Factory registry fixture
@pytest.fixture
def factory_registry() -> FactoryRegistry:
    """Fixture providing factory registry."""
    return FactoryRegistry()


# Mock fixtures using pytest-mock
@pytest.fixture
def mock_logger(mocker: MockerFixture) -> object:
    """Fixture providing mocked logger."""
    return mocker.patch("flext_core.legacy.FlextLogger")


@pytest.fixture
def mock_database(mocker: MockerFixture) -> object:
    """Fixture providing mocked database."""
    mock_db = mocker.MagicMock()
    mock_db.connect.return_value = True
    mock_db.execute.return_value = {"affected_rows": 1}
    return mock_db


@pytest.fixture
def mock_external_service(mocker: MockerFixture) -> object:
    """Fixture providing mocked external service."""
    mock_service = mocker.MagicMock()
    mock_service.call_api.return_value = {"status": "success", "data": "test"}
    return mock_service


# Parametrized fixtures for comprehensive testing
@pytest.fixture(
    params=[
        "development",
        "testing",
        "staging",
        "production",
    ],
)
def environment(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture providing different environments."""
    return str(request.param)


@pytest.fixture(
    params=[
        {"debug": True, "log_level": "DEBUG"},
        {"debug": False, "log_level": "INFO"},
        {"debug": False, "log_level": "WARNING"},
        {"debug": False, "log_level": "ERROR"},
    ],
)
def config_variants(request: pytest.FixtureRequest) -> FlextTypes.Core.Dict:
    """Parametrized fixture providing config variants."""
    return dict(request.param)


@pytest.fixture(params=[1, 10, 100, 1000])
def batch_sizes(request: pytest.FixtureRequest) -> int:
    """Parametrized fixture providing different batch sizes."""
    return int(request.param)


# Scope fixtures for different test levels
@pytest.fixture(scope="session")
def session_container() -> FlextContainer:
    """Session-scoped container for integration tests."""
    container = FlextContainer()
    # Configure session-level services
    container.register("session_service", SessionTestService())
    return container


@pytest.fixture(scope="module")
def module_config() -> FlextConfig:
    """Module-scoped config for module-level tests."""
    return FlextConfigFactory.create_test_config()


@pytest.fixture(scope="class")
def class_database() -> FlextTypes.Core.Dict:
    """Class-scoped database for class-level tests."""
    return {
        "connection_string": "sqlite:///:memory:",
        "pool_size": 1,
        "connected": True,
    }


class SessionTestService:
    """Test service for session-scoped testing."""

    def __init__(self) -> None:
        self._data: FlextTypes.Core.Dict = {}
        self._session_data: dict[str, FlextTypes.Core.Dict] = {}

    def store(self, key: str, value: object) -> None:
        """Store data in session service."""
        self._data[key] = value

    def retrieve(self, key: str) -> object:
        """Retrieve data from session service."""
        return self._data.get(key)

    def clear(self) -> None:
        """Clear session data."""
        self._data.clear()
        self._session_data.clear()

    def create_session(self, session_id: str) -> None:
        """Create a new session."""
        self._session_data[session_id] = {"created_at": datetime.now(UTC), "data": {}}

    def get_session(self, session_id: str) -> FlextTypes.Core.Dict | None:
        """Get session by ID."""
        return self._session_data.get(session_id)

    def update_session(self, session_id: str, data: FlextTypes.Core.Dict) -> None:
        """Update session data."""
        if session_id in self._session_data:
            session_data = self._session_data[session_id]["data"]
            if isinstance(session_data, dict):
                session_data.update(data)

    def delete_session(self, session_id: str) -> None:
        """Delete session."""
        self._session_data.pop(session_id, None)

    def cleanup_sessions(self) -> None:
        """Cleanup all sessions."""
        self._session_data.clear()


# HTTP testing fixtures using pytest-httpx
@pytest.fixture
def http_client() -> object:
    """Fixture providing HTTP client for testing."""
    return httpx.AsyncClient()


@pytest.fixture
def mock_http_responses() -> FlextTypes.Core.Dict:
    """Fixture providing mock HTTP responses."""
    return {
        "success": {"status_code": 200, "json": {"result": "success"}},
        "not_found": {"status_code": 404, "json": {"error": "not_found"}},
        "server_error": {"status_code": 500, "json": {"error": "internal_error"}},
        "timeout": {"status_code": 408, "json": {"error": "timeout"}},
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_environment() -> Generator[None]:
    """Auto-use fixture for environment cleanup."""
    # Setup
    original_env: dict[str, str | None] = {}

    yield

    # Teardown - restore environment

    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# Randomly ordered fixtures for pytest-randomly
@pytest.fixture
def random_data() -> FlextTypes.Core.Dict:
    """Fixture providing random test data (works with pytest-randomly)."""
    return {
        "random_int": random.randint(1, 1000),
        "random_float": random.uniform(0.0, 100.0),
        "random_string": f"random_{random.randint(1000, 9999)}",
        "random_bool": random.choice([True, False]),
    }


# Marker-based fixtures
@pytest.fixture
def slow_operation_data() -> FlextTypes.Core.Dict:
    """Fixture for data requiring slow operations (use with @pytest.mark.slow)."""
    return PerformanceDataFactory.create_large_payload(size_mb=10)


@pytest.fixture
def integration_services() -> FlextTypes.Core.Dict:
    """Fixture for integration test services (use with @pytest.mark.integration)."""
    return {
        "database": {"url": "postgresql://test:test@localhost/integration_test"},
        "cache": {"url": "redis://localhost:6379/1"},
        "queue": {"url": "amqp://guest:guest@localhost:5672/test"},
    }


# Conditional fixtures
@pytest.fixture
def database_url() -> str:
    """Fixture providing database URL based on environment."""
    return os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture
def skip_if_no_network() -> None:
    """Fixture that skips test if no network available."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except OSError:
        pytest.skip("No network connection available")


# Advanced fixtures for specific patterns
@pytest.fixture
def command_pipeline() -> list[FlextCommands.Models.Command]:
    """Fixture providing command pipeline for CQRS testing."""
    return [
        CommandFactory.create_validation_command(),
        CommandFactory.create_processing_command(),
        CommandFactory.create_test_command(),
        CommandFactory.create_batch_command(),
    ]


@pytest.fixture
def event_sourcing_data() -> list[FlextTypes.Core.Dict]:
    """Fixture providing event sourcing test data."""
    base_time = datetime.now(UTC)

    return [
        {
            "event_id": str(uuid.uuid4()),
            "event_type": f"test_event_{i}",
            "timestamp": (base_time.timestamp() + i * 60),
            "data": {"sequence": i, "action": f"action_{i}"},
            "metadata": {"version": 1, "source": "test"},
        }
        for i in range(5)
    ]


# Main unified class
# Repository classes for functional testing
class InMemoryUserRepository:
    """In-memory user repository for functional testing."""

    def __init__(self) -> None:
        self._users: dict[str, FlextTypes.Core.Dict] = {}

    def save(
        self, user_data: FlextTypes.Core.Dict
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Save user data."""
        user_id = user_data.get("id")
        if not user_id:
            return FlextResult[FlextTypes.Core.Dict].fail(
                "User ID required", error_code="VALIDATION_ERROR"
            )

        self._users[str(user_id)] = dict(user_data)
        return FlextResult[FlextTypes.Core.Dict].ok(self._users[str(user_id)])

    def find_by_id(self, user_id: str) -> FlextResult[FlextTypes.Core.Dict]:
        """Find user by ID."""
        user = self._users.get(user_id)
        if not user:
            return FlextResult[FlextTypes.Core.Dict].fail(
                f"User not found: {user_id}", error_code="NOT_FOUND"
            )
        return FlextResult[FlextTypes.Core.Dict].ok(user)

    def find_by_username(self, username: str) -> FlextResult[FlextTypes.Core.Dict]:
        """Find user by username (or name field)."""
        for user in self._users.values():
            # Check both username and name fields for compatibility
            if user.get("username") == username or user.get("name") == username:
                return FlextResult[FlextTypes.Core.Dict].ok(user)
        return FlextResult[FlextTypes.Core.Dict].fail(
            f"User not found: {username}", error_code="NOT_FOUND"
        )

    def find_all(self) -> FlextResult[list[FlextTypes.Core.Dict]]:
        """Find all users."""
        return FlextResult[list[FlextTypes.Core.Dict]].ok(list(self._users.values()))

    def delete(self, user_id: str) -> FlextResult[None]:
        """Delete user by ID."""
        if user_id not in self._users:
            return FlextResult[None].fail(
                f"User not found: {user_id}", error_code="NOT_FOUND"
            )
        del self._users[user_id]
        return FlextResult[None].ok(None)

    def clear(self) -> None:
        """Clear all users."""
        self._users.clear()


class FailingUserRepository:
    """Repository that always fails for error scenario testing."""

    def save(
        self, user_data: FlextTypes.Core.Dict
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Always fail save operation."""
        return FlextResult[FlextTypes.Core.Dict].fail(
            "Repository operation failed", error_code="REPOSITORY_ERROR"
        )

    def find_by_id(self, user_id: str) -> FlextResult[FlextTypes.Core.Dict]:
        """Always fail find operation."""
        return FlextResult[FlextTypes.Core.Dict].fail(
            "Repository operation failed", error_code="REPOSITORY_ERROR"
        )

    def find_by_username(self, username: str) -> FlextResult[FlextTypes.Core.Dict]:
        """Always fail find by username operation."""
        return FlextResult[FlextTypes.Core.Dict].fail(
            "Repository operation failed", error_code="REPOSITORY_ERROR"
        )

    def find_all(self) -> FlextResult[list[FlextTypes.Core.Dict]]:
        """Always fail find all operation."""
        return FlextResult[list[FlextTypes.Core.Dict]].fail(
            "Repository operation failed", error_code="REPOSITORY_ERROR"
        )

    def delete(self, user_id: str) -> FlextResult[None]:
        """Always fail delete operation."""
        return FlextResult[None].fail(
            "Repository operation failed", error_code="REPOSITORY_ERROR"
        )


# Service classes for functional testing
class RealEmailService:
    """Real email service for functional testing."""

    def __init__(self) -> None:
        self._sent_emails: list[FlextTypes.Core.Dict] = []

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        from_email: str | None = None,
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Send email (mock implementation for testing)."""
        if not to or not subject:
            return FlextResult[FlextTypes.Core.Dict].fail(
                "Email recipient and subject required", error_code="VALIDATION_ERROR"
            )

        email_data: FlextTypes.Core.Dict = {
            "id": str(uuid.uuid4()),
            "to": to,
            "subject": subject,
            "body": body,
            "from_email": from_email or "test@example.com",
            "sent_at": datetime.now(UTC).isoformat(),
        }

        self._sent_emails.append(email_data)
        return FlextResult[FlextTypes.Core.Dict].ok(email_data)

    def get_sent_emails(self) -> list[FlextTypes.Core.Dict]:
        """Get all sent emails for testing verification."""
        return list(self._sent_emails)

    def clear_sent_emails(self) -> None:
        """Clear sent emails for test isolation."""
        self._sent_emails.clear()


class RealAuditService:
    """Real audit service for functional testing."""

    def __init__(self) -> None:
        self._audit_logs: list[FlextTypes.Core.Dict] = []

    def log_event(
        self,
        event_type: str,
        entity_id: str,
        details: FlextTypes.Core.Dict | None = None,
        user_id: str | None = None,
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Log audit event."""
        if not event_type or not entity_id:
            return FlextResult[FlextTypes.Core.Dict].fail(
                "Event type and entity ID required", error_code="VALIDATION_ERROR"
            )

        audit_entry: FlextTypes.Core.Dict = {
            "id": str(uuid.uuid4()),
            "event_type": event_type,
            "entity_id": entity_id,
            "details": details or {},
            "user_id": user_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self._audit_logs.append(audit_entry)
        return FlextResult[FlextTypes.Core.Dict].ok(audit_entry)

    def get_audit_logs(self) -> list[FlextTypes.Core.Dict]:
        """Get all audit logs for testing verification."""
        return list(self._audit_logs)

    def get_audit_logs_for_entity(self, entity_id: str) -> list[FlextTypes.Core.Dict]:
        """Get audit logs for specific entity."""
        return [log for log in self._audit_logs if log["entity_id"] == entity_id]

    def clear_audit_logs(self) -> None:
        """Clear audit logs for test isolation."""
        self._audit_logs.clear()


class FlextTestsFixtures:
    """Unified test fixtures for FLEXT ecosystem.

    Consolidates all fixture patterns into a single class interface.
    """

    # Note: Fixtures are functions decorated with @pytest.fixture,
    # so we can't directly assign them as class attributes.
    # Instead, we provide factory methods that return the fixture functions.

    @staticmethod
    def get_all_fixtures() -> FlextTypes.Core.Dict:
        """Get all available fixtures."""
        return {
            "benchmark_config": benchmark_config,
            "config_builder": config_builder,
            "failure_result": failure_result,
            "flext_config": flext_config,
            "flext_container": flext_container,
            "large_dataset": large_dataset,
            "nested_data": nested_data,
            "performance_data": performance_data,
            "result_chain": result_chain,
            "success_result": success_result,
            "temp_config_file": temp_config_file,
            "temp_directory": temp_directory,
            "test_command": test_command,
            "test_entity": test_entity,
            "test_value_object": test_value_object,
        }


__all__ = [
    "BenchmarkFixture",
    "FailingUserRepository",
    "FlextTestsFixtures",
    "InMemoryUserRepository",
    "RealAuditService",
    "RealEmailService",
    "benchmark_config",
    "config_builder",
    "failure_result",
    "flext_config",
    "flext_container",
    "large_dataset",
    "nested_data",
    "performance_data",
    "result_chain",
    "success_result",
    "temp_config_file",
    "temp_directory",
    "test_command",
    "test_entity",
    "test_value_object",
]
