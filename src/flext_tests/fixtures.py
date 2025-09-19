"""Unified fixtures for flext-core tests using massive pytest ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import string
import uuid
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from pydantic import Field

from flext_core import (
    FlextConfig,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextTypes,
)

logger = FlextLogger(__name__)


class FlextTestsFixtures:
    """Unified test fixtures for FLEXT ecosystem.

    Consolidates all fixture patterns into a single class interface.
    Provides pytest fixtures, factory methods, and testing utilities
    for good test coverage across the ecosystem.
    """

    # === Protocol Definitions ===

    class BenchmarkFixture(Protocol):  # pragma: no cover - typing only
        """Benchmark fixture protocol."""

        group: str

        def __call__(self, func: object, /, *args: object, **kwargs: object) -> object:
            """Call method for benchmark fixture."""
            ...

        def pedantic(self, func: object, /, *args: object, **kwargs: object) -> object:
            """Pedantic method for benchmark fixture."""
            ...

    # === Factory Classes ===

    class FactoryRegistry:
        """Factory registry for testing."""

        def __init__(self) -> None:
            """Initialize factory registry."""
            self.factories: FlextTypes.Core.Dict = {}

        def register(self, name: str, factory: object) -> None:
            """Register a factory."""
            self.factories[name] = factory

        def get(self, name: str) -> object:
            """Get a factory."""
            return self.factories.get(name)

    class SequenceFactory:
        """Factory for creating sequence test data."""

        @staticmethod
        def create_sequence(
            length: int = 10,
            prefix: str = "",
            count: int | None = None,
        ) -> list[str]:
            """Create a sequence of test data.

            Args:
                length: Length of the sequence (default 10)
                prefix: Prefix for each item
                count: Override for length if provided

            Returns:
                List of string items

            """
            actual_length = count if count is not None else length
            if prefix:
                return [f"{prefix}_{i}" for i in range(actual_length)]
            return [str(i) for i in range(actual_length)]

        @staticmethod
        def create_timeline_events(count: int = 5) -> list[dict[str, object]]:
            """Create timeline events for testing.

            Args:
                count: Number of events to create

            Returns:
                List of event dictionaries

            """
            base_time = datetime.now(UTC)
            events = []

            for i in range(count):
                event = {
                    "id": f"event_{i}",
                    "timestamp": base_time + timedelta(hours=i),
                    "type": "test_event",
                    "data": {"index": i},
                }
                events.append(event)

            return events

    class PerformanceDataFactory:
        """Factory for creating performance test data."""

        @staticmethod
        def create_large_payload(size_mb: float = 1.0) -> dict[str, object]:
            """Create a large payload for performance testing.

            Args:
                size_mb: Size in megabytes

            Returns:
                Dictionary with large data

            """
            # Calculate approximate size in bytes
            size_bytes = int(size_mb * 1024 * 1024)

            # Create large string data
            chunk_size = 1024  # 1KB chunks
            num_chunks = size_bytes // chunk_size

            large_string = (string.ascii_letters * 20)[:chunk_size] * num_chunks

            return {
                "data": large_string,
                "size_mb": size_mb,
                "chunks": num_chunks,
            }

        @staticmethod
        def create_nested_structure(depth: int = 3) -> dict[str, object]:
            """Create a nested data structure for testing.

            Args:
                depth: Nesting depth

            Returns:
                Nested dictionary

            """

            def create_nested(current_depth: int) -> dict[str, object]:
                if current_depth <= 1:
                    return {"value": f"depth_{current_depth}"}

                return {
                    "value": f"depth_{current_depth}",
                    "nested": create_nested(current_depth - 1),
                    "data": {"depth": current_depth},
                }

            return create_nested(depth)

    class ErrorSimulationFactory:
        """Factory for creating error scenarios."""

        @staticmethod
        def create_timeout_error() -> TimeoutError:
            """Create a timeout error for testing."""
            return TimeoutError("Simulated timeout error for testing")

        @staticmethod
        def create_connection_error() -> ConnectionError:
            """Create a connection error for testing."""
            return ConnectionError("Simulated connection error for testing")

        @staticmethod
        def create_validation_error() -> ValueError:
            """Create a validation error for testing."""
            return ValueError("Simulated validation error for testing")

        @staticmethod
        def create_error_scenario(error_type: str) -> dict[str, object]:
            """Create an error scenario for testing.

            Args:
                error_type: Type of error scenario

            Returns:
                Error scenario dictionary

            """
            scenarios = {
                "ValidationError": {
                    "type": "validation",
                    "message": "Validation failed",
                    "code": "VAL_001",
                    "details": {"field": "test_field", "reason": "invalid"},
                },
                "NetworkError": {
                    "type": "network",
                    "message": "Network error",
                    "code": "NET_001",
                    "details": {"retry_after": 60},
                },
                "TimeoutError": {
                    "type": "timeout",
                    "message": "Operation timed out",
                    "code": 408,
                    "details": {"timeout_seconds": 30},
                },
            }

            if error_type in scenarios:
                return cast("dict[str, object]", scenarios[error_type])
            return {
                "type": "unknown",
                "message": "Unknown error",
                "code": "UNK_001",
                "details": {},
            }

    class SessionTestService:
        """Service for managing test sessions."""

        def __init__(self) -> None:
            """Initialize session service."""
            self._data: dict[str, dict[str, object]] = {}

        def create_session(
            self,
            session_id: str,
            data: dict[str, object] | None = None,
        ) -> dict[str, object]:
            """Create a new session.

            Args:
                session_id: Session identifier
                data: Initial session data

            Returns:
                Session data

            """
            session_data = data or {}
            session_data["id"] = session_id
            session_data["created_at"] = "2024-01-01T00:00:00Z"
            session: dict[str, object] = {
                "id": session_id,
                "created_at": "2024-01-01T00:00:00Z",
                "data": session_data,
            }
            self._data[session_id] = session
            return session

        def get_session(self, session_id: str) -> dict[str, object] | None:
            """Get session by ID.

            Args:
                session_id: Session identifier

            Returns:
                Session data or None

            """
            return self._data.get(session_id)

        def update_session(self, session_id: str, data: dict[str, object]) -> bool:
            """Update session data.

            Args:
                session_id: Session identifier
                data: Data to update

            Returns:
                True if updated, False otherwise

            """
            if session_id in self._data:
                session = self._data[session_id]
                if "data" in session and isinstance(session["data"], dict):
                    session["data"].update(data)
                else:
                    session["data"] = data
                return True
            return False

        def delete_session(self, session_id: str) -> bool:
            """Delete a session.

            Args:
                session_id: Session identifier

            Returns:
                True if deleted, False otherwise

            """
            if session_id in self._data:
                del self._data[session_id]
                return True
            return False

        def cleanup_sessions(self, prefix: str | None = None) -> int:
            """Clean up sessions.

            Args:
                prefix: Optional prefix to filter sessions

            Returns:
                Number of sessions cleaned

            """
            if prefix:
                to_delete = [k for k in self._data if k.startswith(prefix)]
                for key in to_delete:
                    del self._data[key]
                return len(to_delete)

            count = len(self._data)
            self._data.clear()
            return count

    class FlextConfigFactory:
        """Config factory for testing."""

        @staticmethod
        def create_test_config() -> FlextConfig:
            """Create test configuration."""
            return FlextConfig.create(
                constants={"environment": "test", "debug": True},
            ).unwrap()

        @staticmethod
        def create_development_config() -> FlextConfig:
            """Create development configuration."""
            return FlextConfig.create(
                constants={"environment": "development", "debug": True},
            ).unwrap()

        @staticmethod
        def create_production_config() -> FlextConfig:
            """Create production configuration."""
            return FlextConfig.create(
                constants={"environment": "production", "debug": False},
            ).unwrap()

    # === Command Classes ===

    class TestCommand(FlextModels.Command):
        """Test command for fixtures."""

        command_type: str = Field(default="test", description="Type of command")
        config: object = Field(default_factory=dict)

    class BatchCommand(FlextModels.Command):
        """Batch command for fixtures."""

        command_type: str = Field(default="batch", description="Type of command")
        config: object = Field(default_factory=dict)

    class ValidationCommand(FlextModels.Command):
        """Validation command for fixtures."""

        command_type: str = Field(default="validation", description="Type of command")
        config: object = Field(default_factory=dict)

    # === Service Classes ===

    class AsyncTestService:
        """Test service for async testing."""

        def __init__(self) -> None:
            """Initialize async test service."""
            self._executor = FlextTestsFixtures.AsyncExecutor()

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
            return {
                "transformed": True,
                "output": f"transformed_{data.get('input', '')}",
            }

        async def fail_operation(self) -> FlextResult[object]:
            """Simulate async failure."""
            await asyncio.sleep(0.001)
            return FlextResult[object].fail("async_operation_failed")

    class AsyncExecutor:
        """Async executor for testing."""

        def __init__(self) -> None:
            """Initialize async executor."""
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

        async def execute_batch(
            self,
            coros: FlextTypes.Core.List,
        ) -> FlextTypes.Core.List:
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

    class TestDataService:
        """Test data service for testing scenarios."""

        def __init__(self) -> None:
            """Initialize test data service."""
            self._data_store: dict[str, object] = {}
            self._call_count = 0
            self._operations_log: list[str] = []

        def store(self, key: str, value: object) -> None:
            """Store data and log operation."""
            self._data_store[key] = value
            self._operations_log.append(f"store:{key}")

        def retrieve(self, key: str) -> object | None:
            """Retrieve data and log operation."""
            self._call_count += 1
            self._operations_log.append(f"retrieve:{key}")
            return self._data_store.get(key)

        def exists(self, key: str) -> bool:
            """Check if key exists."""
            self._operations_log.append(f"exists:{key}")
            return key in self._data_store

        def delete(self, key: str) -> bool:
            """Delete key and return success status."""
            self._operations_log.append(f"delete:{key}")
            if key in self._data_store:
                del self._data_store[key]
                return True
            return False

        def get_all_keys(self) -> list[str]:
            """Get all stored keys."""
            return list(self._data_store.keys())

        def get_call_count(self) -> int:
            """Get number of retrieve calls made."""
            return self._call_count

        def get_operations_log(self) -> list[str]:
            """Get log of all operations performed."""
            return self._operations_log.copy()

        def reset(self) -> None:
            """Reset service state completely."""
            self._data_store.clear()
            self._call_count = 0
            self._operations_log.clear()

    # === Test Data Models ===

    class TestUser:
        """Test user model."""

        def __init__(self, user_id: str, name: str, email: str) -> None:
            """Initialize test user."""
            self.id = user_id
            self.name = name
            self.email = email
            self.created_at = datetime.now(UTC)

    class TestOrder:
        """Test order model."""

        def __init__(self, order_id: str, user_id: str, amount: float) -> None:
            """Initialize test order."""
            self.id = order_id
            self.user_id = user_id
            self.amount = amount
            self.status = "pending"
            self.created_at = datetime.now(UTC)

    class TestProduct:
        """Test product model."""

        def __init__(self, product_id: str, name: str, price: float) -> None:
            """Initialize test product."""
            self.id = product_id
            self.name = name
            self.price = price
            self.in_stock = True

    # === Utility Methods ===

    @staticmethod
    def generate_test_id() -> str:
        """Generate unique test ID."""
        return str(uuid.uuid4())

    @staticmethod
    def create_test_user(name: str = "Test User") -> TestUser:
        """Create test user."""
        return FlextTestsFixtures.TestUser(
            user_id=FlextTestsFixtures.generate_test_id(),
            name=name,
            email=f"{name.lower().replace(' ', '.')}@test.com",
        )

    @staticmethod
    def create_test_order(user_id: str, amount: float = 100.0) -> TestOrder:
        """Create test order."""
        return FlextTestsFixtures.TestOrder(
            order_id=FlextTestsFixtures.generate_test_id(),
            user_id=user_id,
            amount=amount,
        )

    @staticmethod
    def create_test_product(
        name: str = "Test Product",
        price: float = 50.0,
    ) -> TestProduct:
        """Create test product."""
        return FlextTestsFixtures.TestProduct(
            product_id=FlextTestsFixtures.generate_test_id(),
            name=name,
            price=price,
        )

    @staticmethod
    def create_test_data() -> FlextTypes.Core.Dict:
        """Create test data dictionary."""
        return {
            "id": FlextTestsFixtures.generate_test_id(),
            "name": "test_data",
            "value": 42,
            "nested": {
                "key": "value",
                "number": 123,
            },
            "list": [1, 2, 3, "test"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def create_failure_result(error_message: str = "test_error") -> FlextResult[object]:
        """Create failure result for testing."""
        return FlextResult[object].fail(error_message)

    @staticmethod
    def create_success_result(data: object = "test_data") -> FlextResult[object]:
        """Create success result for testing."""
        return FlextResult[object].ok(data)

    # === Factory Classes ===

    class CommandFactory:
        """Factory for creating test commands."""

        @staticmethod
        def create_processing_command(
            data: dict[str, object] | None = None,
        ) -> dict[str, object]:
            """Create processing command for testing."""
            command_data = data or {"test": "value"}

            return {
                "command": "process",
                "data": command_data,
                "timestamp": datetime.now(UTC).isoformat(),
                "config": command_data,  # Use the data as config
            }


# Export only the unified class
__all__ = [
    "FlextTestsFixtures",
]
