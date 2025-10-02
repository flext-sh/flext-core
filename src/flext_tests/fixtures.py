"""Unified fixtures for flext-core tests using massive pytest ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import string
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast, override

from pydantic import Field

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextTypes,
)


# Lazy logger initialization to avoid configuration issues
class _LoggerSingleton:
    """Singleton logger instance."""

    _instance: FlextLogger | None = None

    @classmethod
    def get_logger(cls) -> FlextLogger:
        """Get logger instance with lazy initialization."""
        if cls._instance is None:
            cls._instance = FlextLogger(__name__)
        return cls._instance


def get_logger() -> FlextLogger:
    """Get logger instance with lazy initialization."""
    return _LoggerSingleton.get_logger()


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

        def pedantic(self, func: object, /, *args: object, **kwargs: object) -> object:
            """Pedantic method for benchmark fixture."""

    # === Factory Classes ===

    class FactoryRegistry:
        """Factory registry for testing."""

        @override
        def __init__(self) -> None:
            """Initialize factory registry."""
            self.factories: FlextTypes.Core.Dict = {}

        def register(self, name: str, factory: object) -> None:
            """Register a factory."""
            self.factories[name] = factory

        def get(self, name: str) -> object:
            """Get a factory.

            Returns:
                object: Factory object or None if not found

            """
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
        def create_timeline_events(
            count: int = 5,
        ) -> list[FlextTypes.Core.Dict]:
            """Create timeline events using ``list[FlextTypes.Core.Dict]``.

            Args:
                count: Number of events to create

            Returns:
                list[FlextTypes.Core.Dict]: Event payloads leveraging the official alias

            """
            base_time = datetime.now(UTC)
            events: list[FlextTypes.Core.Dict] = []

            for i in range(count):
                event: FlextTypes.Core.Dict = {
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
        def create_large_payload(size_mb: float = 1.0) -> FlextTypes.Core.Dict:
            """Create a large payload via the ``FlextTypes.Core.Dict`` alias.

            Args:
                size_mb: Size in megabytes

            Returns:
                FlextTypes.Core.Dict: Large payload data using the official alias

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
        def create_nested_structure(depth: int = 3) -> FlextTypes.Core.Dict:
            """Create a nested data structure with ``FlextTypes.Core.Dict``.

            Args:
                depth: Nesting depth

            Returns:
                FlextTypes.Core.Dict: Nested payload expressed with the alias

            """

            def create_nested(current_depth: int) -> FlextTypes.Core.Dict:
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
            """Create a timeout error for testing.

            Returns:
                TimeoutError: Simulated timeout error instance

            """
            return TimeoutError("Simulated timeout error for testing")

        @staticmethod
        def create_connection_error() -> ConnectionError:
            """Create a connection error for testing.

            Returns:
                ConnectionError: Simulated connection error instance

            """
            return ConnectionError("Simulated connection error for testing")

        @staticmethod
        def create_validation_error() -> ValueError:
            """Create a validation error for testing.

            Returns:
                ValueError: Simulated validation error instance

            """
            return ValueError("Simulated validation error for testing")

        @staticmethod
        def create_error_scenario(error_type: str) -> FlextTypes.Core.Dict:
            """Create an error scenario using ``FlextTypes.Core.Dict``.

            Args:
                error_type: Type of error scenario

            Returns:
                FlextTypes.Core.Dict: Error scenario payload with official alias

            """
            scenarios: dict[str, FlextTypes.Core.Dict] = {
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
                return scenarios[error_type]
            return {
                "type": "unknown",
                "message": "Unknown error",
                "code": "UNK_001",
                "details": {},
            }

    class SessionTestService:
        """Service for managing test sessions."""

        @override
        def __init__(self) -> None:
            """Initialize session service."""
            self._data: dict[str, FlextTypes.Core.Dict] = {}

        def create_session(
            self,
            session_id: str,
            data: FlextTypes.Core.Dict | None = None,
        ) -> FlextTypes.Core.Dict:
            """Create a new session using ``FlextTypes.Core.Dict``.

            Args:
                session_id: Session identifier
                data: Initial session data

            Returns:
                FlextTypes.Core.Dict: Session payload stored with the official alias

            """
            session_data: FlextTypes.Core.Dict = data or {}
            session_data["id"] = session_id
            session_data["created_at"] = "2024-01-01T00:00:00Z"
            session: FlextTypes.Core.Dict = {
                "id": session_id,
                "created_at": "2024-01-01T00:00:00Z",
                "data": session_data,
            }
            self._data[session_id] = session
            return session

        def get_session(self, session_id: str) -> FlextTypes.Core.Dict | None:
            """Get session by ID using ``FlextTypes.Core.Dict``.

            Args:
                session_id: Session identifier

            Returns:
                FlextTypes.Core.Dict | None: Stored session data if present

            """
            return self._data.get(session_id)

        def update_session(
            self,
            session_id: str,
            data: FlextTypes.Core.Dict,
        ) -> bool:
            """Update session data expressed as ``FlextTypes.Core.Dict``.

            Args:
                session_id: Session identifier
                data: Data to update using the official alias

            Returns:
                True if updated, False otherwise

            """
            if session_id in self._data:
                session = self._data[session_id]
                if "data" in session and isinstance(session["data"], dict):
                    cast("dict[str, object]", session["data"]).update(data)
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
            """Create test configuration.

            Returns:
                FlextConfig: Test configuration instance

            """
            return FlextConfig.create(
                environment="test",
                debug=True,
            )

        @staticmethod
        def create_development_config() -> FlextConfig:
            """Create development configuration.

            Returns:
                FlextConfig: Development configuration instance

            """
            return FlextConfig.create(
                environment=FlextConstants.Environment.ConfigEnvironment.DEVELOPMENT,
                debug=True,
            )

        @staticmethod
        def create_production_config() -> FlextConfig:
            """Create production configuration.

            Returns:
                FlextConfig: Production configuration instance

            """
            return FlextConfig.create(environment="production", debug=False)

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

    class TestService:
        """Test service for testing."""

        @override
        def __init__(self) -> None:
            """Initialize test service."""
            self._executor = FlextTestsFixtures.Executor()

        def process(self, data: object) -> FlextTypes.Core.Dict:
            """Process data hronously."""
            time.sleep(0.001)  # Simulate work
            return {"processed": True, "original": data}

        def validate(self, data: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Validate data hronously."""
            time.sleep(0.001)  # Simulate work
            has_required = "required_field" in data
            return {"valid": has_required}

        def transform(self, data: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Transform data hronously."""
            time.sleep(0.001)  # Simulate work
            return {
                "transformed": True,
                "output": f"transformed_{data.get('input', '')}",
            }

        def fail_operation(self) -> FlextResult[object]:
            """Simulate failure."""
            time.sleep(0.001)
            return FlextResult[object].fail("operation_failed")

    class Executor:
        """executor for testing."""

        @override
        def __init__(self) -> None:
            """Initialize executor."""
            self._running: bool = False
            self._tasks: list[object] = []

        def start(self) -> None:
            """Start executor."""
            self._running = True

        def stop(self) -> None:
            """Stop executor and cancel tasks."""
            self._running = False
            # Synchronous stub - no async operations to cancel

        def execute(self, coro: object) -> object:
            """Execute coroutine (sync stub)."""
            if not self._running:
                self.start()

            # Synchronous stub - return the input object
            # Real async operations should be converted to sync alternatives
            self._tasks.append(coro)
            return coro

        def execute_batch(
            self,
            coros: FlextTypes.Core.List,
        ) -> FlextTypes.Core.List:
            """Execute multiple coroutines in batch (sync stub)."""
            if not self._running:
                self.start()

            # Synchronous stub - return the input coroutines as results
            # Real async operations should be converted to sync alternatives
            for coro in coros:
                self._tasks.append(coro)
            return list(coros)

        def cleanup(self) -> None:
            """Cleanup completed tasks."""
            self._tasks.clear()

    class TestDataService:
        """Test data service for testing scenarios."""

        @override
        def __init__(self) -> None:
            """Initialize test data service."""
            self._data_store: FlextTypes.Core.Dict = {}
            self._call_count: int = 0
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

        @override
        def __init__(self, user_id: str, name: str, email: str) -> None:
            """Initialize test user."""
            self.id = user_id
            self.name = name
            self.email = email
            self.created_at = datetime.now(UTC)

    class TestOrder:
        """Test order model."""

        @override
        def __init__(self, order_id: str, user_id: str, amount: float) -> None:
            """Initialize test order."""
            self.id = order_id
            self.user_id = user_id
            self.amount = amount
            self.status = "pending"
            self.created_at = datetime.now(UTC)

    class TestProduct:
        """Test product model."""

        @override
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
        """Create test data using the ``FlextTypes.Core.Dict`` alias."""
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
            data: FlextTypes.Core.Dict | None = None,
        ) -> FlextTypes.Core.Dict:
            """Create processing command using ``FlextTypes.Core.Dict``."""
            command_data: FlextTypes.Core.Dict = data or {"test": "value"}

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
