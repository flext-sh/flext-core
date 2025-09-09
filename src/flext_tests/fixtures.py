"""Unified fixtures for flext-core tests using massive pytest ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Protocol, cast

from pydantic import Field

from flext_core import (
    FlextCommands,
    FlextConfig,
    FlextLogger,
    FlextResult,
    FlextTypes,
)

logger = FlextLogger(__name__)


class FlextTestsFixtures:
    """Unified test fixtures for FLEXT ecosystem.

    Consolidates all fixture patterns into a single class interface.
    Provides pytest fixtures, factory methods, and testing utilities
    for comprehensive test coverage across the entire ecosystem.
    """

    # === Protocol Definitions ===

    class BenchmarkFixture(Protocol):  # pragma: no cover - typing only
        """Benchmark fixture protocol."""

        group: str

        def __call__(
            self, func: object, /, *args: object, **kwargs: object
        ) -> object: ...

        def pedantic(
            self, func: object, /, *args: object, **kwargs: object
        ) -> object: ...

    # === Factory Classes ===

    class PerformanceDataFactory:
        """Performance data factory for testing."""

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
        """Error simulation factory for testing."""

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
        """Sequence factory for testing."""

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
        """Factory registry for testing."""

        def __init__(self) -> None:
            self.factories: FlextTypes.Core.Dict = {}

        def register(self, name: str, factory: object) -> None:
            """Register a factory."""
            self.factories[name] = factory

        def get(self, name: str) -> object:
            """Get a factory."""
            return self.factories.get(name)

    class FlextConfigFactory:
        """Config factory for testing."""

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

    # === Command Classes ===

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
        """Command factory for testing."""

        @staticmethod
        def create_test_command(
            config: object | None = None,
        ) -> FlextTestsFixtures.TestCommand:
            """Create test command."""
            return FlextTestsFixtures.TestCommand(config=config or {})

        @staticmethod
        def create_batch_command(
            config: object | None = None,
        ) -> FlextTestsFixtures.BatchCommand:
            """Create batch command."""
            return FlextTestsFixtures.BatchCommand(config=config or {})

        @staticmethod
        def create_validation_command(
            config: object | None = None,
        ) -> FlextTestsFixtures.ValidationCommand:
            """Create validation command."""
            return FlextTestsFixtures.ValidationCommand(config=config or {})

        @staticmethod
        def create_processing_command(
            config: object | None = None,
        ) -> FlextTestsFixtures.ProcessingCommand:
            """Create processing command."""
            return FlextTestsFixtures.ProcessingCommand(config=config or {})

    # === Service Classes ===

    class AsyncTestService:
        """Test service for async testing."""

        def __init__(self) -> None:
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
            self, coros: FlextTypes.Core.List
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
            self._session_data[session_id] = {
                "created_at": datetime.now(UTC),
                "data": {},
            }

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

    # === Repository Classes ===

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
            return FlextResult[list[FlextTypes.Core.Dict]].ok(
                list(self._users.values())
            )

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
                    "Email recipient and subject required",
                    error_code="VALIDATION_ERROR",
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

        def get_audit_logs_for_entity(
            self, entity_id: str
        ) -> list[FlextTypes.Core.Dict]:
            """Get audit logs for specific entity."""
            return [log for log in self._audit_logs if log["entity_id"] == entity_id]

        def clear_audit_logs(self) -> None:
            """Clear audit logs for test isolation."""
            self._audit_logs.clear()

    # === Fixture Methods (Static Factory Methods for Fixture Creation) ===

    # === Convenience Factory Methods ===

    @staticmethod
    def create_performance_data() -> FlextTypes.Core.Dict:
        """Create performance data for testing."""
        return FlextTestsFixtures.PerformanceDataFactory.create_large_payload()

    @staticmethod
    def create_error_scenario(error_type: str) -> FlextTypes.Core.Dict:
        """Create error scenario for testing."""
        return FlextTestsFixtures.ErrorSimulationFactory.create_error_scenario(
            error_type
        )

    @staticmethod
    def create_sequence(
        length: int = 10, prefix: str = ""
    ) -> FlextTypes.Core.StringList:
        """Create sequence for testing."""
        return FlextTestsFixtures.SequenceFactory.create_sequence(
            length=length, prefix=prefix
        )

    @staticmethod
    def create_timeline_events(count: int = 10) -> list[FlextTypes.Core.Dict]:
        """Create timeline events for testing."""
        return FlextTestsFixtures.SequenceFactory.create_timeline_events(count=count)

    @staticmethod
    def create_test_config() -> FlextConfig:
        """Create test config."""
        return FlextTestsFixtures.FlextConfigFactory.create_test_config()

    @staticmethod
    def create_in_memory_repository() -> FlextTestsFixtures.InMemoryUserRepository:
        """Create in-memory repository for testing."""
        return FlextTestsFixtures.InMemoryUserRepository()

    @staticmethod
    def create_email_service() -> FlextTestsFixtures.RealEmailService:
        """Create email service for testing."""
        return FlextTestsFixtures.RealEmailService()

    @staticmethod
    def create_audit_service() -> FlextTestsFixtures.RealAuditService:
        """Create audit service for testing."""
        return FlextTestsFixtures.RealAuditService()


# Export only the unified class
__all__ = [
    "FlextTestsFixtures",
]
