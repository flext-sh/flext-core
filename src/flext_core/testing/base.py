"""Base test classes for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides base test classes that implement common patterns
used across all FLEXT projects.
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from uuid import UUID

if TYPE_CHECKING:
    from pydantic import BaseModel
else:
    # NO FALLBACKS - SEMPRE usar implementações originais conforme instrução
    from pydantic import BaseModel

from flext_core.domain.shared_types import ServiceResult

# Pytest import handled conditionally to avoid DEP004 issues
if TYPE_CHECKING:
    import pytest
else:
    # NO FALLBACKS - SEMPRE usar implementações originais conforme instrução
    import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


class BaseTestCase(unittest.TestCase):
    """Base test case for synchronous tests with common patterns."""

    # Configuration - override in subclasses
    maxDiff = None  # Show full diffs

    def setUp(self) -> None:
        """Set up test case with common configuration."""
        super().setUp()
        # Reset any global state
        self._cleanup_global_state()

    def tearDown(self) -> None:
        """Clean up after test case."""
        super().tearDown()
        self._cleanup_global_state()

    def _cleanup_global_state(self) -> None:
        """Clean up any global state that could affect tests."""
        # Override in subclasses if needed

    def assert_service_result_success(
        self,
        result: Any,
        expected_data: Any = None,
    ) -> None:
        """Assert that a ServiceResult is successful."""
        if not isinstance(result, ServiceResult):
            self.fail(f"Expected ServiceResult, got {type(result)}")

        if not result.success:
            self.fail(f"Expected success, got error: {result.error}")

        if expected_data is not None and result.data != expected_data:
            self.fail(f"Expected data {expected_data}, got {result.data}")

    def assert_service_result_failure(
        self,
        result: Any,
        expected_error: str | None = None,
    ) -> None:
        """Assert that a ServiceResult is a failure."""
        if not isinstance(result, ServiceResult):
            self.fail(f"Expected ServiceResult, got {type(result)}")

        if result.success:
            self.fail(f"Expected failure, got success: {result.data}")

        if expected_error is not None and expected_error not in str(result.error):
            self.fail(
                f"Expected error containing '{expected_error}', got '{result.error}'",
            )

    def assert_entity_id_valid(self, entity_id: Any) -> None:
        """Assert that an entity ID is valid UUID."""
        if isinstance(entity_id, UUID):
            return  # Already valid UUID

        try:
            UUID(str(entity_id))
        except ValueError as e:
            self.fail(f"Invalid entity ID: {entity_id} - {e}")

    def assert_timestamp_recent(
        self,
        timestamp: Any,
        max_age_seconds: float = 60.0,
    ) -> None:
        """Assert that timestamp is recent (within max_age_seconds)."""
        if not isinstance(timestamp, datetime):
            self.fail(f"Expected datetime, got {type(timestamp)}")

        now = datetime.now(UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        age_seconds = (now - timestamp).total_seconds()
        if age_seconds > max_age_seconds:
            self.fail(
                f"Timestamp {timestamp} is {age_seconds}s old, max allowed: {max_age_seconds}s",
            )

    def assert_config_valid(self, config: Any) -> None:
        """Assert that configuration object is valid."""
        # NO FALLBACKS - SEMPRE usar implementações originais conforme instrução
        if not isinstance(config, BaseModel):
            self.fail(f"Expected Pydantic model, got {type(config)}")

        # Validate by accessing model_fields
        try:
            _ = config.model_dump()
        except (ValueError, TypeError, AttributeError) as e:
            self.fail(f"Configuration validation failed: {e}")


class AsyncTestCase(BaseTestCase):
    """Base test case for async tests with proper event loop management."""

    def setUp(self) -> None:
        """Set up async test case with event loop."""
        super().setUp()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        """Clean up async test case."""
        try:
            if self.loop and not self.loop.is_closed():
                # Cancel all pending tasks
                pending = [t for t in asyncio.all_tasks(self.loop) if not t.done()]
                for task in pending:
                    task.cancel()

                if pending:
                    self.loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True),
                    )

                self.loop.close()
        finally:
            super().tearDown()

    def run_async(self, coro: Any) -> Any:
        """Run async coroutine in test event loop."""
        return self.loop.run_until_complete(coro)

    async def assert_raises_async(
        self,
        exception_class: type[Exception],
        coro: Any,
    ) -> None:
        """Assert that async coroutine raises specific exception."""
        # pytest is always available in test environment
        with pytest.raises(exception_class):
            await coro


def event_loop() -> Generator[asyncio.AbstractEventLoop]:
    """Create event loop for pytest-asyncio."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        # Cancel all pending tasks
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()

        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        loop.close()


# Apply pytest.fixture decorator only if pytest is available
if pytest is not None:
    event_loop = pytest.fixture(event_loop)


# Pytest helpers for async testing
def pytest_run_async(coro: Any) -> Any:
    """Run async coroutine in pytest context."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)
