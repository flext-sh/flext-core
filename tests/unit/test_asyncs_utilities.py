"""Comprehensive tests for flext_tests.asyncs module to achieve 100% coverage.

Real functional tests using actual async functionality without mocks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from contextlib import AbstractAsyncContextManager
from typing import Protocol, cast

import pytest

from flext_core import FlextTypes, T_co
from flext_tests import FlextTestsAsyncs, FlextTestsMatchers


class AsyncMockCallable(Protocol[T_co]):
    """Protocol for async mock callables."""

    def __call__(self, *args: object, **kwargs: object) -> Awaitable[T_co]:
        """Call the async mock function."""
        ...


# Type alias for async context managers
AsyncTestContext = AbstractAsyncContextManager[object]


class TestAsyncTestUtils:
    """Test FlextTestsAsyncs with real async functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_condition_success(self) -> None:
        """Test waiting for a condition that becomes true."""
        counter = 0

        def condition() -> bool:
            nonlocal counter
            counter += 1
            return counter >= 3

        await FlextTestsAsyncs.wait_for_condition(
            condition,
            timeout_seconds=1.0,
            poll_interval=0.1,
        )
        assert counter >= 3

    @pytest.mark.asyncio
    async def test_wait_for_condition_timeout(self) -> None:
        """Test timeout when condition never becomes true."""

        def always_false() -> bool:
            return False

        with pytest.raises(TimeoutError, match="Condition not met"):
            await FlextTestsAsyncs.wait_for_condition(
                always_false,
                timeout_seconds=0.2,
                poll_interval=0.05,
            )

    @pytest.mark.asyncio
    async def test_wait_for_condition_async(self) -> None:
        """Test waiting for an async condition."""
        counter = 0

        async def async_condition() -> bool:
            nonlocal counter
            counter += 1
            await asyncio.sleep(0.01)
            return counter >= 2

        await FlextTestsAsyncs.wait_for_condition(
            async_condition,
            timeout_seconds=1.0,
        )
        assert counter >= 2

    @pytest.mark.asyncio
    async def test_wait_for_condition_with_exception(self) -> None:
        """Test condition that raises exceptions initially."""
        attempts = 0

        def flaky_condition() -> bool:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                msg = "Not ready yet"
                raise ValueError(msg)
            return True

        await FlextTestsAsyncs.wait_for_condition(
            flaky_condition,
            timeout_seconds=1.0,
            poll_interval=0.1,
        )
        assert attempts >= 3

    @pytest.mark.asyncio
    async def test_run_with_timeout_success(self) -> None:
        """Test running coroutine within timeout."""

        async def quick_task() -> str:
            await asyncio.sleep(0.1)
            return "completed"

        result = await FlextTestsAsyncs.run_with_timeout(
            quick_task(),
            timeout_seconds=1.0,
        )
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_run_with_timeout_failure(self) -> None:
        """Test timeout when coroutine takes too long."""

        async def slow_task() -> str:
            await asyncio.sleep(2.0)
            return "never reached"

        with pytest.raises(TimeoutError, match="timed out after"):
            await FlextTestsAsyncs.run_with_timeout(
                slow_task(),
                timeout_seconds=0.1,
            )

    @pytest.mark.asyncio
    async def test_run_concurrently_success(self) -> None:
        """Test running multiple coroutines concurrently."""

        async def task(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        results = await FlextTestsAsyncs.run_concurrently(
            [
                task(1),
                task(2),
                task(3),
            ]
        )
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_run_concurrently_empty(self) -> None:
        """Test running concurrently with no coroutines."""
        results: FlextTypes.Core.List = await FlextTestsAsyncs.run_concurrently([])
        assert results == []

    @pytest.mark.asyncio
    async def test_run_concurrently_with_exceptions(self) -> None:
        """Test running concurrently with some failures."""

        async def good_task(value: int) -> int:
            return value

        async def bad_task() -> None:
            msg = "Task failed"
            raise ValueError(msg)

        results = await FlextTestsAsyncs.run_concurrently(
            [good_task(1), bad_task(), good_task(2)],
            return_exceptions=True,
        )
        # Only successful results should be returned
        assert results == [1, 2]

    @pytest.mark.asyncio
    async def test_run_concurrently_raise_on_exception(self) -> None:
        """Test that exceptions are raised when return_exceptions=False."""

        async def failing_task() -> None:
            msg = "Failed"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Failed"):
            await FlextTestsAsyncs.run_concurrently(
                [failing_task()],
                return_exceptions=False,
            )

    @pytest.mark.asyncio
    async def test_run_with_retry(self) -> None:
        """Test running with retry logic."""
        attempts = 0

        async def flaky_task() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                msg = "Not ready"
                raise ValueError(msg)
            return "success"

        # This should succeed after retries - call function instead of coroutine
        result = await FlextTestsMatchers.run_with_timeout(
            flaky_task,
            timeout_seconds=1.0,
        )
        assert result == "success"


class TestAsyncMockUtils:
    """Test AsyncMockUtils functionality."""

    @pytest.mark.asyncio
    async def test_create_async_mock_with_return_value(self) -> None:
        """Test creating async mock with return value."""
        mock = cast(
            "AsyncMockCallable[str]",
            FlextTestsAsyncs.create_async_mock(return_value="test_result"),
        )
        result = await mock()
        assert result == "test_result"

    @pytest.mark.asyncio
    async def test_create_async_mock_with_side_effect_list(self) -> None:
        """Test async mock with list of side effects."""
        mock = cast(
            "AsyncMockCallable[int]",
            FlextTestsAsyncs.create_async_mock_with_side_effect(side_effect=[1, 2, 3]),
        )

        assert await mock() == 1
        assert await mock() == 2
        assert await mock() == 3

    @pytest.mark.asyncio
    async def test_create_async_mock_with_side_effect_exception(self) -> None:
        """Test async mock that raises exception."""
        mock = cast(
            "AsyncMockCallable[object]",
            FlextTestsAsyncs.create_async_mock(side_effect=ValueError("Test error")),
        )

        with pytest.raises(ValueError, match="Test error"):
            await mock()

    @pytest.mark.asyncio
    async def test_create_async_mock_with_callable_side_effect(self) -> None:
        """Test async mock with callable side effect."""
        call_count = 0

        async def side_effect_func(*_args: object, **_kwargs: object) -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        mock = cast(
            "AsyncMockCallable[int]",
            FlextTestsAsyncs.create_async_mock_with_side_effect(
                side_effect=side_effect_func
            ),
        )

        assert await mock() == 1
        assert await mock() == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_create_delayed_response(self) -> None:
        """Test creating delayed response mock."""
        start_time = time.time()
        mock = FlextTestsAsyncs.create_delayed_async_mock(
            return_value="result",
            delay=0.1,
        )

        result = await mock()
        elapsed = time.time() - start_time

        assert result == "result"
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_create_flaky_mock(self) -> None:
        """Test creating flaky mock that fails initially."""
        mock = cast(
            "AsyncMockCallable[str]",
            FlextTestsAsyncs.create_flaky_async_mock(
                success_value="success",
                failure_count=2,
                exception_type=ValueError,
            ),
        )

        # First two calls should fail
        with pytest.raises(ValueError):
            await mock()
        with pytest.raises(ValueError):
            await mock()

        # Third call should succeed
        result = await mock()
        assert result == "success"


class TestConcurrencyTestHelper:
    """Test ConcurrencyTestHelper functionality."""

    @pytest.mark.asyncio
    async def test_run_parallel_tasks(self) -> None:
        """Test running tasks in parallel."""

        async def task(n: object) -> object:
            await asyncio.sleep(0.01)
            if isinstance(n, int):
                return n * 2
            return 0

        results = await FlextTestsAsyncs.run_parallel_tasks(
            [1, 2, 3, 4],
            task,
        )
        assert results == [2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_run_parallel_tasks_empty(self) -> None:
        """Test running parallel tasks with empty inputs."""

        async def task(n: object) -> object:
            return n

        results = await FlextTestsAsyncs.run_parallel_tasks(
            [],
            task,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_test_race_condition(self) -> None:
        """Test race condition detection."""
        shared_counter = 0
        lock = asyncio.Lock()

        async def safe_increment() -> int:
            nonlocal shared_counter
            async with lock:
                temp = shared_counter
                await asyncio.sleep(0.001)
                shared_counter = temp + 1
                return shared_counter

        result = await FlextTestsAsyncs.test_race_condition_simple(
            safe_increment,
            concurrent_count=5,
        )

        # All results should be unique due to lock
        assert len(set(result)) == 5
        assert shared_counter == 5

    @pytest.mark.asyncio
    async def test_measure_concurrency_performance(self) -> None:
        """Test measuring concurrency performance."""

        async def task() -> None:
            await asyncio.sleep(0.01)

        metrics = await FlextTestsAsyncs.measure_concurrency_performance(
            task,
            iterations=3,
        )

        assert "total_time" in metrics
        assert "average_time" in metrics
        assert "throughput" in metrics
        avg_time = metrics["average_time"]
        throughput = metrics["throughput"]
        assert isinstance(avg_time, (int, float))
        assert avg_time > 0
        assert isinstance(throughput, (int, float))
        assert throughput > 0


class TestAsyncContextManager:
    """Test AsyncContextManager functionality."""

    @pytest.mark.asyncio
    async def test_create_test_context_success(self) -> None:
        """Test creating async context manager."""
        resource_created = False
        resource_cleaned = False

        async def setup() -> str:
            nonlocal resource_created
            resource_created = True
            return "test_resource"

        setup_coro = setup()

        async def teardown(resource: str) -> None:
            nonlocal resource_cleaned
            assert resource == "test_resource"
            resource_cleaned = True

        context = cast(
            "AsyncTestContext",
            FlextTestsAsyncs.create_test_context(
                setup_coro=setup_coro,
                teardown_func=teardown,
            ),
        )

        async with context as resource:
            assert resource == "test_resource"
            assert resource_created
            assert not resource_cleaned

        assert resource_cleaned

    @pytest.mark.asyncio
    async def test_create_test_context_with_exception(self) -> None:
        """Test context manager cleanup on exception."""
        cleaned_up = False

        async def setup() -> str:
            return "resource"

        setup_coro = setup()

        async def teardown(_resource: str) -> None:
            nonlocal cleaned_up
            cleaned_up = True

        context = cast(
            "AsyncTestContext",
            FlextTestsAsyncs.create_test_context(
                setup_coro=setup_coro,
                teardown_func=teardown,
            ),
        )

        error_msg = "Test error"
        with pytest.raises(ValueError, match=error_msg):
            async with context:
                raise ValueError(error_msg)

        # Teardown should still be called
        assert cleaned_up

    @pytest.mark.asyncio
    async def test_managed_resource_simple(self) -> None:
        """Test simple managed resource."""
        async with cast(
            "AsyncTestContext", FlextTestsAsyncs.managed_resource("test_value")
        ) as value:
            assert value == "test_value"

    @pytest.mark.asyncio
    async def test_managed_resource_with_cleanup(self) -> None:
        """Test managed resource with cleanup function."""
        cleaned = False

        async def cleanup(value: str) -> None:
            nonlocal cleaned
            assert value == "test"
            cleaned = True

        async with cast(
            "AsyncTestContext",
            FlextTestsAsyncs.managed_resource(
                "test",
                cleanup_func=cleanup,
            ),
        ) as value:
            assert value == "test"
            assert not cleaned

        assert cleaned

    @pytest.mark.asyncio
    async def test_timeout_context(self) -> None:
        """Test timeout context manager."""
        # Should complete within timeout
        async with cast("AsyncTestContext", FlextTestsAsyncs.timeout_context(1.0)):
            await asyncio.sleep(0.01)
            result = "completed"

        assert result == "completed"

    @pytest.mark.asyncio
    async def test_timeout_context_exceeded(self) -> None:
        """Test timeout context when time is exceeded."""
        with pytest.raises(asyncio.TimeoutError):
            async with cast("AsyncTestContext", FlextTestsAsyncs.timeout_context(0.1)):
                await asyncio.sleep(1.0)
