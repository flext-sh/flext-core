# ruff: noqa: PLC0415
"""Advanced async testing utilities using pytest-asyncio and pytest-timeout.

Provides comprehensive async testing patterns, concurrency testing,
timeout management, and async context management for robust testing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Protocol, TypeGuard, TypeVar

import pytest

from flext_core import FlextLogger

T = TypeVar("T")


def _is_not_exception[T](value: T | BaseException) -> TypeGuard[T]:
    """Type guard to filter out BaseException instances."""
    return not isinstance(value, BaseException)


class AsyncMockProtocol(Protocol):
    """Protocol for async mock functions."""

    async def __call__(self, *args: object, **kwargs: object) -> object: ...


logger = FlextLogger(__name__)


class AsyncTestUtils:
    """Comprehensive async testing utilities with timeout and concurrency support."""

    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool | Awaitable[bool]],
        *,
        timeout_seconds: float = 5.0,
        poll_interval: float = 0.1,
        error_message: str = "Condition not met within timeout",
    ) -> None:
        """Wait for a condition to become true with timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                if asyncio.iscoroutinefunction(condition):
                    result = await condition()
                else:
                    result = condition()

                if result:
                    return

            except Exception as e:
                # Log exceptions during condition checking for debugging
                logger.debug("Exception during condition check: %s", e)

            await asyncio.sleep(poll_interval)

        raise TimeoutError(error_message)

    @staticmethod
    async def run_with_timeout(
        coro: Awaitable[T],
        *,
        timeout_seconds: float = 5.0,
    ) -> T:
        """Run coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except TimeoutError as e:
            msg = f"Operation timed out after {timeout_seconds} seconds"
            raise TimeoutError(msg) from e

    @staticmethod
    async def run_concurrently(
        *coroutines: Awaitable[T],
        return_exceptions: bool = False,
    ) -> list[T]:
        """Run multiple coroutines concurrently."""
        if not coroutines:
            return []

        if return_exceptions:
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            # Filter out exceptions and return only successful results using type guard
            filtered_results: list[T] = [r for r in results if _is_not_exception(r)]
            return filtered_results

        # When return_exceptions=False, asyncio.gather will raise on first exception
        return await asyncio.gather(*coroutines, return_exceptions=False)

    @staticmethod
    async def run_concurrent(
        coroutines: list[Awaitable[T]],
        *,
        return_exceptions: bool = False,
    ) -> list[T]:
        """Run coroutines from a list concurrently."""
        return await AsyncTestUtils.run_concurrently(
            *coroutines,
            return_exceptions=return_exceptions,
        )

    @staticmethod
    async def sleep_with_timeout(duration: float) -> None:
        """Sleep for specified duration (alias for asyncio.sleep)."""
        await asyncio.sleep(duration)

    @staticmethod
    async def simulate_delay(duration: float) -> None:
        """Simulate delay for testing purposes (alias for asyncio.sleep)."""
        await asyncio.sleep(duration)

    @staticmethod
    async def run_concurrent_tasks(
        tasks: list[Awaitable[T]],
        *,
        return_exceptions: bool = False,
    ) -> list[T]:
        """Run a list of coroutines concurrently."""
        if not tasks:
            return []
        if return_exceptions:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions and return only successful results using type guard
            filtered_results: list[T] = [r for r in results if _is_not_exception(r)]
            return filtered_results
        # When return_exceptions=False, asyncio.gather will raise on first exception
        return await asyncio.gather(*tasks, return_exceptions=False)
        # Type assertion: when return_exceptions=False, results is guaranteed to be list[T]

    @staticmethod
    async def retry_async(
        coro_func: Callable[[], Awaitable[T]],
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> T:
        """Retry async operation with exponential backoff."""
        last_exception = None
        current_delay = delay

        for attempt in range(max_attempts):
            try:
                return await coro_func()
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

        if last_exception:
            raise last_exception

        msg = "All retry attempts failed"
        raise RuntimeError(msg)


class AsyncContextManagers:
    """Async context managers for testing scenarios."""

    @staticmethod
    @asynccontextmanager
    async def async_timer(duration: float = 5.0) -> AsyncGenerator[None]:
        """Async context manager with timeout."""
        start_time = time.time()

        async def check_timeout() -> None:
            while True:
                if time.time() - start_time > duration:
                    msg = f"Operation exceeded timeout of {duration} seconds"
                    raise TimeoutError(msg)
                await asyncio.sleep(0.1)

        timeout_task = asyncio.create_task(check_timeout())

        try:
            yield
        finally:
            timeout_task.cancel()
            with suppress(asyncio.CancelledError):
                await timeout_task

    @staticmethod
    @asynccontextmanager
    async def async_resource_manager(
        setup_func: Callable[[], Awaitable[T]],
        cleanup_func: Callable[[T], Awaitable[None]],
    ) -> AsyncGenerator[T]:
        """Generic async resource manager."""
        resource = await setup_func()
        try:
            yield resource
        finally:
            await cleanup_func(resource)

    @staticmethod
    @asynccontextmanager
    async def async_background_task(
        task_func: Callable[[], Awaitable[None]],
    ) -> AsyncGenerator[asyncio.Task[None]]:
        """Run background task during test execution."""
        awaitable = task_func()
        # Ensure we have a coroutine for create_task
        if not asyncio.iscoroutine(awaitable):

            async def _wrapper() -> None:
                return await awaitable

            coro = _wrapper()
        else:
            coro = awaitable
        task = asyncio.create_task(coro)

        try:
            yield task
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    @staticmethod
    @asynccontextmanager
    async def async_event_waiter(
        event: asyncio.Event,
        *,
        wait_duration: float = 5.0,
    ) -> AsyncGenerator[None]:
        """Wait for event with timeout."""
        try:
            await asyncio.wait_for(event.wait(), timeout=wait_duration)
            yield
        except TimeoutError as e:
            msg = f"Event not set within {wait_duration} seconds"
            raise TimeoutError(msg) from e


class AsyncMockUtils:
    """Utilities for mocking async functions and testing async behavior."""

    @staticmethod
    def create_async_mock(
        return_value: object = None,
        side_effect: Exception | None = None,
    ) -> AsyncMockProtocol:
        """Create async mock function."""

        async def async_mock(*_args: object, **_kwargs: object) -> object:
            if side_effect:
                raise side_effect
            return return_value

        return async_mock

    @staticmethod
    def create_delayed_async_mock(
        return_value: object = None,
        delay: float = 0.1,
        side_effect: Exception | None = None,
    ) -> object:
        """Create async mock with delay."""

        async def delayed_async_mock(*_args: object, **_kwargs: object) -> object:
            await asyncio.sleep(delay)
            if side_effect:
                raise side_effect
            return return_value

        return delayed_async_mock

    @staticmethod
    def create_flaky_async_mock(
        return_value: object = None,
        failure_rate: float = 0.3,
        exception: Exception | None = None,
    ) -> object:
        """Create async mock that fails randomly."""
        import random

        async def flaky_async_mock(*_args: object, **_kwargs: object) -> object:
            if random.random() < failure_rate:
                test_exception = exception or RuntimeError("Flaky operation failed")
                raise test_exception
            return return_value

        return flaky_async_mock


class AsyncFixtureUtils:
    """Utilities for creating async fixtures."""

    @staticmethod
    async def async_setup_teardown(
        setup: Callable[[], Awaitable[T]],
        teardown: Callable[[T], Awaitable[None]],
    ) -> AsyncGenerator[T]:
        """Helper for async setup/teardown patterns."""
        resource = await setup()
        try:
            yield resource
        finally:
            await teardown(resource)

    @staticmethod
    async def create_async_test_client() -> object:
        """Create async test client (placeholder for actual implementation)."""

        # This would typically create an async HTTP client or similar
        class AsyncTestClient:
            async def get(self, url: str) -> dict[str, object]:
                await asyncio.sleep(0.01)  # Simulate network delay
                return {"url": url, "status": 200}

            async def post(
                self,
                url: str,
                data: dict[str, object],
            ) -> dict[str, object]:
                await asyncio.sleep(0.01)
                return {"url": url, "data": data, "status": 201}

            async def close(self) -> None:
                await asyncio.sleep(0.01)

        client = AsyncTestClient()
        try:
            yield client
        finally:
            await client.close()

    @staticmethod
    async def create_async_event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop]:
        """Create isolated event loop for testing."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield loop
        finally:
            loop.close()


class AsyncConcurrencyTesting:
    """Advanced concurrency testing patterns."""

    @staticmethod
    async def test_race_condition(
        func1: Callable[[], Awaitable[object]],
        func2: Callable[[], Awaitable[object]],
        iterations: int = 100,
    ) -> dict[str, object]:
        """Test for race conditions between two async functions."""
        results: list[dict[str, object]] = []

        for _ in range(iterations):
            # Start both functions simultaneously
            awaitable1 = func1()
            awaitable2 = func2()

            # Ensure we have coroutines for create_task
            if not asyncio.iscoroutine(awaitable1):

                async def _wrapper1(result: object = awaitable1) -> object:
                    return result

                coro1 = _wrapper1()
            else:
                coro1 = awaitable1

            if not asyncio.iscoroutine(awaitable2):

                async def _wrapper2(result: object = awaitable2) -> object:
                    return result

                coro2 = _wrapper2()
            else:
                coro2 = awaitable2

            task1 = asyncio.create_task(coro1)
            task2 = asyncio.create_task(coro2)

            # Wait for both to complete
            result1, result2 = await asyncio.gather(
                task1,
                task2,
                return_exceptions=True,
            )

            results.append(
                {
                    "result1": result1,
                    "result2": result2,
                    "error1": isinstance(result1, Exception),
                    "error2": isinstance(result2, Exception),
                },
            )

        return {
            "total_iterations": iterations,
            "results": results,
            "error_rate1": sum(1 for r in results if r.get("error1", False))
            / iterations,
            "error_rate2": sum(1 for r in results if r.get("error2", False))
            / iterations,
        }

    @staticmethod
    async def test_concurrent_access(
        func: Callable[[], Awaitable[object]],
        concurrency_level: int = 10,
        iterations_per_worker: int = 10,
    ) -> dict[str, object]:
        """Test concurrent access to a resource."""

        async def worker() -> list[dict[str, object]]:
            results: list[dict[str, object]] = []
            for _ in range(iterations_per_worker):
                try:
                    result = await func()
                    results.append({"result": result, "success": True})
                except Exception as e:
                    results.append({"error": str(e), "success": False})
            return results

        # Start all workers concurrently
        workers = [worker() for _ in range(concurrency_level)]
        worker_results = await asyncio.gather(*workers)

        # Flatten results
        all_results: list[dict[str, object]] = []
        for worker_result in worker_results:
            all_results.extend(worker_result)

        success_count = sum(1 for r in all_results if r.get("success", False))
        total_operations = concurrency_level * iterations_per_worker

        return {
            "total_operations": total_operations,
            "successful_operations": success_count,
            "success_rate": success_count / total_operations,
            "concurrency_level": concurrency_level,
            "detailed_results": all_results,
        }

    @staticmethod
    async def test_deadlock_detection(
        operations: list[Callable[[], Awaitable[object]]],
        deadline: float = 5.0,
    ) -> dict[str, object]:
        """Test for potential deadlocks in async operations."""
        start_time = time.time()

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[op() for op in operations]),
                timeout=deadline,
            )
            end_time = time.time()

            return {
                "completed": True,
                "execution_time": end_time - start_time,
                "results": results,
                "potential_deadlock": False,
            }

        except TimeoutError:
            end_time = time.time()

            return {
                "completed": False,
                "execution_time": end_time - start_time,
                "timeout": deadline,
                "potential_deadlock": True,
            }


# Pytest markers for async tests
class AsyncMarkers:
    """Custom pytest markers for async tests."""

    asyncio = pytest.mark.asyncio
    timeout = pytest.mark.timeout
    concurrency = pytest.mark.concurrency
    race_condition = pytest.mark.race_condition

    @staticmethod
    def async_timeout(seconds: float) -> object:
        """Mark async test with specific timeout."""
        return pytest.mark.timeout(seconds)

    @staticmethod
    def async_slow() -> object:
        """Mark async test as slow."""
        return pytest.mark.slow


# Export utilities
__all__ = [
    "AsyncConcurrencyTesting",
    "AsyncContextManagers",
    "AsyncFixtureUtils",
    "AsyncMarkers",
    "AsyncMockUtils",
    "AsyncTestUtils",
]
