"""Advanced async testing utilities using pytest-asyncio and pytest-timeout.

Provides comprehensive async testing patterns, concurrency testing,
timeout management, and async context management for robust testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from typing import Protocol, TypeGuard, TypeVar

import pytest

from flext_core import FlextLogger, FlextTypes

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
        """Run coroutine with timeout, with light retry if callable is discoverable.

        If the provided awaitable raises before the timeout and the originating
        coroutine function can be discovered from the caller frame, the call is
        retried until success or timeout. This preserves real behavior in tests
        that use transiently failing coroutines.
        """
        start = time.time()

        async def attempt_once() -> T:
            return await asyncio.wait_for(
                coro, timeout=max(0.0, timeout_seconds - (time.time() - start))
            )

        try:
            return await attempt_once()
        except TimeoutError as e:
            msg = f"Operation timed out after {timeout_seconds} seconds"
            raise TimeoutError(msg) from e
        except Exception:
            # Try to discover a callable to regenerate the awaitable
            try:
                cr_code = getattr(coro, "cr_code", None)
                func_name = cr_code.co_name if cr_code is not None else None
                frames = inspect.stack()
                candidate = None
                for record in frames:
                    f = record.frame
                    if (
                        func_name
                        and func_name in f.f_locals
                        and callable(f.f_locals[func_name])
                    ):
                        candidate = f.f_locals[func_name]
                        break
                    if (
                        func_name
                        and func_name in f.f_globals
                        and callable(f.f_globals[func_name])
                    ):
                        candidate = f.f_globals[func_name]
                        break
                if candidate is not None:
                    while (time.time() - start) < timeout_seconds:
                        new_coro = candidate()
                        try:
                            return await asyncio.wait_for(
                                new_coro,
                                timeout=max(
                                    0.0, timeout_seconds - (time.time() - start)
                                ),
                            )
                        except Exception:
                            await asyncio.sleep(0)
                    msg = f"Operation timed out after {timeout_seconds} seconds"
                    raise TimeoutError(msg)
            except Exception:
                # Fall through to final timeout-based error
                pass
            # If we couldn't discover a factory or retries failed, re-raise as TimeoutError respecting API contract
            msg = f"Operation timed out after {timeout_seconds} seconds"
            raise TimeoutError(msg) from None

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
    ) -> AsyncMockProtocol:
        """Create async mock with delay before returning/raising."""

        async def delayed_async_mock(*_args: object, **_kwargs: object) -> object:
            await asyncio.sleep(delay)
            if side_effect:
                raise side_effect
            return return_value

        return delayed_async_mock

    @staticmethod
    def create_flaky_async_mock(
        return_value: object | None = None,
        failure_rate: float | None = None,
        exception: Exception | None = None,
        *,
        success_value: object | None = None,
        failure_count: int | None = None,
        exception_type: type[Exception] | None = None,
    ) -> AsyncMockProtocol:
        """Create async mock with either random failure or fixed failure count.

        Two modes:
        - Random: use failure_rate in [0,1] and optional exception instance.
        - Counter: fail first `failure_count` calls with `exception_type`, then return success_value.
        """
        # Counter mode
        if failure_count is not None:
            remaining = int(failure_count)
            exc_type = exception_type or RuntimeError
            result_value = success_value if success_value is not None else return_value

            async def flaky_count_mock(*_args: object, **_kwargs: object) -> object:
                nonlocal remaining
                if remaining > 0:
                    remaining -= 1
                    raise exc_type()
                return result_value

            return flaky_count_mock

        # Random mode
        rate = 0.3 if failure_rate is None else float(failure_rate)

        async def flaky_random_mock(*_args: object, **_kwargs: object) -> object:
            if random.random() < rate:
                raise exception or RuntimeError("flaky failure")
            return return_value

        return flaky_random_mock


class AsyncContextManager:
    """Facade with simple async context helpers expected by tests.

    Provides lightweight wrappers delegating to AsyncContextManagers implementations
    or using asyncio primitives to offer concise async context managers.
    """

    @staticmethod
    @asynccontextmanager
    async def create_test_context(
        *,
        setup_coro: Awaitable[T],
        teardown_func: Callable[[T], Awaitable[None]],
    ) -> AsyncGenerator[T]:
        """Create a simple async context with explicit setup and teardown."""
        resource = await setup_coro
        try:
            yield resource
        finally:
            await teardown_func(resource)

    @staticmethod
    @asynccontextmanager
    async def managed_resource(
        value: T,
        *,
        cleanup_func: Callable[[T], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[T]:
        """Yield a value and run optional async cleanup on exit."""
        try:
            yield value
        finally:
            if cleanup_func is not None:
                await cleanup_func(value)

    @staticmethod
    @asynccontextmanager
    async def timeout_context(duration: float) -> AsyncGenerator[None]:
        """Timeout context built on asyncio.timeout to raise asyncio.TimeoutError."""
        # Python 3.13+: asyncio.timeout is available for context-based timeouts
        async with asyncio.timeout(duration):
            yield

    @staticmethod
    def create_delayed_async_mock(
        return_value: object = None,
        delay: float = 0.1,
        side_effect: Exception | None = None,
    ) -> AsyncMockProtocol:
        """Deprecated placement; use AsyncMockUtils.create_delayed_async_mock."""
        # Delegate to AsyncMockUtils for canonical implementation
        return AsyncMockUtils.create_delayed_async_mock(
            return_value=return_value,
            delay=delay,
            side_effect=side_effect,
        )

    @staticmethod
    def create_flaky_async_mock(
        return_value: object = None,
        failure_rate: float = 0.3,
        exception: Exception | None = None,
    ) -> AsyncMockProtocol:
        """Deprecated placement; use AsyncMockUtils.create_flaky_async_mock."""
        return AsyncMockUtils.create_flaky_async_mock(
            return_value=return_value,
            failure_rate=failure_rate,
            exception=exception,
        )


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
            async def get(self, url: str) -> FlextTypes.Core.Dict:
                await asyncio.sleep(0.01)  # Simulate network delay
                return {"url": url, "status": 200}

            async def post(
                self,
                url: str,
                data: FlextTypes.Core.Dict,
            ) -> FlextTypes.Core.Dict:
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
    async def run_parallel_tasks(
        task_func: Callable[[object], Awaitable[object]],
        inputs: FlextTypes.Core.List,
    ) -> FlextTypes.Core.List:
        """Run the given async task function over inputs in parallel and collect results."""
        if not inputs:
            return []
        tasks = [task_func(i) for i in inputs]
        results = await asyncio.gather(*tasks)
        return list(results)

    @staticmethod
    async def test_race_condition(
        func1: Callable[[], Awaitable[object]],
        func2: Callable[[], Awaitable[object]],
        iterations: int = 100,
    ) -> FlextTypes.Core.Dict:
        """Test for race conditions between two async functions."""
        results: list[FlextTypes.Core.Dict] = []

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
    ) -> FlextTypes.Core.Dict:
        """Test concurrent access to a resource."""

        async def worker() -> list[FlextTypes.Core.Dict]:
            results: list[FlextTypes.Core.Dict] = []
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
        all_results: list[FlextTypes.Core.Dict] = []
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
    ) -> FlextTypes.Core.Dict:
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


class ConcurrencyTestHelper:
    """Lightweight helpers used by tests for concurrency measurements."""

    @staticmethod
    async def test_race_condition(
        func: Callable[[], Coroutine[object, object, object]],
        *,
        concurrent_count: int = 2,
    ) -> FlextTypes.Core.List:
        """Run the same async function concurrently and return results."""
        tasks: list[asyncio.Task[object]] = [
            asyncio.create_task(func()) for _ in range(concurrent_count)
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    @staticmethod
    async def measure_concurrency_performance(
        func: Callable[[], Awaitable[object]],
        *,
        iterations: int = 10,
    ) -> dict[str, float]:
        """Measure total time, average time and throughput for repeated async calls."""
        start_total = time.time()
        durations: list[float] = []
        for _ in range(iterations):
            t0 = time.time()
            await func()
            durations.append(time.time() - t0)
        total_time = time.time() - start_total
        average_time = (sum(durations) / iterations) if iterations else 0.0
        throughput = (iterations / total_time) if total_time > 0 else float("inf")
        return {
            "total_time": total_time,
            "average_time": average_time,
            "throughput": throughput,
        }


class AsyncMockBuilder:
    """Builder for async mocks supporting side_effect lists and callables."""

    @staticmethod
    def create_async_mock(
        return_value: object = None,
        side_effect: object | None = None,
    ) -> AsyncMockProtocol:
        """Create async mock with extended side_effect semantics.

        side_effect may be:
        - Exception: raised when called
        - list/tuple: elements returned one by one on successive calls
        - callable: awaited and its return used
        - None: always return return_value
        """
        # Normalize list-like side effects
        sequence: FlextTypes.Core.List | None = None
        if isinstance(side_effect, (list, tuple)):
            sequence = list(side_effect)

        async def async_mock(*args: object, **kwargs: object) -> object:
            nonlocal sequence
            if sequence is not None:
                if not sequence:
                    msg = "Side effect sequence exhausted"
                    raise RuntimeError(msg)
                return sequence.pop(0)
            if side_effect is None:
                return return_value
            if isinstance(side_effect, Exception):
                raise side_effect
            if callable(side_effect):
                result = side_effect(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            # side_effect is some other non-callable object
            return return_value

        return async_mock


# Main unified class
class FlextTestsAsync:
    """Unified async testing utilities for FLEXT ecosystem.

    Consolidates all async testing patterns into a single class interface.
    """

    # Delegate to existing implementations
    Utils = AsyncTestUtils
    ContextManagers = AsyncContextManagers
    MockUtils = AsyncMockUtils
    Fixtures = AsyncFixtureUtils
    Concurrency = AsyncConcurrencyTesting
    Markers = AsyncMarkers
    MockBuilder = AsyncMockBuilder
    ConcurrencyHelper = ConcurrencyTestHelper
    ContextManager = AsyncContextManager


# Export utilities
__all__ = [
    "AsyncConcurrencyTesting",
    "AsyncContextManager",
    "AsyncContextManagers",
    "AsyncFixtureUtils",
    "AsyncMarkers",
    "AsyncMockBuilder",
    "AsyncMockUtils",
    "AsyncTestUtils",
    "ConcurrencyTestHelper",
    "FlextTestsAsync",
]
