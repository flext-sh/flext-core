"""Advanced async testing utilities using pytest-asyncio and pytest-timeout.

Provides async testing patterns, concurrency testing,
timeout management, and async context management for robust testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from typing import Protocol, TypeGuard

import pytest

from flext_core import FlextLogger, FlextTypes, T

logger = FlextLogger(__name__)


class FlextTestsAsyncs:
    """Unified async testing utilities for FLEXT ecosystem.

    Consolidates all async testing patterns, concurrency testing, timeout management,
    async context management, and mock utilities into a single unified class.
    """

    @staticmethod
    def _is_not_exception(obj: object | Exception) -> TypeGuard[object]:
        """Type guard to check if object is not an exception."""
        return not isinstance(obj, Exception)

    # === Core Async Testing Utilities ===

    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool | Awaitable[bool]],
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
        timeout_seconds: float = 5.0,
    ) -> T:
        """Run coroutine with timeout, with light retry if callable is discoverable."""
        start = time.time()

        async def attempt_once() -> T:
            """attempt_once method."""
            return await asyncio.wait_for(
                coro,
                timeout=max(0.0, timeout_seconds - (time.time() - start)),
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
                                    0.0,
                                    timeout_seconds - (time.time() - start),
                                ),
                            )
                        except Exception:
                            await asyncio.sleep(0)
                    msg = f"Operation timed out after {timeout_seconds} seconds"
                    raise TimeoutError(msg)
            except Exception as e:
                # Fall through to final timeout-based error
                # Log the exception for debugging but continue to timeout error
                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Exception during timeout operation, falling through to timeout: {e}",
                )
            # If we couldn't discover a factory or retries failed, re-raise as TimeoutError respecting API contract
            msg = f"Operation timed out after {timeout_seconds} seconds"
            raise TimeoutError(msg) from None

    @staticmethod
    async def run_concurrently(
        coroutines: list[Awaitable[T]],
        *,
        return_exceptions: bool = False,
    ) -> list[T]:
        """Run multiple coroutines concurrently."""
        if not coroutines:
            return []

        if return_exceptions:
            results: list[T | BaseException] = await asyncio.gather(
                *coroutines,
                return_exceptions=True,
            )
            # Filter out exceptions and return only successful results
            return [r for r in results if not isinstance(r, BaseException)]

        # When return_exceptions=False, asyncio.gather will raise on first exception
        return await asyncio.gather(*coroutines, return_exceptions=False)

    @staticmethod
    async def run_concurrent(duration: float) -> None:
        """Sleep for specified duration during concurrent operations."""
        await asyncio.sleep(duration)

    @staticmethod
    async def simulate_delay(
        tasks: list[Awaitable[T]],
        *,
        return_exceptions: bool = False,
    ) -> list[T]:
        """Run a list of coroutines concurrently."""
        if not tasks:
            return []
        if return_exceptions:
            results: list[T | BaseException] = await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )
            # Filter out exceptions and return only successful results
            return [r for r in results if not isinstance(r, BaseException)]
        # When return_exceptions=False, asyncio.gather will raise on first exception
        return await asyncio.gather(*tasks, return_exceptions=False)

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

    # === Async Context Managers ===

    @staticmethod
    @asynccontextmanager
    async def async_timer(duration: float) -> AsyncGenerator[None]:
        """Async context manager with timeout."""
        start_time = time.time()

        async def check_timeout() -> None:
            """check_timeout method."""
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
                """_wrapper method."""
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
        wait_duration: float = 5.0,
    ) -> AsyncGenerator[None]:
        """Wait for event with timeout."""
        try:
            await asyncio.wait_for(event.wait(), timeout=wait_duration)
            yield
        except TimeoutError as e:
            msg = f"Event not set within {wait_duration} seconds"
            raise TimeoutError(msg) from e

    @staticmethod
    @asynccontextmanager
    async def create_test_context(
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

    # === Mock Utilities ===

    @staticmethod
    def create_async_mock(
        return_value: object | None = None,
        side_effect: Exception | None = None,
    ) -> FlextTestsAsyncs.AsyncMockProtocol:
        """async_mock method."""

        async def async_mock(*_args: object, **_kwargs: object) -> object:
            if side_effect:
                raise side_effect
            return return_value

        # Return the async function directly as it's compatible with FlextTestsAsyncs.AsyncMockProtocol
        return async_mock

    @staticmethod
    def create_delayed_async_mock(
        delay: float,
        return_value: object | None = None,
        side_effect: Exception | None = None,
    ) -> FlextTestsAsyncs.AsyncMockProtocol:
        """delayed_async_mock method."""

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
    ) -> FlextTestsAsyncs.AsyncMockProtocol:
        """Create async mock with either random failure or fixed failure count."""
        # Counter mode
        if failure_count is not None:
            remaining = int(failure_count)
            exc_type = exception_type or RuntimeError
            result_value = success_value if success_value is not None else return_value

            async def flaky_count_mock(*_args: object, **_kwargs: object) -> object:
                """flaky_count_mock method."""
                nonlocal remaining
                if remaining > 0:
                    remaining -= 1
                    raise exc_type()
                return result_value

            return flaky_count_mock

        # Random mode
        rate = 0.3 if failure_rate is None else float(failure_rate)

        async def flaky_random_mock(*_args: object, **_kwargs: object) -> object:
            """flaky_random_mock method."""
            if random.random() < rate:
                raise exception or RuntimeError("flaky failure")
            return return_value

        return flaky_random_mock

    @staticmethod
    def create_async_mock_with_side_effect(
        side_effect: object | None = None,
        return_value: object | None = None,
    ) -> FlextTestsAsyncs.AsyncMockProtocol:
        """Create async mock with extended side_effect semantics."""
        # Normalize list-like side effects
        sequence: FlextTypes.Core.List | None = None
        if isinstance(side_effect, (list, tuple)):
            sequence = list(side_effect)

        async def async_mock(*args: object, **kwargs: object) -> object:
            """async_mock method."""
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

    # === Fixture Utilities ===

    @staticmethod
    @asynccontextmanager
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
    @asynccontextmanager
    async def create_async_test_client() -> AsyncGenerator[object]:
        """Create async test client (placeholder for actual implementation)."""

        # This would typically create an async HTTP client or similar
        class AsyncTestClient:
            async def get(self, url: str) -> dict[str, object]:
                """Get method."""
                await asyncio.sleep(0.01)  # Simulate network delay
                return {"url": url, "status": 200}

            async def post(self, url: str, data: object) -> dict[str, object]:
                """Post method."""
                await asyncio.sleep(0.01)
                return {"url": url, "data": data, "status": 201}

            async def close(self) -> None:
                """Close method."""
                await asyncio.sleep(0.01)

        client = AsyncTestClient()
        try:
            yield client
        finally:
            await client.close()

    @staticmethod
    @asynccontextmanager
    async def create_async_event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop]:
        """Create isolated event loop for testing."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield loop
        finally:
            loop.close()

    # === Concurrency Testing ===

    @staticmethod
    async def run_parallel_tasks(
        inputs: list[T],
        task_func: Callable[[T], Awaitable[object]],
    ) -> list[object]:
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
                # Create wrapper with proper variable binding
                def make_wrapper1(
                    aw: Awaitable[object],
                ) -> Callable[[], Coroutine[None, None, object]]:
                    async def _wrapper1() -> object:
                        """_wrapper1 method."""
                        return await aw

                    return _wrapper1

                coro1 = make_wrapper1(awaitable1)()
            else:
                coro1 = awaitable1

            if not asyncio.iscoroutine(awaitable2):
                # Create wrapper with proper variable binding
                def make_wrapper2(
                    aw: Awaitable[object],
                ) -> Callable[[], Coroutine[None, None, object]]:
                    async def _wrapper2() -> object:
                        """_wrapper2 method."""
                        return await aw

                    return _wrapper2

                coro2 = make_wrapper2(awaitable2)()
            else:
                coro2 = awaitable2

            # Create tasks - coro1 and coro2 are guaranteed to be coroutines here
            task1: asyncio.Task[object] = asyncio.create_task(coro1)
            task2: asyncio.Task[object] = asyncio.create_task(coro2)

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
        concurrency_level: int = 5,
        iterations_per_worker: int = 10,
    ) -> FlextTypes.Core.Dict:
        """Test concurrent access to a resource."""

        async def worker() -> list[FlextTypes.Core.Dict]:
            """Worker method."""
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
        deadline: float = 10.0,
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

    @staticmethod
    async def test_race_condition_simple(
        func: Callable[[], Awaitable[object]],
        concurrent_count: int = 5,
    ) -> list[object]:
        """Run the same async function concurrently and return results."""
        # Create coroutines and ensure they are proper coroutines
        coroutines = []
        for _ in range(concurrent_count):
            result = func()
            if asyncio.iscoroutine(result):
                coroutines.append(result)
            else:
                # Wrap non-coroutine awaitables
                async def _wrapper(aw: Awaitable[object] = result) -> object:
                    """Wrapper for non-coroutine awaitable."""
                    return await aw

                coroutines.append(_wrapper())

        tasks: list[asyncio.Task[object]] = [
            asyncio.create_task(coro) for coro in coroutines
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    @staticmethod
    async def measure_concurrency_performance(
        func: Callable[[], Awaitable[None]],
        iterations: int = 100,
    ) -> FlextTypes.Core.Dict:
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

    # === Pytest Markers ===
    asyncio = pytest.mark.asyncio
    timeout = pytest.mark.timeout
    concurrency = pytest.mark.concurrency
    race_condition = pytest.mark.race_condition

    @staticmethod
    def async_timeout(seconds: float) -> pytest.MarkDecorator:
        """Mark async test with specific timeout."""
        return pytest.mark.timeout(seconds)

    @staticmethod
    def async_slow() -> pytest.MarkDecorator:
        """Ultra-simple alias for test compatibility."""
        return pytest.mark.slow

    class AsyncMockProtocol(Protocol):
        """Protocol for async mock functions."""

        async def __call__(self, *args: object, **kwargs: object) -> object:
            """Async callable interface for mock functions."""
            ...


# Export only the unified class
__all__ = [
    "FlextTestsAsyncs",
]
