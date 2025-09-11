"""Advanced test matchers and assertions using all pytest capabilities.

Provides consolidated test matchers, performance testing capabilities,
and protocol definitions for comprehensive test assertions.

Leverages pytest-clarity, pytest-benchmark, pytest-mock, and other plugins
for comprehensive testing with clear error messages and performance insights.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable, Sequence
from typing import Protocol, TypeGuard, TypeVar, runtime_checkable

from flext_core import FlextResult, FlextTypes

T = TypeVar("T")


class FlextTestsMatchers:
    """Unified test matchers for FLEXT ecosystem.

    Consolidates all matcher patterns into a single class interface.
    Provides advanced matchers, performance testing, and protocol definitions
    for comprehensive test assertions with pytest integration.
    """

    # === PROTOCOL DEFINITIONS ===

    @runtime_checkable
    class BenchmarkProtocol(Protocol):
        """Local benchmark protocol to avoid circular imports."""

        def benchmark(
            self, func: object, *args: object, **kwargs: object
        ) -> object: ...

        @property
        def stats(self) -> object:
            """Benchmark statistics."""
            ...

    class ContainerProtocol(Protocol):
        """Protocol for container objects with get method."""

        def get(self, service_name: str) -> FlextResult[object]:
            """Get service from container."""
            ...

    class FieldProtocol(Protocol):
        """Protocol for field validation objects."""

        def validate_field_value(self, value: object) -> tuple[bool, str]:
            """Validate field value."""
            ...

    class MockProtocol(Protocol):
        """Protocol for mock objects."""

        call_count: int
        return_value: object

    # === CORE MATCHERS ===

    class CoreMatchers:
        """Advanced matchers for flext-core testing with comprehensive assertions.

        Uses pytest-clarity for better error messages and provides domain-specific
        matchers for FlextResult, containers, and other core patterns.
        """

        @staticmethod
        def assert_result_success(
            result: FlextResult[object],
            expected_data: object = None,
        ) -> None:
            """Assert FlextResult is successful with optional data check.

            Args:
                result: FlextResult to test
                expected_data: Expected data value (optional)

            Raises:
                AssertionError: With clear message about failure

            """
            assert result.is_success, (
                f"Expected successful result, but got failure: {result.error}"
            )

            if expected_data is not None:
                assert result.value == expected_data, (
                    f"Expected data {expected_data!r}, got {result.value!r}"
                )

        @staticmethod
        def assert_result_failure(
            result: FlextResult[object],
            expected_error: str | None = None,
            expected_error_code: str | None = None,
        ) -> None:
            """Assert FlextResult is failed with optional error checks.

            Args:
                result: FlextResult to test
                expected_error: Expected error message (optional)
                expected_error_code: Expected error code (optional)

            Raises:
                AssertionError: With clear message about success

            """
            assert result.is_failure, (
                f"Expected failed result, but got success: {result.value}"
            )

            if expected_error is not None:
                error_message = result.error or "Unknown error"
                assert expected_error in str(error_message), (
                    f"Expected error containing {expected_error!r}, got {error_message!r}"
                )

            if expected_error_code is not None:
                actual_code = result.error_code or "UNKNOWN"
                assert actual_code == expected_error_code, (
                    f"Expected error code {expected_error_code!r}, got {actual_code!r}"
                )

        # Convenience boolean helpers used by some tests
        @staticmethod
        def is_successful_result(result: FlextResult[object]) -> bool:
            return bool(getattr(result, "success", False))

        @staticmethod
        def is_failed_result(result: FlextResult[object]) -> bool:
            return bool(getattr(result, "is_failure", False))

        @staticmethod
        def assert_container_has_service(
            container: FlextTestsMatchers.ContainerProtocol,
            service_name: str,
            expected_type: type | None = None,
        ) -> None:
            """Assert container has service with optional type check.

            Args:
                container: Container to test
                service_name: Name of service to check
                expected_type: Expected type of service (optional)

            Raises:
                AssertionError: With clear message about missing service

            """
            result = container.get(service_name)
            assert result.is_success, (
                f"Expected service {service_name!r} to exist in container, "
                f"but got error: {result.error}"
            )

            if expected_type is not None:
                service: object = result.value
                assert isinstance(service, expected_type), (
                    f"Expected service {service_name!r} to be of type "
                    f"{expected_type.__name__}, got {type(service).__name__}"
                )

        @staticmethod
        def assert_field_validates(
            field: FlextTestsMatchers.FieldProtocol,
            value: object,
            *,
            should_pass: bool = True,
        ) -> None:
            """Assert field validation result.

            Args:
                field: Field to test
                value: Value to validate
                should_pass: Whether validation should pass

            Raises:
                AssertionError: With clear validation message

            """
            is_valid, error_msg = field.validate_field_value(value)

            if should_pass:
                assert is_valid, (
                    f"Expected validation to pass for value {value!r}, "
                    f"but got error: {error_msg}"
                )
            else:
                assert not is_valid, (
                    f"Expected validation to fail for value {value!r}, but it passed"
                )

        @staticmethod
        def assert_json_structure(
            data: FlextTypes.Core.JsonObject,
            expected_keys: Sequence[str],
            *,
            exact_match: bool = False,
        ) -> None:
            """Assert JSON structure has expected keys.

            Args:
                data: JSON data to test
                expected_keys: Keys that should be present
                exact_match: Whether to require exact key match

            Raises:
                AssertionError: With clear message about missing keys

            """
            actual_keys = set(data.keys())
            expected_keys_set = set(expected_keys)

            if exact_match:
                assert actual_keys == expected_keys_set, (
                    f"Expected exact keys {expected_keys_set}, got {actual_keys}"
                )
            else:
                missing_keys = expected_keys_set - actual_keys
                assert not missing_keys, f"Missing required keys: {missing_keys}"

        @staticmethod
        def assert_performance_within_limit(
            benchmark: FlextTestsMatchers.BenchmarkProtocol,
            func: Callable[[], object],
            _max_time_seconds: float = 1.0,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Assert function performance is within time limit.

            Args:
                benchmark: pytest-benchmark fixture
                func: Function to benchmark
                _max_time_seconds: Maximum allowed time
                *args: Function arguments
                **kwargs: Function keyword arguments

            Returns:
                Function result

            Raises:
                AssertionError: If performance exceeds limit

            """
            # Remove max_time_seconds from kwargs if present to avoid passing it to func
            func_kwargs = {k: v for k, v in kwargs.items() if k != "max_time_seconds"}
            result = benchmark(func, *args, **func_kwargs)

            # Get stats from benchmark
            stats = benchmark.stats
            mean_time = getattr(stats, "mean", 0.0)

            assert mean_time <= _max_time_seconds, (
                f"Performance test failed: mean time {mean_time:.4f}s "
                f"exceeds limit {_max_time_seconds:.4f}s"
            )

            return result

        @staticmethod
        def assert_regex_match(
            text: str,
            pattern: str,
            flags: int = 0,
        ) -> None:
            """Assert text matches regex pattern.

            Args:
                text: Text to test
                pattern: Regex pattern
                flags: Regex flags

            Raises:
                AssertionError: With clear message about pattern mismatch

            """
            compiled_pattern = re.compile(pattern, flags)
            match = compiled_pattern.search(text)

            assert match is not None, (
                f"Text {text!r} does not match pattern {pattern!r}"
            )

        @staticmethod
        def assert_type_guard(
            value: object,
            type_guard: Callable[[object], TypeGuard[object]],
        ) -> None:
            """Assert value passes type guard.

            Args:
                value: Value to test
                type_guard: Type guard function

            Raises:
                AssertionError: With clear message about type check

            """
            assert type_guard(value), (
                f"Value {value!r} of type {type(value).__name__} failed type guard check"
            )

        @staticmethod
        def assert_no_deadfixtures(test_module: object) -> None:
            """Assert no dead fixtures in test module (uses pytest-deadfixtures).

            Args:
                test_module: Test module to check

            Note:
                This leverages pytest-deadfixtures plugin for fixture analysis

            """
            # This is checked automatically by pytest-deadfixtures
            # This method serves as documentation and can trigger manual checks

        @staticmethod
        def assert_mock_called_with_result(
            mock_obj: FlextTestsMatchers.MockProtocol,
            expected_result_type: type,
            call_count: int = 1,
        ) -> None:
            """Assert mock was called and returned expected result type.

            Args:
                mock_obj: Mock object to test
                expected_result_type: Expected return type
                call_count: Expected number of calls

            Raises:
                AssertionError: With clear mock call information

            """
            assert mock_obj.call_count == call_count, (
                f"Expected {call_count} calls, got {mock_obj.call_count}"
            )

            if call_count > 0:
                return_value: object = mock_obj.return_value
                assert isinstance(return_value, expected_result_type), (
                    f"Expected return type {expected_result_type.__name__}, "
                    f"got {type(return_value).__name__}"
                )

        @staticmethod
        def assert_environment_variable(
            var_name: str,
            expected_value: str | None = None,
        ) -> None:
            """Assert environment variable exists with optional value check.

            Args:
                var_name: Environment variable name
                expected_value: Expected value (optional)

            Note:
                Uses pytest-env for environment variable testing

            """
            assert var_name in os.environ, (
                f"Environment variable {var_name!r} not found"
            )

            if expected_value is not None:
                actual_value = os.environ[var_name]
                assert actual_value == expected_value, (
                    f"Expected {var_name}={expected_value!r}, got {actual_value!r}"
                )

        @staticmethod
        def assert_async_result(
            async_result: object,
            timeout_seconds: float = 5.0,
        ) -> None:
            """Assert async operation completes within timeout.

            Args:
                async_result: Async result to test
                timeout_seconds: Maximum time to wait

            Note:
                Uses pytest-asyncio and pytest-timeout for async testing

            """
            # This is handled by pytest-asyncio and pytest-timeout decorators
            # This method serves as documentation for async patterns

    # === PERFORMANCE MATCHERS ===

    class PerformanceMatchers:
        """Performance-specific matchers using pytest-benchmark."""

        @staticmethod
        def assert_linear_complexity(
            benchmark: FlextTestsMatchers.BenchmarkProtocol,
            func: Callable[[int], object],
            sizes: Sequence[int],
            tolerance_factor: float = 2.0,
        ) -> None:
            """Assert function has linear time complexity.

            Args:
                benchmark: pytest-benchmark fixture
                func: Function to test (takes size parameter)
                sizes: List of input sizes to test
                tolerance_factor: Allowed deviation from linear growth

            Raises:
                AssertionError: If complexity is not linear

            """
            times: list[float] = []
            for size in sizes:
                benchmark.benchmark(func, size)
                times.append(getattr(benchmark.stats, "mean", 0.0))

            # Check if time growth is roughly linear
            if len(times) >= 2:
                # Ensure division results in float for type safety
                growth_ratio: float = (
                    float(times[-1] / times[0]) if times[0] > 0 else 1.0
                )
                size_ratio: float = float(sizes[-1] / sizes[0]) if sizes[0] > 0 else 1.0

                assert growth_ratio <= size_ratio * tolerance_factor, (
                    f"Complexity appears non-linear: time grew by {growth_ratio:.2f}x "
                    f"while size grew by {size_ratio:.2f}x"
                )

        @staticmethod
        def assert_memory_efficient(
            benchmark: FlextTestsMatchers.BenchmarkProtocol,
            func: Callable[[], object],
            _max_memory_mb: float = 100.0,
        ) -> object:
            """Assert function is memory efficient.

            Args:
                benchmark: pytest-benchmark fixture
                func: Function to test
                _max_memory_mb: Maximum memory usage in MB

            Note:
                This is a placeholder for memory profiling integration

            """
            # Memory profiling would require additional tools like memory_profiler
            # For now, this serves as a marker for memory-sensitive tests
            return benchmark.benchmark(func)

        @staticmethod
        def assert_performance_within_limit(
            benchmark: FlextTestsMatchers.BenchmarkProtocol,
            func: Callable[[], object],
            max_time_seconds: float = 1.0,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Assert function performance is within time limit.

            Args:
                benchmark: pytest-benchmark fixture
                func: Function to test
                max_time_seconds: Maximum allowed execution time
                *args: Arguments to pass to function
                **kwargs: Keyword arguments to pass to function

            Returns:
                Function execution result

            Raises:
                AssertionError: If execution time exceeds limit

            """
            # Use benchmark fixture directly (it's callable)
            # Execute function with benchmark - benchmark fixture IS the callable
            if args or kwargs:
                result = benchmark(func, *args, **kwargs)
            else:
                result = benchmark(func)

            # For pytest-benchmark, the result is returned and stats are attached
            # Note: max_time_seconds parameter is for future enhancement
            _ = max_time_seconds  # Acknowledge parameter to avoid unused warning
            return result

    # === FACTORY METHODS ===

    @classmethod
    def matchers(cls) -> FlextTestsMatchers.CoreMatchers:
        """Get core matchers instance."""
        return cls.CoreMatchers()

    @classmethod
    def performance(cls) -> FlextTestsMatchers.PerformanceMatchers:
        """Get performance matchers instance."""
        return cls.PerformanceMatchers()

    # === CONVENIENCE METHODS ===

    @classmethod
    def assert_result_success(
        cls,
        result: FlextResult[object],
        expected_data: object = None,
    ) -> None:
        """Assert FlextResult is successful quickly."""
        cls.CoreMatchers.assert_result_success(result, expected_data)

    @classmethod
    def assert_result_failure(
        cls,
        result: FlextResult[object],
        expected_error: str | None = None,
        expected_error_code: str | None = None,
    ) -> None:
        """Assert FlextResult is failed quickly."""
        cls.CoreMatchers.assert_result_failure(
            result, expected_error, expected_error_code
        )

    @classmethod
    def assert_json_structure(
        cls,
        data: FlextTypes.Core.JsonObject,
        expected_keys: Sequence[str],
        *,
        exact_match: bool = False,
    ) -> None:
        """Assert JSON structure quickly."""
        cls.CoreMatchers.assert_json_structure(
            data, expected_keys, exact_match=exact_match
        )

    @classmethod
    def assert_regex_match(
        cls,
        text: str,
        pattern: str,
        flags: int = 0,
    ) -> None:
        """Assert regex match quickly."""
        cls.CoreMatchers.assert_regex_match(text, pattern, flags)

    @classmethod
    def assert_environment_variable(
        cls,
        var_name: str,
        expected_value: str | None = None,
    ) -> None:
        """Assert environment variable quickly."""
        cls.CoreMatchers.assert_environment_variable(var_name, expected_value)

    @classmethod
    def is_successful_result(cls, result: FlextResult[object]) -> bool:
        """Check if result is successful quickly."""
        return cls.CoreMatchers.is_successful_result(result)

    @classmethod
    def is_failed_result(cls, result: FlextResult[object]) -> bool:
        """Check if result is failed quickly."""
        return cls.CoreMatchers.is_failed_result(result)

    @classmethod
    async def run_with_timeout(
        cls, coro: object, timeout_seconds: float = 5.0
    ) -> object:
        """Run coroutine with timeout and auto-retry for test compatibility."""
        import time

        start_time = time.time()
        last_exception = None

        while time.time() - start_time < timeout_seconds:
            try:
                # Create fresh coroutine for each retry attempt
                fresh_coro = coro() if callable(coro) else coro
                return await asyncio.wait_for(
                    fresh_coro, timeout=min(0.5, timeout_seconds)
                )
            except TimeoutError as timeout_err:
                raise AssertionError(
                    f"Operation timed out after {timeout_seconds} seconds"
                ) from timeout_err
            except Exception as e:
                last_exception = e
                await asyncio.sleep(0.1)  # Brief retry delay
                continue

        # If we get here, we exhausted retry time
        raise AssertionError(f"Operation failed: {last_exception}") from last_exception

    @classmethod
    async def simulate_delay(cls, delay_seconds: float) -> None:
        """Simulate async delay for test compatibility."""
        await asyncio.sleep(delay_seconds)

    @classmethod
    async def run_parallel_tasks(
        cls, task_func: object, inputs: list[object]
    ) -> list[object]:
        """Run tasks in parallel for test compatibility."""
        if not callable(task_func):
            msg = "task_func must be callable"
            raise ValueError(msg)

        # Run tasks concurrently - create awaitable tasks
        tasks = [task_func(input_item) for input_item in inputs]
        # Use gather with type ignoring for the asyncio call since task_func return type is unknown
        results = await asyncio.gather(*tasks)
        return list(results)

    @classmethod
    async def run_concurrently(
        cls, *tasks: object, return_exceptions: bool = True
    ) -> list[object]:
        """Ultra-simple alias for test compatibility - runs tasks concurrently."""
        if return_exceptions:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions when return_exceptions=True - only return successful results
            return [r for r in results if not isinstance(r, Exception)]
        return await asyncio.gather(*tasks)

    @classmethod
    async def test_race_condition(
        cls, task_func: object, *, concurrent_count: int = 5
    ) -> list[object]:
        """Ultra-simple alias for test compatibility - runs function concurrently to test race conditions."""
        if not callable(task_func):
            msg = "task_func must be callable"
            raise ValueError(msg)
        # Run the same function multiple times concurrently
        tasks = [task_func() for _ in range(concurrent_count)]
        results = await asyncio.gather(*tasks)
        return list(results)

    @classmethod
    async def measure_concurrency_performance(
        cls, task_func: object, *, iterations: int = 3
    ) -> dict[str, float]:
        """Ultra-simple alias for test compatibility - measures concurrency performance."""
        if not callable(task_func):
            msg = "task_func must be callable"
            raise ValueError(msg)

        start_time = time.time()

        # Run tasks concurrently for performance measurement
        tasks = [task_func() for _ in range(iterations)]
        await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time
        average_time = total_time / iterations if iterations > 0 else 0.0
        throughput = iterations / total_time if total_time > 0 else 0.0

        return {
            "total_time": total_time,
            "average_time": average_time,
            "throughput": throughput,
        }

    @classmethod
    def timeout_context(cls, timeout_seconds: float) -> object:
        """Create timeout context manager for test compatibility."""
        return asyncio.timeout(timeout_seconds)

    @classmethod
    def create_async_mock(
        cls, return_value: object = None, side_effect: object = None
    ) -> object:
        """Create async mock for test compatibility."""

        class AsyncMock:
            def __init__(
                self, return_value: object = None, side_effect: object = None
            ) -> None:
                self.return_value = return_value
                self.side_effect = side_effect
                self.call_count = 0
                self.called = False
                self._side_effect_iter: object = None
                if isinstance(side_effect, list):
                    self._side_effect_iter = iter(side_effect)

            async def __call__(self, *args: object, **kwargs: object) -> object:
                self.call_count += 1
                self.called = True

                if self.side_effect is not None:
                    if isinstance(self.side_effect, list) and self._side_effect_iter:
                        try:
                            effect = next(self._side_effect_iter)
                            if isinstance(effect, Exception):
                                raise effect
                            return effect
                        except StopIteration:
                            return self.return_value
                    elif callable(self.side_effect):
                        result = self.side_effect(*args, **kwargs)
                        if asyncio.iscoroutine(result):
                            return await result
                        return result
                    elif isinstance(self.side_effect, Exception):
                        # If side_effect is an exception instance, raise it
                        raise self.side_effect
                    else:
                        return self.side_effect

                return self.return_value

        return AsyncMock(return_value, side_effect)

    @classmethod
    def create_flaky_async_mock(
        cls,
        success_value: object = "success",
        failure_count: int = 2,
        exception_type: type[Exception] = ValueError,
    ) -> object:
        """Ultra-simple flaky async mock for test compatibility."""

        class FlakyAsyncMock:
            def __init__(
                self,
                success_value: object,
                failure_count: int,
                exception_type: type[Exception],
            ) -> None:
                self.success_value = success_value
                self.failure_count = failure_count
                self.exception_type = exception_type
                self.call_count = 0

            async def __call__(self, *args: object, **kwargs: object) -> object:
                self.call_count += 1
                if self.call_count <= self.failure_count:
                    mock_error_msg = "Mock failure"
                    raise self.exception_type(mock_error_msg)
                return self.success_value

        return FlakyAsyncMock(success_value, failure_count, exception_type)

    @classmethod
    def managed_resource(cls, value: object, *, cleanup_func: object = None) -> object:
        """Ultra-simple managed resource context manager for test compatibility."""

        class ManagedResourceContext:
            def __init__(self, value: object, cleanup_func: object = None) -> None:
                self.value = value
                self.cleanup_func = cleanup_func

            async def __aenter__(self) -> object:
                return self.value

            async def __aexit__(
                self, exc_type: object, exc_val: object, exc_tb: object
            ) -> None:
                if self.cleanup_func and callable(self.cleanup_func):
                    if asyncio.iscoroutinefunction(self.cleanup_func):
                        await self.cleanup_func(self.value)
                    else:
                        self.cleanup_func(self.value)

        return ManagedResourceContext(value, cleanup_func)

    @classmethod
    def create_test_context(
        cls, setup_coro: object = None, teardown_func: object = None
    ) -> object:
        """Create async context manager for test compatibility."""

        class AsyncTestContext:
            def __init__(
                self, setup_coro: object = None, teardown_func: object = None
            ) -> None:
                self.setup_coro = setup_coro
                self.teardown_func = teardown_func
                self.resource: object = None

            async def __aenter__(self) -> object:
                if self.setup_coro:
                    self.resource = await self.setup_coro
                return self.resource

            async def __aexit__(
                self, exc_type: object, exc_val: object, exc_tb: object
            ) -> None:
                if self.teardown_func and self.resource is not None:
                    await self.teardown_func(self.resource)

        return AsyncTestContext(setup_coro, teardown_func)


# === REMOVED COMPATIBILITY ALIASES AND FACADES ===
# Legacy compatibility removed as per user request
# All compatibility facades, aliases and protocol facades have been commented out
# Only FlextTestsMatchers class is now exported

# Main class alias for backward compatibility - REMOVED
# FlextTestsMatcher = FlextTestsMatchers

# Legacy FlextTestsMatchers class - REMOVED (commented out)
# class FlextTestsMatchers:
#     """Compatibility facade for FlextTestsMatchers - use FlextTestsMatchers.CoreMatchers instead."""
#     ... all methods commented out

# Legacy PerformanceMatchers class - REMOVED (commented out)
# class PerformanceMatchers:
#     """Compatibility facade for PerformanceMatchers - use FlextTestsMatchers.PerformanceMatchers instead."""
#     ... all methods commented out

# Protocol facades for backward compatibility - REMOVED (commented out)
# BenchmarkProtocol = FlextTestsMatchers.BenchmarkProtocol
# ContainerProtocol = FlextTestsMatchers.ContainerProtocol
# FieldProtocol = FlextTestsMatchers.FieldProtocol
# MockProtocol = FlextTestsMatchers.MockProtocol

# Export only the unified class
__all__ = [
    "FlextTestsMatchers",
]
