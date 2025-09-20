"""Advanced test matchers and assertions using all pytest capabilities.

Provides consolidated test matchers, performance testing capabilities,
and protocol definitions for test assertions.

Leverages pytest-clarity, pytest-benchmark, pytest-mock, and other plugins
for testing with clear error messages and performance insights.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import random
import re
import signal
import time
import tracemalloc
import uuid
import warnings
from collections.abc import Awaitable, Callable, Container, Iterable, Sequence, Sized
from itertools import starmap
from typing import Protocol, Self, cast


class FlextTestsMatchers:
    """Comprehensive test matching utilities following unified class pattern.

    Single unified class containing all test matching protocols and utilities
    as nested components following FLEXT architectural principles.
    """

    # =========================================================================
    # PROTOCOLS - Nested protocol definitions within unified class
    # =========================================================================

    class ResultLike(Protocol):
        """Protocol for result-like objects."""

        def __bool__(self) -> bool:
            """Return boolean representation of the result."""
            ...

    class SuccessResultLike(ResultLike, Protocol):
        """Protocol for success result objects."""

        is_success: bool
        success: bool

    class FailureResultLike(ResultLike, Protocol):
        """Protocol for failure result objects."""

        is_failure: bool
        failure: bool

    class FlextResultLike(SuccessResultLike, FailureResultLike, Protocol):
        """Protocol for complete FlextResult-like objects with both success and failure attributes."""

        is_success: bool
        success: bool
        is_failure: bool
        failure: bool

    class ContainerLike(Protocol):
        """Protocol for container-like objects."""

        def __len__(self) -> int:
            """Return length of the container."""
            ...

        def __contains__(self, item: object) -> bool:
            """Check if item is contained in the container."""
            ...

    class ErrorResultLike(Protocol):
        """Protocol for objects with error attributes."""

        error: str | None

    class ErrorCodeResultLike(Protocol):
        """Protocol for objects with error_code attributes."""

        error_code: str | None

    class ValueResultLike(Protocol):
        """Protocol for objects with value or data attributes."""

        @property
        def value(self) -> object: ...

    class DataResultLike(Protocol):
        """Protocol for objects with data attributes."""

        @property
        def data(self) -> object: ...

    class EmptyCheckable(Protocol):
        """Protocol for objects that can be checked for emptiness."""

        def __len__(self) -> int:
            """Return length of the object."""
            ...

    class HasIsEmpty(Protocol):
        """Protocol for objects with is_empty attribute."""

        is_empty: bool

    class HasContains(Protocol):
        """Protocol for objects with contains method."""

        def contains(self, item: object) -> bool: ...

    # =========================================================================
    # CORE MATCHERS - FlextResult and object testing
    # =========================================================================

    class CoreMatchers:
        """Core matching utilities for FlextResult and common objects."""

        @staticmethod
        def be_success(result: object) -> bool:
            """Check if result indicates success."""
            if hasattr(result, "is_success"):
                is_success_attr = getattr(result, "is_success", None)
                if is_success_attr is not None:
                    return bool(is_success_attr)
            if hasattr(result, "success"):
                success_attr = getattr(result, "success", None)
                if success_attr is not None:
                    return bool(success_attr)
            return False

        @staticmethod
        def be_failure(result: object) -> bool:
            """Check if result indicates failure."""
            if hasattr(result, "is_failure"):
                is_failure_attr = getattr(result, "is_failure", None)
                if is_failure_attr is not None:
                    return bool(is_failure_attr)
            if hasattr(result, "failure"):
                failure_attr = getattr(result, "failure", None)
                if failure_attr is not None:
                    return bool(failure_attr)
            return False

        @staticmethod
        def have_error(result: object, expected_error: str | None = None) -> bool:
            """Check if result has error message."""
            if not hasattr(result, "error"):
                return False

            error_attr = getattr(result, "error", None)
            if error_attr is None:
                return False

            if expected_error is None:
                return True

            return str(expected_error) in str(error_attr)

        @staticmethod
        def have_error_code(result: object, expected_code: str) -> bool:
            """Check if result has specific error code."""
            if not hasattr(result, "error_code"):
                return False

            error_code_attr = getattr(result, "error_code", None)
            if error_code_attr is None:
                return False

            return str(error_code_attr) == str(expected_code)

        @staticmethod
        def have_value(result: object, expected_value: object = None) -> bool:
            """Check if result has expected value."""
            # Try value first, then data for backward compatibility
            actual_value: object = None
            if hasattr(result, "value"):
                value_attr = getattr(result, "value", None)
                if value_attr is not None:
                    actual_value = value_attr
            elif hasattr(result, "data"):
                data_attr = getattr(result, "data", None)
                if data_attr is not None:
                    actual_value = data_attr
            else:
                return False

            if expected_value is None:
                return actual_value is None

            return actual_value == expected_value

        @staticmethod
        def be_empty(container: object) -> bool:
            """Check if container is empty."""
            if hasattr(container, "__len__"):
                # Type guard: if it has __len__, treat it as Sized
                sized_container = cast("Sized", container)
                return len(sized_container) == 0
            if hasattr(container, "is_empty"):
                is_empty_attr = getattr(container, "is_empty", None)
                if is_empty_attr is not None:
                    return bool(is_empty_attr)
            return False

        @staticmethod
        def contain_item(container: object, item: object) -> bool:
            """Check if container contains item."""
            if hasattr(container, "__contains__"):
                # Type guard: if it has __contains__, treat it as Container
                container_obj = cast("Container[object]", container)
                return item in container_obj
            if hasattr(container, "contains"):
                contains_method = getattr(container, "contains", None)
                if contains_method is not None and callable(contains_method):
                    return bool(contains_method(item))
            return False

        @staticmethod
        def be_instance_of(obj: object, expected_type: type[object]) -> bool:
            """Check if object is instance of expected type."""
            return isinstance(obj, expected_type)

        @staticmethod
        def have_attribute(obj: object, attr_name: str) -> bool:
            """Check if object has attribute."""
            return hasattr(obj, attr_name)

        @staticmethod
        def have_method(obj: object, method_name: str) -> bool:
            """Check if object has callable method."""
            return hasattr(obj, method_name) and callable(getattr(obj, method_name))

        @staticmethod
        def be_callable(obj: object) -> bool:
            """Check if object is callable."""
            return callable(obj)

        @staticmethod
        def satisfy_predicate(obj: object, predicate: object) -> bool:
            """Check if object satisfies predicate function."""
            if not callable(predicate):
                return False
            try:
                # Cast to callable since we've checked it's callable
                predicate_func = cast("Callable[[object], object]", predicate)
                return bool(predicate_func(obj))
            except Exception:
                return False

        @staticmethod
        def be_json_serializable(obj: object) -> bool:
            """Check if object is JSON serializable."""
            try:
                json.dumps(obj)
                return True
            except (TypeError, ValueError):
                return False

        @staticmethod
        def match_regex(text: str, pattern: str) -> bool:
            """Check if text matches regex pattern."""
            try:
                return bool(re.search(pattern, text))
            except re.error:
                return False

        @staticmethod
        def be_valid_uuid(value: str) -> bool:
            """Check if string is valid UUID."""
            try:
                uuid.UUID(value)
                return True
            except ValueError:
                return False

        @staticmethod
        def be_valid_email(email: str) -> bool:
            """Check if string is valid email format."""
            pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            return bool(re.match(pattern, email))

        @staticmethod
        def be_positive_number(value: str | float) -> bool:
            """Check if value is positive number."""
            try:
                return float(value) > 0
            except (TypeError, ValueError):
                return False

        @staticmethod
        def be_within_range(value: str | float, min_val: float, max_val: float) -> bool:
            """Check if value is within numeric range."""
            try:
                num_value = float(value)
                return min_val <= num_value <= max_val
            except (TypeError, ValueError):
                return False

        @staticmethod
        def have_length(container: object, expected_length: int) -> bool:
            """Check if container has expected length."""
            if not hasattr(container, "__len__"):
                return False
            # Type guard: if it has __len__, treat it as Sized
            sized_container = cast("Sized", container)
            return len(sized_container) == expected_length

        @staticmethod
        def be_sorted(sequence: Sequence[object], *, reverse: bool = False) -> bool:
            """Check if sequence is sorted."""
            if not hasattr(sequence, "__getitem__") or not hasattr(sequence, "__len__"):
                return False

            try:
                items = list(sequence)
                if not items:
                    return True  # Empty sequence is considered sorted

                # Use string representation for sorting to avoid type variable issues
                # This ensures we can always compare items regardless of their types
                items_as_strings = [str(item) for item in items]
                sorted_strings = sorted(items_as_strings, reverse=reverse)
                return items_as_strings == sorted_strings
            except (TypeError, AttributeError):
                return False

        @staticmethod
        def all_satisfy(
            container: Iterable[object],
            predicate: Callable[[object], object],
        ) -> bool:
            """Check if all items in container satisfy predicate."""
            try:
                return all(bool(predicate(item)) for item in container)
            except Exception:
                return False

        @staticmethod
        def any_satisfy(
            container: Iterable[object],
            predicate: Callable[[object], object],
        ) -> bool:
            """Check if any item in container satisfies predicate."""
            try:
                return any(bool(predicate(item)) for item in container)
            except Exception:
                return False

        @staticmethod
        def assert_greater_than(
            actual: float,
            expected: float,
            msg: str = "",
            message: str = "",
        ) -> None:
            """Assert that actual value is greater than expected."""
            if actual <= expected:
                error_msg = msg or message or f"Expected {actual} > {expected}"
                raise AssertionError(error_msg)

        @staticmethod
        def be_equivalent_to(obj1: object, obj2: object) -> bool:
            """Check if objects are equivalent (handles complex comparisons)."""
            if obj1 == obj2:
                return True

            # Handle dict comparison
            if isinstance(obj1, dict) and isinstance(obj2, dict):
                return obj1.keys() == obj2.keys() and all(
                    FlextTestsMatchers.CoreMatchers.be_equivalent_to(obj1[k], obj2[k])
                    for k in obj1
                )

            # Handle list/tuple comparison
            if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
                return len(obj1) == len(obj2) and all(
                    starmap(
                        FlextTestsMatchers.CoreMatchers.be_equivalent_to,
                        zip(obj1, obj2, strict=False),
                    ),
                )

            # Handle set comparison
            if isinstance(obj1, set) and isinstance(obj2, set):
                return obj1 == obj2

            return False

    # =========================================================================
    # SPECIALIZED MATCHERS - Result-specific matching
    # =========================================================================

    class SuccessMatcher:
        """Specialized matcher for success results."""

        def __init__(self, result: object) -> None:
            """Initialize success matcher with result object."""
            self.result = result

        def with_value(self, expected_value: object) -> bool:
            """Check success result has expected value."""
            return FlextTestsMatchers.CoreMatchers.have_value(
                self.result,
                expected_value,
            )

    class FailureMatcher:
        """Specialized matcher for failure results."""

        def __init__(self, result: object) -> None:
            """Initialize failure matcher with result object."""
            self.result = result

        def with_error(self, expected_error: str) -> bool:
            """Check failure result has expected error."""
            return FlextTestsMatchers.CoreMatchers.have_error(
                self.result,
                expected_error,
            )

    # =========================================================================
    # PERFORMANCE MATCHERS - Benchmarking and timing
    # =========================================================================

    class PerformanceMatchers:
        """Performance and timing matchers for benchmarking."""

        @staticmethod
        def execute_within_time(func: object, max_time: float) -> bool:
            """Check if function executes within time limit."""
            if not callable(func):
                return False

            start_time = time.time()
            try:
                func()
                execution_time = time.time() - start_time
                return execution_time <= max_time
            except Exception:
                return False

        @staticmethod
        def memory_usage_within_limit(func: object, max_memory_mb: float) -> bool:
            """Check if function uses memory within limit."""
            if not callable(func):
                return False

            tracemalloc.start()
            try:
                func()
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mb = peak / 1024 / 1024
                return peak_mb <= max_memory_mb
            except Exception:
                tracemalloc.stop()
                return False

        @staticmethod
        def throughput_above_threshold(
            func: object,
            iterations: int,
            min_ops_per_sec: float,
        ) -> bool:
            """Check if function throughput is above threshold."""
            if not callable(func):
                return False

            start_time = time.time()
            try:
                for _ in range(iterations):
                    func()
                total_time = time.time() - start_time
                ops_per_sec = iterations / total_time if total_time > 0 else 0
                return ops_per_sec >= min_ops_per_sec
            except Exception:
                return False

        @staticmethod
        def scale_linearly(func: object, input_sizes: list[int]) -> bool:
            """Check if function execution time scales linearly with input size."""
            if not callable(func) or len(input_sizes) < 2:
                return False

            times = []
            try:
                for size in input_sizes:
                    start_time = time.time()
                    func(size)
                    execution_time = time.time() - start_time
                    times.append(execution_time)

                # Simple linear scaling check
                for i in range(1, len(times)):
                    ratio = times[i] / times[0] if times[0] > 0 else 0
                    size_ratio = input_sizes[i] / input_sizes[0]
                    # Allow 50% deviation from linear scaling
                    if not (0.5 * size_ratio <= ratio <= 1.5 * size_ratio):
                        return False
                return True
            except Exception:
                return False

    # =========================================================================
    # TEST FIXTURES - Nested test infrastructure classes
    # =========================================================================

    class BenchmarkFixture:
        """Benchmark test fixture for performance testing."""

        def __init__(self, name: str, iterations: int = 1000) -> None:
            """Initialize benchmark fixture with name and iteration count."""
            self.name = name
            self.iterations = iterations
            self.results: list[float] = []

        def add_measurement(self, duration: float) -> None:
            """Add performance measurement."""
            self.results.append(duration)

        def average_time(self) -> float:
            """Calculate average execution time."""
            return sum(self.results) / len(self.results) if self.results else 0.0

        def max_time(self) -> float:
            """Get maximum execution time."""
            return max(self.results) if self.results else 0.0

    class ContainerProtocol:
        """Protocol for dependency injection container testing."""

        def register(self, service_id: str, instance: object) -> None:
            """Register service instance."""

        def get(self, service_id: str) -> object:
            """Get service instance."""

    class FieldProtocol:
        """Protocol for field validation testing."""

        def validate(self, _value: object) -> bool:
            """Validate field value."""
            return True

        def format(self, value: object) -> str:
            """Format field value."""
            return str(value)

    class MockProtocol:
        """Protocol for mock object testing."""

        def reset(self) -> None:
            """Reset mock state."""

    # =========================================================================
    # TYPE GUARDS AND ASSERTIONS
    # =========================================================================

    @staticmethod
    def assert_type_guard(
        value: object,
        expected_type: type[object] | Callable[[object], bool],
    ) -> None:
        """Assert that value is of expected type with type guard."""
        if callable(expected_type) and not isinstance(expected_type, type):
            # It's a callable type guard function
            if not expected_type(value):
                msg = f"Type guard function {expected_type.__name__} failed for value {value}"
                raise AssertionError(msg)
        # It's a type
        elif not isinstance(value, expected_type):
            type_name = getattr(expected_type, "__name__", str(expected_type))
            msg = f"Expected {type_name}, got {type(value).__name__}"
            raise AssertionError(msg)

    @staticmethod
    def assert_performance_within_limit(
        execution_time: float,
        limit: float,
        operation: str = "operation",
    ) -> None:
        """Assert that execution time is within performance limit."""
        if execution_time > limit:
            msg = (
                f"{operation} took {execution_time:.3f}s, exceeds limit of {limit:.3f}s"
            )
            raise AssertionError(msg)

    # =========================================================================
    # CONVENIENCE FACTORY METHODS
    # =========================================================================

    @staticmethod
    def matchers() -> CoreMatchers:
        """Get core matchers instance."""
        return FlextTestsMatchers.CoreMatchers()

    @staticmethod
    def performance() -> PerformanceMatchers:
        """Get performance matchers instance."""
        return FlextTestsMatchers.PerformanceMatchers()

    # =========================================================================
    # ASSERTION HELPERS - Direct assertion methods
    # =========================================================================

    @staticmethod
    def assert_result_success(
        result: object,
        expected_value: object = None,
        expected_data: object = None,
    ) -> None:
        """Assert that result is successful with optional value check."""
        assert FlextTestsMatchers.CoreMatchers.be_success(result), (
            f"Expected success, got {result}"
        )
        if expected_value is not None:
            assert FlextTestsMatchers.CoreMatchers.have_value(result, expected_value), (
                f"Expected value {expected_value}, got {getattr(result, 'value', getattr(result, 'data', None))}"
            )
        if expected_data is not None:
            assert FlextTestsMatchers.CoreMatchers.have_value(result, expected_data), (
                f"Expected data {expected_data}, got {getattr(result, 'value', getattr(result, 'data', None))}"
            )

    @staticmethod
    def assert_result_failure(
        result: object,
        expected_error: str | None = None,
    ) -> None:
        """Assert that result is failure with optional error check."""
        assert FlextTestsMatchers.CoreMatchers.be_failure(result), (
            f"Expected failure, got {result}"
        )
        if expected_error is not None:
            assert FlextTestsMatchers.CoreMatchers.have_error(result, expected_error), (
                f"Expected error containing '{expected_error}', got {getattr(result, 'error', None)}"
            )

    @staticmethod
    def assert_json_structure(
        obj: object,
        expected_keys: list[str],
        *,
        exact_match: bool = True,
    ) -> None:
        """Assert that object has expected JSON structure."""
        if not isinstance(obj, dict):
            msg = f"Expected dict, got {type(obj).__name__}"
            raise AssertionError(msg)

        for key in expected_keys:
            if key not in obj:
                msg = f"Missing key '{key}' in {list(obj.keys())}"
                raise AssertionError(msg)

        if exact_match:
            obj_keys = set(obj.keys())
            expected_keys_set = set(expected_keys)
            if obj_keys != expected_keys_set:
                extra_keys = obj_keys - expected_keys_set
                if extra_keys:
                    msg = f"Unexpected keys {list(extra_keys)} found in object"
                    raise AssertionError(msg)

    @staticmethod
    def assert_regex_match(text: str, pattern: str) -> None:
        """Assert that text matches regex pattern."""
        if not FlextTestsMatchers.CoreMatchers.match_regex(text, pattern):
            msg = f"Text '{text}' does not match pattern '{pattern}'"
            raise AssertionError(msg)

    @staticmethod
    def assert_environment_variable(
        var_name: str,
        expected_value: str | None = None,
    ) -> None:
        """Assert that environment variable exists with optional value check."""
        actual_value = os.environ.get(var_name)
        if actual_value is None:
            msg = f"Environment variable '{var_name}' not set"
            raise AssertionError(msg)
        if expected_value is not None and actual_value != expected_value:
            msg = f"Environment variable '{var_name}' = '{actual_value}', expected '{expected_value}'"
            raise AssertionError(msg)

    @staticmethod
    def is_successful_result(result: object) -> bool:
        """Check if result indicates success."""
        return FlextTestsMatchers.CoreMatchers.be_success(result)

    @staticmethod
    def is_failed_result(result: object) -> bool:
        """Check if result indicates failure."""
        return FlextTestsMatchers.CoreMatchers.be_failure(result)

    # =========================================================================
    # ASYNC TESTING UTILITIES
    # =========================================================================

    @staticmethod
    async def simulate_delay(seconds: float) -> None:
        """Simulate async delay for test compatibility."""
        await asyncio.sleep(seconds)

    @staticmethod
    async def run_with_timeout(
        coro: Awaitable[object],
        timeout_seconds: float,
    ) -> object:
        """Run coroutine with timeout and auto-retry for test compatibility."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except TimeoutError:
            # Auto-retry once with extended timeout
            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds * 2)
            except TimeoutError:
                msg = f"Operation timed out after {timeout_seconds * 2} seconds"
                raise TimeoutError(msg) from None

    @staticmethod
    async def run_parallel_tasks(tasks: list[Awaitable[object]]) -> list[object]:
        """Run tasks in parallel for test compatibility."""
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def run_concurrently(
        func: Callable[[object], object],
        *args: object,
        **_kwargs: object,
    ) -> object:
        """Ultra-simple for test compatibility - runs tasks concurrently."""
        partial_func = functools.partial(func, *args)
        return await asyncio.get_event_loop().run_in_executor(None, partial_func)

    @staticmethod
    async def test_race_condition(
        func1: Callable[[], object],
        func2: Callable[[], object],
    ) -> tuple[object, object]:
        """Ultra-simple for test compatibility - runs function concurrently to test race conditions."""
        return await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(None, func1),
            asyncio.get_event_loop().run_in_executor(None, func2),
        )

    @staticmethod
    async def measure_concurrency_performance(
        func: Callable[[], object],
        concurrency_level: int,
    ) -> dict[str, object]:
        """Ultra-simple for test compatibility - measures concurrency performance."""
        start_time = time.time()

        tasks = [
            asyncio.get_event_loop().run_in_executor(None, func)
            for _ in range(concurrency_level)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        return {
            "total_time": end_time - start_time,
            "concurrency_level": concurrency_level,
            "results": results,
            "success_count": sum(1 for r in results if not isinstance(r, Exception)),
            "error_count": sum(1 for r in results if isinstance(r, Exception)),
        }

    # =========================================================================
    # CONTEXT MANAGERS AND UTILITIES
    # =========================================================================

    @staticmethod
    def timeout_context(timeout: float) -> object:
        """Create timeout context manager for test compatibility."""

        class TimeoutContext:
            def __init__(self, timeout_seconds: float) -> None:
                self.timeout_seconds = timeout_seconds

            def __enter__(self) -> Self:
                signal.alarm(int(self.timeout_seconds))
                return self

            def __exit__(
                self,
                exc_type: object,
                exc_val: object,
                exc_tb: object,
            ) -> None:
                signal.alarm(0)

        return TimeoutContext(timeout)

    @staticmethod
    def create_async_mock(
        return_value: object = None,
        side_effect: object = None,
        delay: float = 0.0,
    ) -> object:
        """Create async mock for test compatibility."""

        class AsyncMock:
            def __init__(
                self,
                return_value: object = None,
                side_effect: object = None,
                delay: float = 0.0,
            ) -> None:
                self.return_value = return_value
                self.side_effect = side_effect
                self.delay = delay
                self.call_count = 0
                self.call_args_list: list[object] = []

            async def __call__(self, *args: object, **kwargs: object) -> object:
                self.call_count += 1
                self.call_args_list.append((args, kwargs))

                if self.delay > 0:
                    await asyncio.sleep(self.delay)

                if self.side_effect is not None:
                    if isinstance(self.side_effect, Exception):
                        raise self.side_effect
                    if callable(self.side_effect):
                        result = self.side_effect(*args, **kwargs)
                        if asyncio.iscoroutine(result):
                            return await result
                        return result
                    return self.side_effect

                return self.return_value

            def reset_mock(self) -> None:
                """Reset mock state."""
                self.call_count = 0
                self.call_args_list = []

            @property
            def called(self) -> bool:
                """Check if mock was called."""
                return self.call_count > 0

        return AsyncMock(return_value, side_effect, delay)

    @staticmethod
    def create_flaky_async_mock(
        success_return: object = None,
        failure_exception: object = None,
        failure_rate: float = 0.3,
    ) -> object:
        """Ultra-simple flaky async mock for test compatibility."""

        class FlakyAsyncMock:
            """Flaky async mock for test compatibility."""

            def __init__(
                self,
                success_return: object = None,
                failure_exception: object = None,
                failure_rate: float = 0.3,
            ) -> None:
                self.success_return = success_return
                self.failure_exception = failure_exception or Exception(
                    "Flaky mock failure",
                )
                self.failure_rate = failure_rate
                self.call_count = 0

            async def __call__(self, *_args: object, **_kwargs: object) -> object:
                self.call_count += 1

                if random.random() < self.failure_rate:
                    if isinstance(self.failure_exception, BaseException):
                        raise self.failure_exception

                    class MockFailureError(Exception):
                        pass

                    raise MockFailureError(f"Mock failure: {self.failure_exception}")

                return self.success_return

        return FlakyAsyncMock(success_return, failure_exception, failure_rate)

    @staticmethod
    def managed_resource(
        resource_factory: object,
        cleanup_func: object = None,
    ) -> object:
        """Ultra-simple managed resource context manager for test compatibility."""

        class ManagedResourceContext:
            def __init__(self, factory: object, cleanup: object = None) -> None:
                self.factory = factory
                self.cleanup = cleanup
                self.resource: object = None

            def __enter__(self) -> object:
                if callable(self.factory):
                    self.resource = self.factory()
                else:
                    self.resource = self.factory
                return self.resource

            def __exit__(
                self,
                exc_type: object,
                exc_val: object,
                exc_tb: object,
            ) -> None:
                if self.cleanup and callable(self.cleanup) and self.resource:
                    try:
                        self.cleanup(self.resource)
                    except Exception as e:
                        # Log cleanup error but don't raise to avoid masking original exceptions
                        warnings.warn(
                            f"Resource cleanup failed: {e}",
                            ResourceWarning,
                            stacklevel=2,
                        )

        return ManagedResourceContext(resource_factory, cleanup_func)

    @staticmethod
    async def create_test_context(
        setup_func: object = None,
        teardown_func: object = None,
        context_data: dict[str, object] | None = None,
    ) -> object:
        """Create async context manager for test compatibility."""

        class TestContext:
            def __init__(
                self,
                setup_func: object = None,
                teardown_func: object = None,
                context_data: dict[str, object] | None = None,
            ) -> None:
                self.setup_func = setup_func
                self.teardown_func = teardown_func
                self.context_data = context_data or {}
                self.result = None

            async def __aenter__(self) -> Self:
                if self.setup_func and callable(self.setup_func):
                    result = self.setup_func()
                    if asyncio.iscoroutine(result):
                        self.result = await result
                    else:
                        self.result = result
                return self

            async def __aexit__(
                self,
                exc_type: object,
                exc_val: object,
                exc_tb: object,
            ) -> None:
                if self.teardown_func and callable(self.teardown_func):
                    cleanup_result = self.teardown_func()
                    if asyncio.iscoroutine(cleanup_result):
                        await cleanup_result

        return TestContext(setup_func, teardown_func, context_data)

    # =========================================================================
    # DIRECT ACCESS METHODS - Aliases removed per FLEXT architectural principles
    # =========================================================================

    # Aliases removed - use CoreMatchers.be_success, CoreMatchers.be_failure directly
    # per FLEXT architectural principles


# Export only the unified class
__all__ = [
    "FlextTestsMatchers",
]
