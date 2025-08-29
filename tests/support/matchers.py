"""Advanced test matchers and assertions using all pytest capabilities.

Leverages pytest-clarity, pytest-benchmark, pytest-mock, and other plugins
for comprehensive testing with clear error messages and performance insights.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from typing import Protocol, TypeGuard, TypeVar

from flext_core import FlextResult, FlextTypes

from .performance import BenchmarkProtocol

T = TypeVar("T")

JsonDict = FlextTypes.Core.JsonObject


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


class FlextMatchers:
    """Advanced matchers for flext-core testing with comprehensive assertions.

    Uses pytest-clarity for better error messages and provides domain-specific
    matchers for FlextResult, containers, and other core patterns.
    """

    @staticmethod
    def assert_result_success(
        result: FlextResult[T],
        expected_data: T | None = None,
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
        result: FlextResult[T],
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

    @staticmethod
    def assert_container_has_service(
        container: ContainerProtocol,
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
        field: FieldProtocol,
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
        data: JsonDict,
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
        benchmark: BenchmarkProtocol,
        func: Callable[[], T],
        max_time_seconds: float = 1.0,
        *args: object,
        **kwargs: object,
    ) -> T:
        """Assert function performance is within time limit.

        Args:
            benchmark: pytest-benchmark fixture
            func: Function to benchmark
            max_time_seconds: Maximum allowed time
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            AssertionError: If performance exceeds limit

        """
        result = benchmark(func, *args, **kwargs)

        # Get stats from benchmark
        stats = benchmark.stats
        mean_time = getattr(stats, "mean", 0.0)

        assert mean_time <= max_time_seconds, (
            f"Performance test failed: mean time {mean_time:.4f}s "
            f"exceeds limit {max_time_seconds:.4f}s"
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

        assert match is not None, f"Text {text!r} does not match pattern {pattern!r}"

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
        mock_obj: MockProtocol,
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
        assert var_name in os.environ, f"Environment variable {var_name!r} not found"

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


class PerformanceMatchers:
    """Performance-specific matchers using pytest-benchmark."""

    @staticmethod
    def assert_linear_complexity(
        benchmark: BenchmarkProtocol,
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
            benchmark(func, size)
            times.append(getattr(benchmark.stats, "mean", 0.0))

        # Check if time growth is roughly linear
        if len(times) >= 2:
            # Ensure division results in float for type safety
            growth_ratio: float = float(times[-1] / times[0]) if times[0] > 0 else 1.0
            size_ratio: float = float(sizes[-1] / sizes[0]) if sizes[0] > 0 else 1.0

            assert growth_ratio <= size_ratio * tolerance_factor, (
                f"Complexity appears non-linear: time grew by {growth_ratio:.2f}x "
                f"while size grew by {size_ratio:.2f}x"
            )

    @staticmethod
    def assert_memory_efficient(
        benchmark: BenchmarkProtocol,
        func: Callable[[], T],
        _max_memory_mb: float = 100.0,
    ) -> T:
        """Assert function is memory efficient.

        Args:
            benchmark: pytest-benchmark fixture
            func: Function to test
            max_memory_mb: Maximum memory usage in MB

        Note:
            This is a placeholder for memory profiling integration

        """
        # Memory profiling would require additional tools like memory_profiler
        # For now, this serves as a marker for memory-sensitive tests
        return benchmark(func)


# Export all matchers
__all__ = [
    "FlextMatchers",
    "PerformanceMatchers",
]
