"""Modern tests for flext_core.decorators - Enterprise Decorator Patterns.

Refactored test suite using comprehensive testing libraries for decorator functionality.
Demonstrates SOLID principles, decorator patterns, and extensive test automation.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import Mock

import pytest
from hypothesis import given, strategies as st
from tests.conftest import TestScenario
from tests.support.async_utils import AsyncTestUtils
from tests.support.domain_factories import UserDataFactory
from tests.support.factory_boy_factories import (
    EdgeCaseGenerators,
    UserFactory,
    create_validation_test_cases,
)
from tests.support.performance_utils import BenchmarkUtils, PerformanceProfiler

from flext_core.decorators import FlextDecorators

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# CORE DECORATOR FUNCTIONALITY TESTS
# ============================================================================


class TestFlextDecoratorsCore:
    """Test core decorator functionality with factory patterns."""

    def test_safe_result_decorator_success(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test safe_result decorator with successful function."""
        user_data = user_data_factory.build()

        @FlextDecorators.safe_result
        def process_user_data(data: dict[str, Any]) -> str:
            return f"Processed: {data['name']}"

        result = process_user_data(user_data)

        assert result.success
        assert f"Processed: {user_data['name']}" in result.value

    def test_safe_result_decorator_failure(self) -> None:
        """Test safe_result decorator with failing function."""

        @FlextDecorators.safe_result
        def failing_function() -> str:
            msg = "Test error"
            raise ValueError(msg)

        result = failing_function()

        assert result.is_failure
        assert "Test error" in result.error

    def test_safe_result_decorator_with_factory_data(self) -> None:
        """Test safe_result decorator with factory_boy generated data."""
        user = UserFactory()

        @FlextDecorators.safe_result
        def validate_user_email(email: str) -> str:
            if "@" not in email:
                msg = "Invalid email format"
                raise ValueError(msg)
            return f"Valid email: {email}"

        result = validate_user_email(user.email)

        assert result.success
        assert "Valid email:" in result.value

    @pytest.mark.parametrize("test_case", create_validation_test_cases())
    def test_safe_result_with_validation_cases(self, test_case: dict) -> None:
        """Test safe_result with comprehensive validation test cases."""

        @FlextDecorators.safe_result
        def validate_data(data: Any) -> str:
            if data is None:
                msg = "Data cannot be None"
                raise ValueError(msg)
            return f"Validated: {data}"

        result = validate_data(test_case["data"])

        if test_case["expected_valid"]:
            assert result.success
            assert "Validated:" in result.value
        else:
            # May succeed or fail depending on the data
            pass


# ============================================================================
# PERFORMANCE DECORATOR TESTS
# ============================================================================


class TestFlextDecoratorsPerformance:
    """Test decorator performance characteristics using benchmark utilities."""

    def test_cached_decorator_performance(self, benchmark: object) -> None:
        """Benchmark cached decorator performance."""
        call_count = 0

        @FlextDecorators.cached_with_timing(max_size=100)
        def expensive_calculation(n: int) -> int:
            nonlocal call_count
            call_count += 1
            return sum(range(n))

        def benchmark_operations() -> list[int]:
            # First calls (cache misses)
            results = [expensive_calculation(i) for i in range(10)]
            # Repeated calls (cache hits)
            results.extend([expensive_calculation(i) for i in range(10)])
            return results

        results = BenchmarkUtils.benchmark_with_warmup(
            benchmark, benchmark_operations, warmup_rounds=3
        )

        assert len(results) == 20
        # Verify caching worked - only 10 unique calls
        assert call_count == 10

    def test_timing_decorator_overhead(self) -> None:
        """Test timing decorator overhead with performance profiler."""
        profiler = PerformanceProfiler()

        @FlextDecorators.cached_with_timing()
        def simple_operation(x: int) -> int:
            return x * 2

        with profiler.profile_time("timing_decorator_overhead"):
            for i in range(1000):
                simple_operation(i % 10)  # Ensure cache hits

        profiler.assert_time_efficient(
            max_time_ms=50.0, operation_name="timing_decorator_overhead"
        )

    def test_memory_efficiency_of_decorators(self) -> None:
        """Test memory efficiency of decorator applications."""
        profiler = PerformanceProfiler()

        with profiler.profile_memory("decorator_memory"):
            # Apply decorators to many functions
            decorated_functions = []
            for i in range(100):

                @FlextDecorators.safe_result
                def func(x: int = i) -> int:
                    return x * 2

                decorated_functions.append(func)

        # Assert reasonable memory usage
        profiler.assert_memory_efficient(
            max_memory_mb=10.0, operation_name="decorator_memory"
        )


# ============================================================================
# VALIDATION DECORATOR TESTS
# ============================================================================


class TestFlextDecoratorsValidation:
    """Test validation decorator patterns."""

    def test_validation_decorator_with_factory_models(self) -> None:
        """Test validation decorator with factory_boy models."""

        def validate_user_data(data: dict) -> bool:
            required_fields = ["name", "email", "age"]
            return all(field in data for field in required_fields)

        @FlextDecorators.validated_with_result(validate_user_data)
        def create_user_profile(data: dict) -> str:
            return f"Profile created for {data['name']}"

        # Valid data from factory
        user = UserFactory()
        user_data = {
            "name": user.name,
            "email": user.email,
            "age": user.age,
        }

        result = create_user_profile(user_data)

        assert result.success
        assert "Profile created for" in result.value

    def test_validation_decorator_edge_cases(self) -> None:
        """Test validation decorator with edge case data."""

        def validate_positive_number(value: Any) -> bool:
            return isinstance(value, (int, float)) and value > 0

        @FlextDecorators.validated_with_result(validate_positive_number)
        def calculate_square_root(value: float) -> float:
            return value**0.5

        # Test with edge case values
        edge_cases = EdgeCaseGenerators.boundary_numbers()

        for test_value in edge_cases:
            result = calculate_square_root(test_value)

            if test_value > 0:
                assert result.success
                assert result.value == test_value**0.5
            else:
                assert result.is_failure


# ============================================================================
# ASYNC DECORATOR TESTS
# ============================================================================


class TestFlextDecoratorsAsync:
    """Test decorators in async contexts."""

    @pytest.mark.asyncio
    async def test_safe_result_decorator_async(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test safe_result decorator with async functions."""
        user_data = user_data_factory.build()

        @FlextDecorators.safe_result
        async def async_process_user(data: dict) -> str:
            await AsyncTestUtils.sleep_with_timeout(0.01)
            return f"Async processed: {data['name']}"

        result = await async_process_user(user_data)

        assert result.success
        assert f"Async processed: {user_data['name']}" in result.value

    @pytest.mark.asyncio
    async def test_async_decorator_error_handling(self) -> None:
        """Test async decorator error handling."""

        @FlextDecorators.safe_result
        async def async_failing_operation() -> str:
            await AsyncTestUtils.sleep_with_timeout(0.01)
            msg = "Async operation failed"
            raise ValueError(msg)

        result = await async_failing_operation()

        assert result.is_failure
        assert "Async operation failed" in result.error

    @pytest.mark.asyncio
    async def test_concurrent_decorated_operations(self) -> None:
        """Test concurrent execution of decorated async operations."""

        @FlextDecorators.safe_result
        async def async_calculation(value: int) -> int:
            await AsyncTestUtils.sleep_with_timeout(0.001)
            return value * 2

        # Test concurrent execution
        results = await AsyncTestUtils.run_concurrent([
            async_calculation(1),
            async_calculation(2),
            async_calculation(3),
        ])

        assert len(results) == 3
        assert all(r.success for r in results)
        assert {r.value for r in results} == {2, 4, 6}


# ============================================================================
# PROPERTY-BASED DECORATOR TESTS
# ============================================================================


class TestFlextDecoratorsProperties:
    """Property-based tests for decorator invariants."""

    @given(st.integers())
    def test_safe_result_preserves_values(self, value: int) -> None:
        """Property: safe_result preserves function return values."""

        @FlextDecorators.safe_result
        def identity_function(x: int) -> int:
            return x

        result = identity_function(value)
        assert result.success
        assert result.value == value

    @given(st.text())
    def test_safe_result_captures_exceptions(self, error_message: str) -> None:
        """Property: safe_result captures all exceptions."""

        @FlextDecorators.safe_result
        def failing_function() -> str:
            raise ValueError(error_message)

        result = failing_function()
        assert result.is_failure
        assert error_message in result.error

    @given(st.lists(st.integers(), min_size=1, max_size=10))
    def test_cached_decorator_consistency(self, values: list[int]) -> None:
        """Property: cached decorator returns consistent results."""
        call_count = 0

        @FlextDecorators.cached_with_timing(max_size=100)
        def process_value(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 3

        # Call with each value twice
        first_results = [process_value(v) for v in values]
        second_results = [process_value(v) for v in values]

        # Results should be identical
        assert first_results == second_results
        # Should only have called function once per unique value
        assert call_count == len(set(values))


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestFlextDecoratorsIntegration:
    """Integration tests using test scenarios."""

    def test_decorator_composition_integration(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test composition of multiple decorators."""
        user_data = user_data_factory.build()

        def validate_data(data: dict) -> bool:
            return "name" in data and "email" in data

        @FlextDecorators.cached_with_timing(max_size=10)
        @FlextDecorators.validated_with_result(validate_data)
        @FlextDecorators.safe_result
        def complex_user_operation(data: dict) -> str:
            return f"Complex operation for {data['name']}"

        result = complex_user_operation(user_data)

        assert result.success
        assert f"Complex operation for {user_data['name']}" in result.value

    def test_error_handling_scenarios(self, test_scenarios: list[TestScenario]) -> None:
        """Test decorator error handling with various scenarios."""
        error_scenario = next(
            (s for s in test_scenarios if s.scenario_type == "error"), None
        )
        if not error_scenario:
            pytest.skip("No error scenario available")

        @FlextDecorators.safe_result
        def scenario_processor(data: Any) -> str:
            if not data:
                msg = "Empty data"
                raise ValueError(msg)
            return f"Processed: {data}"

        result = scenario_processor("")

        assert result.is_failure
        assert "Empty data" in result.error


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestFlextDecoratorsEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.unicode_strings())
    def test_unicode_handling_in_decorators(self, edge_value: str) -> None:
        """Test decorator handling of unicode strings."""

        @FlextDecorators.safe_result
        def process_unicode(text: str) -> str:
            return f"Processed: {text}"

        result = process_unicode(edge_value)

        assert result.success
        assert "Processed:" in result.value

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.boundary_numbers())
    def test_boundary_number_handling(self, edge_value: float) -> None:
        """Test decorator handling of boundary numbers."""

        @FlextDecorators.safe_result
        def mathematical_operation(x: float) -> float:
            if x == 0:
                msg = "Cannot divide by zero"
                raise ZeroDivisionError(msg)
            return 1.0 / x

        result = mathematical_operation(edge_value)

        if edge_value == 0:
            assert result.is_failure
            assert "divide by zero" in result.error.lower()
        else:
            assert result.success
            assert result.value == 1.0 / edge_value

    @pytest.mark.parametrize("empty_value", EdgeCaseGenerators.empty_values())
    def test_empty_value_handling(self, empty_value: Any) -> None:
        """Test decorator handling of empty/null values."""

        @FlextDecorators.safe_result
        def process_empty_value(value: Any) -> str:
            if value is None or value == "":
                msg = "Value cannot be empty"
                raise ValueError(msg)
            return f"Value: {value}"

        result = process_empty_value(empty_value)

        if empty_value is None or empty_value == "":
            assert result.is_failure
            assert "cannot be empty" in result.error
        else:
            assert result.success
            assert "Value:" in result.value


# ============================================================================
# MOCKING AND SIDE EFFECTS TESTS
# ============================================================================


class TestFlextDecoratorsMocking:
    """Test decorators with mocking and side effects."""

    def test_decorator_with_mocked_dependencies(self, mocker) -> None:
        """Test decorated functions with mocked dependencies."""
        mock_service = mocker.Mock()
        mock_service.process_data.return_value = "mocked_result"

        @FlextDecorators.safe_result
        def service_operation(data: str) -> str:
            return mock_service.process_data(data)

        result = service_operation("test_data")

        assert result.success
        assert result.value == "mocked_result"
        mock_service.process_data.assert_called_once_with("test_data")

    def test_decorator_side_effects_tracking(self, mocker) -> None:
        """Test tracking of side effects in decorated functions."""
        side_effect_tracker = Mock()

        @FlextDecorators.safe_result
        def function_with_side_effects(value: int) -> int:
            side_effect_tracker.record(f"processing_{value}")
            return value * 2

        result = function_with_side_effects(5)

        assert result.success
        assert result.value == 10
        side_effect_tracker.record.assert_called_once_with("processing_5")


# ============================================================================
# STRESS TESTING
# ============================================================================


class TestFlextDecoratorsStress:
    """Stress tests for decorator patterns."""

    def test_high_frequency_decorator_calls(self) -> None:
        """Test decorator performance under high frequency calls."""
        call_count = 0

        @FlextDecorators.cached_with_timing(max_size=1000)
        def high_frequency_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x**2

        # Make many calls
        for _ in range(10000):
            value = time.time_ns() % 100  # Vary the input
            result = high_frequency_operation(int(value))
            assert isinstance(result, int)

        # Should have cached many calls
        assert call_count < 10000

    def test_decorator_memory_under_stress(self) -> None:
        """Test decorator memory usage under stress conditions."""
        profiler = PerformanceProfiler()

        with profiler.profile_memory("stress_test"):
            # Create many decorated functions
            functions = []
            for i in range(1000):

                @FlextDecorators.safe_result
                def stress_function(x: int = i) -> int:
                    return x + 1

                functions.append(stress_function)

            # Call all functions
            results = [func() for func in functions]
            assert len(results) == 1000

        # Memory should be reasonable
        profiler.assert_memory_efficient(
            max_memory_mb=50.0, operation_name="stress_test"
        )
