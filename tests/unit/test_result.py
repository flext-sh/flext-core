"""Modern tests for FlextResult - Railway Pattern Implementation.

Refactored test suite using comprehensive testing libraries for FlextResult functionality.
Demonstrates SOLID principles, modern pytest patterns, and extensive test automation.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st
from tests.support.async_utils import AsyncTestUtils
from tests.support.domain_factories import FlextResultFactory, UserDataFactory
from tests.support.factory_boy_factories import (
    EdgeCaseGenerators,
    UserFactory,
    create_validation_test_cases,
)
from tests.support.performance_utils import BenchmarkUtils, PerformanceProfiler

# Import TestScenario from conftest.py (local enum)
# from tests.support.test_patterns import TestScenario
from flext_core import FlextConstants, FlextResult, FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================


class TestFlextResultCore:
    """Test core FlextResult functionality with factory patterns."""

    def test_ok_result_creation(self, user_data_factory: UserDataFactory) -> None:
        """Test successful result creation using factories."""
        test_data = user_data_factory.build()
        result = FlextResult.ok(test_data)

        assert result.success
        assert not result.is_failure
        assert result.value == test_data
        assert result.error is None

    def test_fail_result_creation(self) -> None:
        """Test failure result creation."""
        error_msg = FlextConstants.Messages.OPERATION_FAILED
        result = FlextResult[None].fail(error_msg)

        assert result.is_failure
        assert not result.success
        assert result.error == error_msg
        # Note: Cannot access .value on failed result - it raises TypeError

    @pytest.mark.parametrize(
        "test_data", [42, "string_value", [1, 2, 3], {"key": "value"}, None]
    )
    def test_ok_with_various_types(self, test_data: FlextTypes.Core.Object) -> None:
        """Test ok() with various data types using parametrization."""
        result = FlextResult.ok(test_data)
        assert result.success
        assert result.value == test_data

    def test_result_factory_integration(self) -> None:
        """Test integration with FlextResultFactory."""
        success_result = FlextResultFactory.create_success(
            FlextConstants.Messages.SUCCESS
        )
        failure_result = FlextResultFactory.create_failure(
            FlextConstants.Messages.OPERATION_FAILED,
            FlextConstants.Errors.VALIDATION_ERROR,
        )

        assert success_result.success
        assert success_result.value == FlextConstants.Messages.SUCCESS

        assert failure_result.is_failure
        assert failure_result.error == FlextConstants.Messages.OPERATION_FAILED
        assert failure_result.error_code == FlextConstants.Errors.VALIDATION_ERROR


# ============================================================================
# RAILWAY PATTERN TESTS
# ============================================================================


class TestFlextResultRailway:
    """Test railway-oriented programming patterns."""

    def test_map_success_transformation(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test map transformation on successful results."""
        user_data = user_data_factory.build()
        result = FlextResult.ok(user_data["name"])

        mapped_result = result.map(lambda name: name.upper())

        assert mapped_result.success
        assert mapped_result.value == user_data["name"].upper()

    def test_map_failure_passthrough(self) -> None:
        """Test map doesn't transform failure results."""
        result = FlextResult[None].fail(FlextConstants.Messages.UNKNOWN_ERROR)
        mapped_result = result.map(lambda x: x.upper())

        assert mapped_result.is_failure
        assert mapped_result.error == FlextConstants.Messages.UNKNOWN_ERROR

    def test_flat_map_chaining(self, user_data_factory: UserDataFactory) -> None:
        """Test flat_map for monadic chaining."""
        user_data = user_data_factory.build()

        def validate_name(
            name: FlextTypes.Core.String,
        ) -> FlextResult[FlextTypes.Core.String]:
            if len(name) > 2:
                return FlextResult.ok(name.title())
            return FlextResult[None].fail(FlextConstants.Messages.VALIDATION_FAILED)

        result = FlextResult.ok(user_data["name"]).flat_map(validate_name)

        if len(user_data["name"]) > 2:
            assert result.success
            assert result.value == user_data["name"].title()
        else:
            assert result.is_failure
            assert result.error == FlextConstants.Messages.VALIDATION_FAILED

    def test_railway_pattern_composition(self) -> None:
        """Test complex railway pattern composition."""
        result = (
            FlextResult.ok("  hello world  ")
            .map(str.strip)
            .map(str.title)
            .flat_map(
                lambda s: FlextResult.ok(f"Processed: {s}")
                if s
                else FlextResult[None].fail(FlextConstants.Messages.VALUE_EMPTY)
            )
            .map(lambda s: s.upper())
        )

        assert result.success
        assert result.value == "PROCESSED: HELLO WORLD"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestFlextResultErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_unwrap_success(self, user_data_factory: UserDataFactory) -> None:
        """Test unwrap on successful results."""
        test_data = user_data_factory.build()
        result = FlextResult.ok(test_data)

        assert result.value == test_data

    def test_value_failure_raises(self) -> None:
        """Test value property raises on failure results."""
        result = FlextResult[None].fail(FlextConstants.Messages.UNKNOWN_ERROR)

        with pytest.raises(TypeError, match=FlextConstants.Messages.UNKNOWN_ERROR):
            _ = result.value

    def test_unwrap_or_default(self, user_data_factory: UserDataFactory) -> None:
        """Test unwrap_or provides default on failure."""
        success_data = user_data_factory.build()
        default_data = {"default": True}

        success_result = FlextResult.ok(success_data)
        failure_result = FlextResult[None].fail(FlextConstants.Messages.UNKNOWN_ERROR)

        assert success_result.unwrap_or(default_data) == success_data
        assert failure_result.unwrap_or(default_data) == default_data

    def test_error_propagation_chain(self) -> None:
        """Test error propagation through transformation chains."""

        def failing_transform(
            _x: FlextTypes.Core.String,
        ) -> FlextResult[FlextTypes.Core.String]:
            return FlextResult[None].fail(FlextConstants.Messages.OPERATION_FAILED)

        result = (
            FlextResult.ok("input").flat_map(failing_transform).map(str.upper)
        )  # Should not execute

        assert result.is_failure
        assert result.error == FlextConstants.Messages.OPERATION_FAILED


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestFlextResultPerformance:
    """Test performance characteristics using benchmark utilities."""

    def test_creation_performance(self, benchmark: object) -> None:
        """Benchmark result creation performance."""

        def create_results() -> FlextTypes.Core.List[
            FlextResult[FlextTypes.Core.String]
        ]:
            return [FlextResult.ok(f"value_{i}") for i in range(100)]

        results = BenchmarkUtils.benchmark_with_warmup(
            benchmark, create_results, warmup_rounds=3
        )

        assert len(results) == 100
        assert all(r.success for r in results)

    def test_chaining_performance(self, benchmark: object) -> None:
        """Benchmark railway pattern chaining performance."""

        def chain_operations() -> FlextResult[str]:
            return (
                FlextResult.ok("test")
                .map(lambda x: x.upper())
                .map(lambda x: f"PREFIX_{x}")
                .map(lambda x: x.lower())
                .flat_map(lambda x: FlextResult.ok(f"{x}_suffix"))
            )

        result = BenchmarkUtils.benchmark_with_warmup(
            benchmark, chain_operations, warmup_rounds=5
        )

        assert result.success
        assert result.value == "prefix_test_suffix"

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of result operations."""
        profiler = PerformanceProfiler()

        with profiler.profile_memory("result_operations"):
            # Create and chain many results
            results = []
            for i in range(1000):
                result = (
                    FlextResult.ok(f"data_{i}")
                    .map(str.upper)
                    .flat_map(lambda x: FlextResult.ok(f"processed_{x}"))
                )
                results.append(result)

        # Assert reasonable memory usage (< 5MB for 1000 results)
        profiler.assert_memory_efficient(
            max_memory_mb=5.0, operation_name="result_operations"
        )


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================


class TestFlextResultProperties:
    """Property-based tests for FlextResult invariants."""

    @given(st.text())
    def test_ok_preserves_value(self, value: FlextTypes.Core.String) -> None:
        """Property: ok(value).value == value."""
        result = FlextResult.ok(value)
        assert result.value == value
        assert result.success

    @given(st.text())
    def test_fail_preserves_error(self, error: FlextTypes.Core.String) -> None:
        """Property: fail(error).error == error (with empty string handling)."""
        result = FlextResult[None].fail(error)

        # FlextResult converts empty/whitespace-only errors to default message
        expected_error = error.strip() if error else ""
        if not expected_error:
            expected_error = FlextConstants.Messages.UNKNOWN_ERROR

        assert result.error == expected_error
        assert result.is_failure

    @given(st.text(), st.text())
    def test_map_composition_law(
        self, value: FlextTypes.Core.String, prefix: FlextTypes.Core.String
    ) -> None:
        """Property: result.map(f).map(g) == result.map(lambda x: g(f(x)))."""

        def f(x: FlextTypes.Core.String) -> FlextTypes.Core.String:
            return f"{prefix}_{x}"

        def g(x: FlextTypes.Core.String) -> FlextTypes.Core.String:
            return x.upper()

        result = FlextResult.ok(value)

        composed1 = result.map(f).map(g)
        composed2 = result.map(lambda x: g(f(x)))

        assert composed1.value == composed2.value
        assert composed1.success == composed2.success


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestFlextResultIntegration:
    """Integration tests using test scenarios."""

    def test_user_validation_pipeline(self, user_data_factory: UserDataFactory) -> None:
        """Test complete user validation pipeline."""
        user_data = user_data_factory.build()

        def validate_user_data(
            data: FlextTypes.Core.Dict,
        ) -> FlextResult[FlextTypes.Core.Dict]:
            return (
                FlextResult.ok(data)
                .flat_map(
                    lambda d: FlextResult.ok(d)
                    if d.get("name")
                    else FlextResult[None].fail(FlextConstants.Messages.NAME_EMPTY)
                )
                .flat_map(
                    lambda d: FlextResult.ok(d)
                    if d.get("email")
                    else FlextResult[None].fail(FlextConstants.Messages.INVALID_INPUT)
                )
                .map(lambda d: {**d, "validated": True})
            )

        result = validate_user_data(user_data)

        assert result.success
        assert result.value["validated"] is True
        assert result.value["name"] == user_data["name"]

    def test_error_handling_scenarios(
        self, test_scenarios: FlextTypes.Core.List
    ) -> None:
        """Test various error handling scenarios."""
        # Check if we have error scenarios (look for ERROR_CASE enum value)
        has_error_scenario = any(
            str(scenario).endswith("ERROR_CASE") for scenario in test_scenarios
        )
        if not has_error_scenario:
            pytest.skip("No error scenario available")

        def process_with_validation(
            data: FlextTypes.Core.String,
        ) -> FlextResult[FlextTypes.Core.String]:
            if not data:
                return FlextResult[None].fail(FlextConstants.Messages.VALUE_EMPTY)
            if len(data) < 3:
                return FlextResult[None].fail(FlextConstants.Messages.VALIDATION_FAILED)
            return FlextResult.ok(data.upper())

        result = process_with_validation("")
        assert result.is_failure
        assert FlextConstants.Messages.VALUE_EMPTY in result.error


# ============================================================================
# ASYNC RESULT TESTS
# ============================================================================


class TestFlextResultAsync:
    """Test FlextResult in async contexts."""

    @pytest.mark.asyncio
    async def test_async_result_processing(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test FlextResult usage in async functions."""
        user_data = user_data_factory.build()

        async def async_process(
            data: FlextTypes.Core.Dict,
        ) -> FlextResult[FlextTypes.Core.Dict]:
            # Simulate async processing
            await AsyncTestUtils.sleep_with_timeout(0.01)
            return FlextResult.ok({**data, "processed_async": True})

        result = await async_process(user_data)

        assert result.success
        assert result.value["processed_async"] is True

    @pytest.mark.asyncio
    async def test_async_error_handling(self) -> None:
        """Test async error handling with FlextResult."""

        async def async_failing_operation() -> FlextResult[FlextTypes.Core.String]:
            await AsyncTestUtils.sleep_with_timeout(0.01)
            return FlextResult[None].fail(FlextConstants.Messages.OPERATION_FAILED)

        result = await async_failing_operation()

        assert result.is_failure
        assert result.error == FlextConstants.Messages.OPERATION_FAILED

    @pytest.mark.asyncio
    async def test_async_concurrency_handling(self) -> None:
        """Test concurrent async operations with FlextResult."""

        async def create_async_result(
            value: FlextTypes.Core.String,
        ) -> FlextResult[FlextTypes.Core.String]:
            await AsyncTestUtils.sleep_with_timeout(0.001)
            return FlextResult.ok(f"processed_{value}")

        # Test concurrent execution
        results = await AsyncTestUtils.run_concurrent_tasks(
            [
                create_async_result("a"),
                create_async_result("b"),
                create_async_result("c"),
            ]
        )

        assert len(results) == 3
        assert all(r.success for r in results)
        assert {r.value for r in results} == {
            "processed_a",
            "processed_b",
            "processed_c",
        }


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestFlextResultEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.unicode_strings())
    def test_unicode_handling(self, edge_value: FlextTypes.Core.String) -> None:
        """Test FlextResult with unicode strings."""
        result = FlextResult.ok(edge_value)
        assert result.success
        assert result.value == edge_value

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.boundary_numbers())
    def test_boundary_numbers(self, edge_value: FlextTypes.Core.Float) -> None:
        """Test FlextResult with boundary number values."""
        result = FlextResult.ok(edge_value)
        assert result.success
        assert result.value == edge_value

    @pytest.mark.parametrize("empty_value", EdgeCaseGenerators.empty_values())
    def test_empty_values(self, empty_value: FlextTypes.Core.Object) -> None:
        """Test FlextResult with empty/null values."""
        result = FlextResult.ok(empty_value)
        assert result.success
        assert result.value == empty_value

    def test_large_data_handling(self) -> None:
        """Test FlextResult with large data structures."""
        large_data = EdgeCaseGenerators.large_values()[0]  # Large string
        result = FlextResult.ok(large_data)

        assert result.success
        assert len(result.value) == 10000
        assert result.value == large_data


# ============================================================================
# FACTORY BOY INTEGRATION TESTS
# ============================================================================


class TestFlextResultFactoryIntegration:
    """Test integration with factory_boy factories."""

    def test_user_factory_integration(self) -> None:
        """Test FlextResult with factory_boy user creation."""
        user = UserFactory()
        result = FlextResult.ok(user)

        assert result.success
        assert hasattr(result.value, "name")
        assert hasattr(result.value, "email")
        assert hasattr(result.value, "age")

    def test_batch_factory_processing(self) -> None:
        """Test FlextResult with batch factory data."""
        users = UserFactory.create_batch(5)

        def process_users(
            user_list: FlextTypes.Core.List,
        ) -> FlextResult[FlextTypes.Core.List]:
            if not user_list:
                return FlextResult[None].fail(FlextConstants.Messages.INVALID_INPUT)
            return FlextResult.ok([f"processed_{user.name}" for user in user_list])

        result = process_users(users)

        assert result.success
        assert len(result.value) == 5
        assert all(name.startswith("processed_") for name in result.value)

    def test_validation_test_cases(self) -> None:
        """Test FlextResult with comprehensive validation test cases."""
        test_cases = create_validation_test_cases()

        for case in test_cases:
            result = FlextResult.ok(case["data"])
            assert result.success
            assert result.value == case["data"]
