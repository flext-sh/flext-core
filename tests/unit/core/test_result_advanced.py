"""Advanced tests for FlextResult with modern patterns.

Tests using parametrized scenarios, property-based testing,
performance benchmarking, and advanced fixtures.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from flext_core.result import FlextResult
from tests.conftest import TestCase, TestScenario

# ============================================================================
# Parametrized Testing with Test Cases
# ============================================================================


class TestFlextResultParametrized:
    """Tests using advanced parametrization patterns."""

    @pytest.fixture
    def result_test_cases(self) -> list[TestCase]:
        """Define test cases for FlextResult operations."""
        return [
            TestCase(
                id="success_with_string",
                description="Successful result with string data",
                input_data={"value": "test_data", "success": True},
                expected_output="test_data",
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="success_with_dict",
                description="Successful result with dict data",
                input_data={"value": {"key": "value"}, "success": True},
                expected_output={"key": "value"},
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="failure_with_error",
                description="Failed result with error message",
                input_data={"error": "Operation failed", "success": False},
                expected_output=None,
                expected_error="Operation failed",
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="none_value_success",
                description="Successful result with None value",
                input_data={"value": None, "success": True},
                expected_output=None,
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                id="empty_string_failure",
                description="Failed result with empty error",
                input_data={"error": "", "success": False},
                expected_output=None,
                expected_error="Unknown error occurred",
                scenario=TestScenario.BOUNDARY,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_result_creation_parametrized(self, result_test_cases):
        """Test FlextResult creation with various scenarios."""
        for test_case in result_test_cases:
            if test_case.input_data.get("success"):
                result = FlextResult.ok(test_case.input_data.get("value"))
                assert result.success
                assert result.data == test_case.expected_output
                assert result.error is None
            else:
                result = FlextResult.fail(test_case.input_data.get("error", ""))
                assert result.is_failure
                assert result.error == test_case.expected_error
                assert result.data is None

    @pytest.mark.parametrize(
        ("operation", "initial_value", "transform", "expected"),
        [
            ("map", 5, lambda x: x * 2, 10),
            ("map", "test", lambda x: x.upper(), "TEST"),
            ("map", [1, 2, 3], len, 3),
            ("map", {"a": 1}, lambda x: x.get("a"), 1),
        ],
        ids=["map_integer", "map_string", "map_list", "map_dict"],
    )
    @pytest.mark.happy_path
    def test_result_transformations(
        self, operation, initial_value, transform, expected
    ):
        """Test FlextResult transformation operations."""
        result = FlextResult.ok(initial_value)

        if operation == "map":
            transformed = result.map(transform)
            assert transformed.success
            assert transformed.data == expected


# ============================================================================
# Property-Based Testing with Hypothesis
# ============================================================================


class TestFlextResultPropertyBased:
    """Property-based tests using Hypothesis."""

    @pytest.mark.hypothesis
    @given(
        value=st.one_of(
            st.text(),
            st.integers(),
            st.floats(allow_nan=False),
            st.booleans(),
            st.none(),
        )
    )
    def test_result_ok_preserves_value(self, value):
        """Property: FlextResult.ok preserves any value."""
        result = FlextResult.ok(value)
        assert result.success
        assert result.data == value
        assert result.error is None

    @pytest.mark.hypothesis
    @given(error_msg=st.text(min_size=1).filter(lambda s: s.strip()))
    def test_result_fail_preserves_error(self, error_msg):
        """Property: FlextResult.fail preserves non-empty error message."""
        result = FlextResult.fail(error_msg)
        assert result.is_failure
        assert result.error == error_msg.strip()
        assert result.data is None

    @pytest.mark.hypothesis
    @given(
        value=st.integers(),
        func1=st.sampled_from(
            [
                lambda x: x + 1,
                lambda x: x * 2,
                lambda x: x - 1,
            ]
        ),
        func2=st.sampled_from(
            [
                lambda x: x * 3,
                lambda x: x + 10,
                lambda x: x // 2 if x != 0 else 0,
            ]
        ),
    )
    def test_result_map_composition(self, value, func1, func2):
        """Property: map operations compose correctly."""
        result = FlextResult.ok(value)

        # Apply functions separately
        result1 = result.map(func1).map(func2)

        # Apply composed function
        def composed(x):
            return func2(func1(x))

        result2 = result.map(composed)

        # Results should be identical
        assert result1.data == result2.data


# ============================================================================
# Performance Testing
# ============================================================================


class TestFlextResultPerformance:
    """Performance benchmarks for FlextResult operations."""

    @pytest.mark.benchmark
    def test_result_creation_performance(
        self, performance_monitor, performance_threshold
    ):
        """Benchmark FlextResult creation performance."""

        def create_results():
            return [FlextResult.ok(f"value_{i}") for i in range(1000)]

        metrics = performance_monitor(create_results)

        assert (
            metrics["execution_time"] < performance_threshold["result_creation"] * 1000
        )
        assert len(metrics["result"]) == 1000

    @pytest.mark.benchmark
    def test_result_chain_performance(self, performance_monitor, performance_threshold):
        """Benchmark chained FlextResult operations."""

        def chain_operations():
            return (
                FlextResult.ok(1)
                .map(lambda x: x + 1)
                .map(lambda x: x * 2)
                .map(lambda x: x**2)
                .map(str)
                .map(len)
            )

        metrics = performance_monitor(chain_operations)

        assert metrics["execution_time"] < 0.01  # 10ms for chain
        assert metrics["result"].data == 2  # len("16")


# ============================================================================
# Snapshot Testing
# ============================================================================


class TestFlextResultSnapshot:
    """Snapshot tests for complex result structures."""

    @pytest.mark.snapshot
    def test_complex_result_snapshot(self, snapshot_manager):
        """Test complex result structure with snapshot."""
        # Create complex result structure
        result = FlextResult.ok(
            {
                "users": [
                    {"id": 1, "name": "Alice", "active": True},
                    {"id": 2, "name": "Bob", "active": False},
                ],
                "metadata": {
                    "total": 2,
                    "page": 1,
                    "timestamp": "2024-01-01T00:00:00Z",
                },
            }
        )

        # Transform the result
        transformed = result.map(
            lambda data: {
                **data,
                "active_users": sum(1 for u in data["users"] if u["active"]),
            }
        )

        # Snapshot the final structure
        snapshot_manager(
            "complex_result",
            {
                "success": transformed.success,
                "data": transformed.data,
                "error": transformed.error,
            },
        )


# ============================================================================
# Integration with Advanced Fixtures
# ============================================================================


class TestFlextResultWithFixtures:
    """Tests using advanced fixtures from conftest."""

    def test_result_with_entity_factory(self, entity_factory, assert_helpers):
        """Test FlextResult with entity factory fixture."""
        # Create entity using factory
        entity = entity_factory("test-123", {"name": "Test Entity", "value": 42})

        # Wrap in result
        result = FlextResult.ok(entity)

        # Use assert helpers
        assert_helpers.assert_result_ok(result)
        assert_helpers.assert_entity_valid(result.data)

    def test_result_with_mock_factory(self, mock_factory):
        """Test FlextResult with mock factory."""
        # Create mock service
        service = mock_factory("test_service")

        # Service returns FlextResult
        result = service.process("test_input")

        assert result.success
        assert result.data == "processed"

    def test_result_with_test_builder(self, test_builder):
        """Test FlextResult with test data builder."""
        # Build test data
        data = (
            test_builder()
            .with_id("test-456")
            .with_field("name", "Test")
            .with_field("value", 100)
            .build()
        )

        # Create result with built data
        result = FlextResult.ok(data)

        assert result.data["id"] == "test-456"
        assert result.data["name"] == "Test"
        assert result.data["value"] == 100


# ============================================================================
# Error Path Testing
# ============================================================================


class TestFlextResultErrorPaths:
    """Comprehensive error path testing."""

    @pytest.mark.error_path
    @pytest.mark.parametrize(
        ("error_type", "error_message", "recovery_strategy"),
        [
            ("validation", "Invalid input", lambda: FlextResult.ok("default")),
            ("network", "Connection timeout", lambda: FlextResult.fail("Retry later")),
            ("permission", "Access denied", lambda: FlextResult.fail("Unauthorized")),
        ],
    )
    def test_error_recovery_strategies(
        self, error_type, error_message, recovery_strategy
    ):
        """Test various error recovery strategies."""
        # Simulate error
        result = FlextResult.fail(error_message)

        # Apply recovery strategy
        recovered = result.or_else_get(recovery_strategy)

        # Verify recovery behavior
        if error_type == "validation":
            assert recovered.success
            assert recovered.data == "default"
        else:
            assert recovered.is_failure
            assert recovered.error in {"Retry later", "Unauthorized"}


# ============================================================================
# Async Testing
# ============================================================================


class TestFlextResultAsync:
    """Async tests for FlextResult patterns."""

    @pytest.mark.asyncio
    async def test_async_result_processing(self, async_client):
        """Test FlextResult with async operations."""
        # Fetch data asynchronously
        response = await async_client.get("/api/data")

        # Wrap in FlextResult
        result = (
            FlextResult.ok(response)
            if response["status"] == "success"
            else FlextResult.fail("Failed")
        )

        # Process result
        processed = result.map(lambda r: r.get("url", ""))

        assert processed.success
        assert processed.data == "/api/data"


# ============================================================================
# Context Manager Testing
# ============================================================================


class TestFlextResultContext:
    """Test FlextResult with context managers."""

    def test_result_with_performance_context(self):
        """Test FlextResult within performance context."""
        from tests.conftest import assert_performance

        with assert_performance(max_time=0.001, max_memory=10000):
            result = FlextResult.ok("test")
            transformed = result.map(str.upper)
            assert transformed.data == "TEST"


# ============================================================================
# Boundary Testing
# ============================================================================


class TestFlextResultBoundary:
    """Boundary condition tests."""

    @pytest.mark.boundary
    @pytest.mark.parametrize(
        ("value", "description"),
        [
            ("" * 0, "empty string"),
            ("x" * 10000, "very long string"),
            ([], "empty list"),
            ([None] * 1000, "list of Nones"),
            ({}, "empty dict"),
            ({f"key_{i}": i for i in range(1000)}, "large dict"),
        ],
    )
    def test_boundary_values(self, value, description):
        """Test FlextResult with boundary values."""
        result = FlextResult.ok(value)

        # Verify result handles boundary values
        assert result.success
        assert result.data == value

        # Test transformations on boundary values
        if isinstance(value, (str, list, dict)):
            transformed = result.map(len)
            assert transformed.data == len(value)
