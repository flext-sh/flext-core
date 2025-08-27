"""Modern tests for FlextUtilities - Advanced Utility Functions.

Refactored test suite using comprehensive testing libraries for utility functionality.
Demonstrates SOLID principles, property-based testing, and extensive automation.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from hypothesis import assume, given, strategies as st
from pytest_benchmark.fixture import BenchmarkFixture
from tests.support.async_utils import AsyncTestUtils
from tests.support.domain_factories import UserDataFactory
from tests.support.factory_boy_factories import (
    EdgeCaseGenerators,
    create_validation_test_cases,
)
from tests.support.performance_utils import BenchmarkUtils, PerformanceProfiler

# Direct imports avoiding problematic paths
from flext_core.utilities import FlextUtilities

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# CORE UTILITY TESTS
# ============================================================================


class TestFlextUtilitiesCore:
    """Test core utility functionality with factory patterns."""

    def test_id_generation_with_factories(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test ID generation using factory patterns."""
        # Generate multiple users to test ID uniqueness
        users = [user_data_factory.build() for _ in range(100)]
        ids = [FlextUtilities.generate_id() for _ in users]

        # Test uniqueness
        assert len(set(ids)) == len(ids), "All IDs should be unique"

        # Test format
        for id_value in ids:
            assert isinstance(id_value, str)
            assert len(id_value) > 0

    def test_uuid_generation_with_validation(self) -> None:
        """Test UUID generation with comprehensive validation."""
        generated_uuids = [FlextUtilities.generate_uuid() for _ in range(50)]

        for uuid_str in generated_uuids:
            # Validate UUID format
            uuid_obj = uuid.UUID(uuid_str)
            assert str(uuid_obj) == uuid_str
            assert len(uuid_str) == 36
            assert uuid_str.count("-") == 4

    @given(st.text(min_size=1, max_size=1000))
    def test_truncate_property_based(self, text: str) -> None:
        """Property-based testing of truncate function."""
        assume(len(text) > 0)

        max_length = 50
        truncated = FlextUtilities.truncate(text, max_length)

        # Properties that should always hold
        assert len(truncated) <= max_length
        assert isinstance(truncated, str)

        if len(text) <= max_length:
            assert truncated == text
        else:
            assert truncated.endswith("...")
            assert len(truncated) == max_length

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.unicode_strings())
    def test_unicode_handling(self, edge_value: str) -> None:
        """Test utility functions with Unicode edge cases."""
        # Test truncate with Unicode
        truncated = FlextUtilities.truncate(edge_value, 20)
        assert isinstance(truncated, str)
        assert len(truncated) <= 20

        # Test that utilities handle Unicode gracefully
        correlation_id = FlextUtilities.generate_correlation_id()
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestFlextUtilitiesPerformance:
    """Test utility performance characteristics."""

    def test_id_generation_performance(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark ID generation performance."""

        def generate_batch() -> list[str]:
            return [FlextUtilities.generate_id() for _ in range(1000)]

        ids = BenchmarkUtils.benchmark_with_warmup(
            benchmark, generate_batch, warmup_rounds=3
        )

        assert len(ids) == 1000
        assert len(set(ids)) == 1000  # All unique

    def test_uuid_generation_performance(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark UUID generation performance."""

        def generate_uuid_batch() -> list[str]:
            return [FlextUtilities.generate_uuid() for _ in range(1000)]

        uuids = BenchmarkUtils.benchmark_with_warmup(
            benchmark, generate_uuid_batch, warmup_rounds=3
        )

        assert len(uuids) == 1000
        assert len(set(uuids)) == 1000  # All unique

    def test_truncate_performance(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark truncate function performance."""
        long_text = "A" * 10000

        def truncate_batch() -> list[str]:
            return [
                FlextUtilities.truncate(long_text, length)
                for length in range(10, 100, 10)
            ]

        results = BenchmarkUtils.benchmark_with_warmup(
            benchmark, truncate_batch, warmup_rounds=3
        )

        assert len(results) == 9
        assert all(isinstance(r, str) for r in results)

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of utility operations."""
        profiler = PerformanceProfiler()

        with profiler.profile_memory("utility_operations"):
            # Generate many IDs and UUIDs
            ids = [FlextUtilities.generate_id() for _ in range(10000)]
            uuids = [FlextUtilities.generate_uuid() for _ in range(10000)]

            # Truncate operations
            text = "Sample text for truncation testing" * 100
            truncated = [
                FlextUtilities.truncate(text, length) for length in range(10, 200, 10)
            ]

        profiler.assert_memory_efficient(
            max_memory_mb=50.0, operation_name="utility_operations"
        )

        # Verify results
        assert len(ids) == 10000
        assert len(uuids) == 10000
        assert len(truncated) == 19


# ============================================================================
# ASYNC UTILITY TESTS
# ============================================================================


class TestFlextUtilitiesAsync:
    """Test utility functions in async contexts."""

    @pytest.mark.asyncio
    async def test_async_id_generation(self, async_test_utils: AsyncTestUtils) -> None:
        """Test ID generation in async context."""

        async def generate_async_ids() -> list[str]:
            return [FlextUtilities.generate_id() for _ in range(100)]

        # Test with timeout
        ids = await async_test_utils.run_with_timeout(
            generate_async_ids(), timeout_seconds=5.0
        )

        assert len(ids) == 100
        assert len(set(ids)) == 100

    @pytest.mark.asyncio
    async def test_concurrent_id_generation(
        self, async_test_utils: AsyncTestUtils
    ) -> None:
        """Test concurrent ID generation."""

        async def generate_ids() -> list[str]:
            return [FlextUtilities.generate_id() for _ in range(50)]

        # Run multiple concurrent generators
        results = await async_test_utils.run_concurrent_tasks(
            [generate_ids() for _ in range(10)]
        )

        # Flatten results and check uniqueness
        all_ids = [id_val for result in results for id_val in result]
        assert len(all_ids) == 500
        assert len(set(all_ids)) == 500  # All should be unique


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestFlextUtilitiesEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("length", [0, 1, 2, 3, 10, 100, 1000])
    def test_truncate_boundary_lengths(self, length: int) -> None:
        """Test truncate with various boundary lengths."""
        text = "Sample text for testing truncation boundaries"

        if length < 0:
            with pytest.raises(ValueError):
                FlextUtilities.truncate(text, length)
        else:
            result = FlextUtilities.truncate(text, length)
            assert isinstance(result, str)
            # CORRECTED: Mathematical property must always hold
            assert len(result) <= length

            # For lengths >= 3, expect suffix when text is truncated
            if length >= 3 and len(text) > length:
                assert result.endswith("...")
            # For lengths < 3, can't fit suffix, so just truncate text
            elif length < 3 and len(text) > length:
                assert result == text[:length]

    def test_truncate_empty_string(self) -> None:
        """Test truncate with empty string."""
        result = FlextUtilities.truncate("", 10)
        assert result == ""

    def test_truncate_exact_length(self) -> None:
        """Test truncate when text is exactly max length."""
        text = "exact"
        result = FlextUtilities.truncate(text, 5)
        assert result == text

    @pytest.mark.parametrize("edge_case", EdgeCaseGenerators.boundary_numbers())
    def test_numeric_edge_cases(self, edge_case: float) -> None:
        """Test utilities with numeric edge cases."""
        # Convert to string and test truncation
        text = str(edge_case)
        truncated = FlextUtilities.truncate(text, 20)

        assert isinstance(truncated, str)
        assert len(truncated) <= 20


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestFlextUtilitiesIntegration:
    """Integration tests using comprehensive scenarios."""

    def test_complete_utility_workflow(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test complete utility workflow with real data."""
        # Generate user data
        user_data = user_data_factory.build()

        # Generate IDs for the user
        user_id = FlextUtilities.generate_id()
        correlation_id = FlextUtilities.generate_correlation_id()

        # Process user data with utilities
        display_name = FlextUtilities.truncate(user_data["name"], 20)

        # Verify integration
        assert len(user_id) > 0
        assert len(correlation_id) > 0
        assert len(display_name) <= 20
        assert isinstance(display_name, str)

    def test_validation_integration(self) -> None:
        """Test integration with validation scenarios."""
        test_cases = create_validation_test_cases()

        for case in test_cases:
            if case["expected_valid"]:
                # Test utilities with valid data
                data_str = str(case["data"])
                truncated = FlextUtilities.truncate(data_str, 50)
                assert isinstance(truncated, str)

    def test_stress_testing(self) -> None:
        """Stress test utility functions."""
        # Generate large numbers of IDs
        ids = set()
        for _ in range(10000):
            new_id = FlextUtilities.generate_id()
            assert new_id not in ids, "Duplicate ID generated"
            ids.add(new_id)

        # Test with very long strings
        very_long_text = "x" * 100000
        truncated = FlextUtilities.truncate(very_long_text, 1000)
        assert len(truncated) <= 1000


# ============================================================================
# HYPOTHESIS PROPERTY-BASED TESTS
# ============================================================================


class TestFlextUtilitiesProperties:
    """Property-based tests using Hypothesis."""

    @given(st.text(), st.integers(min_value=0, max_value=1000))
    def test_truncate_properties(self, text: str, max_length: int) -> None:
        """Property-based test for truncate function."""
        result = FlextUtilities.truncate(text, max_length)

        # Properties that must always hold
        assert isinstance(result, str)
        assert len(result) <= max_length

        if len(text) <= max_length:
            assert result == text

        if max_length >= 3 and len(text) > max_length:
            assert result.endswith("...")

    @given(st.integers(min_value=1, max_value=1000))
    def test_id_generation_properties(self, count: int) -> None:
        """Property-based test for ID generation."""
        ids = [FlextUtilities.generate_id() for _ in range(count)]

        # Properties
        assert len(ids) == count
        assert all(isinstance(id_val, str) for id_val in ids)
        assert all(len(id_val) > 0 for id_val in ids)
        assert len(set(ids)) == count  # All unique


# ============================================================================
# TYPE GUARDS TESTS
# ============================================================================


class TestFlextTypeGuards:
    """Test type guard utilities if available."""

    def test_type_guard_basic(self) -> None:
        """Test basic type guard functionality."""
        # Test with various types
        assert FlextUtilities.is_string("test")
        assert not FlextUtilities.is_string(123)
        assert not FlextUtilities.is_string(None)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("string", True),
            (123, False),
            ([], False),
            ({}, False),
            (None, False),
        ],
    )
    def test_string_type_guard(self, value: object, *, expected: bool) -> None:
        """Test string type guard with various inputs."""
        result = FlextUtilities.is_string(value)
        assert result == expected

    def test_type_guard_with_factories(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test type guards with factory-generated data."""
        user_data = user_data_factory.build()

        # Test type guards on factory data
        assert FlextUtilities.is_string(user_data["name"])
        assert FlextUtilities.is_string(user_data["email"])

        if "age" in user_data:
            assert isinstance(user_data["age"], int)


# ============================================================================
# TIMESTAMP AND DATE UTILITIES
# ============================================================================


class TestFlextTimestampUtilities:
    """Test timestamp and date utility functions."""

    def test_iso_timestamp_generation(self) -> None:
        """Test ISO timestamp generation."""
        timestamp = FlextUtilities.generate_iso_timestamp()

        # Validate format
        assert isinstance(timestamp, str)
        assert "T" in timestamp
        assert timestamp.endswith("Z") or "+" in timestamp or "-" in timestamp[-6:]

    def test_timestamp_uniqueness(self) -> None:
        """Test that timestamps are reasonably unique."""
        timestamps = [FlextUtilities.generate_iso_timestamp() for _ in range(100)]

        # Most should be unique (allowing for some duplicates due to timing)
        unique_count = len(set(timestamps))
        assert unique_count >= 95  # Allow for some timing overlaps

    def test_timestamp_parsing(self) -> None:
        """Test that generated timestamps can be parsed."""
        timestamp = FlextUtilities.generate_iso_timestamp()

        # Should be parseable as datetime
        parsed = datetime.fromisoformat(timestamp)
        assert isinstance(parsed, datetime)
        assert parsed.tzinfo is not None
