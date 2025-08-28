"""Tests for FlextResult - Railway Pattern Implementation.

Clean test suite for FlextResult functionality focusing on the current API.
Tests the railway-oriented programming pattern without external dependencies.
"""

from __future__ import annotations

import pytest

from flext_core import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextResultCore:
    """Test core FlextResult functionality."""

    def test_ok_result_creation(self) -> None:
        """Test successful result creation."""
        test_data = {"name": "test", "value": 42}
        result = FlextResult.ok(test_data)

        assert result.success
        assert not result.is_failure
        assert result.value == test_data
        assert result.error is None

    def test_fail_result_creation(self) -> None:
        """Test failure result creation."""
        error_msg = "Operation failed"
        result = FlextResult[None].fail(error_msg)

        assert result.is_failure
        assert not result.success
        assert result.error == error_msg
        # Test that accessing value on failure raises TypeError
        with pytest.raises(TypeError, match="Attempted to access value on failed result"):
            _ = result.value

    def test_map_on_success(self) -> None:
        """Test map operation on successful result."""
        result = FlextResult.ok(5)
        mapped = result.map(lambda x: x * 2)

        assert mapped.success
        assert mapped.value == 10

    def test_map_on_failure(self) -> None:
        """Test map operation preserves failure."""
        result = FlextResult[None].fail("error")
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_failure
        assert mapped.error == "error"

    def test_flat_map_on_success(self) -> None:
        """Test flat_map operation on successful result."""
        result = FlextResult.ok(5)
        flat_mapped = result.flat_map(lambda x: FlextResult.ok(x * 2))

        assert flat_mapped.success
        assert flat_mapped.value == 10

    def test_flat_map_on_failure(self) -> None:
        """Test flat_map operation preserves failure."""
        result = FlextResult[None].fail("error")
        flat_mapped = result.flat_map(lambda x: FlextResult.ok(x * 2))

        assert flat_mapped.is_failure
        assert flat_mapped.error == "error"

    def test_unwrap_success(self) -> None:
        """Test unwrap on successful result."""
        result = FlextResult.ok("test_value")
        assert result.unwrap() == "test_value"

    def test_unwrap_failure_raises(self) -> None:
        """Test unwrap on failure raises exception."""
        result = FlextResult[None].fail("error")

        with pytest.raises(RuntimeError, match="error"):
            result.unwrap()

    def test_unwrap_or_with_success(self) -> None:
        """Test unwrap_or with successful result."""
        result = FlextResult.ok("success")
        assert result.unwrap_or("default") == "success"

    def test_unwrap_or_with_failure(self) -> None:
        """Test unwrap_or with failure returns default."""
        result = FlextResult[None].fail("error")
        assert result.unwrap_or("default") == "default"


class TestFlextResultChaining:
    """Test result chaining operations."""

    def test_railway_pattern_success_chain(self) -> None:
        """Test successful railway pattern chain."""
        result = (
            FlextResult.ok(10)
            .map(lambda x: x * 2)
            .flat_map(lambda x: FlextResult.ok(x + 5))
            .map(str)
        )

        assert result.success
        assert result.value == "25"

    def test_railway_pattern_failure_chain(self) -> None:
        """Test railway pattern with failure in chain."""
        result = (
            FlextResult.ok(10)
            .map(lambda x: x * 2)
            .flat_map(lambda _: FlextResult[None].fail("chain_error"))
            .map(str)
        )

        assert result.is_failure
        assert result.error == "chain_error"

    def test_early_failure_stops_chain(self) -> None:
        """Test that early failure stops the chain."""
        result = (
            FlextResult[None].fail("initial_error")
            .map(lambda x: x * 2)  # Should not execute
            .flat_map(lambda x: FlextResult.ok(x + 5))  # Should not execute
        )

        assert result.is_failure
        assert result.error == "initial_error"


class TestFlextResultProperties:
    """Test FlextResult property access."""

    def test_success_properties(self) -> None:
        """Test properties of successful result."""
        result = FlextResult.ok("test")

        assert result.success
        assert not result.is_failure
        assert result.value == "test"
        assert result.error is None

    def test_failure_properties(self) -> None:
        """Test properties of failure result."""
        result = FlextResult[None].fail("test_error")

        assert not result.success
        assert result.is_failure
        assert result.error == "test_error"
        # Test that accessing value on failure raises TypeError
        with pytest.raises(TypeError, match="Attempted to access value on failed result"):
            _ = result.value

    def test_data_property_compatibility(self) -> None:
        """Test .data property for backward compatibility."""
        result = FlextResult.ok("test_data")

        # Test that .data property exists and works
        assert hasattr(result, "data")
        assert result.data == "test_data"

    def test_fail_preserves_error(self) -> None:
        """Test that failure result preserves error message."""
        error_msg = "specific error message"
        result = FlextResult[None].fail(error_msg)

        assert result.error == error_msg
        assert str(result.error) == error_msg


class TestFlextResultAsync:
    """Test FlextResult with async operations."""

    @pytest.mark.asyncio
    async def test_async_result_usage(self) -> None:
        """Test FlextResult works with async functions."""
        async def async_operation(value: int) -> FlextResult[int]:
            return FlextResult.ok(value * 2)

        result = await async_operation(5)
        assert result.success
        assert result.value == 10

    @pytest.mark.asyncio
    async def test_async_concurrency_handling(self) -> None:
        """Test concurrent async result operations."""
        import asyncio

        async def async_multiply(x: int) -> FlextResult[int]:
            await asyncio.sleep(0.01)  # Simulate async work
            return FlextResult.ok(x * 2)

        results = await asyncio.gather(
            async_multiply(1),
            async_multiply(2),
            async_multiply(3)
        )

        assert all(r.success for r in results)
        assert [r.value for r in results] == [2, 4, 6]


class TestFlextResultPerformance:
    """Test FlextResult performance characteristics."""

    def test_result_creation_performance(self) -> None:
        """Test that result creation is efficient."""
        # Create many results quickly
        results = [FlextResult.ok(i) for i in range(1000)]

        assert len(results) == 1000
        assert all(r.success for r in results)
        assert [r.value for r in results[:5]] == [0, 1, 2, 3, 4]

    def test_chaining_performance(self) -> None:
        """Test that chaining operations are efficient."""
        # Long chain of operations
        result = FlextResult.ok(1)
        for _ in range(100):
            result = result.map(lambda x: x + 1)

        assert result.success
        assert result.value == 101


class TestFlextResultErrorHandling:
    """Test error handling patterns with FlextResult."""

    def test_error_propagation(self) -> None:
        """Test error propagation through operations."""
        def divide(x: int, y: int) -> FlextResult[float]:
            if y == 0:
                return FlextResult[None].fail("Division by zero")
            return FlextResult.ok(x / y)

        result = divide(10, 0)
        assert result.is_failure
        assert "Division by zero" in result.error

    def test_error_recovery(self) -> None:
        """Test error recovery patterns."""
        def safe_divide(x: int, y: int) -> FlextResult[float]:
            if y == 0:
                return FlextResult[None].fail("Division by zero")
            return FlextResult.ok(x / y)

        result = safe_divide(10, 0)
        recovered_value = result.unwrap_or(0.0)

        assert result.is_failure
        assert recovered_value == 0.0

    def test_multiple_error_scenarios(self) -> None:
        """Test handling multiple error scenarios."""
        def validate_and_process(data: dict[str, object]) -> FlextResult[dict[str, object]]:
            if not data:
                return FlextResult[None].fail("Data is empty")
            if "name" not in data:
                return FlextResult[None].fail("Name field missing")
            if not isinstance(data["name"], str):
                return FlextResult[None].fail("Name must be string")
            return FlextResult.ok({"processed": data["name"]})

        # Test empty data
        result1 = validate_and_process({})
        assert result1.is_failure
        assert result1.error == "Data is empty"

        # Test missing name
        result2 = validate_and_process({"other": "value"})
        assert result2.is_failure
        assert result2.error == "Name field missing"

        # Test invalid name type
        result3 = validate_and_process({"name": 123})
        assert result3.is_failure
        assert result3.error == "Name must be string"

        # Test success
        result4 = validate_and_process({"name": "test"})
        assert result4.success
        assert result4.value == {"processed": "test"}


class TestFlextResultIntegration:
    """Integration tests for FlextResult patterns."""

    def test_business_logic_integration(self) -> None:
        """Test FlextResult integration with business logic."""
        class User:
            def __init__(self, name: str, email: str) -> None:
                self.name = name
                self.email = email

        def create_user(data: dict[str, str]) -> FlextResult[User]:
            if not data.get("name"):
                return FlextResult[None].fail("Name required")
            if not data.get("email") or "@" not in data["email"]:
                return FlextResult[None].fail("Valid email required")

            return FlextResult.ok(User(data["name"], data["email"]))

        def format_user(user: User) -> FlextResult[dict[str, str]]:
            return FlextResult.ok({
                "display_name": f"{user.name} <{user.email}>",
                "user_id": f"user_{user.name.lower()}"
            })

        # Test complete flow
        result = (
            create_user({"name": "John", "email": "john@test.com"})
            .flat_map(format_user)
        )

        assert result.success
        formatted = result.value
        assert formatted["display_name"] == "John <john@test.com>"
        assert formatted["user_id"] == "user_john"

    def test_error_handling_scenarios(self) -> None:
        """Test various error handling scenarios."""
        def risky_operation(value: int) -> FlextResult[str]:
            if value < 0:
                return FlextResult[None].fail("Negative values not allowed")
            if value > 100:
                return FlextResult[None].fail("Value too large")
            return FlextResult.ok(f"processed_{value}")

        # Test different error paths
        test_cases = [
            (-1, "Negative values not allowed"),
            (150, "Value too large"),
            (50, None)  # Success case
        ]

        for value, expected_error in test_cases:
            result = risky_operation(value)

            if expected_error:
                assert result.is_failure
                assert result.error == expected_error
            else:
                assert result.success
                assert result.value == f"processed_{value}"
