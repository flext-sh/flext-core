"""Comprehensive FlextResult tests - covering all functionality."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flext_core import FlextResult


class TestFlextResult:
    """Test FlextResult for all usage patterns."""

    def test_success_creation(self) -> None:
        """Test creating successful FlextResult."""
        result = FlextResult.ok("test data")

        assert result.is_success
        assert not result.is_failure
        assert result.data == "test data"

    def test_failure_creation(self) -> None:
        """Test creating failed FlextResult."""
        result: FlextResult[str] = FlextResult.fail("error message")

        assert result.is_failure
        assert not result.is_success
        assert result.error == "error message"

    def test_with_none_data(self) -> None:
        """Test FlextResult with None data (common pattern for void
        operations)."""
        result = FlextResult.ok(None)

        assert result.is_success
        assert result.data is None

    def test_equality(self) -> None:
        """Test FlextResult equality comparison."""
        result1 = FlextResult.ok("data")
        result2 = FlextResult.ok("data")
        result3 = FlextResult.ok("different")
        failure: FlextResult[str] = FlextResult.fail("error")

        assert result1 == result2
        assert result1 != result3
        assert result1 != failure

    def test_boolean_conversion(self) -> None:
        """Test FlextResult boolean conversion."""
        success = FlextResult.ok("data")
        failure: FlextResult[str] = FlextResult.fail("error")

        assert bool(success) is True
        assert bool(failure) is False

    def test_immutability(self) -> None:
        """Test that FlextResult is immutable."""
        result = FlextResult.ok("data")

        # FlextResult should be frozen/immutable
        try:
            result.data = "changed"  # type: ignore[misc]  # This should fail
            msg = "FlextResult should be immutable"
            raise AssertionError(msg)
        except (AttributeError, TypeError, ValidationError):
            pass  # Expected - frozen pydantic model

    def test_unwrap_success(self) -> None:
        """Test unwrapping successful result."""
        result = FlextResult.ok("test data")
        assert result.unwrap() == "test data"

    def test_unwrap_failure_raises(self) -> None:
        """Test unwrapping failure result raises ValueError."""
        result: FlextResult[str] = FlextResult.fail("error")
        with pytest.raises(ValueError, match="Cannot unwrap failure result"):
            result.unwrap()

    def test_unwrap_or_success(self) -> None:
        """Test unwrap_or with successful result."""
        result = FlextResult.ok("test data")
        assert result.unwrap_or("default") == "test data"

    def test_unwrap_or_failure(self) -> None:
        """Test unwrap_or with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")
        assert result.unwrap_or("default") == "default"

    def test_map_success(self) -> None:
        """Test mapping successful result."""
        result = FlextResult.ok("test")
        mapped = result.map(lambda x: x.upper())
        assert mapped.is_success
        assert mapped.data == "TEST"

    def test_map_failure(self) -> None:
        """Test mapping failure result."""
        result: FlextResult[str] = FlextResult.fail("error")
        mapped = result.map(lambda x: x.upper())
        assert mapped.is_failure
        assert mapped.error == "error"

    def test_map_transformation_error(self) -> None:
        """Test map with transformation that raises exception."""
        result = FlextResult.ok(5)
        mapped = result.map(lambda x: x / 0)  # Will raise ZeroDivisionError
        assert mapped.is_failure
        assert mapped.error is not None
        assert "Transformation failed" in mapped.error

    def test_flat_map_success(self) -> None:
        """Test flat_map with successful result."""
        result = FlextResult.ok("test")
        chained = result.flat_map(lambda x: FlextResult.ok(x.upper()))
        assert chained.is_success
        assert chained.data == "TEST"

    def test_flat_map_failure(self) -> None:
        """Test flat_map with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")
        chained = result.flat_map(lambda x: FlextResult.ok(x.upper()))
        assert chained.is_failure
        assert chained.error == "error"

    def test_flat_map_chain_failure(self) -> None:
        """Test flat_map where chained operation fails."""
        result = FlextResult.ok("test")
        chained = result.flat_map(lambda x: FlextResult.fail("chain error"))
        assert chained.is_failure
        assert chained.error == "chain error"

    def test_validation_success_with_error(self) -> None:
        """Test validation prevents success result with error."""
        with pytest.raises(ValueError, match="Success result cannot contain error"):
            FlextResult(success=True, data="test", error="error")

    def test_validation_failure_with_data(self) -> None:
        """Test validation prevents failure result with data."""
        with pytest.raises(ValueError, match="Failure result cannot contain data"):
            FlextResult(success=False, data="test", error="error")

    def test_validation_failure_without_error(self) -> None:
        """Test validation requires error message for failure."""
        with pytest.raises(
            ValueError,
            match="Failure result must contain descriptive error message",
        ):
            FlextResult(success=False, data=None, error=None)

    def test_fail_empty_error_gets_default(self) -> None:
        """Test that empty error message gets default value."""
        result: FlextResult[None] = FlextResult.fail("")
        assert result.error == "Unknown error occurred"

        result2: FlextResult[None] = FlextResult.fail("   ")
        assert result2.error == "Unknown error occurred"

    def test_fail_strips_whitespace(self) -> None:
        """Test that error message whitespace is stripped."""
        result: FlextResult[None] = FlextResult.fail("  error message  ")
        assert result.error == "error message"
