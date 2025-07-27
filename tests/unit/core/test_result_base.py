"""Tests for _result_base module."""

import pytest

from flext_core._exceptions_base import _FlextBaseError
from flext_core._result_base import _BaseResult


class TestBaseResult:
    """Test _BaseResult class."""

    def test_ok_creation(self) -> None:
        """Test creation of successful result."""
        result = _BaseResult.ok("test_data")
        assert result.success is True
        assert result.data == "test_data"
        assert result.error is None

    def test_fail_creation(self) -> None:
        """Test creation of failed result."""
        result = _BaseResult.fail("test_error")
        assert result.success is False
        assert result.data is None
        assert result.error == "test_error"

    def test_unwrap_success(self) -> None:
        """Test unwrapping successful result."""
        result = _BaseResult.ok("test_data")
        data = result._unwrap()
        assert data == "test_data"

    def test_unwrap_failure(self) -> None:
        """Test unwrapping failed result raises exception."""
        result = _BaseResult.fail("test_error")
        with pytest.raises(_FlextBaseError):
            result._unwrap()

    def test_unwrap_or_success(self) -> None:
        """Test unwrap_or with successful result."""
        result = _BaseResult.ok("test_data")
        data = result._unwrap_or("default")
        assert data == "test_data"

    def test_unwrap_or_failure(self) -> None:
        """Test unwrap_or with failed result."""
        result = _BaseResult.fail("test_error")
        data = result._unwrap_or("default")
        assert data == "default"

    def test_is_success(self) -> None:
        """Test is_success method."""
        success_result = _BaseResult.ok("data")
        failure_result = _BaseResult.fail("error")

        assert success_result._is_success() is True
        assert failure_result._is_success() is False

    def test_is_failure(self) -> None:
        """Test is_failure method."""
        success_result = _BaseResult.ok("data")
        failure_result = _BaseResult.fail("error")

        assert success_result._is_failure() is False
        assert failure_result._is_failure() is True

    def test_with_context(self) -> None:
        """Test adding context to result."""
        result = _BaseResult.ok("data")
        new_result = result._with_context(operation="test")

        assert new_result.success is True
        assert new_result.data == "data"
        assert new_result.context["operation"] == "test"

    def test_map_success(self) -> None:
        """Test mapping over successful result."""
        result = _BaseResult.ok(5)
        mapped = result._map(lambda x: x * 2)

        assert mapped.success is True
        assert mapped.data == 10

    def test_map_failure(self) -> None:
        """Test mapping over failed result."""
        result = _BaseResult.fail("error")
        mapped = result._map(lambda x: x * 2)

        assert mapped.success is False
        assert mapped.error == "error"

    def test_bool_conversion(self) -> None:
        """Test boolean conversion."""
        success_result = _BaseResult.ok("data")
        failure_result = _BaseResult.fail("error")

        assert bool(success_result) is True
        assert bool(failure_result) is False

    def test_string_representation(self) -> None:
        """Test string representation."""
        success_result = _BaseResult.ok("data")
        failure_result = _BaseResult.fail("error")

        assert "Ok" in str(success_result)
        assert "Err" in str(failure_result)
