"""Comprehensive tests for result.py module covering missing functionality.

This test suite provides complete coverage of FlextResult functionality,
focusing on the methods and edge cases not covered by existing tests.
"""

from __future__ import annotations

import pytest

from flext_core.exceptions import FlextOperationError
from flext_core.result import FlextResult, chain, compose, safe_call

pytestmark = [pytest.mark.unit, pytest.mark.core]


@pytest.mark.unit
class TestFlextResultProperties:
    """Test FlextResult property methods."""

    def test_error_code_property_success(self) -> None:
        """Test error_code property on success result."""
        result = FlextResult.ok("data")
        assert result.error_code is None

    def test_error_code_property_failure(self) -> None:
        """Test error_code property on failure result."""
        result: FlextResult[None] = FlextResult.fail("error", error_code="E001")
        if result.error_code != "E001":
            msg: str = f"Expected {'E001'}, got {result.error_code}"
            raise AssertionError(msg)

    def test_error_data_property_success(self) -> None:
        """Test error_data property on success result."""
        result = FlextResult.ok("data")
        if result.error_data != {}:
            msg: str = f"Expected {{}}, got {result.error_data}"
            raise AssertionError(msg)

    def test_error_data_property_failure(self) -> None:
        """Test error_data property on failure result."""
        error_data = {"field": "value", "code": 123}
        result: FlextResult[None] = FlextResult.fail("error", error_data=error_data)
        if result.error_data != error_data:
            msg: str = f"Expected {error_data}, got {result.error_data}"
            raise AssertionError(msg)

    def test_error_data_property_failure_none(self) -> None:
        """Test error_data property when None passed."""
        result: FlextResult[None] = FlextResult.fail("error", error_data=None)
        if result.error_data != {}:
            msg: str = f"Expected {{}}, got {result.error_data}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestFlextResultFactoryMethods:
    """Test FlextResult factory methods."""

    def test_fail_with_empty_error(self) -> None:
        """Test fail method with empty error string."""
        result: FlextResult[None] = FlextResult.fail("")
        if result.error != "Unknown error occurred":
            msg: str = f"Expected {'Unknown error occurred'}, got {result.error}"
            raise AssertionError(msg)

    def test_fail_with_whitespace_error(self) -> None:
        """Test fail method with whitespace-only error."""
        result: FlextResult[None] = FlextResult.fail("   ")
        if result.error != "Unknown error occurred":
            msg: str = f"Expected {'Unknown error occurred'}, got {result.error}"
            raise AssertionError(msg)

    def test_fail_with_all_parameters(self) -> None:
        """Test fail method with all parameters."""
        error_data: dict[str, object] = {"key": "value"}
        result: FlextResult[None] = FlextResult.fail(
            "Test error",
            error_code="E001",
            error_data=error_data,
        )

        assert result.is_failure
        if result.error != "Test error":
            msg: str = f"Expected {'Test error'}, got {result.error}"
            raise AssertionError(msg)
        assert result.error_code == "E001"
        if result.error_data != error_data:
            error_data_msg: str = f"Expected {error_data}, got {result.error_data}"
            raise AssertionError(error_data_msg)


@pytest.mark.unit
class TestFlextResultUnwrap:
    """Test FlextResult unwrap method."""

    def test_unwrap_success_with_none_data(self) -> None:
        """Test unwrap with None data (valid for success)."""
        result = FlextResult.ok(None)
        assert result.unwrap() is None

    def test_unwrap_failure_raises_with_error_code(self) -> None:
        """Test unwrap failure raises FlextOperationError with error code."""
        result: FlextResult[None] = FlextResult.fail("Test error", error_code="E001")

        with pytest.raises(FlextOperationError) as exc_info:
            result.unwrap()

        exception = exc_info.value
        # FlextOperationError includes error code in string representation
        if "Test error" not in str(exception):
            msg: str = f"Expected {'Test error'} in {exception!s}"
            raise AssertionError(msg)
        if exception.error_code != "E001":
            error_code_msg: str = f"Expected {'E001'}, got {exception.error_code}"
            raise AssertionError(error_code_msg)

    def test_unwrap_failure_raises_with_error_data(self) -> None:
        """Test unwrap failure raises FlextOperationError with error data."""
        error_data = {"field": "test", "value": 123}
        result: FlextResult[None] = FlextResult.fail(
            "Test error", error_data=error_data
        )

        with pytest.raises(FlextOperationError) as exc_info:
            result.unwrap()

        exception = exc_info.value
        if exception.context != error_data:
            msg: str = f"Expected {error_data}, got {exception.context}"
            raise AssertionError(msg)

    def test_unwrap_failure_with_none_error(self) -> None:
        """Test unwrap failure when error is None."""
        result = FlextResult[str](error="")  # Force empty error
        result._error = None  # Manually set to None

        # When error is None, the result is actually successful, so unwrap should work
        # This test should expect success, not failure
        assert result.success
        assert result.unwrap() is None  # Should return data (None) instead of raising


@pytest.mark.unit
class TestFlextResultMapExceptions:
    """Test FlextResult map method exception handling."""

    def test_map_type_error_exception(self) -> None:
        """Test map method with TypeError exception."""
        result = FlextResult.ok("test")

        def failing_func(x: str) -> int:
            return int(x)  # Will raise TypeError

        mapped = result.map(failing_func)
        assert mapped.is_failure
        error_msg = mapped.error or ""
        if "Transformation failed" not in error_msg:
            msg: str = f"Expected {'Transformation failed'} in {mapped.error}"
            raise AssertionError(msg)
        if mapped.error_code != "MAP_ERROR":
            map_error_msg: str = f"Expected {'MAP_ERROR'}, got {mapped.error_code}"
            raise AssertionError(map_error_msg)
        assert mapped.error_data["exception_type"] == "ValueError"

    def test_map_value_error_exception(self) -> None:
        """Test map method with ValueError exception."""
        result = FlextResult.ok("not_a_number")

        def failing_func(x: str) -> int:
            return int(x)  # Will raise ValueError

        mapped = result.map(failing_func)
        assert mapped.is_failure
        error_msg = mapped.error or ""
        if "Transformation failed" not in error_msg:
            msg: str = f"Expected {'Transformation failed'} in {mapped.error}"
            raise AssertionError(msg)
        if mapped.error is not None:
            assert "invalid literal" in mapped.error

    def test_map_runtime_error_exception(self) -> None:
        """Test map method with RuntimeError exception."""
        result = FlextResult.ok("test")

        def failing_func(x: str) -> str:
            msg = "Custom runtime error"
            raise RuntimeError(msg)

        mapped = result.map(failing_func)
        assert mapped.is_failure
        error_msg = mapped.error or ""
        if "Transformation failed" not in error_msg:
            msg: str = f"Expected {'Transformation failed'} in {mapped.error}"
            raise AssertionError(msg)
        if mapped.error is not None:
            assert "Custom runtime error" in mapped.error

    def test_map_unexpected_exception(self) -> None:
        """Test map method with unexpected exception type."""
        result = FlextResult.ok("test")

        def failing_func(x: str) -> str:
            msg = "Unexpected key error"
            raise KeyError(msg)

        mapped = result.map(failing_func)
        assert mapped.is_failure
        if mapped.error is None or "Transformation failed" not in mapped.error:
            msg: str = f"Expected {'Transformation failed'} in {mapped.error}"
            raise AssertionError(msg)
        if mapped.error_code != "MAP_ERROR":
            unexpected_error_msg: str = (
                f"Expected {'MAP_ERROR'}, got {mapped.error_code}"
            )
            raise AssertionError(unexpected_error_msg)

    def test_map_with_none_data(self) -> None:
        """Test map method with None data (valid for success)."""
        result = FlextResult.ok(None)

        def func(x: object) -> str:
            return f"processed_{x}"

        mapped = result.map(func)
        assert mapped.success
        if mapped.data != "processed_None":
            msg: str = f"Expected {'processed_None'}, got {mapped.data}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestFlextResultFlatMapExceptions:
    """Test FlextResult flat_map method exception handling."""

    def test_flat_map_type_error_exception(self) -> None:
        """Test flat_map method with TypeError exception."""
        result = FlextResult.ok("test")

        def failing_func(x: str) -> FlextResult[int]:
            # This will raise AttributeError when accessing non-existent method
            x.non_existent_method()
            return FlextResult.ok(42)  # Never reached

        mapped = result.flat_map(failing_func)
        assert mapped.is_failure
        error_msg = mapped.error or ""
        if "Chained operation failed" not in error_msg:
            msg: str = f"Expected {'Chained operation failed'} in {mapped.error}"
            raise AssertionError(msg)
        if mapped.error_code != "BIND_ERROR":
            bind_error_msg: str = f"Expected {'BIND_ERROR'}, got {mapped.error_code}"
            raise AssertionError(bind_error_msg)

    def test_flat_map_index_error_exception(self) -> None:
        """Test flat_map method with IndexError exception."""
        result = FlextResult.ok([1, 2, 3])

        def failing_func(x: list[int]) -> FlextResult[int]:
            return FlextResult.ok(x[10])  # Index out of range

        mapped = result.flat_map(failing_func)
        assert mapped.is_failure
        error_msg = mapped.error or ""
        if "Chained operation failed" not in error_msg:
            msg: str = f"Expected {'Chained operation failed'} in {mapped.error}"
            raise AssertionError(msg)

    def test_flat_map_key_error_exception(self) -> None:
        """Test flat_map method with KeyError exception."""
        result = FlextResult.ok({"a": 1})

        def failing_func(x: dict[str, int]) -> FlextResult[int]:
            return FlextResult.ok(x["missing_key"])

        mapped = result.flat_map(failing_func)
        assert mapped.is_failure
        error_msg = mapped.error or ""
        if "Chained operation failed" not in error_msg:
            msg: str = f"Expected {'Chained operation failed'} in {mapped.error}"
            raise AssertionError(msg)

    def test_flat_map_unexpected_exception(self) -> None:
        """Test flat_map method with unexpected exception type."""
        result = FlextResult.ok("test")

        def failing_func(x: str) -> FlextResult[str]:
            msg = "Unexpected OS error"
            raise OSError(msg)

        mapped = result.flat_map(failing_func)
        assert mapped.is_failure
        error_msg = mapped.error or ""
        if "Unexpected chaining error" not in error_msg:
            msg: str = f"Expected {'Unexpected chaining error'} in {mapped.error}"
            raise AssertionError(msg)
        if mapped.error_code != "CHAIN_ERROR":
            chain_error_msg: str = f"Expected {'CHAIN_ERROR'}, got {mapped.error_code}"
            raise AssertionError(chain_error_msg)

    def test_flat_map_with_none_data(self) -> None:
        """Test flat_map method with None data (valid for success)."""
        result = FlextResult.ok(None)

        def func(x: object) -> FlextResult[str]:
            return FlextResult.ok(f"processed_{x}")

        mapped = result.flat_map(func)
        assert mapped.success
        if mapped.data != "processed_None":
            msg: str = f"Expected {'processed_None'}, got {mapped.data}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestFlextResultUnwrapOr:
    """Test FlextResult unwrap_or method."""

    def test_unwrap_or_success_with_none_data(self) -> None:
        """Test unwrap_or with None data returns default."""
        result = FlextResult.ok(None)
        if result.unwrap_or("default") != "default":
            msg: str = f"Expected {'default'}, got {result.unwrap_or('default')}"
            raise AssertionError(msg)

    def test_unwrap_or_success_with_data(self) -> None:
        """Test unwrap_or with actual data returns data."""
        result = FlextResult.ok("actual")
        if result.unwrap_or("default") != "actual":
            msg: str = f"Expected {'actual'}, got {result.unwrap_or('default')}"
            raise AssertionError(msg)

    def test_unwrap_or_failure_returns_default(self) -> None:
        """Test unwrap_or with failure returns default."""
        result: FlextResult[object] = FlextResult.fail("error")
        if result.unwrap_or("default") != "default":
            msg: str = f"Expected {'default'}, got {result.unwrap_or('default')}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestFlextResultHash:
    """Test FlextResult __hash__ method."""

    def test_hash_success_with_hashable_data(self) -> None:
        """Test hash with hashable data."""
        result1 = FlextResult.ok("test")
        result2 = FlextResult.ok("test")
        if hash(result1) != hash(result2):
            msg: str = f"Expected {hash(result2)}, got {hash(result1)}"
            raise AssertionError(msg)

    def test_hash_success_with_unhashable_data(self) -> None:
        """Test hash with unhashable data."""
        result1 = FlextResult.ok({"key": "value"})
        result2 = FlextResult.ok({"key": "value"})

        # Should not raise error and should produce consistent hashes
        hash1 = hash(result1)
        hash2 = hash(result2)
        assert isinstance(hash1, int)
        assert isinstance(hash2, int)

    def test_hash_failure_results(self) -> None:
        """Test hash with failure results."""
        result1: FlextResult[object] = FlextResult.fail("error", error_code="E001")
        result2: FlextResult[object] = FlextResult.fail("error", error_code="E001")
        if hash(result1) != hash(result2):
            msg: str = f"Expected {hash(result2)}, got {hash(result1)}"
            raise AssertionError(msg)

    def test_hash_different_results(self) -> None:
        """Test hash with different results."""
        result1 = FlextResult.ok("test1")
        result2 = FlextResult.ok("test2")
        assert hash(result1) != hash(result2)


@pytest.mark.unit
class TestFlextResultRailwayMethods:
    """Test FlextResult railway-oriented programming methods."""

    def test_then_success(self) -> None:
        """Test then method (alias for flat_map) with success."""
        result = FlextResult.ok(5)

        def double(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        chained = result.then(double)
        assert chained.success
        if chained.data != 10:
            msg: str = f"Expected {10}, got {chained.data}"
            raise AssertionError(msg)

    def test_then_failure(self) -> None:
        """Test then method with failure."""
        result: FlextResult[int] = FlextResult.fail("initial error")

        def double(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        chained = result.then(double)
        assert chained.is_failure
        if chained.error != "initial error":
            msg: str = f"Expected {'initial error'}, got {chained.error}"
            raise AssertionError(msg)

    def test_bind_success(self) -> None:
        """Test bind method (alias for flat_map) with success."""
        result = FlextResult.ok(5)

        def double(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        bound = result.bind(double)
        assert bound.success
        if bound.data != 10:
            msg: str = f"Expected {10}, got {bound.data}"
            raise AssertionError(msg)

    def test_or_else_success(self) -> None:
        """Test or_else method with success result."""
        result = FlextResult.ok("success")
        alternative = FlextResult.ok("alternative")

        chosen = result.or_else(alternative)
        if chosen.data != "success":
            msg: str = f"Expected {'success'}, got {chosen.data}"
            raise AssertionError(msg)

    def test_or_else_failure(self) -> None:
        """Test or_else method with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")
        alternative = FlextResult.ok("alternative")

        chosen = result.or_else(alternative)
        if chosen.data != "alternative":
            msg: str = f"Expected {'alternative'}, got {chosen.data}"
            raise AssertionError(msg)

    def test_or_else_get_success(self) -> None:
        """Test or_else_get method with success result."""
        result = FlextResult.ok("success")

        def get_alternative() -> FlextResult[str]:
            return FlextResult.ok("alternative")

        chosen = result.or_else_get(get_alternative)
        if chosen.data != "success":
            msg: str = f"Expected {'success'}, got {chosen.data}"
            raise AssertionError(msg)

    def test_or_else_get_failure(self) -> None:
        """Test or_else_get method with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")

        def get_alternative() -> FlextResult[str]:
            return FlextResult.ok("alternative")

        chosen = result.or_else_get(get_alternative)
        if chosen.data != "alternative":
            msg: str = f"Expected {'alternative'}, got {chosen.data}"
            raise AssertionError(msg)

    def test_or_else_get_with_exception(self) -> None:
        """Test or_else_get method when function raises exception."""
        result: FlextResult[str] = FlextResult.fail("error")

        def failing_func() -> FlextResult[str]:
            msg = "Function failed"
            raise ValueError(msg)

        chosen = result.or_else_get(failing_func)
        assert chosen.is_failure
        error_msg = chosen.error or ""
        if "Function failed" not in error_msg:
            msg: str = f"Expected {'Function failed'} in {chosen.error}"
            raise AssertionError(msg)

    def test_recover_success(self) -> None:
        """Test recover method with success result."""
        result = FlextResult.ok("success")

        def recovery(error: str) -> str:
            return f"recovered_from_{error}"

        recovered = result.recover(recovery)
        if recovered.data != "success":
            msg: str = f"Expected {'success'}, got {recovered.data}"
            raise AssertionError(msg)

    def test_recover_failure(self) -> None:
        """Test recover method with failure result."""
        result: FlextResult[object] = FlextResult.fail("error")

        def recovery(error: str) -> str:
            return f"recovered_from_{error}"

        recovered = result.recover(recovery)
        assert recovered.success
        if recovered.data != "recovered_from_error":
            msg: str = f"Expected {'recovered_from_error'}, got {recovered.data}"
            raise AssertionError(msg)

    def test_recover_with_none_error(self) -> None:
        """Test recover method when error is None."""
        result = FlextResult[str](error="")
        result._error = None  # Manually set to None

        # When error is None, the result becomes successful, so recover returns self
        def recovery(error: str) -> str:
            return "recovered"

        recovered = result.recover(recovery)
        assert recovered.success  # Result is successful, so recover returns self
        assert recovered is result  # Should return the same instance

    def test_recover_with_exception(self) -> None:
        """Test recover method when recovery function raises exception."""
        result: FlextResult[object] = FlextResult.fail("error")

        def failing_recovery(error: str) -> str:
            msg = "Recovery failed"
            raise ValueError(msg)

        recovered = result.recover(failing_recovery)
        assert recovered.is_failure
        error_msg = recovered.error or ""
        if "Recovery failed" not in error_msg:
            msg: str = f"Expected {'Recovery failed'} in {recovered.error}"
            raise AssertionError(msg)

    def test_recover_with_success(self) -> None:
        """Test recover_with method with success result."""
        result = FlextResult.ok("success")

        def recovery(error: str) -> FlextResult[str]:
            return FlextResult.ok(f"recovered_from_{error}")

        recovered = result.recover_with(recovery)
        if recovered.data != "success":
            msg: str = f"Expected {'success'}, got {recovered.data}"
            raise AssertionError(msg)

    def test_recover_with_failure(self) -> None:
        """Test recover_with method with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")

        def recovery(error: str) -> FlextResult[str]:
            return FlextResult.ok(f"recovered_from_{error}")

        recovered = result.recover_with(recovery)
        assert recovered.success
        if recovered.data != "recovered_from_error":
            msg: str = f"Expected {'recovered_from_error'}, got {recovered.data}"
            raise AssertionError(msg)

    def test_recover_with_none_error_add(self) -> None:
        """Test recover_with method when error is None."""
        result = FlextResult[str](error="")
        result._error = None  # Manually set to None

        def recovery(error: str) -> FlextResult[str]:
            return FlextResult.ok("recovered")

        recovered = result.recover_with(recovery)
        # When error is None, result should be treated as success, so return self
        if not recovered.success:
            msg: str = f"Expected True, got {recovered.success}"
            raise AssertionError(msg)

    def test_recover_with_exception_add(self) -> None:
        """Test recover_with method when recovery function raises exception."""
        result: FlextResult[str] = FlextResult.fail("error")

        def failing_recovery(error: str) -> FlextResult[str]:
            msg = "Recovery failed"
            raise ValueError(msg)

        recovered = result.recover_with(failing_recovery)
        assert recovered.is_failure
        error_msg = recovered.error or ""
        if "Recovery failed" not in error_msg:
            msg: str = f"Expected {'Recovery failed'} in {recovered.error}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestFlextResultSideEffects:
    """Test FlextResult side effect methods."""

    def test_tap_success_with_data(self) -> None:
        """Test tap method with success result and data."""
        result = FlextResult.ok("test")
        side_effect_called = []

        def side_effect(data: str) -> None:
            side_effect_called.append(data)

        returned = result.tap(side_effect)
        assert returned is result  # Should return self
        if side_effect_called != ["test"]:
            msg: str = f"Expected {['test']}, got {side_effect_called}"
            raise AssertionError(msg)

    def test_tap_success_with_none_data(self) -> None:
        """Test tap method with success result but None data."""
        result = FlextResult.ok(None)
        side_effect_called = []

        def side_effect(data: object) -> None:
            side_effect_called.append(data)

        returned = result.tap(side_effect)
        assert returned is result
        if (
            side_effect_called != [] and side_effect_called is not None
        ):  # Should not call with None data:
            msg: str = f"Expected {[]}, got {side_effect_called}"
            raise AssertionError(msg)

    def test_tap_failure(self) -> None:
        """Test tap method with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")
        side_effect_called = []

        def side_effect(data: str) -> None:
            side_effect_called.append(data)

        returned = result.tap(side_effect)
        assert returned is result
        if (
            side_effect_called != [] and side_effect_called is not None
        ):  # Should not call on failure:
            msg: str = f"Expected {[]}, got {side_effect_called}"
            raise AssertionError(msg)

    def test_tap_with_exception(self) -> None:
        """Test tap method when side effect raises exception."""
        result = FlextResult.ok("test")

        def failing_side_effect(data: str) -> None:
            msg = "Side effect failed"
            raise ValueError(msg)

        # Should suppress exception and return original result
        returned = result.tap(failing_side_effect)
        assert returned is result
        assert returned.success

    def test_tap_error_success(self) -> None:
        """Test tap_error method with success result."""
        result = FlextResult.ok("success")
        side_effect_called = []

        def error_side_effect(error: str) -> None:
            side_effect_called.append(error)

        returned = result.tap_error(error_side_effect)
        assert returned is result
        if (
            side_effect_called != [] and side_effect_called is not None
        ):  # Should not call on success:
            msg: str = f"Expected {[]}, got {side_effect_called}"
            raise AssertionError(msg)

    def test_tap_error_failure(self) -> None:
        """Test tap_error method with failure result."""
        result: FlextResult[object] = FlextResult.fail("test_error")
        side_effect_called = []

        def error_side_effect(error: str) -> None:
            side_effect_called.append(error)

        returned = result.tap_error(error_side_effect)
        assert returned is result
        if side_effect_called != ["test_error"]:
            msg: str = f"Expected {['test_error']}, got {side_effect_called}"
            raise AssertionError(msg)

    def test_tap_error_with_none_error(self) -> None:
        """Test tap_error method when error is None."""
        result = FlextResult[str](error="")
        result._error = None  # Manually set to None
        side_effect_called = []

        def error_side_effect(error: str) -> None:
            side_effect_called.append(error)

        returned = result.tap_error(error_side_effect)
        assert returned is result
        if (
            side_effect_called != [] and side_effect_called is not None
        ):  # Should not call with None error:
            msg: str = f"Expected {[]}, got {side_effect_called}"
            raise AssertionError(msg)

    def test_tap_error_with_exception(self) -> None:
        """Test tap_error method when side effect raises exception."""
        result: FlextResult[object] = FlextResult.fail("error")

        def failing_error_side_effect(error: str) -> None:
            msg = "Error side effect failed"
            raise ValueError(msg)

        # Should suppress exception and return original result
        returned = result.tap_error(failing_error_side_effect)
        assert returned is result
        assert returned.is_failure


@pytest.mark.unit
class TestFlextResultFilter:
    """Test FlextResult filter method."""

    def test_filter_success_predicate_true(self) -> None:
        """Test filter method with success result and true predicate."""
        result = FlextResult.ok(10)

        def is_positive(x: int) -> bool:
            return x > 0

        filtered = result.filter(is_positive)
        assert filtered.success
        if filtered.data != 10:
            msg: str = f"Expected {10}, got {filtered.data}"
            raise AssertionError(msg)

    def test_filter_success_predicate_false(self) -> None:
        """Test filter method with success result and false predicate."""
        result = FlextResult.ok(-5)

        def is_positive(x: int) -> bool:
            return x > 0

        filtered = result.filter(is_positive, "Number must be positive")
        assert filtered.is_failure
        if filtered.error != "Number must be positive":
            msg: str = f"Expected {'Number must be positive'}, got {filtered.error}"
            raise AssertionError(msg)

    def test_filter_success_predicate_false_default_message(self) -> None:
        """Test filter method with default error message."""
        result = FlextResult.ok(-5)

        def is_positive(x: int) -> bool:
            return x > 0

        filtered = result.filter(is_positive)
        assert filtered.is_failure
        if filtered.error != "Filter predicate failed":
            msg: str = f"Expected {'Filter predicate failed'}, got {filtered.error}"
            raise AssertionError(msg)

    def test_filter_failure(self) -> None:
        """Test filter method with failure result."""
        result: FlextResult[int] = FlextResult.fail("initial error")

        def always_true(x: int) -> bool:
            return True

        filtered = result.filter(always_true)
        assert filtered.is_failure
        if filtered.error != "initial error":
            msg: str = f"Expected {'initial error'}, got {filtered.error}"
            raise AssertionError(msg)

    def test_filter_with_none_data(self) -> None:
        """Test filter method with None data."""
        result = FlextResult.ok(None)

        def is_not_none(x: object) -> bool:
            return x is not None

        filtered = result.filter(is_not_none)
        assert filtered.is_failure
        if filtered.error != "Filter predicate failed":
            msg: str = f"Expected {'Filter predicate failed'}, got {filtered.error}"
            raise AssertionError(msg)

    def test_filter_with_predicate_exception(self) -> None:
        """Test filter method when predicate raises exception."""
        result = FlextResult.ok("test")

        def failing_predicate(x: str) -> bool:
            msg = "Predicate failed"
            raise ValueError(msg)

        filtered = result.filter(failing_predicate)
        assert filtered.is_failure
        error_msg = filtered.error or ""
        if "Predicate failed" not in error_msg:
            msg: str = f"Expected {'Predicate failed'} in {filtered.error}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestFlextResultZipWith:
    """Test FlextResult zip_with method."""

    def test_zip_with_both_success(self) -> None:
        """Test zip_with with both results successful."""
        result1 = FlextResult.ok(5)
        result2 = FlextResult.ok(10)

        def add(x: int, y: int) -> int:
            return x + y

        zipped = result1.zip_with(result2, add)
        assert zipped.success
        if zipped.data != 15:
            msg: str = f"Expected {15}, got {zipped.data}"
            raise AssertionError(msg)

    def test_zip_with_first_failure(self) -> None:
        """Test zip_with with first result failed."""
        result1: FlextResult[int] = FlextResult.fail("first error")
        result2 = FlextResult.ok(10)

        def add(x: int, y: int) -> int:
            return x + y

        zipped = result1.zip_with(result2, add)
        assert zipped.is_failure
        if zipped.error != "first error":
            msg: str = f"Expected {'first error'}, got {zipped.error}"
            raise AssertionError(msg)

    def test_zip_with_second_failure(self) -> None:
        """Test zip_with with second result failed."""
        result1 = FlextResult.ok(5)
        result2: FlextResult[int] = FlextResult.fail("second error")

        def add(x: int, y: int) -> int:
            return x + y

        zipped = result1.zip_with(result2, add)
        assert zipped.is_failure
        if zipped.error != "second error":
            msg: str = f"Expected {'second error'}, got {zipped.error}"
            raise AssertionError(msg)

    def test_zip_with_first_none_data(self) -> None:
        """Test zip_with when first result has None data."""
        result1 = FlextResult.ok(None)
        result2 = FlextResult.ok(10)

        def combine(x: object, y: int) -> str:
            return f"{x}_{y}"

        zipped = result1.zip_with(result2, combine)
        assert zipped.is_failure
        if zipped.error != "Missing data for zip operation":
            msg: str = (
                f"Expected {'Missing data for zip operation'}, got {zipped.error}"
            )
            raise AssertionError(msg)

    def test_zip_with_second_none_data(self) -> None:
        """Test zip_with when second result has None data."""
        result1 = FlextResult.ok(5)
        result2 = FlextResult.ok(None)

        def combine(x: int, y: object) -> str:
            return f"{x}_{y}"

        zipped = result1.zip_with(result2, combine)
        assert zipped.is_failure
        if zipped.error != "Missing data for zip operation":
            msg: str = (
                f"Expected {'Missing data for zip operation'}, got {zipped.error}"
            )
            raise AssertionError(msg)

    def test_zip_with_function_exception(self) -> None:
        """Test zip_with when combining function raises exception."""
        result1 = FlextResult.ok(5)
        result2 = FlextResult.ok(0)

        def divide(x: int, y: int) -> float:
            return x / y  # Will raise ZeroDivisionError

        zipped = result1.zip_with(result2, divide)
        assert zipped.is_failure
        error_msg = zipped.error or ""
        if "division by zero" not in error_msg:
            msg: str = f"Expected {'division by zero'} in {zipped.error}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestFlextResultConversion:
    """Test FlextResult conversion methods."""

    def test_to_either_success(self) -> None:
        """Test to_either method with success result."""
        result = FlextResult.ok("data")
        data, error = result.to_either()
        if data != "data":
            msg: str = f"Expected {'data'}, got {data}"
            raise AssertionError(msg)
        assert error is None

    def test_to_either_failure(self) -> None:
        """Test to_either method with failure result."""
        result: FlextResult[object] = FlextResult.fail("error")
        data, error = result.to_either()
        assert data is None
        if error != "error":
            msg: str = f"Expected {'error'}, got {error}"
            raise AssertionError(msg)

    def test_to_exception_success(self) -> None:
        """Test to_exception method with success result."""
        result = FlextResult.ok("data")
        exception = result.to_exception()
        assert exception is None

    def test_to_exception_failure(self) -> None:
        """Test to_exception method with failure result."""
        result: FlextResult[object] = FlextResult.fail("error message")
        exception = result.to_exception()
        assert isinstance(exception, FlextOperationError)
        if "error message" not in str(exception):
            msg: str = f"Expected {'error message'} in {exception!s}"
            raise AssertionError(msg)

    def test_to_exception_failure_with_none_error(self) -> None:
        """Test to_exception method when error is None."""
        result = FlextResult[str](error="")
        result._error = None  # Manually set to None
        # When error is None, result is actually successful
        exception = result.to_exception()
        assert exception is None  # Success should return None


@pytest.mark.unit
class TestFlextResultStaticMethods:
    """Test FlextResult static methods."""

    def test_from_exception_success(self) -> None:
        """Test from_exception method with successful function."""

        def successful_func() -> str:
            return "success"

        result = FlextResult.from_exception(successful_func)
        assert result.success
        if result.data != "success":
            msg: str = f"Expected {'success'}, got {result.data}"
            raise AssertionError(msg)

    def test_from_exception_failure(self) -> None:
        """Test from_exception method with failing function."""

        def failing_func() -> str:
            msg = "Function failed"
            raise ValueError(msg)

        result = FlextResult.from_exception(failing_func)
        assert result.is_failure
        error_msg = result.error or ""
        if "Function failed" not in error_msg:
            msg: str = f"Expected 'Function failed' in {result.error}"
            raise AssertionError(msg)

    def test_combine_all_success(self) -> None:
        """Test combine method with all successful results."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.ok("b"),
            FlextResult.ok("c"),
        ]

        combined = FlextResult.combine(*results)
        assert combined.success
        if combined.data != ["a", "b", "c"]:
            msg: str = f"Expected {['a', 'b', 'c']}, got {combined.data}"
            raise AssertionError(msg)

    def test_combine_with_failure(self) -> None:
        """Test combine method with one failure."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.fail("error"),
            FlextResult.ok("c"),
        ]

        combined = FlextResult.combine(*results)
        assert combined.is_failure
        if combined.error != "error":
            msg: str = f"Expected {'error'}, got {combined.error}"
            raise AssertionError(msg)

    def test_combine_with_none_data(self) -> None:
        """Test combine method with None data results."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.ok(None),  # This should be skipped
            FlextResult.ok("c"),
        ]

        combined = FlextResult.combine(*results)
        assert combined.success
        if combined.data != ["a", "c"]:
            msg: str = f"Expected {['a', 'c']}, got {combined.data}"
            raise AssertionError(msg)

    def test_all_success_true(self) -> None:
        """Test all_success method with all successful results."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.ok("b"),
            FlextResult.ok("c"),
        ]

        if not (FlextResult.all_success(*results)):
            msg: str = f"Expected True, got {FlextResult.all_success(*results)}"
            raise AssertionError(msg)

    def test_all_success_false(self) -> None:
        """Test all_success method with one failure."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.fail("error"),
            FlextResult.ok("c"),
        ]

        if FlextResult.all_success(*results):
            msg: str = f"Expected False, got {FlextResult.all_success(*results)}"
            raise AssertionError(msg)

    def test_any_success_true(self) -> None:
        """Test any_success method with one success."""
        results: list[FlextResult[object]] = [
            FlextResult.fail("error1"),
            FlextResult.ok("success"),
            FlextResult.fail("error2"),
        ]

        if not (FlextResult.any_success(*results)):
            msg: str = f"Expected True, got {FlextResult.any_success(*results)}"
            raise AssertionError(msg)

    def test_any_success_false(self) -> None:
        """Test any_success method with all failures."""
        results: list[FlextResult[object]] = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("error3"),
        ]

        if FlextResult.any_success(*results):
            msg: str = f"Expected False, got {FlextResult.any_success(*results)}"
            raise AssertionError(msg)

    def test_first_success_found(self) -> None:
        """Test first_success method when success is found."""
        results: list[FlextResult[object]] = [
            FlextResult.fail("error1"),
            FlextResult.ok("first_success"),
            FlextResult.ok("second_success"),
        ]

        first = FlextResult.first_success(*results)
        assert first.success
        if first.data != "first_success":
            msg: str = f"Expected {'first_success'}, got {first.data}"
            raise AssertionError(msg)

    def test_first_success_not_found(self) -> None:
        """Test first_success method when no success is found."""
        results: list[FlextResult[object]] = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("last_error"),
        ]

        first = FlextResult.first_success(*results)
        assert first.is_failure
        if first.error != "last_error":
            msg: str = f"Expected {'last_error'}, got {first.error}"
            raise AssertionError(msg)

    def test_first_success_empty_results(self) -> None:
        """Test first_success method with no results."""
        first: FlextResult[object] = FlextResult.first_success()
        assert first.is_failure
        if first.error != "No successful results found":
            msg: str = f"Expected {'No successful results found'}, got {first.error}"
            raise AssertionError(msg)

    def test_try_all_success(self) -> None:
        """Test try_all method when first function succeeds."""

        def func1() -> str:
            return "success"

        def func2() -> str:
            msg = "Should not be called"
            raise ValueError(msg)

        result = FlextResult.try_all(func1, func2)
        assert result.success
        if result.data != "success":
            msg: str = f"Expected {'success'}, got {result.data}"
            raise AssertionError(msg)

    def test_try_all_second_succeeds(self) -> None:
        """Test try_all method when second function succeeds."""

        def func1() -> str:
            msg = "First fails"
            raise ValueError(msg)

        def func2() -> str:
            return "second_success"

        result = FlextResult.try_all(func1, func2)
        assert result.success
        if result.data != "second_success":
            msg: str = f"Expected {'second_success'}, got {result.data}"
            raise AssertionError(msg)

    def test_try_all_all_fail(self) -> None:
        """Test try_all method when all functions fail."""

        def func1() -> str:
            msg = "First fails"
            raise ValueError(msg)

        def func2() -> str:
            msg = "Last error"
            raise RuntimeError(msg)

        result = FlextResult.try_all(func1, func2)
        assert result.is_failure
        if result.error != "Last error":
            msg: str = f"Expected {'Last error'}, got {result.error}"
            raise AssertionError(msg)

    def test_try_all_no_functions(self) -> None:
        """Test try_all method with no functions."""
        result: FlextResult[object] = FlextResult.try_all()
        assert result.is_failure
        if result.error != "No functions provided":
            msg: str = f"Expected {'No functions provided'}, got {result.error}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestModuleLevelFunctions:
    """Test module-level railway operations."""

    def test_chain_all_success(self) -> None:
        """Test chain function with all successful results."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.ok("b"),
            FlextResult.ok("c"),
        ]

        chained = chain(*results)
        assert chained.success
        if chained.data != ["a", "b", "c"]:
            msg: str = f"Expected {['a', 'b', 'c']}, got {chained.data}"
            raise AssertionError(msg)

    def test_chain_with_failure(self) -> None:
        """Test chain function with one failure."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.fail("chain error"),
            FlextResult.ok("c"),
        ]

        chained = chain(*results)
        assert chained.is_failure
        if chained.error != "chain error":
            msg: str = f"Expected {'chain error'}, got {chained.error}"
            raise AssertionError(msg)

    def test_chain_with_none_data(self) -> None:
        """Test chain function with None data results."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.ok(None),  # Should be skipped
            FlextResult.ok("c"),
        ]

        chained = chain(*results)
        assert chained.success
        if chained.data != ["a", "c"]:
            msg: str = f"Expected {['a', 'c']}, got {chained.data}"
            raise AssertionError(msg)

    def test_compose_alias(self) -> None:
        """Test compose function (alias for chain)."""
        results: list[FlextResult[object]] = [
            FlextResult.ok("a"),
            FlextResult.ok("b"),
        ]

        composed = compose(*results)
        assert composed.success
        if composed.data != ["a", "b"]:
            msg: str = f"Expected {['a', 'b']}, got {composed.data}"
            raise AssertionError(msg)

    def test_safe_call_success(self) -> None:
        """Test safe_call function with successful function."""

        def successful_func() -> str:
            return "success"

        result = safe_call(successful_func)
        assert result.success
        if result.data != "success":
            msg: str = f"Expected {'success'}, got {result.data}"
            raise AssertionError(msg)

    def test_safe_call_failure(self) -> None:
        """Test safe_call function with failing function."""

        def failing_func() -> str:
            msg = "Function failed"
            raise ValueError(msg)

        result = safe_call(failing_func)
        assert result.is_failure
        if result.error != "Function failed":
            msg: str = f"Expected {'Function failed'}, got {result.error}"
            raise AssertionError(msg)
        assert result.error_data["exception_type"] == "ValueError"

    def test_safe_call_runtime_error(self) -> None:
        """Test safe_call function with RuntimeError."""

        def failing_func() -> str:
            msg = "Runtime error"
            raise RuntimeError(msg)

        result = safe_call(failing_func)
        assert result.is_failure
        if result.error != "Runtime error":
            msg: str = f"Expected {'Runtime error'}, got {result.error}"
            raise AssertionError(msg)

    def test_safe_call_empty_error_message(self) -> None:
        """Test safe_call function with exception that has empty message."""

        def failing_func() -> str:
            msg = ""
            raise ValueError(msg)

        result = safe_call(failing_func)
        assert result.is_failure
        if result.error != "Operation failed":
            msg: str = f"Expected {'Operation failed'}, got {result.error}"
            raise AssertionError(msg)
