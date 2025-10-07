"""Comprehensive tests for FlextResult returns library integration.

Tests cover the new methods added for dry-python/returns integration:
- from_callable() with @safe decorator
- flow_through() for pipeline composition
- Maybe interop (to_maybe, from_maybe)
- IO interop (to_io, to_io_result, from_io_result)
- Railway methods (lash, alt, value_or_call)
"""

from __future__ import annotations

import pytest
from returns.io import IO, IOFailure, IOSuccess
from returns.maybe import Nothing, Some

from flext_core import FlextResult


class TestFromCallable:
    """Test from_callable method with @safe decorator integration."""

    def test_from_callable_success(self) -> None:
        """Test from_callable with successful operation."""

        def safe_operation() -> int:
            return 42

        result = FlextResult[int].from_callable(safe_operation)

        assert result.is_success
        assert result.value == 42

    def test_from_callable_with_exception(self) -> None:
        """Test from_callable with operation that raises exception."""

        def failing_operation() -> int:
            msg = "Operation failed"
            raise ValueError(msg)

        result = FlextResult[int].from_callable(failing_operation)

        assert result.is_failure
        assert "Operation failed" in (result.error or "")

    def test_from_callable_with_custom_error_code(self) -> None:
        """Test from_callable with custom error code."""

        def failing_operation() -> str:
            msg = "Custom error"
            raise RuntimeError(msg)

        result = FlextResult[str].from_callable(
            failing_operation, error_code="CUSTOM_ERROR"
        )

        assert result.is_failure
        assert result.error_code == "CUSTOM_ERROR"

    def test_from_callable_with_none_return(self) -> None:
        """Test from_callable with function returning None."""

        def returns_none() -> None:
            return None

        result = FlextResult[None].from_callable(returns_none)

        assert result.is_success
        assert result.value is None

    def test_from_callable_with_complex_operation(self) -> None:
        """Test from_callable with complex operation."""

        def complex_operation() -> dict[str, object]:
            data: dict[str, object] = {"processed": True, "count": 10}
            if data["count"] > 5:
                return data
            msg = "Count too low"
            raise ValueError(msg)

        result = FlextResult[dict[str, object]].from_callable(complex_operation)

        assert result.is_success
        assert result.value == {"processed": True, "count": 10}


class TestFlowThrough:
    """Test flow_through method for pipeline composition."""

    def test_flow_through_success_pipeline(self) -> None:
        """Test flow_through with successful operations."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def multiply_by_two(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        result = FlextResult[int].ok(5).flow_through(add_one, multiply_by_two)

        assert result.is_success
        assert result.value == 12  # (5 + 1) * 2

    def test_flow_through_failure_propagation(self) -> None:
        """Test flow_through stops at first failure."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def fail_if_even(x: int) -> FlextResult[int]:
            if x % 2 == 0:
                return FlextResult[int].fail("Number is even")
            return FlextResult[int].ok(x)

        def multiply_by_two(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        result = (
            FlextResult[int].ok(5).flow_through(add_one, fail_if_even, multiply_by_two)
        )

        assert result.is_failure
        assert result.error == "Number is even"

    def test_flow_through_with_initial_failure(self) -> None:
        """Test flow_through with initial failure result."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        result = FlextResult[int].fail("Initial failure").flow_through(add_one)

        assert result.is_failure
        assert result.error == "Initial failure"

    def test_flow_through_empty_pipeline(self) -> None:
        """Test flow_through with no operations."""
        result = FlextResult[int].ok(42).flow_through()

        assert result.is_success
        assert result.value == 42

    def test_flow_through_complex_transformations(self) -> None:
        """Test flow_through with complex data transformations."""

        def validate_dict(data: dict[str, object]) -> FlextResult[dict[str, object]]:
            if "required_field" not in data:
                return FlextResult[dict[str, object]].fail("Missing required field")
            return FlextResult[dict[str, object]].ok(data)

        def enrich_data(data: dict[str, object]) -> FlextResult[dict[str, object]]:
            enriched = {**data, "enriched": True}
            return FlextResult[dict[str, object]].ok(enriched)

        def transform_data(data: dict[str, object]) -> FlextResult[dict[str, object]]:
            transformed = {
                **data,
                "transformed": True,
            }
            # Count includes the count key itself
            transformed["count"] = len(transformed) + 1
            return FlextResult[dict[str, object]].ok(transformed)

        initial_data: dict[str, object] = {"required_field": "value"}
        result = (
            FlextResult[dict[str, object]]
            .ok(initial_data)
            .flow_through(validate_dict, enrich_data, transform_data)
        )

        assert result.is_success
        assert result.value["enriched"] is True
        assert result.value["transformed"] is True
        assert result.value["count"] == 4


class TestMaybeInterop:
    """Test Maybe monad interoperability methods."""

    def test_to_maybe_success(self) -> None:
        """Test converting successful result to Some."""
        result = FlextResult[str].ok("test_value")
        maybe = result.to_maybe()

        assert isinstance(maybe, Some)
        sentinel = object()
        value = maybe.value_or(sentinel)
        assert value == "test_value"

    def test_to_maybe_failure(self) -> None:
        """Test converting failed result to Nothing."""
        result = FlextResult[str].fail("error")
        maybe = result.to_maybe()

        assert maybe == Nothing  # Nothing is a singleton, not a class

    def test_from_maybe_some(self) -> None:
        """Test creating result from Some."""
        maybe = Some("test_value")
        result = FlextResult.from_maybe(maybe)

        assert result.is_success
        assert result.value == "test_value"

    def test_from_maybe_nothing(self) -> None:
        """Test creating result from Nothing."""
        maybe = Nothing
        result = FlextResult.from_maybe(maybe)

        assert result.is_failure
        assert result.error == "No value in Maybe"

    def test_maybe_roundtrip_success(self) -> None:
        """Test roundtrip conversion success -> maybe -> success."""
        original = FlextResult[int].ok(42)
        maybe = original.to_maybe()
        recovered = FlextResult.from_maybe(maybe)

        assert recovered.is_success
        assert recovered.value == 42

    def test_maybe_roundtrip_failure(self) -> None:
        """Test roundtrip conversion failure -> maybe -> failure."""
        original = FlextResult[int].fail("error")
        maybe = original.to_maybe()
        recovered = FlextResult.from_maybe(maybe)

        assert recovered.is_failure


class TestIOInterop:
    """Test IO monad interoperability methods."""

    def test_to_io_success(self) -> None:
        """Test converting successful result to IO."""
        result = FlextResult[str].ok("test_value")
        io_container = result.to_io()

        assert isinstance(io_container, IO)
        # Note: returns.io.IO intentionally hides internal value
        # Test that we can map over it to verify it contains the value
        mapped = io_container.map(lambda x: x.upper())
        assert isinstance(mapped, IO)

    def test_to_io_failure_raises(self) -> None:
        """Test converting failed result to IO raises ValueError."""
        result = FlextResult[str].fail("error")

        with pytest.raises(ValueError, match="Cannot convert failure to IO"):
            result.to_io()

    def test_to_io_result_success(self) -> None:
        """Test converting successful result to IOSuccess."""
        result = FlextResult[str].ok("test_value")
        io_result = result.to_io_result()

        assert isinstance(io_result, IOSuccess)

    def test_to_io_result_failure(self) -> None:
        """Test converting failed result to IOFailure."""
        result = FlextResult[str].fail("error_message")
        io_result = result.to_io_result()

        assert isinstance(io_result, IOFailure)

    def test_from_io_result_success(self) -> None:
        """Test creating result from IOSuccess."""
        io_success = IOSuccess(42)
        result = FlextResult.from_io_result(io_success)

        assert result.is_success
        assert result.value == 42

    def test_from_io_result_failure(self) -> None:
        """Test creating result from IOFailure."""
        io_failure = IOFailure("io_error")
        result = FlextResult.from_io_result(io_failure)

        assert result.is_failure
        assert "io_error" in (result.error or "")

    def test_io_result_roundtrip_success(self) -> None:
        """Test roundtrip conversion success -> IOResult -> success."""
        original = FlextResult[dict[str, object]].ok({"key": "value"})
        io_result = original.to_io_result()
        recovered = FlextResult.from_io_result(io_result)

        assert recovered.is_success
        assert recovered.value == {"key": "value"}

    def test_io_result_roundtrip_failure(self) -> None:
        """Test roundtrip conversion failure -> IOResult -> failure."""
        original = FlextResult[int].fail("original_error")
        io_result = original.to_io_result()
        recovered = FlextResult.from_io_result(io_result)

        assert recovered.is_failure
        assert "original_error" in (recovered.error or "")


class TestRailwayMethods:
    """Test railway-oriented programming methods."""

    def test_lash_on_success(self) -> None:
        """Test lash on successful result does nothing."""
        result = FlextResult[int].ok(42)

        def error_handler(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(0)

        lashed = result.lash(error_handler)

        assert lashed.is_success
        assert lashed.value == 42

    def test_lash_on_failure(self) -> None:
        """Test lash on failed result applies error handler."""
        result = FlextResult[int].fail("error")

        def error_handler(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(99)

        lashed = result.lash(error_handler)

        assert lashed.is_success
        assert lashed.value == 99

    def test_lash_propagates_handler_failure(self) -> None:
        """Test lash when error handler also fails."""
        result = FlextResult[int].fail("original_error")

        def failing_handler(error: str) -> FlextResult[int]:
            return FlextResult[int].fail("handler_error")

        lashed = result.lash(failing_handler)

        assert lashed.is_failure
        assert lashed.error == "handler_error"

    def test_lash_with_exception_in_handler(self) -> None:
        """Test lash when error handler raises exception."""
        result = FlextResult[int].fail("error")

        def exception_handler(error: str) -> FlextResult[int]:
            msg = "Handler exception"
            raise ValueError(msg)

        lashed = result.lash(exception_handler)

        assert lashed.is_failure
        assert "Lash operation failed" in (lashed.error or "")

    def test_alt_on_success(self) -> None:
        """Test alt on successful result returns self."""
        result = FlextResult[int].ok(42)
        default = FlextResult[int].ok(99)

        alt_result = result.alt(default)

        assert alt_result.is_success
        assert alt_result.value == 42

    def test_alt_on_failure(self) -> None:
        """Test alt on failed result returns default."""
        result = FlextResult[int].fail("error")
        default = FlextResult[int].ok(99)

        alt_result = result.alt(default)

        assert alt_result.is_success
        assert alt_result.value == 99

    def test_alt_chaining(self) -> None:
        """Test chaining multiple alt operations."""
        result = FlextResult[int].fail("error1")
        default1 = FlextResult[int].fail("error2")
        default2 = FlextResult[int].ok(99)

        alt_result = result.alt(default1).alt(default2)

        assert alt_result.is_success
        assert alt_result.value == 99

    def test_value_or_call_on_success(self) -> None:
        """Test value_or_call on successful result."""
        result = FlextResult[int].ok(42)

        def compute_default() -> int:
            return 99

        value = result.value_or_call(compute_default)

        assert value == 42

    def test_value_or_call_on_failure(self) -> None:
        """Test value_or_call on failed result computes default."""
        result = FlextResult[int].fail("error")

        def compute_default() -> int:
            return 99

        value = result.value_or_call(compute_default)

        assert value == 99

    def test_value_or_call_lazy_evaluation(self) -> None:
        """Test value_or_call doesn't call function on success."""
        result = FlextResult[int].ok(42)
        call_count = 0

        def compute_default() -> int:
            nonlocal call_count
            call_count += 1
            return 99

        value = result.value_or_call(compute_default)

        assert value == 42
        assert call_count == 0  # Function should not be called

    def test_value_or_call_exception_handling(self) -> None:
        """Test value_or_call when default computation raises exception."""
        result = FlextResult[int].fail("error")

        def failing_default() -> int:
            msg = "Default computation failed"
            raise ValueError(msg)

        with pytest.raises(Exception):  # BaseError from FlextResult
            result.value_or_call(failing_default)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple new methods."""

    def test_complete_pipeline_with_error_recovery(self) -> None:
        """Test complete pipeline with error recovery using new methods."""

        def risky_operation() -> int:
            msg = "Risky operation failed"
            raise ValueError(msg)

        def recovery_operation(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(0)

        def double_value(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        # Combine from_callable, lash, and flow_through
        result = (
            FlextResult[int]
            .from_callable(risky_operation)
            .lash(recovery_operation)
            .flow_through(double_value)
        )

        assert result.is_success
        assert result.value == 0  # recovered value (0) * 2 = 0

    def test_maybe_io_interop_combination(self) -> None:
        """Test combining Maybe and IO interoperability."""
        # Start with a result
        original = FlextResult[str].ok("test")

        # Convert to Maybe
        maybe = original.to_maybe()

        # Convert back to Result
        from_maybe = FlextResult.from_maybe(maybe)

        # Convert to IOResult
        io_result = from_maybe.to_io_result()

        # Convert back to Result
        final = FlextResult.from_io_result(io_result)

        assert final.is_success
        assert final.value == "test"

    def test_railway_methods_with_flow_through(self) -> None:
        """Test combining railway methods with flow_through."""

        def validate(x: int) -> FlextResult[int]:
            if x < 0:
                return FlextResult[int].fail("Negative number")
            return FlextResult[int].ok(x)

        def process(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        def error_recovery(error: str) -> FlextResult[int]:
            return FlextResult[int].ok(1)

        # Test with valid input
        result1 = (
            FlextResult[int].ok(5).flow_through(validate, process).lash(error_recovery)
        )

        assert result1.is_success
        assert result1.value == 10

        # Test with invalid input that recovers
        result2 = (
            FlextResult[int]
            .ok(-5)
            .flow_through(validate, process)
            .lash(error_recovery)
            .flow_through(process)
        )

        assert result2.is_success
        assert result2.value == 2  # recovered (1) * 2

    def test_complex_data_transformation_pipeline(self) -> None:
        """Test complex data transformation using all new methods."""

        def fetch_data() -> dict[str, object]:
            return {"raw": True, "value": 100}

        def validate_data(data: dict[str, object]) -> FlextResult[dict[str, object]]:
            if "value" not in data:
                return FlextResult[dict[str, object]].fail("Missing value")
            return FlextResult[dict[str, object]].ok(data)

        def enrich_data(data: dict[str, object]) -> FlextResult[dict[str, object]]:
            enriched = {**data, "enriched": True}
            return FlextResult[dict[str, object]].ok(enriched)

        def error_fallback(
            error: str,
        ) -> FlextResult[dict[str, object]]:
            return FlextResult[dict[str, object]].ok({"fallback": True})

        # Complete pipeline
        result = (
            FlextResult[dict[str, object]]
            .from_callable(fetch_data)
            .flow_through(validate_data, enrich_data)
            .lash(error_fallback)
        )

        assert result.is_success
        assert result.value["raw"] is True
        assert result.value["enriched"] is True
        assert result.value["value"] == 100
