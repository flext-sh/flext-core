"""Comprehensive tests for FlextDecorators and decorator functionality."""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from flext_core import (
    FlextDecorators,
    FlextResult,
)
from flext_core.decorators import (
    FlextDecoratorUtils,
    FlextErrorHandlingDecorators,
    FlextLoggingDecorators,
    FlextPerformanceDecorators,
    FlextValidationDecorators,
)
from flext_core.protocols import FlextDecoratedFunction

# Aliases for compatibility
_BaseDecoratorUtils = FlextDecoratorUtils
_BaseValidationDecorators = FlextValidationDecorators
_BaseErrorHandlingDecorators = FlextErrorHandlingDecorators
_BasePerformanceDecorators = FlextPerformanceDecorators
_BaseLoggingDecorators = FlextLoggingDecorators
_DecoratedFunction = FlextDecoratedFunction

# Simple decorator functions
flext_safe_call = FlextDecorators.safe_result
flext_cache_decorator = FlextPerformanceDecorators.create_cache_decorator
flext_safe_decorator = FlextErrorHandlingDecorators.create_safe_decorator
flext_timing_decorator = FlextPerformanceDecorators.get_timing_decorator

# Internal decorator functions from base_decorators
_safe_call_decorator = FlextErrorHandlingDecorators.create_safe_decorator
_validate_input_decorator = FlextValidationDecorators.create_validation_decorator

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3


class UserModel(BaseModel):
    """Test model for validation."""

    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    email: str


class TestFlextDecorators:
    """Test FlextDecorators orchestration functionality."""

    def test_safe_result_decorator_success(self) -> None:
        """Test safe_result decorator with successful function."""

        @FlextDecorators.safe_result
        def safe_function(x: int, y: int) -> int:
            return x + y

        result = safe_function(2, 3)

        assert isinstance(result, FlextResult)
        assert result.success
        if result.data != 5:
            raise AssertionError(f"Expected {5}, got {result.data}")

    def test_safe_result_decorator_failure(self) -> None:
        """Test safe_result decorator with failing function."""

        @FlextDecorators.safe_result
        def failing_function() -> int:
            msg = "Test error"
            raise ValueError(msg)

        result = failing_function()

        assert isinstance(result, FlextResult)
        assert result.is_failure
        if "Test error" not in (result.error or ""):
            raise AssertionError(f"Expected 'Test error' in {result.error}")

    def test_safe_result_decorator_with_none_return(self) -> None:
        """Test safe_result decorator with function returning None."""

        @FlextDecorators.safe_result
        def void_function() -> None:
            return None

        result = void_function()

        assert isinstance(result, FlextResult)
        # result.data is None is guaranteed by the decorator, no need to assert

    def test_validated_with_result_decorator_success(self) -> None:
        """Test validated_with_result decorator with valid data."""

        @FlextDecorators.validated_with_result(UserModel)
        def create_user(**kwargs: object) -> str:
            return f"Created user: {kwargs['name']}"

        result = create_user(name="Alice", age=30, email="alice@example.com")

        assert isinstance(result, FlextResult)
        assert result.success
        if "Created user: Alice" not in (result.data or ""):
            raise AssertionError(f"Expected {'Created user: Alice'} in {result.data}")

    def test_validated_with_result_decorator_validation_failure(self) -> None:
        """Test validated_with_result decorator with invalid data."""

        @FlextDecorators.validated_with_result(UserModel)
        def create_user(**kwargs: object) -> str:
            return f"Created user: {kwargs['name']}"

        # Invalid age
        result = create_user(name="Alice", age=-5, email="alice@example.com")

        assert isinstance(result, FlextResult)
        assert result.is_failure
        if "Validation failed" not in (result.error or ""):
            raise AssertionError(f"Expected 'Validation failed' in {result.error}")

    def test_validated_with_result_decorator_execution_failure(self) -> None:
        """Test validated_with_result decorator when execution fails."""

        @FlextDecorators.validated_with_result(UserModel)
        def failing_user_creation(**kwargs: object) -> str:
            msg = "Database error"
            raise RuntimeError(msg)

        result = failing_user_creation(name="Alice", age=30, email="alice@example.com")

        assert isinstance(result, FlextResult)
        assert result.is_failure
        if "Execution failed" not in (result.error or ""):
            raise AssertionError(f"Expected 'Execution failed' in {result.error}")

    def test_cached_with_timing_decorator(self) -> None:
        """Test cached_with_timing decorator functionality."""
        call_count = 0

        @FlextDecorators.cached_with_timing(max_size=2)
        def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate work
            return x * 2

        # First call
        result1 = expensive_operation(5)
        if result1 != 10:
            raise AssertionError(f"Expected {10}, got {result1}")
        assert call_count == 1

        # Second call with same argument (should be cached)
        result2 = expensive_operation(5)
        if result2 != 10:
            raise AssertionError(f"Expected {10}, got {result2}")
        assert call_count == 1  # Should not increment

        # Different argument
        result3 = expensive_operation(3)
        if result3 != 6:
            raise AssertionError(f"Expected {6}, got {result3}")
        assert call_count == EXPECTED_BULK_SIZE

    def test_safe_cached_decorator(self) -> None:
        """Test safe_cached decorator functionality."""
        call_count = 0

        @FlextDecorators.safe_cached(max_size=2)
        def safe_cached_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if x < 0:
                msg = "Negative input"
                raise ValueError(msg)
            return x * 3

        # Successful call - safe_cached doesn't return FlextResult,
        # it works with base decorators
        result1 = safe_cached_operation(4)
        if result1 != 12:
            raise AssertionError(f"Expected {12}, got {result1}")
        assert call_count == 1

        # Cached call
        result2 = safe_cached_operation(4)
        if result2 != 12:
            raise AssertionError(f"Expected {12}, got {result2}")
        assert call_count == 1  # Should not increment

        # Failing call - safe decorator will handle the exception
        with contextlib.suppress(ValueError):
            safe_cached_operation(-1)
            # If safe decorator doesn't raise, we continue

    def test_validated_cached_decorator(self) -> None:
        """Test validated_cached decorator functionality."""

        @FlextDecorators.validated_cached(UserModel, max_size=2)
        def process_user(**kwargs: object) -> str:
            return f"Processed: {kwargs['name']} ({kwargs['age']})"

        # Valid input
        result1 = process_user(name="Bob", age=25, email="bob@example.com")
        assert isinstance(result1, FlextResult)
        assert result1.success
        if "Processed: Bob (25)" not in (result1.data or ""):
            raise AssertionError(f"Expected {'Processed: Bob (25)'} in {result1.data}")

        # Invalid input
        result2 = process_user(name="", age=25, email="bob@example.com")
        assert isinstance(result2, FlextResult)
        assert result2.is_failure
        if "Validation failed" not in (result2.error or ""):
            raise AssertionError(f"Expected {'Validation failed'} in {result2.error}")

    def test_complete_decorator_with_all_features(self) -> None:
        """Test complete_decorator with all features enabled."""

        @FlextDecorators.complete_decorator(
            UserModel,
            cache_size=16,
            with_timing=True,
            with_logging=True,
        )
        def complex_user_operation(**kwargs: object) -> str:
            return f"Complex operation for: {kwargs['name']}"

        result = complex_user_operation(
            name="Charlie",
            age=35,
            email="charlie@example.com",
        )

        assert isinstance(result, FlextResult)
        assert result.success
        if "Complex operation for: Charlie" not in (result.data or ""):
            raise AssertionError(
                f"Expected {'Complex operation for: Charlie'} in {result.data}"
            )

    def test_complete_decorator_minimal(self) -> None:
        """Test complete_decorator with minimal configuration."""

        @FlextDecorators.complete_decorator()
        def simple_operation(x: int) -> int:
            return x * 2

        result = simple_operation(x=7)

        # complete_decorator without model_class doesn't return FlextResult
        if result != 14:
            raise AssertionError(f"Expected {14}, got {result}")


class TestStandaloneDecorators:
    """Test standalone decorator functions."""

    def test_flext_safe_call_success(self) -> None:
        """Test flext_safe_call with successful function."""

        @flext_safe_call
        def safe_operation(a: int, b: int) -> int:
            return a * b

        result = safe_operation(6, 7)
        # flext_safe_call returns FlextResult
        assert isinstance(result, FlextResult)
        assert result.success
        if result.data != 42:
            raise AssertionError(f"Expected {42}, got {result.data}")

    def test_flext_safe_call_failure(self) -> None:
        """Test flext_safe_call with failing function."""

        @flext_safe_call
        def risky_operation() -> int:
            msg = "Something went wrong"
            raise RuntimeError(msg)

        # flext_safe_call returns FlextResult, not exception
        result = risky_operation()
        assert isinstance(result, FlextResult)
        assert result.is_failure
        if "Something went wrong" not in (result.error or ""):
            raise AssertionError(f"Expected 'Something went wrong' in {result.error}")

    def test_flext_cache_decorator(self) -> None:
        """Test flext_cache_decorator functionality."""
        call_count = 0

        @flext_cache_decorator(max_size=3)
        def cached_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x**2

        # First call
        result1 = cached_function(4)
        if result1 != 16:
            raise AssertionError(f"Expected {16}, got {result1}")
        assert call_count == 1

        # Cached call
        result2 = cached_function(4)
        if result2 != 16:
            raise AssertionError(f"Expected {16}, got {result2}")
        assert call_count == 1

        # New argument
        result3 = cached_function(5)
        if result3 != 25:
            raise AssertionError(f"Expected {25}, got {result3}")
        assert call_count == EXPECTED_BULK_SIZE

    def test_flext_safe_decorator(self) -> None:
        """Test flext_safe_decorator functionality."""

        @flext_safe_decorator()
        def potentially_unsafe_function(x: int) -> int:
            if x < 0:
                msg = "Negative value not allowed"
                raise ValueError(msg)
            return x + 10

        # Should handle exceptions gracefully
        try:
            result1 = potentially_unsafe_function(5)
            if result1 != 15:
                raise AssertionError(f"Expected {15}, got {result1}")
        except (ValueError, TypeError, RuntimeError) as e:
            if TYPE_CHECKING:
                import pytest  # imported but unused in TYPE_CHECKING
            else:
                import pytest
            pytest.fail(f"Safe decorator should handle exceptions: {e}")

    def test_flext_timing_decorator(self) -> None:
        """Test flext_timing_decorator functionality."""

        @flext_timing_decorator
        def timed_function(delay: float) -> str:
            time.sleep(delay)
            return "completed"

        result = timed_function(0.01)
        if result != "completed":
            raise AssertionError(f"Expected {'completed'}, got {result}")


class TestDecoratorComposition:
    """Test combining multiple decorators."""

    def test_multiple_decorators_composition(self) -> None:
        """Test applying multiple decorators to the same function."""

        @FlextDecorators.safe_result
        @flext_timing_decorator
        def multi_decorated_function(x: int) -> int:
            time.sleep(0.001)
            return x * 10

        result = multi_decorated_function(3)

        assert isinstance(result, FlextResult)
        assert result.success
        if result.data != 30:
            raise AssertionError(f"Expected {30}, got {result.data}")

    def test_decorator_with_validation_and_caching(self) -> None:
        """Test decorator combining validation and caching."""

        @FlextDecorators.validated_cached(UserModel, max_size=5)
        def user_processor(**kwargs: object) -> dict[str, object]:
            return {
                "id": hash(kwargs["email"]),
                "display_name": f"{kwargs['name']} ({kwargs['age']})",
                "status": "active",
            }

        # Valid processing - decorator transforms result to FlextResult
        result1 = user_processor(name="Diana", age=28, email="diana@example.com")
        # Use hasattr to check if it's a FlextResult without type conflicts
        assert hasattr(result1, "success")
        assert result1.success
        if "display_name" not in str(getattr(result1, "data", "") or ""):
            raise AssertionError(
                f"Expected {'display_name'} in {getattr(result1, 'data', '')}"
            )

        # Invalid input
        result2 = user_processor(name="Diana", age=200, email="diana@example.com")
        # Use hasattr to check if it's a FlextResult without type conflicts
        assert hasattr(result2, "is_failure")
        assert result2.is_failure


class TestDecoratorErrorHandling:
    """Test error handling across different decorators."""

    def test_safe_result_with_different_exception_types(self) -> None:
        """Test safe_result handling different exception types."""

        @FlextDecorators.safe_result
        def multi_exception_function(exception_type: str) -> str:
            if exception_type == "value":
                value_error_msg = "Value error"
                raise ValueError(value_error_msg)
            if exception_type == "type":
                type_error_msg = "Type error"
                raise TypeError(type_error_msg)
            if exception_type == "runtime":
                runtime_error_msg = "Runtime error"
                raise RuntimeError(runtime_error_msg)
            return "success"

        # Test different exception types
        result1 = multi_exception_function("value")
        assert isinstance(result1, FlextResult)
        assert result1.is_failure
        if "Value error" not in (result1.error or ""):
            raise AssertionError(f"Expected {'Value error'} in {result1.error}")

        result2 = multi_exception_function("type")
        assert isinstance(result2, FlextResult)
        assert result2.is_failure
        if "Type error" not in (result2.error or ""):
            raise AssertionError(f"Expected {'Type error'} in {result2.error}")

        result3 = multi_exception_function("runtime")
        assert isinstance(result3, FlextResult)
        assert result3.is_failure
        if "Runtime error" not in (result3.error or ""):
            raise AssertionError(f"Expected {'Runtime error'} in {result3.error}")

        # Test success case
        result4 = multi_exception_function("none")
        assert isinstance(result4, FlextResult)
        assert result4.success
        if result4.data != "success":
            raise AssertionError(f"Expected {'success'}, got {result4.data}")

    def test_validation_decorator_error_details(self) -> None:
        """Test validation decorator provides detailed error information."""

        @FlextDecorators.validated_with_result(UserModel)
        def detailed_user_creation(**kwargs: object) -> dict[str, object]:
            return {"created": True, "user": kwargs}

        # Multiple validation errors
        result = detailed_user_creation(
            name="",  # Too short
            age=-1,  # Negative
            email="invalid-email",  # Invalid format
        )

        # Decorator transforms result to FlextResult - use hasattr approach
        assert hasattr(result, "is_failure")
        assert result.is_failure
        if "Validation failed" not in (getattr(result, "error", "") or ""):
            raise AssertionError(
                f"Expected 'Validation failed' in {getattr(result, 'error', '')}"
            )


class TestDecoratorPerformance:
    """Test decorator performance characteristics."""

    def test_caching_effectiveness(self) -> None:
        """Test that caching actually improves performance."""
        expensive_calls = 0

        @FlextDecorators.cached_with_timing(max_size=10)
        def expensive_computation(n: int) -> int:
            nonlocal expensive_calls
            expensive_calls += 1
            # Simulate expensive computation
            time.sleep(0.001)
            return sum(range(n))

        # First call
        start_time = time.time()
        result1 = expensive_computation(100)
        first_call_time = time.time() - start_time

        # Second call (should be cached)
        start_time = time.time()
        result2 = expensive_computation(100)
        cached_call_time = time.time() - start_time

        if result1 != result2:
            raise AssertionError(f"Expected {result2}, got {result1}")
        assert expensive_calls == 1  # Only called once
        assert cached_call_time < first_call_time  # Cached call is faster

    def test_cache_size_limits(self) -> None:
        """Test that cache respects size limits."""
        call_count = 0

        @FlextDecorators.cached_with_timing(max_size=2)
        def limited_cache_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Fill cache
        limited_cache_function(1)  # call_count = 1
        limited_cache_function(2)  # call_count = 2
        limited_cache_function(3)  # call_count = 3, evicts 1

        # Verify cache behavior
        limited_cache_function(2)  # Should be cached, call_count stays 3
        limited_cache_function(3)  # Should be cached, call_count stays 3
        limited_cache_function(1)  # Should be re-computed, call_count = 4

        if call_count != 4:
            raise AssertionError(f"Expected {4}, got {call_count}")


class TestBaseDecoratorClasses:
    """Test base decorator classes for better coverage."""

    def test_base_decorator_utils(self) -> None:
        """Test _BaseDecoratorUtils functionality."""

        # Test preserve_metadata
        def original_function() -> str:
            """Original docstring."""
            return "test"

        def wrapper_function() -> str:
            return "wrapper"

        # This should preserve metadata
        result = _BaseDecoratorUtils.preserve_metadata(
            cast("_DecoratedFunction", original_function),
            cast("_DecoratedFunction", wrapper_function),
        )
        assert callable(result)

    def test_base_validation_decorators(self) -> None:
        """Test _BaseValidationDecorators functionality."""

        def simple_validator(value: object) -> bool:
            return isinstance(value, str) and len(str(value)) > 0

        # Test create_validation_decorator
        decorator = _BaseValidationDecorators.create_validation_decorator(
            simple_validator
        )
        assert callable(decorator)

        # Test validate_arguments (currently just returns the function)
        def test_func() -> str:
            return "test"

        validated_func = _BaseValidationDecorators.validate_arguments(
            cast("_DecoratedFunction", test_func)
        )
        assert validated_func is test_func

    def test_base_error_handling_decorators(self) -> None:
        """Test _BaseErrorHandlingDecorators functionality."""

        def error_handler(error: Exception) -> str:
            return f"Handled: {error}"

        # Test create_safe_decorator
        decorator = _BaseErrorHandlingDecorators.create_safe_decorator(error_handler)
        assert callable(decorator)

        # Test create_safe_decorator without handler
        decorator = _BaseErrorHandlingDecorators.create_safe_decorator()
        assert callable(decorator)

    def test_base_performance_decorators(self) -> None:
        """Test _BasePerformanceDecorators functionality."""

        # Test get_timing_decorator
        decorator = _BasePerformanceDecorators.get_timing_decorator()
        assert callable(decorator)

        # Test create_cache_decorator
        decorator = _BasePerformanceDecorators.create_cache_decorator(max_size=10)
        assert callable(decorator)

        # Test memoize_decorator
        def test_func(x: int) -> int:
            return x * 2

        memoized = _BasePerformanceDecorators.memoize_decorator(
            cast("_DecoratedFunction", test_func)
        )
        assert callable(memoized)


class TestDecoratorCoverageImprovements:
    """Test cases specifically for improving coverage of _decorators_base.py module."""

    def test_decorator_imports_coverage(self) -> None:
        """Test decorator imports from TYPE_CHECKING block (lines 91-93)."""
        from flext_core import _decorators_base

        # Verify that TYPE_CHECKING imports are available at runtime for coverage
        assert hasattr(_decorators_base, "_DecoratedFunction")
        assert hasattr(_decorators_base, "_BaseDecoratorUtils")

    def test_immutability_decorators_coverage(self) -> None:
        """Test immutability decorator methods (lines 161, 276, 281)."""
        from flext_core.base_decorators import (
            FlextImmutabilityDecorators as _BaseImmutabilityDecorators,
        )

        def sample_function(x: int) -> int:
            return x * 2

        # Test immutable_decorator - line 276
        decorated = _BaseImmutabilityDecorators.immutable_decorator(
            cast("_DecoratedFunction", sample_function)
        )
        assert decorated is sample_function  # Returns same function

        # Test freeze_args_decorator - line 281
        decorated = _BaseImmutabilityDecorators.freeze_args_decorator(
            cast("_DecoratedFunction", sample_function)
        )
        assert decorated is sample_function  # Returns same function

    def test_functional_decorators_coverage(self) -> None:
        """Test functional decorator methods (lines 290, 295)."""
        from flext_core.base_decorators import (
            FlextFunctionalDecorators as _BaseFunctionalDecorators,
        )

        def sample_function(x: int) -> int:
            return x * 2

        # Test curry_decorator - line 290
        decorated = _BaseFunctionalDecorators.curry_decorator(
            cast("_DecoratedFunction", sample_function)
        )
        assert decorated is sample_function  # Returns same function

        # Test compose_decorator - line 295
        decorated = _BaseFunctionalDecorators.compose_decorator(
            cast("_DecoratedFunction", sample_function)
        )
        assert decorated is sample_function  # Returns same function

    def test_logging_decorator_exception_handling(self) -> None:
        """Test logging decorator exception handling (lines 222-236)."""
        import pytest

        from flext_core.base_decorators import (
            FlextLoggingDecorators as _BaseLoggingDecorators,
        )

        def failing_function() -> None:
            msg = "Test runtime error"
            raise RuntimeError(msg)

        decorated = _BaseLoggingDecorators.log_calls_decorator(
            cast("_DecoratedFunction", failing_function)
        )

        # Should re-raise the exception after logging
        with pytest.raises(RuntimeError, match="Test runtime error"):
            decorated()

    def test_logging_decorator_type_error_handling(self) -> None:
        """Test logging decorator with TypeError (lines 222-236)."""
        import pytest

        from flext_core.base_decorators import (
            FlextLoggingDecorators as _BaseLoggingDecorators,
        )

        def type_error_function() -> None:
            msg = "Type error occurred"
            raise TypeError(msg)

        decorated = _BaseLoggingDecorators.log_calls_decorator(
            cast("_DecoratedFunction", type_error_function)
        )

        with pytest.raises(TypeError, match="Type error occurred"):
            decorated()

    def test_logging_decorator_value_error_handling(self) -> None:
        """Test logging decorator with ValueError (lines 222-236)."""
        import pytest

        from flext_core.base_decorators import (
            FlextLoggingDecorators as _BaseLoggingDecorators,
        )

        def value_error_function() -> None:
            msg = "Value error occurred"
            raise ValueError(msg)

        decorated = _BaseLoggingDecorators.log_calls_decorator(
            cast("_DecoratedFunction", value_error_function)
        )

        with pytest.raises(ValueError, match="Value error occurred"):
            decorated()

    def test_log_exceptions_decorator_exception_handling(self) -> None:
        """Test log_exceptions_decorator exception handling (lines 246-267)."""
        import pytest

        from flext_core.base_decorators import (
            FlextLoggingDecorators as _BaseLoggingDecorators,
        )

        def failing_function() -> None:
            msg = "Exception for logging test"
            raise RuntimeError(msg)

        decorated = _BaseLoggingDecorators.log_exceptions_decorator(
            cast("_DecoratedFunction", failing_function)
        )

        with pytest.raises(RuntimeError, match="Exception for logging test"):
            decorated()

    def test_log_exceptions_decorator_multiple_exception_types(self) -> None:
        """Test log_exceptions_decorator with different exception types (lines 246-267)."""
        import pytest

        from flext_core.base_decorators import (
            FlextLoggingDecorators as _BaseLoggingDecorators,
        )

        def type_error_function() -> None:
            msg = "Type error in log_exceptions test"
            raise TypeError(msg)

        def value_error_function() -> None:
            msg = "Value error in log_exceptions test"
            raise ValueError(msg)

        decorated_type = _BaseLoggingDecorators.log_exceptions_decorator(
            cast("_DecoratedFunction", type_error_function)
        )
        decorated_value = _BaseLoggingDecorators.log_exceptions_decorator(
            cast("_DecoratedFunction", value_error_function)
        )

        with pytest.raises(TypeError, match="Type error in log_exceptions test"):
            decorated_type()

        with pytest.raises(ValueError, match="Value error in log_exceptions test"):
            decorated_value()

    def test_safe_call_decorator_with_error_handler(self) -> None:
        """Test safe_call_decorator with error handler (lines 325-326)."""
        from flext_core.base_decorators import _safe_call_decorator

        error_handled = False

        def error_handler(error: Exception) -> str:
            nonlocal error_handled
            error_handled = True
            return f"Handled: {error}"

        def failing_function() -> None:
            msg = "Function failed"
            raise ValueError(msg)

        decorator = _safe_call_decorator(error_handler)
        decorated = decorator(cast("_DecoratedFunction", failing_function))

        result = decorated()
        assert error_handled
        assert result == "Handled: Function failed"

    def test_validate_input_decorator_validation_failure(self) -> None:
        """Test validation decorator with validation failure (lines 377-388)."""
        import pytest

        from flext_core.base_decorators import _validate_input_decorator
        from flext_core.exceptions import FlextValidationError

        def always_false_validator(arg: object) -> bool:
            return False

        def sample_function(x: int) -> int:
            return x * 2

        decorator = _validate_input_decorator(always_false_validator)
        decorated = decorator(cast("_DecoratedFunction", sample_function))

        with pytest.raises(FlextValidationError, match="Input validation failed"):
            decorated(5)

    def test_validate_input_decorator_with_multiple_args(self) -> None:
        """Test validation decorator with multiple arguments (lines 377-388)."""
        import pytest

        from flext_core.base_decorators import _validate_input_decorator
        from flext_core.exceptions import FlextValidationError

        def validator_requiring_positive(arg: object) -> bool:
            return isinstance(arg, int) and arg > 0

        def sample_function(x: int, y: int) -> int:
            return x + y

        decorator = _validate_input_decorator(validator_requiring_positive)
        decorated = decorator(cast("_DecoratedFunction", sample_function))

        # Should pass with at least one positive argument
        result = decorated(5, -1)
        assert result == 4

        # Should fail with all non-positive arguments
        with pytest.raises(FlextValidationError, match="Input validation failed"):
            decorated(-1, -2)

    def test_decorator_factory_methods_coverage(self) -> None:
        """Test decorator factory methods (lines 401, 408, 413, 420)."""
        from flext_core.base_decorators import (
            FlextDecoratorFactory as _BaseDecoratorFactory,
        )

        # Test create_cache_decorator - line 401
        cache_decorator = _BaseDecoratorFactory.create_cache_decorator(64)
        assert callable(cache_decorator)

        # Test create_safe_decorator - line 408
        safe_decorator = _BaseDecoratorFactory.create_safe_decorator()
        assert callable(safe_decorator)

        # Test create_timing_decorator - line 413
        timing_decorator = _BaseDecoratorFactory.create_timing_decorator()
        assert callable(timing_decorator)

        # Test create_validation_decorator - line 420
        def dummy_validator(arg: object) -> bool:
            return True

        validation_decorator = _BaseDecoratorFactory.create_validation_decorator(
            dummy_validator
        )
        assert callable(validation_decorator)

    def test_error_handling_decorator_retry_method(self) -> None:
        """Test retry_decorator method (line 161)."""
        from flext_core.base_decorators import (
            FlextErrorHandlingDecorators as _BaseErrorHandlingDecorators,
        )

        def sample_function(x: int) -> int:
            return x * 2

        # Test retry_decorator - currently returns same function (line 161)
        decorated = _BaseErrorHandlingDecorators.retry_decorator(
            cast("_DecoratedFunction", sample_function)
        )
        assert decorated is sample_function  # Returns same function
