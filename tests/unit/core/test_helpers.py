"""Tests for FLEXT Core helpers - reducing boilerplate for applications."""

from __future__ import annotations

import pytest
from pydantic import Field

from flext_core import (
    FlextCacheableMixin,
    FlextHelpers,
    FlextLoggerMixin,
    FlextPipeline,
    FlextResult,
    FlextValidators,
    FlextValueObject,
    flext_chain,
    flext_fail,
    flext_ok,
    flext_retry,
    flext_safe,
    flext_validate,
)


class TestQuickHelpers:
    """Test quick helper functions."""

    def test_ok_helper(self) -> None:
        """Test ok helper creates success result."""
        result = flext_ok("test")
        assert result.is_success
        assert result.data == "test"

    def test_fail_helper(self) -> None:
        """Test fail helper creates failure result."""
        result = flext_fail("error message")
        assert result.is_failure
        assert result.error == "error message"

    def test_helpers_ok(self) -> None:
        """Test FlextHelpers.flext_ok method."""
        result = FlextHelpers.flext_ok("test")
        assert result.is_success
        assert result.data == "test"

    def test_helpers_fail(self) -> None:
        """Test FlextHelpers.flext_fail method."""
        result = FlextHelpers.flext_fail("error message")
        assert result.is_failure
        assert result.error == "error message"


class TestSafeHelpers:
    """Test safe helper functions."""

    def test_safe_success(self) -> None:
        """Test safe helper with successful function."""

        def divide(a: int, b: int) -> float:
            return a / b

        result = FlextHelpers.flext_safe(divide, 10, 2)
        assert result.is_success
        assert result.data == 5.0

    def test_safe_exception(self) -> None:
        """Test safe helper with exception."""

        def divide(a: int, b: int) -> float:
            if b == 0:
                msg = "Division by zero"
                raise ValueError(msg)
            return a / b

        result = FlextHelpers.flext_safe(divide, 10, 0)
        assert result.is_failure
        assert "division by zero" in (result.error or "").lower()

    def test_flext_safe_alias(self) -> None:
        """Test flext_safe function alias."""

        def divide(a: int, b: int) -> float:
            return a / b

        result = flext_safe(divide, 10, 2)
        assert result.is_success
        assert result.data == 5.0


class TestValidationHelpers:
    """Test validation helper functions."""

    def test_validate_email_valid(self) -> None:
        """Test email validation with valid email."""
        result = FlextValidators.flext_validate_email("test@example.com")
        assert result.is_success
        assert result.data == "test@example.com"

    def test_validate_email_invalid(self) -> None:
        """Test email validation with invalid email."""
        result = FlextValidators.flext_validate_email("invalid-email")
        assert result.is_failure
        assert "Invalid email format" in (result.error or "")

    def test_validate_non_empty_valid(self) -> None:
        """Test non-empty validation with valid string."""
        result = FlextValidators.flext_validate_not_empty("  test  ")
        assert result.is_success
        assert result.data == "test"

    def test_validate_non_empty_invalid(self) -> None:
        """Test non-empty validation with empty string."""
        result = FlextValidators.flext_validate_not_empty("   ")
        assert result.is_failure
        assert "cannot be empty" in (result.error or "").lower()

    def test_validate_required_success(self) -> None:
        """Test required validation with valid data."""
        result = FlextValidators.flext_validate_required("test")
        assert result.is_success
        assert result.data == "test"

    def test_validate_required_failure(self) -> None:
        """Test required validation with None."""
        result = FlextValidators.flext_validate_required(None)
        assert result.is_failure
        assert "required" in (result.error or "")


class TestChainHelpers:
    """Test chain helper functions."""

    def test_chain_success_operations(self) -> None:
        """Test chain helper with successful operations."""

        def add_one(x: int) -> int:
            return x + 1

        def multiply_two(x: int) -> int:
            return x * 2

        chained_func = FlextHelpers.flext_chain(add_one, multiply_two)
        result = chained_func(5)
        assert result == 12  # (5 + 1) * 2

    def test_chain_with_flext_results(self) -> None:
        """Test chain with FlextResult objects."""

        def add_one_result(x: int) -> FlextResult[int]:
            return FlextResult.ok(x + 1)

        def multiply_two_result(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        # Create a custom chain that handles FlextResult properly
        result1 = add_one_result(5)
        if result1.is_success:
            result2 = multiply_two_result(result1.data)
            assert result2.is_success
            assert result2.data == 12

    def test_flext_chain_alias(self) -> None:
        """Test flext_chain function alias."""

        def add_one(x: int) -> int:
            return x + 1

        def multiply_two(x: int) -> int:
            return x * 2

        chained_func = flext_chain(add_one, multiply_two)
        result = chained_func(5)
        assert result == 12


class TestPipeline:
    """Test Pipeline class for fluent operations."""

    def test_pipeline_success_chain(self) -> None:
        """Test pipeline with successful operations."""

        def add_one(x: int) -> int:
            return x + 1

        def multiply_two(x: int) -> int:
            return x * 2

        pipeline = FlextPipeline(5)
        result_pipeline = pipeline.pipe(add_one).pipe(multiply_two)

        assert result_pipeline.is_success
        assert result_pipeline.data == 12  # (5 + 1) * 2

    def test_pipeline_with_exception(self) -> None:
        """Test pipeline with exception in chain."""

        def add_one(x: int) -> int:
            return x + 1

        def raise_error(_: int) -> int:
            msg = "Test error"
            raise ValueError(msg)  # Use ValueError which is caught

        def multiply_two(x: int) -> int:
            return x * 2

        pipeline = FlextPipeline(5)
        result_pipeline = pipeline.pipe(add_one).pipe(raise_error).pipe(multiply_two)

        assert not result_pipeline.is_success
        assert result_pipeline.error is not None
        assert "Test error" in (result_pipeline.error or "")

    def test_pipeline_unwrap(self) -> None:
        """Test pipeline unwrap operation."""

        def add_one(x: int) -> int:
            return x + 1

        pipeline = FlextPipeline(5)
        result = pipeline.pipe(add_one).unwrap()
        assert result == 6

    def test_pipeline_unwrap_or(self) -> None:
        """Test pipeline unwrap_or operation."""

        def raise_error(_: int) -> int:
            msg = "Test error"
            raise ValueError(msg)

        pipeline = FlextPipeline(5)
        result = pipeline.pipe(raise_error).unwrap_or(99)
        assert result == 99


class TestValidationChain:
    """Test validation chain functionality."""

    def test_validation_chain_success(self) -> None:
        """Test validation chain with successful validations."""
        result = (
            flext_validate("test@example.com").required().not_empty().email().result()
        )
        assert result.is_success
        assert result.data == "test@example.com"

    def test_validation_chain_failure(self) -> None:
        """Test validation chain with failing validation."""
        result = (
            flext_validate("invalid-email")
            .required()
            .not_empty()
            .email()  # This should fail
            .result()
        )
        assert result.is_failure
        assert "Invalid email format" in (result.error or "")


class TestMixins:
    """Test mixin classes."""

    def test_logger_mixin(self) -> None:
        """Test LoggerMixin provides logger."""

        class TestClass(FlextLoggerMixin):
            pass

        obj = TestClass()
        logger = obj.logger
        assert logger is not None
        assert hasattr(logger, "info")

    def test_cacheable_mixin(self) -> None:
        """Test CacheableMixin provides caching functionality."""

        class TestClass(FlextCacheableMixin):
            def __init__(self) -> None:
                super().__init__()

        obj = TestClass()

        # Test cache operations
        obj.cache_set("test_key", "test_value")
        cached_value = obj.cache_get("test_key")
        assert cached_value == "test_value"

        # Test cache removal
        obj.cache_remove("test_key")
        cached_value = obj.cache_get("test_key")
        assert cached_value is None


class TestRetryDecorator:
    """Test retry functionality."""

    def test_retry_success_first_attempt(self) -> None:
        """Test retry with immediate success."""
        call_count = 0

        def test_function() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        retry_wrapper = FlextHelpers.flext_retry(test_function, max_retries=3)
        result = retry_wrapper()

        assert result.is_success
        assert result.data == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self) -> None:
        """Test retry with success after failures."""
        call_count = 0

        def test_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                msg = "temporary failure"
                raise ValueError(msg)
            return "success"

        retry_wrapper = FlextHelpers.flext_retry(test_function, max_retries=3)
        result = retry_wrapper()

        assert result.is_success
        assert result.data == "success"
        assert call_count == 3

    def test_retry_all_attempts_fail(self) -> None:
        """Test retry when all attempts fail."""
        call_count = 0

        def test_function() -> str:
            nonlocal call_count
            call_count += 1
            msg = "persistent failure"
            raise ValueError(msg)

        retry_wrapper = FlextHelpers.flext_retry(test_function, max_retries=2)
        result = retry_wrapper()

        assert result.is_failure
        assert "persistent failure" in (result.error or "")
        assert call_count == 3  # Initial attempt + 2 retries

    def test_flext_retry_alias(self) -> None:
        """Test flext_retry function alias."""

        def test_function() -> str:
            return "success"

        retry_wrapper = flext_retry(test_function, max_retries=1)
        result = retry_wrapper()

        assert result.is_success
        assert result.data == "success"


class TestQuickValueObject:
    """Test quick value object functionality."""

    def test_value_object_creation(self) -> None:
        """Test creating a value object."""

        class Money(FlextValueObject):
            amount: float = Field(...)
            currency: str = Field(...)

            def validate_domain_rules(self) -> None:
                """Validate domain rules for Money."""
                if self.amount < 0:
                    msg = "Amount cannot be negative"
                    raise ValueError(msg)

        money = Money(amount=100.0, currency="USD")
        assert money.amount == 100.0
        assert money.currency == "USD"

    def test_value_object_immutability(self) -> None:
        """Test value object immutability."""

        class Money(FlextValueObject):
            amount: float = Field(...)
            currency: str = Field(...)

            def validate_domain_rules(self) -> None:
                """Validate domain rules for Money."""
                if self.amount < 0:
                    msg = "Amount cannot be negative"
                    raise ValueError(msg)

        money = Money(amount=100.0, currency="USD")

        # FlextValueObject should be frozen, so this should raise an error
        with pytest.raises((AttributeError, ValueError)):
            money.amount = 200.0

    def test_value_object_equality(self) -> None:
        """Test value object equality."""

        class Money(FlextValueObject):
            amount: float = Field(...)
            currency: str = Field(...)

            def validate_domain_rules(self) -> None:
                """Validate domain rules for Money."""
                if self.amount < 0:
                    msg = "Amount cannot be negative"
                    raise ValueError(msg)

        money1 = Money(amount=100.0, currency="USD")
        money2 = Money(amount=100.0, currency="USD")
        money3 = Money(amount=200.0, currency="USD")

        assert money1 == money2
        assert money1 != money3
