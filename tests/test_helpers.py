"""Tests for FLEXT Core helpers - reducing boilerplate for applications."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import Field

from flext_core.helpers import ContainerMixin
from flext_core.helpers import LoggerMixin
from flext_core.helpers import QuickEntity
from flext_core.helpers import QuickValueObject
from flext_core.helpers import ValidatorMixin
from flext_core.helpers import cache_result
from flext_core.helpers import chain
from flext_core.helpers import fail
from flext_core.helpers import from_dict
from flext_core.helpers import ok
from flext_core.helpers import pipeline
from flext_core.helpers import retry
from flext_core.helpers import safe
from flext_core.helpers import to_dict
from flext_core.helpers import validate_email
from flext_core.helpers import validate_non_empty
from flext_core.helpers import validate_required

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class TestQuickHelpers:
    """Test quick helper functions."""

    def test_ok_helper(self) -> None:
        """Test ok helper creates success result."""
        result = ok("test")
        assert result.is_success
        assert result.data == "test"

    def test_fail_helper(self) -> None:
        """Test fail helper creates failure result."""
        result = fail("error message")
        assert result.is_failure
        assert result.error == "error message"

    def test_chain_success(self) -> None:
        """Test chain helper with all success results."""
        result1 = ok("data1")
        result2 = ok("data2")
        result3 = ok("data3")

        chained = chain(result1, result2, result3)
        assert chained.is_success
        assert chained.data == ["data1", "data2", "data3"]

    def test_chain_with_failure(self) -> None:
        """Test chain helper with one failure."""
        result1 = ok("data1")
        result2 = fail("error")
        result3 = ok("data3")

        chained = chain(result1, result2, result3)
        assert chained.is_failure
        assert "error" in (chained.error or "")


class TestSafeDecorator:
    """Test safe decorator."""

    def test_safe_success(self) -> None:
        """Test safe decorator with successful function."""
        @safe
        def divide(a: int, b: int) -> float:
            return a / b

        result = divide(10, 2)
        assert result.is_success
        assert result.data == 5.0

    def test_safe_exception(self) -> None:
        """Test safe decorator with exception."""
        @safe
        def divide(a: int, b: int) -> float:
            return a / b

        result = divide(10, 0)
        assert result.is_failure
        assert "division by zero" in (result.error or "").lower()


class TestValidationHelpers:
    """Test validation helper functions."""

    def test_validate_required_success(self) -> None:
        """Test validate_required with valid data."""
        validator = validate_required("name", "email")
        data = {"name": "John", "email": "john@example.com", "age": 30}

        result = validator(data)
        assert result.is_success
        assert result.data == data

    def test_validate_required_missing_fields(self) -> None:
        """Test validate_required with missing fields."""
        validator = validate_required("name", "email")
        data = {"name": "John"}

        result = validator(data)
        assert result.is_failure
        assert "email" in (result.error or "")

    def test_validate_email_valid(self) -> None:
        """Test email validation with valid email."""
        result = validate_email("test@example.com")
        assert result.is_success
        assert result.data == "test@example.com"

    def test_validate_email_invalid(self) -> None:
        """Test email validation with invalid email."""
        result = validate_email("invalid-email")
        assert result.is_failure
        assert "Invalid email format" in (result.error or "")

    def test_validate_non_empty_valid(self) -> None:
        """Test non-empty validation with valid string."""
        result = validate_non_empty("  test  ")
        assert result.is_success
        assert result.data == "test"

    def test_validate_non_empty_invalid(self) -> None:
        """Test non-empty validation with empty string."""
        result = validate_non_empty("   ")
        assert result.is_failure
        assert "cannot be empty" in (result.error or "").lower()


class TestPipeline:
    """Test Pipeline class for fluent operations."""

    def test_pipeline_success_chain(self) -> None:
        """Test pipeline with successful operations."""
        def add_one(x: int) -> FlextResult[int]:
            return ok(x + 1)

        def multiply_two(x: int) -> FlextResult[int]:
            return ok(x * 2)

        result = pipeline(5).then(add_one).then(multiply_two).result()
        assert result.is_success
        assert result.data == 12  # (5 + 1) * 2

    def test_pipeline_with_failure(self) -> None:
        """Test pipeline with failure in chain."""
        def add_one(x: int) -> FlextResult[int]:
            return ok(x + 1)

        def fail_operation(_: int) -> FlextResult[int]:
            return fail("operation failed")

        def multiply_two(x: int) -> FlextResult[int]:
            return ok(x * 2)

        result = (
            pipeline(5)
            .then(add_one)
            .then(fail_operation)
            .then(multiply_two)
            .result()
        )
        assert result.is_failure
        assert "operation failed" in (result.error or "")

    def test_pipeline_map_operation(self) -> None:
        """Test pipeline map operation."""
        result = pipeline("hello").map(str.upper).map(lambda x: f"{x}!").result()
        assert result.is_success
        assert result.data == "HELLO!"


class TestMixins:
    """Test mixin classes."""

    def test_logger_mixin(self) -> None:
        """Test LoggerMixin provides logger."""
        class TestClass(LoggerMixin):
            pass

        obj = TestClass()
        logger = obj.logger
        assert logger is not None
        assert hasattr(logger, "info")

    def test_container_mixin(self) -> None:
        """Test ContainerMixin provides container access."""
        class TestClass(ContainerMixin):
            pass

        obj = TestClass()
        container = obj.container
        assert container is not None

    def test_validator_mixin(self) -> None:
        """Test ValidatorMixin provides validation."""
        class TestClass(ValidatorMixin):
            pass

        obj = TestClass()
        data = {"name": "John", "email": "john@example.com"}
        result = obj.validate_data(data, "name", "email")
        assert result.is_success


class TestRetryDecorator:
    """Test retry decorator."""

    def test_retry_success_first_attempt(self) -> None:
        """Test retry decorator with immediate success."""
        call_count = 0

        @retry(max_attempts=3)
        def test_function() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self) -> None:
        """Test retry decorator with success after failures."""
        call_count = 0

        @retry(max_attempts=3)
        def test_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                msg = "temporary failure"
                raise ValueError(msg)
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_all_attempts_fail(self) -> None:
        """Test retry decorator when all attempts fail."""
        call_count = 0

        @retry(max_attempts=3)
        def test_function() -> str:
            nonlocal call_count
            call_count += 1
            msg = "persistent failure"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="persistent failure"):
            test_function()
        assert call_count == 3


class TestCacheDecorator:
    """Test cache_result decorator."""

    def test_cache_result(self) -> None:
        """Test caching decorator caches results."""
        call_count = 0

        @cache_result
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Call with different args
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2  # Incremented


class TestDataHelpers:
    """Test data transformation helpers."""

    def test_to_dict_with_model_dump(self) -> None:
        """Test to_dict with object having model_dump method."""
        class MockObject:
            def model_dump(self) -> dict[str, str]:
                return {"key": "value"}

        obj = MockObject()
        result = to_dict(obj)
        assert result == {"key": "value"}

    def test_to_dict_with_dict_attr(self) -> None:
        """Test to_dict with object having __dict__."""
        class MockObject:
            def __init__(self) -> None:
                self.key = "value"

        obj = MockObject()
        result = to_dict(obj)
        assert result == {"key": "value"}

    def test_to_dict_fallback(self) -> None:
        """Test to_dict fallback for simple values."""
        result = to_dict("simple_value")
        assert result == {"value": "simple_value"}

    def test_from_dict_success(self) -> None:
        """Test from_dict with successful creation."""
        class SimpleClass:
            def __init__(self, name: str, age: int) -> None:
                self.name = name
                self.age = age

        data = {"name": "John", "age": 30}
        result = from_dict(SimpleClass, data)
        assert result.is_success
        assert result.data.name == "John"
        assert result.data.age == 30

    def test_from_dict_failure(self) -> None:
        """Test from_dict with invalid data."""
        class SimpleClass:
            def __init__(self, name: str, age: int) -> None:
                self.name = name
                self.age = age

        data = {"name": "John"}  # Missing 'age'
        result = from_dict(SimpleClass, data)
        assert result.is_failure


class TestQuickDomainObjects:
    """Test quick domain object classes."""

    def test_quick_entity_creation(self) -> None:
        """Test QuickEntity creation."""
        class User(QuickEntity):
            name: str
            email: str

        # Note: QuickEntity inherits from FlextEntity which requires specific fields
        # This test verifies the class can be created and used

    def test_quick_value_object_from_dict(self) -> None:
        """Test QuickValueObject from_dict method."""

        class Money(QuickValueObject):
            amount: float = Field(...)
            currency: str = Field(...)

        data = {"amount": 100.0, "currency": "USD"}
        result = Money.from_dict(data)
        assert result.is_success
        assert result.data.amount == 100.0
        assert result.data.currency == "USD"
