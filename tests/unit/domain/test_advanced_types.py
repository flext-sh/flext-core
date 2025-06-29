"""Unit tests for advanced domain types.

Tests for ServiceResult, ServiceError, and other advanced patterns.
"""

from __future__ import annotations

import pytest
from flx_core.domain.advanced_types import ServiceError, ServiceResult, ValidationResult

# Python 3.13 type aliases
type TestValue = int | str | dict[str, str]
type TestError = ServiceError


class TestServiceResult:
    """Test ServiceResult monadic pattern."""

    def test_success_creation(self) -> None:
        """Test creating successful result."""
        value = 42
        result = ServiceResult.ok(value)

        assert result.is_success is True
        assert result.is_failure is False
        assert result.value == value
        assert result.error is None

    def test_failure_creation(self) -> None:
        """Test creating failed result."""
        error = ServiceError("TEST_ERROR", "Test error message")
        result: ServiceResult[int] = ServiceResult.fail(error)

        assert result.is_failure is True
        assert result.is_success is False
        assert result.error == error
        assert result.value is None

    def test_map_on_success(self) -> None:
        """Test map operation on successful result."""
        result = ServiceResult.ok(10)
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_success is True
        assert mapped.value == 20

    def test_map_on_failure(self) -> None:
        """Test map operation on failed result."""
        error = ServiceError("ERROR", "Failed")
        result: ServiceResult[int] = ServiceResult.fail(error)
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_failure is True
        assert mapped.error == error

    def test_flat_map_on_success(self) -> None:
        """Test flat_map operation on successful result."""
        result = ServiceResult.ok(10)

        def double_if_positive(x: int) -> ServiceResult[int]:
            if x > 0:
                return ServiceResult.ok(x * 2)
            return ServiceResult.fail(
                ServiceError("NEGATIVE", "Value must be positive")
            )

        mapped = result.flat_map(double_if_positive)

        assert mapped.is_success is True
        assert mapped.value == 20

    def test_flat_map_chain(self) -> None:
        """Test chaining flat_map operations."""
        result = ServiceResult.ok(5)

        def add_one(x: int) -> ServiceResult[int]:
            return ServiceResult.ok(x + 1)

        def multiply_two(x: int) -> ServiceResult[int]:
            return ServiceResult.ok(x * 2)

        final = result.flat_map(add_one).flat_map(multiply_two)

        assert final.is_success is True
        assert final.value == 12  # (5 + 1) * 2

    def test_unwrap_on_success(self) -> None:
        """Test unwrap on successful result."""
        result = ServiceResult.ok("test")
        value = result.unwrap()

        assert value == "test"

    def test_unwrap_on_failure(self) -> None:
        """Test unwrap on failed result raises exception."""
        result: ServiceResult[str] = ServiceResult.fail(ServiceError("ERROR", "Failed"))

        with pytest.raises(ValueError, match="Called unwrap on a failure"):
            result.unwrap()

    def test_unwrap_or_on_success(self) -> None:
        """Test unwrap_or on successful result."""
        result = ServiceResult.ok("success")
        value = result.unwrap_or("default")

        assert value == "success"

    def test_unwrap_or_on_failure(self) -> None:
        """Test unwrap_or on failed result."""
        result: ServiceResult[str] = ServiceResult.fail(ServiceError("ERROR", "Failed"))
        value = result.unwrap_or("default")

        assert value == "default"

    def test_match_pattern(self) -> None:
        """Test match pattern for result handling."""

        def handle_success(result: ServiceResult[int]) -> str:
            if result.is_success:
                return f"Success: {result.value}"
            return f"Error: {result.error.message}"

        success = ServiceResult.ok(42)
        failure: ServiceResult[int] = ServiceResult.fail(ServiceError("ERR", "Failed"))

        assert handle_success(success) == "Success: 42"
        assert handle_success(failure) == "Error: Failed"

    def test_from_optional(self) -> None:
        """Test creating ServiceResult from optional value."""
        # With value
        result1 = ServiceResult.from_optional(42, ServiceError("MISSING", "No value"))
        assert result1.is_success is True
        assert result1.value == 42

        # Without value
        result2 = ServiceResult.from_optional(None, ServiceError("MISSING", "No value"))
        assert result2.is_failure is True
        assert result2.error.code == "MISSING"


class TestServiceError:
    """Test ServiceError value object."""

    def test_error_creation(self) -> None:
        """Test creating service error."""
        error = ServiceError("TEST_CODE", "Test message")

        assert error.code == "TEST_CODE"
        assert error.message == "Test message"
        assert error.details is None
        assert error.timestamp is not None

    def test_error_with_details(self) -> None:
        """Test creating error with details."""
        details = {"field": "email", "reason": "invalid format"}
        error = ServiceError("VALIDATION_ERROR", "Invalid input", details=details)

        assert error.details == details

    def test_error_equality(self) -> None:
        """Test error equality based on code and message."""
        error1 = ServiceError("CODE", "Message")
        error2 = ServiceError("CODE", "Message")
        error3 = ServiceError("CODE", "Different message")
        error4 = ServiceError("DIFFERENT", "Message")

        assert error1 == error2
        assert error1 != error3
        assert error1 != error4

    def test_error_string_representation(self) -> None:
        """Test error string representation."""
        error = ServiceError("TEST_ERROR", "Something went wrong")

        assert str(error) == "[TEST_ERROR] Something went wrong"

    def test_error_with_cause(self) -> None:
        """Test error with cause chain."""
        cause = ServiceError("ROOT_CAUSE", "Original error")
        error = ServiceError("WRAPPER", "Wrapped error", cause=cause)

        assert error.cause == cause
        assert error.cause.code == "ROOT_CAUSE"

    def test_common_error_factories(self) -> None:
        """Test common error factory methods."""
        # Validation error
        validation_error = ServiceError.validation_error("Invalid email format")
        assert validation_error.code == "VALIDATION_ERROR"
        assert "Invalid email format" in validation_error.message

        # Not found error
        not_found = ServiceError.not_found("User", "123")
        assert not_found.code == "NOT_FOUND"
        assert "User" in not_found.message
        assert "123" in not_found.message

        # Permission error
        permission = ServiceError.permission_denied("REDACTED_LDAP_BIND_PASSWORD access")
        assert permission.code == "PERMISSION_DENIED"
        assert "REDACTED_LDAP_BIND_PASSWORD access" in permission.message


class TestValidationResult:
    """Test ValidationResult for domain validation."""

    def test_valid_result(self) -> None:
        """Test creating valid result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result(self) -> None:
        """Test creating invalid result with errors."""
        errors = ["Email is required", "Name too short"]
        warnings = ["Phone number recommended"]

        result = ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_add_error(self) -> None:
        """Test adding error to validation result."""
        result = ValidationResult(is_valid=True)

        result.add_error("New error")

        assert result.is_valid is False
        assert "New error" in result.errors

    def test_add_warning(self) -> None:
        """Test adding warning to validation result."""
        result = ValidationResult(is_valid=True)

        result.add_warning("Consider this")

        assert result.is_valid is True  # Warnings don't affect validity
        assert "Consider this" in result.warnings

    def test_merge_results(self) -> None:
        """Test merging multiple validation results."""
        result1 = ValidationResult(
            is_valid=False, errors=["Error 1"], warnings=["Warning 1"]
        )

        result2 = ValidationResult(is_valid=True, errors=[], warnings=["Warning 2"])

        result3 = ValidationResult(is_valid=False, errors=["Error 2"], warnings=[])

        merged = ValidationResult.merge([result1, result2, result3])

        assert merged.is_valid is False  # Any invalid makes merged invalid
        assert len(merged.errors) == 2
        assert len(merged.warnings) == 2

    def test_from_errors(self) -> None:
        """Test creating validation result from errors list."""
        errors = ["Error 1", "Error 2", "Error 3"]

        result = ValidationResult.from_errors(errors)

        assert result.is_valid is False
        assert len(result.errors) == 3
        assert result.warnings == []

    def test_to_service_result(self) -> None:
        """Test converting validation result to service result."""
        # Valid case
        valid = ValidationResult(is_valid=True)
        service_result = valid.to_service_result("Success")

        assert service_result.is_success is True
        assert service_result.value == "Success"

        # Invalid case
        invalid = ValidationResult(
            is_valid=False, errors=["Field required", "Invalid format"]
        )
        service_result = invalid.to_service_result("Won't be used")

        assert service_result.is_failure is True
        assert service_result.error.code == "VALIDATION_ERROR"
        assert "Field required" in service_result.error.message
        assert "Invalid format" in service_result.error.message
