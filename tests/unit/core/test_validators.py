"""Tests for FlextValidators and validation utilities.

Comprehensive tests for all validation patterns, chains, and utility functions.
"""

from __future__ import annotations

import pytest

from flext_core import FlextResult, FlextValidators, flext_validate


class TestFlextValidators:
    """Test FlextValidators static methods."""

    def test_validate_email_success(self) -> None:
        """Test email validation with valid email."""
        result = FlextValidators.flext_validate_email("test@example.com")

        assert result.is_success
        assert result.data == "test@example.com"

    def test_validate_email_failure(self) -> None:
        """Test email validation with invalid email."""
        result = FlextValidators.flext_validate_email("invalid-email")

        assert result.is_failure
        assert "Invalid email format" in (result.error or "")

    def test_validate_not_empty_success(self) -> None:
        """Test not empty validation with valid string."""
        result = FlextValidators.flext_validate_not_empty("test")

        assert result.is_success
        assert result.data == "test"

    def test_validate_not_empty_failure(self) -> None:
        """Test not empty validation with empty string."""
        result = FlextValidators.flext_validate_not_empty("")

        assert result.is_failure
        assert "cannot be empty" in (result.error or "")

    def test_validate_not_empty_whitespace(self) -> None:
        """Test not empty validation with whitespace."""
        result = FlextValidators.flext_validate_not_empty("   ")

        assert result.is_failure
        assert "cannot be empty" in (result.error or "")

    def test_validate_length_success(self) -> None:
        """Test length validation within bounds."""
        result = FlextValidators.flext_validate_length("test", 2, 10)

        assert result.is_success
        assert result.data == "test"

    def test_validate_length_too_short(self) -> None:
        """Test length validation with too short string."""
        result = FlextValidators.flext_validate_length("hi", 5, 10)

        assert result.is_failure
        assert "between 5 and 10" in (result.error or "")

    def test_validate_length_too_long(self) -> None:
        """Test length validation with too long string."""
        result = FlextValidators.flext_validate_length("toolongstring", 2, 5)

        assert result.is_failure
        assert "between 2 and 5" in (result.error or "")

    def test_validate_numeric_success(self) -> None:
        """Test numeric validation with valid number."""
        result = FlextValidators.flext_validate_numeric("123")

        assert result.is_success
        assert result.data == 123

    def test_validate_numeric_failure(self) -> None:
        """Test numeric validation with invalid number."""
        result = FlextValidators.flext_validate_numeric("abc")

        assert result.is_failure
        assert "valid number" in (result.error or "")

    def test_validate_range_success(self) -> None:
        """Test range validation within bounds."""
        result = FlextValidators.flext_validate_range(5, 1, 10)

        assert result.is_success
        assert result.data == 5

    def test_validate_range_too_small(self) -> None:
        """Test range validation with too small value."""
        result = FlextValidators.flext_validate_range(0, 5, 10)

        assert result.is_failure
        assert "between 5 and 10" in (result.error or "")

    def test_validate_range_too_large(self) -> None:
        """Test range validation with too large value."""
        result = FlextValidators.flext_validate_range(15, 5, 10)

        assert result.is_failure
        assert "between 5 and 10" in (result.error or "")

    def test_validate_required_success(self) -> None:
        """Test required validation with valid value."""
        result = FlextValidators.flext_validate_required("test")

        assert result.is_success
        assert result.data == "test"

    def test_validate_required_none(self) -> None:
        """Test required validation with None."""
        result = FlextValidators.flext_validate_required(None)

        assert result.is_failure
        assert "required" in (result.error or "")

    def test_validate_required_empty_string(self) -> None:
        """Test required validation with empty string."""
        result = FlextValidators.flext_validate_required("")

        assert result.is_failure
        assert "required" in (result.error or "")

    def test_validate_choice_success(self) -> None:
        """Test choice validation with valid choice."""
        result = FlextValidators.flext_validate_choice("red", ["red", "green", "blue"])

        assert result.is_success
        assert result.data == "red"

    def test_validate_choice_failure(self) -> None:
        """Test choice validation with invalid choice."""
        result = FlextValidators.flext_validate_choice(
            "yellow",
            ["red", "green", "blue"],
        )

        assert result.is_failure
        assert "must be one of" in (result.error or "")


class TestFlextValidationChain:
    """Test FlextValidationChain functionality."""

    def test_empty_chain(self) -> None:
        """Test validation chain with no validators."""
        result = flext_validate("test").result()

        assert result.is_success
        assert result.data == "test"

    def test_single_validation_success(self) -> None:
        """Test chain with single successful validation."""
        result = flext_validate("test@example.com").email().result()

        assert result.is_success
        assert result.data == "test@example.com"

    def test_single_validation_failure(self) -> None:
        """Test chain with single failing validation."""
        result = flext_validate("invalid-email").email().result()

        assert result.is_failure
        assert "Invalid email format" in (result.error or "")

    def test_required_validation(self) -> None:
        """Test required validation in chain."""
        result = flext_validate("test").required().result()
        assert result.is_success

        result = flext_validate(None).required().result()
        assert result.is_failure

    def test_not_empty_validation(self) -> None:
        """Test not empty validation in chain."""
        result = flext_validate("test").not_empty().result()
        assert result.is_success

        result = flext_validate("").not_empty().result()
        assert result.is_failure

    def test_length_validation(self) -> None:
        """Test length validation in chain."""
        result = flext_validate("test").length(2, 10).result()
        assert result.is_success

        result = flext_validate("x").length(5, 10).result()
        assert result.is_failure

    def test_range_validation(self) -> None:
        """Test range validation in chain."""
        result = flext_validate(5).range(1, 10).result()
        assert result.is_success

        result = flext_validate(15).range(1, 10).result()
        assert result.is_failure

    def test_multiple_validations_success(self) -> None:
        """Test chain with multiple successful validations."""
        result = (
            flext_validate("test@example.com").required().not_empty().email().result()
        )

        assert result.is_success
        assert result.data == "test@example.com"

    def test_multiple_validations_failure(self) -> None:
        """Test chain with failing validation in sequence."""
        result = (
            flext_validate("invalid-email")
            .required()
            .not_empty()
            .email()  # This should fail
            .result()
        )

        assert result.is_failure
        assert "Invalid email format" in (result.error or "")

    def test_chain_with_custom_validator(self) -> None:
        """Test chain with custom validation function."""

        def custom_validator(value: str) -> FlextResult[str]:
            if "test" in value:
                return FlextResult.ok(value)
            return FlextResult.fail("Must contain 'test'")

        result = flext_validate("teststring").validate_with(custom_validator).result()
        assert result.is_success

        result = flext_validate("other").validate_with(custom_validator).result()
        assert result.is_failure


@pytest.mark.integration
class TestValidationIntegration:
    """Integration tests for complex validation scenarios."""

    def test_user_email_validation(self) -> None:
        """Test complete user email validation."""
        # Valid email
        result = (
            flext_validate("user@example.com")
            .required()
            .not_empty()
            .email()
            .length(5, 50)
            .result()
        )
        assert result.is_success

        # Invalid email
        result = (
            flext_validate("invalid")
            .required()
            .not_empty()
            .email()
            .result()  # Should fail at email validation
        )
        assert result.is_failure

    def test_password_validation(self) -> None:
        """Test password validation chain."""

        def has_uppercase(value: str) -> FlextResult[str]:
            if any(c.isupper() for c in value):
                return FlextResult.ok(value)
            return FlextResult.fail("Must contain uppercase letter")

        def has_number(value: str) -> FlextResult[str]:
            if any(c.isdigit() for c in value):
                return FlextResult.ok(value)
            return FlextResult.fail("Must contain number")

        # Valid password
        result = (
            flext_validate("SecurePass123")
            .required()
            .not_empty()
            .length(8, 50)
            .validate_with(has_uppercase)
            .validate_with(has_number)
            .result()
        )
        assert result.is_success

        # Too short password
        result = (
            flext_validate("Short1")
            .required()
            .not_empty()
            .length(8, 50)  # Should fail here
            .result()
        )
        assert result.is_failure

    def test_numeric_validation_chain(self) -> None:
        """Test numeric validation with conversion."""

        # Test with valid numeric string
        def convert_and_validate(value: str) -> FlextResult[int]:
            numeric_result = FlextValidators.flext_validate_numeric(value)
            if numeric_result.is_failure:
                return FlextResult.fail(numeric_result.error or "Not numeric")

            return FlextValidators.flext_validate_range(numeric_result.data, 1, 100)

        result = (
            flext_validate("42")
            .required()
            .not_empty()
            .validate_with(convert_and_validate)
            .result()
        )
        assert result.is_success

        # Test with out of range number
        result = (
            flext_validate("150")
            .required()
            .not_empty()
            .validate_with(convert_and_validate)
            .result()
        )
        assert result.is_failure
