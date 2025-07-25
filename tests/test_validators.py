"""Tests for FlextValidators and validation utilities.

Comprehensive tests for all validation patterns, chains, and utility functions.
"""

from __future__ import annotations

import pytest

from flext_core import FlextResult
from flext_core.validators import ChoiceValidator
from flext_core.validators import EmailValidator
from flext_core.validators import FlextValidationChain
from flext_core.validators import LengthValidator
from flext_core.validators import NotEmptyValidator
from flext_core.validators import NotNoneValidator
from flext_core.validators import RangeValidator
from flext_core.validators import RegexValidator
from flext_core.validators import TypeValidator
from flext_core.validators import validate
from flext_core.validators import validate_all
from flext_core.validators import validate_any
from flext_core.validators import validate_choice
from flext_core.validators import validate_email
from flext_core.validators import validate_number
from flext_core.validators import validate_string
from flext_core.validators import validate_type


class TestNotNoneValidator:
    """Test NotNoneValidator functionality."""

    def test_validate_success(self) -> None:
        """Test validation with non-None value."""
        validator = NotNoneValidator()
        result = validator.validate("test")

        assert result.is_success
        assert result.data == "test"

    def test_validate_failure(self) -> None:
        """Test validation with None value."""
        validator = NotNoneValidator()
        result = validator.validate(None)

        assert result.is_failure
        assert "cannot be None" in (result.error or "")

    def test_callable_interface(self) -> None:
        """Test validator can be called as function."""
        validator = NotNoneValidator()
        result = validator("test")

        assert result.is_success
        assert result.data == "test"


class TestNotEmptyValidator:
    """Test NotEmptyValidator functionality."""

    def test_validate_success(self) -> None:
        """Test validation with non-empty string."""
        validator = NotEmptyValidator()
        result = validator.validate("test")

        assert result.is_success
        assert result.data == "test"

    def test_validate_empty_string(self) -> None:
        """Test validation with empty string."""
        validator = NotEmptyValidator()
        result = validator.validate("")

        assert result.is_failure
        assert "cannot be empty" in (result.error or "")

    def test_validate_whitespace_only(self) -> None:
        """Test validation with whitespace-only string."""
        validator = NotEmptyValidator()
        result = validator.validate("   ")

        assert result.is_failure
        assert "cannot be empty" in (result.error or "")

    def test_validate_whitespace_only_no_strip(self) -> None:
        """Test validation with whitespace when strip disabled."""
        validator = NotEmptyValidator(strip_whitespace=False)
        result = validator.validate("   ")

        assert result.is_success
        assert result.data == "   "

    def test_validate_none(self) -> None:
        """Test validation with None value."""
        validator = NotEmptyValidator()
        result = validator.validate(None)

        assert result.is_failure
        assert "cannot be None" in (result.error or "")


class TestLengthValidator:
    """Test LengthValidator functionality."""

    def test_validate_within_range(self) -> None:
        """Test validation with string within length range."""
        validator = LengthValidator(min_length=2, max_length=10)
        result = validator.validate("test")

        assert result.is_success
        assert result.data == "test"

    def test_validate_too_short(self) -> None:
        """Test validation with string too short."""
        validator = LengthValidator(min_length=5)
        result = validator.validate("hi")

        assert result.is_failure
        assert "too short" in (result.error or "")

    def test_validate_too_long(self) -> None:
        """Test validation with string too long."""
        validator = LengthValidator(max_length=3)
        result = validator.validate("toolong")

        assert result.is_failure
        assert "too long" in (result.error or "")

    def test_validate_exact_length(self) -> None:
        """Test validation with exact length boundaries."""
        validator = LengthValidator(min_length=4, max_length=4)

        # Exact length should pass
        result = validator.validate("test")
        assert result.is_success

        # One character off should fail
        result = validator.validate("tes")
        assert result.is_failure

        result = validator.validate("tests")
        assert result.is_failure

    def test_invalid_constraints(self) -> None:
        """Test validator with invalid constraints."""
        with pytest.raises(ValueError, match="cannot be negative"):
            LengthValidator(min_length=-1)

        with pytest.raises(ValueError, match="cannot be less than"):
            LengthValidator(min_length=10, max_length=5)

    def test_validate_none(self) -> None:
        """Test validation with None value."""
        validator = LengthValidator()
        result = validator.validate(None)

        assert result.is_failure
        assert "cannot be None" in (result.error or "")


class TestEmailValidator:
    """Test EmailValidator functionality."""

    def test_validate_valid_emails(self) -> None:
        """Test validation with valid email addresses."""
        validator = EmailValidator()

        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "test+tag@example.org",
            "123@456.789",
            "a@b.co",
        ]

        for email in valid_emails:
            result = validator.validate(email)
            assert result.is_success, f"Email {email} should be valid"
            assert result.data == email

    def test_validate_invalid_emails(self) -> None:
        """Test validation with invalid email addresses."""
        validator = EmailValidator()

        invalid_emails = [
            "invalid",
            "@domain.com",
            "user@",
            "user@domain",
            "user..name@domain.com",
            "user@domain..com",
            "",
            "   ",
        ]

        for email in invalid_emails:
            result = validator.validate(email)
            assert result.is_failure, f"Email {email} should be invalid"
            assert "Invalid email format" in (result.error or "")

    def test_validate_none(self) -> None:
        """Test validation with None value."""
        validator = EmailValidator()
        result = validator.validate(None)

        assert result.is_failure
        assert "cannot be None" in (result.error or "")

    def test_validate_non_string(self) -> None:
        """Test validation with non-string value."""
        validator = EmailValidator()
        result = validator.validate(123)

        assert result.is_failure
        assert "must be a string" in (result.error or "")


class TestRangeValidator:
    """Test RangeValidator functionality."""

    def test_validate_within_range(self) -> None:
        """Test validation with value within range."""
        validator = RangeValidator(min_value=1, max_value=10)
        result = validator.validate(5)

        assert result.is_success
        assert result.data == 5

    def test_validate_too_small(self) -> None:
        """Test validation with value too small."""
        validator = RangeValidator(min_value=5)
        result = validator.validate(3)

        assert result.is_failure
        assert "too small" in (result.error or "")

    def test_validate_too_large(self) -> None:
        """Test validation with value too large."""
        validator = RangeValidator(max_value=10)
        result = validator.validate(15)

        assert result.is_failure
        assert "too large" in (result.error or "")

    def test_validate_boundary_values(self) -> None:
        """Test validation with boundary values."""
        validator = RangeValidator(min_value=5, max_value=10)

        # Boundary values should pass
        assert validator.validate(5).is_success
        assert validator.validate(10).is_success

        # Just outside boundaries should fail
        assert validator.validate(4).is_failure
        assert validator.validate(11).is_failure

    def test_validate_none(self) -> None:
        """Test validation with None value."""
        validator = RangeValidator()
        result = validator.validate(None)

        assert result.is_failure
        assert "cannot be None" in (result.error or "")

    def test_invalid_range(self) -> None:
        """Test validator with invalid range."""
        with pytest.raises(ValueError, match="cannot be greater than"):
            RangeValidator(min_value=10, max_value=5)


class TestRegexValidator:
    """Test RegexValidator functionality."""

    def test_validate_matching_pattern(self) -> None:
        """Test validation with matching pattern."""
        validator = RegexValidator(r"^[a-z]+$")
        result = validator.validate("hello")

        assert result.is_success
        assert result.data == "hello"

    def test_validate_non_matching_pattern(self) -> None:
        """Test validation with non-matching pattern."""
        validator = RegexValidator(r"^[a-z]+$")
        result = validator.validate("Hello123")

        assert result.is_failure
        assert "does not match pattern" in (result.error or "")

    def test_custom_error_message(self) -> None:
        """Test validator with custom error message."""
        validator = RegexValidator(r"^[a-z]+$", "Only lowercase letters allowed")
        result = validator.validate("Hello")

        assert result.is_failure
        assert result.error == "Only lowercase letters allowed"

    def test_complex_pattern(self) -> None:
        """Test validation with complex regex pattern."""
        # Phone number pattern (flexible for various formats)
        validator = RegexValidator(
            r"^(\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{4}$",
        )

        valid_phones = [
            "+1 (555) 123-4567",
            "555-123-4567",
            "5551234567",
            "+44 20 7946 0958",
        ]

        for phone in valid_phones:
            result = validator.validate(phone)
            assert result.is_success, f"Phone {phone} should be valid"

    def test_invalid_regex_pattern(self) -> None:
        """Test validator with invalid regex pattern."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            RegexValidator("[unclosed")

    def test_validate_none(self) -> None:
        """Test validation with None value."""
        validator = RegexValidator(r".*")
        result = validator.validate(None)

        assert result.is_failure
        assert "cannot be None" in (result.error or "")

    def test_validate_non_string(self) -> None:
        """Test validation with non-string value."""
        validator = RegexValidator(r".*")
        result = validator.validate(123)

        assert result.is_failure
        assert "must be a string" in (result.error or "")


class TestChoiceValidator:
    """Test ChoiceValidator functionality."""

    def test_validate_valid_choice(self) -> None:
        """Test validation with valid choice."""
        validator = ChoiceValidator(["red", "green", "blue"])
        result = validator.validate("red")

        assert result.is_success
        assert result.data == "red"

    def test_validate_invalid_choice(self) -> None:
        """Test validation with invalid choice."""
        validator = ChoiceValidator(["red", "green", "blue"])
        result = validator.validate("yellow")

        assert result.is_failure
        assert "must be one of" in (result.error or "")

    def test_validate_different_types(self) -> None:
        """Test validation with different value types."""
        validator = ChoiceValidator([1, 2, 3, "four"])

        assert validator.validate(2).is_success
        assert validator.validate("four").is_success
        assert validator.validate(5).is_failure
        assert validator.validate("five").is_failure

    def test_empty_choices(self) -> None:
        """Test validator with empty choices."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ChoiceValidator([])


class TestTypeValidator:
    """Test TypeValidator functionality."""

    def test_validate_correct_type(self) -> None:
        """Test validation with correct type."""
        validator = TypeValidator(str)
        result = validator.validate("hello")

        assert result.is_success
        assert result.data == "hello"

    def test_validate_incorrect_type(self) -> None:
        """Test validation with incorrect type."""
        validator = TypeValidator(str)
        result = validator.validate(123)

        assert result.is_failure
        assert "Expected str, got int" in (result.error or "")

    def test_validate_with_inheritance(self) -> None:
        """Test validation with inheritance."""

        class Base:
            pass

        class Derived(Base):
            pass

        validator = TypeValidator(Base)
        derived_instance = Derived()

        result = validator.validate(derived_instance)
        assert result.is_success
        assert result.data is derived_instance


class TestFlextValidationChain:
    """Test FlextValidationChain functionality."""

    def test_empty_chain(self) -> None:
        """Test validation chain with no validators."""
        chain = FlextValidationChain("test")
        result = chain.result()

        assert result.is_success
        assert result.data == "test"

    def test_single_validator_success(self) -> None:
        """Test chain with single successful validator."""
        chain = FlextValidationChain("test")
        result = chain.validate_with(NotEmptyValidator()).result()

        assert result.is_success
        assert result.data == "test"

    def test_single_validator_failure(self) -> None:
        """Test chain with single failing validator."""
        chain = FlextValidationChain("")
        result = chain.validate_with(NotEmptyValidator()).result()

        assert result.is_failure
        assert "cannot be empty" in (result.error or "")

    def test_multiple_validators_success(self) -> None:
        """Test chain with multiple successful validators."""
        email = "test@example.com"
        chain = FlextValidationChain(email)

        result = (
            chain.validate_with(NotEmptyValidator())
            .validate_with(EmailValidator())
            .validate_with(LengthValidator(min_length=5))
            .result()
        )

        assert result.is_success
        assert result.data == email

    def test_multiple_validators_failure(self) -> None:
        """Test chain with failing validator in sequence."""
        chain = FlextValidationChain("invalid-email")

        result = (
            chain.validate_with(NotEmptyValidator())  # Pass
            .validate_with(EmailValidator())  # Fail
            .validate_with(LengthValidator(min_length=5))  # Not executed
            .result()
        )

        assert result.is_failure
        assert "Invalid email format" in (result.error or "")

    def test_conditional_validation(self) -> None:
        """Test conditional validation with validate_if."""
        chain = FlextValidationChain("test@example.com")

        # Only validate email format if string is not empty
        result = (
            chain.validate_with(NotEmptyValidator())
            .validate_if(True, EmailValidator())  # Should validate  # noqa: FBT003
            .validate_if(
                False,  # noqa: FBT003
                LengthValidator(min_length=100),
            )  # Should skip
            .result()
        )

        assert result.is_success

    def test_custom_validation(self) -> None:
        """Test custom validation with lambda function."""
        chain = FlextValidationChain(42)

        result = (
            chain.custom(lambda x: x > 0, "Must be positive")
            .custom(lambda x: x % 2 == 0, "Must be even")
            .result()
        )

        assert result.is_success

    def test_custom_validation_failure(self) -> None:
        """Test custom validation failure."""
        chain = FlextValidationChain(-5)

        result = (
            chain.custom(lambda x: x > 0, "Must be positive")
            .custom(lambda x: x % 2 == 0, "Must be even")
            .result()
        )

        assert result.is_failure
        assert "Must be positive" in (result.error or "")

    def test_unwrap_success(self) -> None:
        """Test unwrapping successful validation."""
        chain = FlextValidationChain("test")
        value = chain.validate_with(NotEmptyValidator()).unwrap()

        assert value == "test"

    def test_unwrap_failure(self) -> None:
        """Test unwrapping failed validation."""
        chain = FlextValidationChain("")

        with pytest.raises(ValueError, match="cannot be empty"):
            chain.validate_with(NotEmptyValidator()).unwrap()


class TestUtilityFunctions:
    """Test validation utility functions."""

    def test_validate_function(self) -> None:
        """Test validate function creates chain."""
        chain = validate("test")
        assert isinstance(chain, FlextValidationChain)

    def test_validate_string_success(self) -> None:
        """Test validate_string with valid string."""
        result = validate_string(
            "hello@example.com",
            not_empty=True,
            min_length=5,
            max_length=50,
            pattern=r".*@.*",
        )

        assert result.is_success

    def test_validate_string_failure(self) -> None:
        """Test validate_string with invalid string."""
        result = validate_string(
            "",
            not_empty=True,
        )

        assert result.is_failure

    def test_validate_email_success(self) -> None:
        """Test validate_email with valid email."""
        result = validate_email("test@example.com")

        assert result.is_success
        assert result.data == "test@example.com"

    def test_validate_email_failure(self) -> None:
        """Test validate_email with invalid email."""
        result = validate_email("invalid")

        assert result.is_failure

    def test_validate_number_success(self) -> None:
        """Test validate_number with valid number."""
        result = validate_number(42, min_value=0, max_value=100)

        assert result.is_success
        assert result.data == 42

    def test_validate_number_failure(self) -> None:
        """Test validate_number with invalid number."""
        result = validate_number(-5, min_value=0)

        assert result.is_failure

    def test_validate_choice_success(self) -> None:
        """Test validate_choice with valid choice."""
        result = validate_choice("red", ["red", "green", "blue"])

        assert result.is_success
        assert result.data == "red"

    def test_validate_choice_failure(self) -> None:
        """Test validate_choice with invalid choice."""
        result = validate_choice("yellow", ["red", "green", "blue"])

        assert result.is_failure

    def test_validate_type_success(self) -> None:
        """Test validate_type with correct type."""
        result = validate_type("hello", str)

        assert result.is_success
        assert result.data == "hello"

    def test_validate_type_failure(self) -> None:
        """Test validate_type with incorrect type."""
        result = validate_type(123, str)

        assert result.is_failure

    def test_validate_all_success(self) -> None:
        """Test validate_all with all successful validations."""
        validations = [
            FlextResult.ok("value1"),
            FlextResult.ok("value2"),
            FlextResult.ok("value3"),
        ]

        result = validate_all(*validations)

        assert result.is_success
        assert result.data == ["value1", "value2", "value3"]

    def test_validate_all_failure(self) -> None:
        """Test validate_all with one failed validation."""
        validations = [
            FlextResult.ok("value1"),
            FlextResult.fail("error"),
            FlextResult.ok("value3"),
        ]

        result = validate_all(*validations)

        assert result.is_failure

    def test_validate_any_success(self) -> None:
        """Test validate_any with at least one success."""
        validations = [
            FlextResult.fail("error1"),
            FlextResult.ok("success"),
            FlextResult.fail("error2"),
        ]

        result = validate_any(*validations)

        assert result.is_success
        assert result.data == "success"

    def test_validate_any_failure(self) -> None:
        """Test validate_any with all failures."""
        validations = [
            FlextResult.fail("error1"),
            FlextResult.fail("error2"),
            FlextResult.fail("error3"),
        ]

        result = validate_any(*validations)

        assert result.is_failure


@pytest.mark.integration
class TestValidationIntegration:
    """Integration tests for complex validation scenarios."""

    def test_user_registration_validation(self) -> None:
        """Test complete user registration validation."""

        def validate_user_data(data: dict[str, str]) -> dict[str, bool]:
            """Validate user registration data."""
            results = {}

            # Validate email
            email_result = validate_email(data.get("email", ""))
            results["email_valid"] = email_result.is_success

            # Validate password
            password_result = validate_string(
                data.get("password", ""),
                min_length=8,
                pattern=r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).*$",
            )
            results["password_valid"] = password_result.is_success

            # Validate username
            username_result = validate_string(
                data.get("username", ""),
                min_length=3,
                max_length=20,
                pattern=r"^[a-zA-Z0-9_]+$",
            )
            results["username_valid"] = username_result.is_success

            # Validate age
            try:
                age = int(data.get("age", "0"))
                age_result = validate_number(age, min_value=13, max_value=120)
                results["age_valid"] = age_result.is_success
            except ValueError:
                results["age_valid"] = False

            return results

        # Valid user data
        valid_data = {
            "email": "user@example.com",
            "password": "SecurePass123",
            "username": "user123",
            "age": "25",
        }

        results = validate_user_data(valid_data)
        assert all(results.values())

        # Invalid user data
        invalid_data = {
            "email": "invalid-email",
            "password": "weak",
            "username": "ab",
            "age": "12",
        }

        results = validate_user_data(invalid_data)
        assert not any(results.values())

    def test_complex_validation_chain(self) -> None:
        """Test complex validation chain with multiple conditions."""

        def validate_product_code(code: str) -> FlextResult[str]:
            """Validate product code with complex rules."""
            return (
                validate(code)
                .validate_with(NotEmptyValidator())
                .validate_with(LengthValidator(min_length=6, max_length=12))
                .validate_with(RegexValidator(r"^[A-Z]{2}-\d{4}-[A-Z]{2}$"))
                .custom(
                    lambda x: x.split("-")[1] != "0000",
                    "Product number cannot be 0000",
                )
                .custom(
                    lambda x: x[:2] in ["PR", "SR", "MR"],
                    "Invalid product category prefix",
                )
                .result()
            )

        # Valid product codes
        valid_codes = ["PR-1234-AB", "SR-5678-CD", "MR-9999-XY"]
        for code in valid_codes:
            result = validate_product_code(code)
            assert result.is_success, f"Code {code} should be valid"

        # Invalid product codes
        invalid_codes = [
            "",  # Empty
            "PR-1234",  # Too short
            "pr-1234-ab",  # Wrong case
            "XX-1234-AB",  # Invalid prefix
            "PR-0000-AB",  # Invalid number
            "PR-12345-AB",  # Number too long
        ]

        for code in invalid_codes:
            result = validate_product_code(code)
            assert result.is_failure, f"Code {code} should be invalid"
