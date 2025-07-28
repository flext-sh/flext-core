"""Comprehensive tests for FlextValidation consolidated functionality.

Tests all consolidated features following "entregar mais com muito menos" approach:
- FlextValidationConfig: Pydantic-based configuration with field validation
- FlextValidationResult: Structured validation results with success/failure handling
- FlextValidators: Consolidated validation functions using extensive validation logic
- FlextPredicates: Functional predicates for filtering and validation patterns
- FlextValidation: Main validation interface with inheritance and composition
- Field validation functions: String, numeric, email, required field validation
- Convenience functions: High-level validation helpers with FlextResult integration
- Backward compatibility: Legacy aliases and function compatibility
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from flext_core.validation import (
    FlextPredicates,
    FlextValidation,
    FlextValidationConfig,
    FlextValidationResult,
    FlextValidators,
    Predicates,
    Validation,
    ValidationResult,
    ValidationResultFactory,
    _validate_email_field,
    _validate_numeric_field,
    _validate_required_field,
    _validate_string_field,
    _ValidationConfig,
    _ValidationResult,
    is_email,
    is_non_empty_string,
    is_not_none,
    is_string,
    validate_email,
    validate_email_field,
    validate_entity_id,
    validate_non_empty_string,
    validate_numeric,
    validate_numeric_field,
    validate_required,
    validate_required_field,
    validate_string,
    validate_string_field,
)

if TYPE_CHECKING:
    from flext_core import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.patterns]


class TestFlextValidationConfig:
    """Test FlextValidationConfig Pydantic-based configuration."""

    def test_valid_config_creation(self) -> None:
        """Test creating valid validation configuration."""
        # Basic config
        config = FlextValidationConfig(field_name="username")
        assert config.field_name == "username"
        assert config.min_length == 0  # default
        assert config.max_length is None  # default
        assert config.min_val is None  # default
        assert config.max_val is None  # default
        assert config.pattern is None  # default

        # Config with all parameters
        config_full = FlextValidationConfig(
            field_name="password",
            min_length=8,
            max_length=128,
            min_val=1.0,
            max_val=100.0,
            pattern=r"^[a-zA-Z0-9]+$",
        )
        assert config_full.field_name == "password"
        assert config_full.min_length == 8
        assert config_full.max_length == 128
        assert config_full.min_val == 1.0
        assert config_full.max_val == 100.0
        assert config_full.pattern == r"^[a-zA-Z0-9]+$"

    def test_config_validation_errors(self) -> None:
        """Test validation errors in configuration."""
        # Empty field name should fail
        with pytest.raises(ValidationError):
            FlextValidationConfig(field_name="")

        # Negative min_length should fail
        with pytest.raises(ValidationError):
            FlextValidationConfig(field_name="test", min_length=-1)

        # Invalid max_length should fail
        with pytest.raises(ValidationError):
            FlextValidationConfig(field_name="test", max_length=0)

    def test_config_max_length_validation(self) -> None:
        """Test max_length validation against min_length."""
        # Test with valid max_length greater than min_length
        valid_config = FlextValidationConfig(
            field_name="test",
            min_length=5,
            max_length=10,
        )
        assert valid_config.max_length == 10

        # Invalid: max_length <= min_length should fail
        with pytest.raises(
            ValidationError,
            match="max_length must be greater than min_length",
        ):
            FlextValidationConfig(
                field_name="test",
                min_length=10,
                max_length=5,
            )

        # Edge case: max_length equal to min_length should fail
        with pytest.raises(ValidationError):
            FlextValidationConfig(
                field_name="test",
                min_length=5,
                max_length=5,
            )

    def test_config_immutability(self) -> None:
        """Test that configuration is immutable (frozen)."""
        config = FlextValidationConfig(field_name="test")

        # Should not be able to modify frozen model
        with pytest.raises(ValidationError):
            config.field_name = "modified"

    def test_config_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            FlextValidationConfig(
                field_name="test",
                extra_field="not_allowed",
            )


class TestFlextValidationResult:
    """Test FlextValidationResult structured validation results."""

    def test_success_result_creation(self) -> None:
        """Test creating successful validation results."""
        # Success using factory method
        success = FlextValidationResult.success()
        assert success.is_valid is True
        assert success.error_message is None
        assert success.field_name is None

        # Success using direct constructor
        success_direct = FlextValidationResult(is_valid=True)
        assert success_direct.is_valid is True
        assert success_direct.error_message is None

    def test_failure_result_creation(self) -> None:
        """Test creating failure validation results."""
        # Failure using factory method
        failure = FlextValidationResult.failure("Validation failed")
        assert failure.is_valid is False
        assert failure.error_message == "Validation failed"
        assert failure.field_name is None

        # Failure with field name
        field_failure = FlextValidationResult.failure(
            "Email format invalid",
            field_name="email",
        )
        assert field_failure.is_valid is False
        assert field_failure.error_message == "Email format invalid"
        assert field_failure.field_name == "email"

        # Failure using direct constructor
        failure_direct = FlextValidationResult(
            is_valid=False,
            error_message="Direct error",
            field_name="direct_field",
        )
        assert failure_direct.is_valid is False
        assert failure_direct.error_message == "Direct error"
        assert failure_direct.field_name == "direct_field"

    def test_result_immutability(self) -> None:
        """Test that validation results are immutable."""
        result = FlextValidationResult.success()

        # Should not be able to modify frozen model
        with pytest.raises(ValidationError):
            result.is_valid = False

    def test_result_string_representation(self) -> None:
        """Test string representation of validation results."""
        success = FlextValidationResult.success()
        assert "is_valid=True" in str(success)

        failure = FlextValidationResult.failure("Test error", "test_field")
        result_str = str(failure)
        assert "is_valid=False" in result_str
        assert "Test error" in result_str
        assert "test_field" in result_str


class TestFlextValidators:
    """Test FlextValidators consolidated validation functions."""

    def test_basic_type_checks(self) -> None:
        """Test basic type checking validators."""
        # is_callable
        assert FlextValidators.is_callable(lambda x: x) is True
        assert FlextValidators.is_callable(str) is True
        assert FlextValidators.is_callable("not_callable") is False
        assert FlextValidators.is_callable(42) is False

        # is_not_none
        assert FlextValidators.is_not_none("test") is True
        assert FlextValidators.is_not_none(0) is True
        false_value = False
        assert FlextValidators.is_not_none(false_value) is True
        assert FlextValidators.is_not_none(None) is False

        # is_string
        assert FlextValidators.is_string("hello") is True
        assert FlextValidators.is_string("") is True
        assert FlextValidators.is_string(42) is False
        assert FlextValidators.is_string(None) is False

        # is_non_empty_string
        assert FlextValidators.is_non_empty_string("hello") is True
        assert FlextValidators.is_non_empty_string("   test   ") is True
        assert FlextValidators.is_non_empty_string("") is False
        assert FlextValidators.is_non_empty_string("   ") is False
        assert FlextValidators.is_non_empty_string(None) is False

    def test_numeric_validators(self) -> None:
        """Test numeric validation functions."""
        # is_int
        assert FlextValidators.is_int(42) is True
        assert FlextValidators.is_int(-5) is True
        assert FlextValidators.is_int(0) is True
        assert FlextValidators.is_int(math.pi) is False
        assert FlextValidators.is_int("42") is False

        # is_positive_int
        assert FlextValidators.is_positive_int(1) is True
        assert FlextValidators.is_positive_int(100) is True
        assert FlextValidators.is_positive_int(0) is False
        assert FlextValidators.is_positive_int(-1) is False
        assert FlextValidators.is_positive_int(math.pi) is False

    def test_collection_validators(self) -> None:
        """Test collection validation functions."""
        # is_list
        assert FlextValidators.is_list([]) is True
        assert FlextValidators.is_list([1, 2, 3]) is True
        assert FlextValidators.is_list("not_list") is False
        assert FlextValidators.is_list({"key": "value"}) is False

        # is_non_empty_list
        assert FlextValidators.is_non_empty_list([1, 2, 3]) is True
        assert FlextValidators.is_non_empty_list(["item"]) is True
        assert FlextValidators.is_non_empty_list([]) is False
        assert FlextValidators.is_non_empty_list("not_list") is False

        # is_dict
        assert FlextValidators.is_dict({}) is True
        assert FlextValidators.is_dict({"key": "value"}) is True
        assert FlextValidators.is_dict([]) is False
        assert FlextValidators.is_dict("not_dict") is False

        # is_non_empty_dict
        assert FlextValidators.is_non_empty_dict({"key": "value"}) is True
        assert FlextValidators.is_non_empty_dict({"a": 1, "b": 2}) is True
        assert FlextValidators.is_non_empty_dict({}) is False
        assert FlextValidators.is_non_empty_dict("not_dict") is False

    def test_pattern_matching_validators(self) -> None:
        """Test pattern matching and regex validators."""
        # is_email
        assert FlextValidators.is_email("user@example.com") is True
        assert FlextValidators.is_email("test.email+tag@domain.co.uk") is True
        assert FlextValidators.is_email("invalid-email") is False
        assert FlextValidators.is_email("@domain.com") is False
        assert FlextValidators.is_email("user@") is False
        assert FlextValidators.is_email(42) is False

        # is_uuid
        assert FlextValidators.is_uuid("123e4567-e89b-12d3-a456-426614174000") is True
        assert FlextValidators.is_uuid("550E8400-E29B-41D4-A716-446655440000") is True
        assert FlextValidators.is_uuid("invalid-uuid") is False
        assert FlextValidators.is_uuid("123-456-789") is False
        assert FlextValidators.is_uuid(42) is False

        # is_url
        assert FlextValidators.is_url("https://example.com") is True
        assert FlextValidators.is_url("http://subdomain.example.com/path") is True
        assert FlextValidators.is_url("ftp://files.example.com") is True
        assert FlextValidators.is_url("invalid-url") is False
        assert FlextValidators.is_url("not a url") is False
        assert FlextValidators.is_url(42) is False

        # matches_pattern
        assert FlextValidators.matches_pattern("abc123", r"^[a-z]+\d+$") is True
        assert FlextValidators.matches_pattern("ABC123", r"^[A-Z]+\d+$") is True
        assert FlextValidators.matches_pattern("invalid", r"^\d+$") is False
        assert FlextValidators.matches_pattern(42, r"^\d+$") is False

        # Invalid regex pattern should return False
        assert FlextValidators.matches_pattern("test", r"[invalid(regex") is False

    def test_range_and_length_validators(self) -> None:
        """Test range and length validation functions."""
        # is_in_range
        assert FlextValidators.is_in_range(5, 1, 10) is True
        assert FlextValidators.is_in_range(1, 1, 10) is True  # boundary
        assert FlextValidators.is_in_range(10, 1, 10) is True  # boundary
        assert FlextValidators.is_in_range(0, 1, 10) is False
        assert FlextValidators.is_in_range(11, 1, 10) is False
        assert FlextValidators.is_in_range(5.5, 1.0, 10.0) is True
        assert FlextValidators.is_in_range("not_number", 1, 10) is False

        # has_min_length
        assert FlextValidators.has_min_length("hello", 3) is True
        assert FlextValidators.has_min_length("hello", 5) is True  # exact
        assert FlextValidators.has_min_length("hi", 5) is False
        assert FlextValidators.has_min_length([1, 2, 3], 2) is True
        assert FlextValidators.has_min_length([], 1) is False
        assert FlextValidators.has_min_length(42, 1) is False  # no __len__

        # has_max_length
        assert FlextValidators.has_max_length("hello", 10) is True
        assert FlextValidators.has_max_length("hello", 5) is True  # exact
        assert FlextValidators.has_max_length("very long string", 5) is False
        assert FlextValidators.has_max_length([1, 2, 3], 5) is True
        assert FlextValidators.has_max_length([1, 2, 3, 4, 5, 6], 5) is False
        assert FlextValidators.has_max_length(42, 10) is False  # no __len__

    def test_instance_and_identifier_validators(self) -> None:
        """Test instance checking and identifier validation."""
        # is_instance_of
        assert FlextValidators.is_instance_of("test", str) is True
        assert FlextValidators.is_instance_of(42, int) is True
        assert FlextValidators.is_instance_of(math.pi, float) is True
        assert FlextValidators.is_instance_of([1, 2, 3], list) is True
        assert FlextValidators.is_instance_of("test", int) is False
        assert FlextValidators.is_instance_of(42, str) is False

        # is_valid_identifier
        assert FlextValidators.is_valid_identifier("valid_name") is True
        assert FlextValidators.is_valid_identifier("_private") is True
        assert FlextValidators.is_valid_identifier("name123") is True
        assert FlextValidators.is_valid_identifier("Class") is True
        assert FlextValidators.is_valid_identifier("123invalid") is False
        assert FlextValidators.is_valid_identifier("with-dash") is False
        assert FlextValidators.is_valid_identifier("with space") is False
        assert FlextValidators.is_valid_identifier("") is False
        assert FlextValidators.is_valid_identifier(42) is False


class TestFlextPredicates:
    """Test FlextPredicates functional predicates for validation patterns."""

    def test_basic_predicates(self) -> None:
        """Test basic predicate functions."""
        # not_none
        not_none_pred = FlextPredicates.not_none()
        assert not_none_pred("test") is True
        assert not_none_pred(0) is True
        false_value = False
        assert not_none_pred(false_value) is True
        assert not_none_pred(None) is False

        # non_empty_string
        non_empty_pred = FlextPredicates.non_empty_string()
        assert non_empty_pred("hello") is True
        assert non_empty_pred("   test   ") is True
        assert non_empty_pred("") is False
        assert non_empty_pred("   ") is False
        assert non_empty_pred(None) is False
        assert non_empty_pred(42) is False

        # positive_number
        positive_pred = FlextPredicates.positive_number()
        assert positive_pred(1) is True
        assert positive_pred(math.pi) is True
        assert positive_pred(0) is False
        assert positive_pred(-1) is False
        assert positive_pred("not_number") is False

    def test_length_predicates(self) -> None:
        """Test length-based predicates."""
        # min_length
        min_5_pred = FlextPredicates.min_length(5)
        assert min_5_pred("hello") is True
        assert min_5_pred("longer string") is True
        assert min_5_pred("hi") is False
        assert min_5_pred([1, 2, 3, 4, 5]) is True
        assert min_5_pred([1, 2]) is False
        assert min_5_pred(42) is False  # no __len__

        # max_length
        max_10_pred = FlextPredicates.max_length(10)
        assert max_10_pred("short") is True
        assert max_10_pred("exactly10!") is True
        assert max_10_pred("this is too long") is False
        assert max_10_pred([1, 2, 3]) is True
        assert max_10_pred(list(range(15))) is False
        assert max_10_pred(42) is False  # no __len__

    def test_regex_predicates(self) -> None:
        """Test regex-based predicates."""
        # matches_regex
        digit_pred = FlextPredicates.matches_regex(r"^\d+$")
        assert digit_pred("123") is True
        assert digit_pred("456789") is True
        assert digit_pred("abc") is False
        assert digit_pred("123abc") is False
        assert digit_pred(123) is False  # not string

        # Complex regex
        email_like_pred = FlextPredicates.matches_regex(r"^[a-z]+@[a-z]+\.[a-z]+$")
        assert email_like_pred("user@domain.com") is True
        assert email_like_pred("test@example.org") is True
        assert email_like_pred("invalid@email") is False
        assert email_like_pred("not-an-email") is False

        # is_email predicate
        email_pred = FlextPredicates.is_email()
        assert email_pred("user@example.com") is True
        assert email_pred("test.email@domain.co.uk") is True
        assert email_pred("invalid-email") is False

        # is_uuid predicate
        uuid_pred = FlextPredicates.is_uuid()
        assert uuid_pred("123e4567-e89b-12d3-a456-426614174000") is True
        assert uuid_pred("invalid-uuid") is False

        # is_url predicate
        url_pred = FlextPredicates.is_url()
        assert url_pred("https://example.com") is True
        assert url_pred("invalid-url") is False

    def test_range_predicates(self) -> None:
        """Test range-based predicates."""
        # in_range
        age_range_pred = FlextPredicates.in_range(0, 150)
        assert age_range_pred(25) is True
        assert age_range_pred(0) is True  # boundary
        assert age_range_pred(150) is True  # boundary
        assert age_range_pred(-1) is False
        assert age_range_pred(151) is False
        assert age_range_pred(50.5) is True
        assert age_range_pred("not_number") is False

        # percentage range
        percent_pred = FlextPredicates.in_range(0.0, 100.0)
        assert percent_pred(50.5) is True
        assert percent_pred(0.0) is True
        assert percent_pred(100.0) is True
        assert percent_pred(-0.1) is False
        assert percent_pred(100.1) is False

    def test_container_predicates(self) -> None:
        """Test container-based predicates."""
        # contains
        contains_a_pred = FlextPredicates.contains("a")
        assert contains_a_pred("abc") is True
        assert contains_a_pred("test a string") is True
        assert contains_a_pred("xyz") is False
        assert contains_a_pred(["a", "b", "c"]) is True
        assert contains_a_pred(["x", "y", "z"]) is False
        assert contains_a_pred({"a": 1, "b": 2}) is True
        assert contains_a_pred({"x": 1, "y": 2}) is False
        assert contains_a_pred(42) is False  # no __contains__

        # starts_with
        starts_test_pred = FlextPredicates.starts_with("test")
        assert starts_test_pred("test_string") is True
        assert starts_test_pred("testing") is True
        assert starts_test_pred("not_test") is False
        assert starts_test_pred("tes") is False
        assert starts_test_pred(42) is False  # not string

        # ends_with
        ends_ing_pred = FlextPredicates.ends_with("ing")
        assert ends_ing_pred("testing") is True
        assert ends_ing_pred("running") is True
        assert ends_ing_pred("test") is False
        assert ends_ing_pred("ingtest") is False
        assert ends_ing_pred(42) is False  # not string

    def test_predicate_composition(self) -> None:
        """Test composing multiple predicates."""
        # Combining predicates with logical operators
        non_empty_string_pred = FlextPredicates.non_empty_string()
        min_length_pred = FlextPredicates.min_length(3)

        # Simulate AND logic
        def combined_predicate(value: object) -> bool:
            return non_empty_string_pred(value) and min_length_pred(value)

        assert combined_predicate("hello") is True
        assert combined_predicate("hi") is False  # too short
        assert combined_predicate("") is False  # empty
        assert combined_predicate(42) is False  # not string


class TestFlextValidation:
    """Test FlextValidation main validation interface with inheritance."""

    def test_inherited_validator_methods(self) -> None:
        """Test that FlextValidation inherits all FlextValidators methods."""
        # Basic type checks (inherited)
        assert FlextValidation.is_string("test") is True
        assert FlextValidation.is_not_none("test") is True
        assert FlextValidation.is_email("user@example.com") is True
        assert FlextValidation.is_positive_int(5) is True

        # Collection checks (inherited)
        assert FlextValidation.is_list([1, 2, 3]) is True
        assert FlextValidation.is_non_empty_dict({"key": "value"}) is True

        # Pattern matching (inherited)
        assert FlextValidation.is_uuid("123e4567-e89b-12d3-a456-426614174000") is True
        assert FlextValidation.matches_pattern("abc123", r"^[a-z]+\d+$") is True

    def test_validator_composition_chain(self) -> None:
        """Test validator composition with chain (AND logic)."""
        # Chain string validators
        string_email_validator = FlextValidation.chain(
            FlextValidation.is_string,
            FlextValidation.is_non_empty_string,
            FlextValidation.is_email,
        )

        assert string_email_validator("user@example.com") is True
        assert string_email_validator("invalid-email") is False  # not email
        assert string_email_validator("") is False  # empty string
        assert string_email_validator(42) is False  # not string

        # Chain numeric validators
        positive_int_validator = FlextValidation.chain(
            FlextValidation.is_int,
            FlextValidation.is_positive_int,
        )

        assert positive_int_validator(5) is True
        assert positive_int_validator(-5) is False  # not positive
        assert positive_int_validator(math.pi) is False  # not int
        assert positive_int_validator("5") is False  # not int

        # Empty chain should return True
        empty_chain = FlextValidation.chain()
        assert empty_chain("anything") is True

    def test_validator_composition_any_of(self) -> None:
        """Test validator composition with any_of (OR logic)."""
        # Any of multiple type validators
        string_or_int_validator = FlextValidation.any_of(
            FlextValidation.is_string,
            FlextValidation.is_int,
        )

        assert string_or_int_validator("test") is True
        assert string_or_int_validator(42) is True
        assert string_or_int_validator(math.pi) is False  # neither string nor int
        assert string_or_int_validator([1, 2, 3]) is False  # neither

        # Any of multiple pattern validators
        email_or_uuid_validator = FlextValidation.any_of(
            FlextValidation.is_email,
            FlextValidation.is_uuid,
        )

        assert email_or_uuid_validator("user@example.com") is True
        assert email_or_uuid_validator("123e4567-e89b-12d3-a456-426614174000") is True
        assert email_or_uuid_validator("invalid") is False  # neither

        # Empty any_of should return False
        empty_any_of = FlextValidation.any_of()
        assert empty_any_of("anything") is False

    def test_validation_config_creation(self) -> None:
        """Test validation configuration creation."""
        # Basic config
        config = FlextValidation.create_validation_config("username")
        assert config.field_name == "username"
        assert config.min_length == 0
        assert config.max_length is None

        # Config with length constraints
        password_config = FlextValidation.create_validation_config(
            "password",
            min_length=8,
            max_length=128,
        )
        assert password_config.field_name == "password"
        assert password_config.min_length == 8
        assert password_config.max_length == 128

    def test_safe_validate_with_flext_result(self) -> None:
        """Test safe_validate with FlextResult error handling."""
        # Successful validation
        success_result = FlextValidation.safe_validate(
            "user@example.com",
            FlextValidation.is_email,
        )
        assert success_result.is_success
        assert success_result.data == "user@example.com"

        # Failed validation
        fail_result = FlextValidation.safe_validate(
            "invalid-email",
            FlextValidation.is_email,
        )
        assert fail_result.is_failure
        assert "validation failed" in fail_result.error.lower()

        # Exception in validator
        def failing_validator(value: object) -> bool:
            msg = "Validator error"
            raise ValueError(msg)

        error_result = FlextValidation.safe_validate("test", failing_validator)
        assert error_result.is_failure
        assert "validation error" in error_result.error.lower()

    def test_entity_validation_methods(self) -> None:
        """Test entity validation methods."""
        # validate_entity_id
        assert FlextValidation.validate_entity_id("user_123") is True
        assert FlextValidation.validate_entity_id("valid-id") is True
        assert FlextValidation.validate_entity_id("") is False
        assert FlextValidation.validate_entity_id("   ") is False
        assert FlextValidation.validate_entity_id(None) is False

        # validate_non_empty_string
        assert FlextValidation.validate_non_empty_string("test") is True
        assert FlextValidation.validate_non_empty_string("") is False
        assert FlextValidation.validate_non_empty_string(None) is False


class TestFieldValidationFunctions:
    """Test field validation functions with Pydantic configuration."""

    def test_validate_required_field(self) -> None:
        """Test validate_required_field function."""
        # Valid required values
        result = validate_required_field("test_value", "username")
        assert result.is_valid is True
        assert result.error_message is None

        result_number = validate_required_field(42, "age")
        assert result_number.is_valid is True

        # None value should fail
        none_result = validate_required_field(None, "required_field")
        assert none_result.is_valid is False
        assert "required but was None" in none_result.error_message
        assert none_result.field_name == "required_field"

        # Empty string should fail
        empty_result = validate_required_field("", "username")
        assert empty_result.is_valid is False
        assert "required but was empty" in empty_result.error_message

        # Whitespace-only string should fail
        whitespace_result = validate_required_field("   ", "username")
        assert whitespace_result.is_valid is False
        assert "required but was empty" in whitespace_result.error_message

    def test_validate_string_field(self) -> None:
        """Test validate_string_field with length constraints."""
        # Valid string within constraints
        result = validate_string_field("hello", "message", min_length=3, max_length=10)
        assert result.is_valid is True
        assert result.error_message is None

        # Valid string at boundaries
        boundary_result = validate_string_field(
            "test",
            "field",
            min_length=4,
            max_length=4,
        )
        assert boundary_result.is_valid is True

        # Non-string value should fail
        non_string_result = validate_string_field(42, "name")
        assert non_string_result.is_valid is False
        assert "must be a string" in non_string_result.error_message
        assert "got int" in non_string_result.error_message

        # String too short should fail
        short_result = validate_string_field("hi", "password", min_length=8)
        assert short_result.is_valid is False
        assert "must be at least 8 characters" in short_result.error_message

        # String too long should fail
        long_result = validate_string_field(
            "this is too long",
            "username",
            max_length=5,
        )
        assert long_result.is_valid is False
        assert "must be at most 5 characters" in long_result.error_message

        # Invalid configuration should fail
        invalid_config_result = validate_string_field(
            "test",
            "",
            min_length=5,
            max_length=3,
        )
        assert invalid_config_result.is_valid is False
        assert (
            "invalid validation config" in invalid_config_result.error_message.lower()
        )

    def test_validate_numeric_field(self) -> None:
        """Test validate_numeric_field with range constraints."""
        # Valid numeric values
        int_result = validate_numeric_field(25, "age", min_val=0, max_val=150)
        assert int_result.is_valid is True

        float_result = validate_numeric_field(
            math.pi,
            "score",
            min_val=0.0,
            max_val=10.0,
        )
        assert float_result.is_valid is True

        # Valid at boundaries
        boundary_result = validate_numeric_field(
            100,
            "percentage",
            min_val=0,
            max_val=100,
        )
        assert boundary_result.is_valid is True

        # Non-numeric value should fail
        non_numeric_result = validate_numeric_field("not_number", "age")
        assert non_numeric_result.is_valid is False
        assert "must be a number" in non_numeric_result.error_message
        assert "got str" in non_numeric_result.error_message

        # Value too small should fail
        small_result = validate_numeric_field(-5, "age", min_val=0)
        assert small_result.is_valid is False
        assert "must be at least 0" in small_result.error_message

        # Value too large should fail
        large_result = validate_numeric_field(200, "age", max_val=150)
        assert large_result.is_valid is False
        assert "must be at most 150" in large_result.error_message

    def test_validate_email_field(self) -> None:
        """Test validate_email_field function."""
        # Valid email addresses
        valid_result = validate_email_field("user@example.com", "email")
        assert valid_result.is_valid is True

        complex_result = validate_email_field(
            "test.email+tag@domain.co.uk",
            "contact_email",
        )
        assert complex_result.is_valid is True

        # Non-string value should fail
        non_string_result = validate_email_field(42, "email")
        assert non_string_result.is_valid is False
        assert "must be a string" in non_string_result.error_message

        # Invalid email format should fail
        invalid_result = validate_email_field("invalid-email", "email")
        assert invalid_result.is_valid is False
        assert "must be a valid email address" in invalid_result.error_message

        # Empty string should fail
        empty_result = validate_email_field("", "email")
        assert empty_result.is_valid is False


class TestConvenienceFunctions:
    """Test convenience functions for high-level validation."""

    def test_validate_required_convenience(self) -> None:
        """Test validate_required convenience function."""
        # Successful validation
        result = validate_required("test_value", "username")
        assert result.is_valid is True

        # Failed validation
        fail_result = validate_required(None, "required_field")
        assert fail_result.is_valid is False
        assert fail_result.field_name == "required_field"

        # Default field name
        default_result = validate_required("test")
        assert default_result.is_valid is True

    def test_validate_string_convenience(self) -> None:
        """Test validate_string convenience function."""
        # Valid string
        result = validate_string("hello", "message", min_length=3, max_length=10)
        assert result.is_valid is True

        # Invalid string
        fail_result = validate_string("hi", "password", min_length=8)
        assert fail_result.is_valid is False

        # Default parameters
        default_result = validate_string("test")
        assert default_result.is_valid is True

    def test_validate_numeric_convenience(self) -> None:
        """Test validate_numeric convenience function."""
        # Valid numeric
        result = validate_numeric(25, "age", min_val=0, max_val=150)
        assert result.is_valid is True

        # Invalid numeric
        fail_result = validate_numeric(-5, "age", min_val=0)
        assert fail_result.is_valid is False

        # Default parameters
        default_result = validate_numeric(42)
        assert default_result.is_valid is True

    def test_validate_email_convenience(self) -> None:
        """Test validate_email convenience function."""
        # Valid email
        result = validate_email("user@example.com", "email")
        assert result.is_valid is True

        # Invalid email
        fail_result = validate_email("invalid-email", "email")
        assert fail_result.is_valid is False

        # Default field name
        default_result = validate_email("test@example.com")
        assert default_result.is_valid is True


class TestEntityValidationFunctions:
    """Test entity validation functions."""

    def test_validate_entity_id_function(self) -> None:
        """Test validate_entity_id function."""
        assert validate_entity_id("user_123") is True
        assert validate_entity_id("valid-entity-id") is True
        assert validate_entity_id("") is False
        assert validate_entity_id("   ") is False
        assert validate_entity_id(None) is False
        assert validate_entity_id(42) is False

    def test_validate_non_empty_string_function(self) -> None:
        """Test validate_non_empty_string function."""
        assert validate_non_empty_string("test") is True
        assert validate_non_empty_string("   valid   ") is True
        assert validate_non_empty_string("") is False
        assert validate_non_empty_string("   ") is False
        assert validate_non_empty_string(None) is False
        assert validate_non_empty_string(42) is False


class TestBackwardCompatibilityAliases:
    """Test backward compatibility aliases and legacy functions."""

    def test_legacy_class_aliases(self) -> None:
        """Test legacy class aliases."""
        # Legacy class access
        assert Validation is FlextValidation
        assert Predicates is FlextPredicates
        assert ValidationResult is FlextValidationResult
        assert ValidationResultFactory is FlextValidationResult

        # Internal legacy aliases
        assert FlextValidators is FlextValidators
        assert FlextPredicates is FlextPredicates
        assert _ValidationConfig is FlextValidationConfig
        assert _ValidationResult is FlextValidationResult

    def test_legacy_function_aliases(self) -> None:
        """Test legacy function aliases."""
        # Legacy function names
        assert is_not_none is FlextValidation.is_not_none
        assert is_string is FlextValidation.is_string
        assert is_non_empty_string is FlextValidation.is_non_empty_string
        assert is_email is FlextValidation.is_email

        # Test functionality works through aliases
        assert is_not_none("test") is True
        assert is_string("hello") is True
        assert is_non_empty_string("test") is True
        assert is_email("user@example.com") is True

        # Internal function aliases
        assert _validate_required_field is validate_required_field
        assert _validate_string_field is validate_string_field
        assert _validate_numeric_field is validate_numeric_field
        assert _validate_email_field is validate_email_field

    def test_legacy_functionality_equivalence(self) -> None:
        """Test that legacy aliases provide equivalent functionality."""
        # Test that legacy classes work the same as new ones
        legacy_validation = Validation()
        new_validation = FlextValidation()

        # Both should have the same methods
        assert hasattr(legacy_validation, "is_email")
        assert hasattr(new_validation, "is_email")
        assert hasattr(legacy_validation, "chain")
        assert hasattr(new_validation, "chain")

        # Test that legacy predicates work
        legacy_predicates = Predicates()
        new_predicates = FlextPredicates()

        assert hasattr(legacy_predicates, "not_none")
        assert hasattr(new_predicates, "not_none")

        # Test legacy result creation
        legacy_result = ValidationResult.success()
        new_result = FlextValidationResult.success()

        assert legacy_result.is_valid is True
        assert new_result.is_valid is True


class TestValidationIntegrationScenarios:
    """Test integration scenarios combining multiple validation features."""

    def test_user_registration_validation_scenario(self) -> None:
        """Test complete user registration validation scenario."""

        def validate_user_registration(
            user_data: dict[str, Any],
        ) -> list[FlextValidationResult]:
            """Validate complete user registration data."""
            results = []

            # Validate username
            username_result = validate_string(
                user_data.get("username"),
                "username",
                min_length=3,
                max_length=20,
            )
            results.append(username_result)

            # Validate email
            email_result = validate_email(user_data.get("email"), "email")
            results.append(email_result)

            # Validate age
            age_result = validate_numeric(
                user_data.get("age"),
                "age",
                min_val=13,
                max_val=120,
            )
            results.append(age_result)

            return results

        # Valid user data
        valid_user = {
            "username": "john_doe",
            "email": "john@example.com",
            "age": 25,
        }

        valid_results = validate_user_registration(valid_user)
        assert all(result.is_valid for result in valid_results)

        # Invalid user data
        invalid_user = {
            "username": "jo",  # too short
            "email": "invalid-email",  # invalid format
            "age": 200,  # too old
        }

        invalid_results = validate_user_registration(invalid_user)
        assert all(not result.is_valid for result in invalid_results)

        # Check specific error messages
        username_error = invalid_results[0]
        assert "must be at least 3 characters" in username_error.error_message

        email_error = invalid_results[1]
        assert "must be a valid email address" in email_error.error_message

        age_error = invalid_results[2]
        assert "must be at most 120" in age_error.error_message

    def test_complex_validator_composition_scenario(self) -> None:
        """Test complex validator composition for business rules."""

        # Create a complex business rule validator
        def create_strong_password_validator() -> callable:
            """Create validator for strong password requirements."""
            return FlextValidation.chain(
                FlextValidation.is_string,
                FlextValidation.is_non_empty_string,
                lambda x: len(x) >= 8,
                lambda x: len(x) <= 128,
                lambda x: any(c.isupper() for c in x),  # Has uppercase
                lambda x: any(c.islower() for c in x),  # Has lowercase
                lambda x: any(c.isdigit() for c in x),  # Has digit
                lambda x: any(c in "!@#$%^&*" for c in x),  # Has special char
            )

        strong_password_validator = create_strong_password_validator()

        # Valid strong passwords
        assert strong_password_validator("StrongPass1!") is True
        assert strong_password_validator("MySecure2@Pass") is True

        # Invalid passwords
        assert strong_password_validator("weak") is False  # too short
        assert (
            strong_password_validator("weakpassword") is False
        )  # no uppercase, digit, special
        assert (
            strong_password_validator("WEAKPASSWORD") is False
        )  # no lowercase, digit, special
        assert strong_password_validator("WeakPassword") is False  # no digit, special
        assert strong_password_validator("WeakPassword1") is False  # no special
        assert strong_password_validator(None) is False  # not string

    def test_predicate_based_filtering_scenario(self) -> None:
        """Test predicate-based data filtering scenario."""
        # Sample data
        users = [
            {"name": "Alice", "age": 25, "email": "alice@example.com"},
            {"name": "", "age": 17, "email": "bob@example.com"},  # invalid name
            {"name": "Charlie", "age": 30, "email": "invalid-email"},  # invalid email
            {"name": "Diana", "age": 22, "email": "diana@example.com"},
            {"name": "Eve", "age": -5, "email": "eve@example.com"},  # invalid age
        ]

        # Create validation predicates
        valid_name_pred = FlextPredicates.non_empty_string()
        valid_age_pred = FlextPredicates.in_range(18, 120)
        valid_email_pred = FlextPredicates.is_email()

        # Filter valid users
        valid_users = [
            user
            for user in users
            if valid_name_pred(user["name"])
            and valid_age_pred(user["age"])
            and valid_email_pred(user["email"])
        ]

        # Should only have Alice and Diana
        assert len(valid_users) == 2
        assert valid_users[0]["name"] == "Alice"
        assert valid_users[1]["name"] == "Diana"

    def test_safe_validation_with_error_recovery_scenario(self) -> None:
        """Test safe validation with error recovery patterns."""

        def safe_validate_user_email(email: object) -> FlextResult[str]:
            """Safely validate user email with detailed error handling."""
            # Use safe validation with automatic error handling
            return FlextValidation.safe_validate(email, FlextValidation.is_email)

        # Valid email
        valid_result = safe_validate_user_email("user@example.com")
        assert valid_result.is_success
        assert valid_result.data == "user@example.com"

        # Invalid email
        invalid_result = safe_validate_user_email("invalid-email")
        assert invalid_result.is_failure
        assert "validation failed" in invalid_result.error.lower()

        # Non-string input
        non_string_result = safe_validate_user_email(42)
        assert non_string_result.is_failure

        # None input
        none_result = safe_validate_user_email(None)
        assert none_result.is_failure
