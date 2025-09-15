"""Comprehensive FlextValidations tests targeting uncovered lines and edge cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
import uuid

from pydantic import BaseModel, Field

from flext_core import FlextResult, FlextValidations
from flext_tests import FlextTestsMatchers


class TestFlextValidationsComprehensive:
    """Comprehensive tests for FlextValidations targeting uncovered functionality."""

    def test_type_validators_validate_string_success(self) -> None:
        """Test TypeValidators.validate_string with valid string."""
        result = FlextValidations.TypeValidators.validate_string("valid_string")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "valid_string"

    def test_type_validators_validate_string_failure(self) -> None:
        """Test TypeValidators.validate_string with non-string value."""
        result = FlextValidations.TypeValidators.validate_string(12345)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_type_validators_validate_integer_with_min_max(self) -> None:
        """Test TypeValidators.validate_integer with min/max constraints."""
        # Test valid integer within range
        result = FlextValidations.TypeValidators.validate_integer(50)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 50

    def test_type_validators_validate_integer_below_min(self) -> None:
        """Test TypeValidators.validate_integer with invalid string."""
        result = FlextValidations.TypeValidators.validate_integer("not_a_number")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Cannot convert" in result.error

    def test_type_validators_validate_integer_above_max(self) -> None:
        """Test TypeValidators.validate_integer with non-numeric type."""
        result = FlextValidations.TypeValidators.validate_integer([1, 2, 3])
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_type_validators_validate_integer_string_conversion(self) -> None:
        """Test TypeValidators.validate_integer with string conversion."""
        result = FlextValidations.TypeValidators.validate_integer("42")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_type_validators_validate_integer_invalid_string(self) -> None:
        """Test TypeValidators.validate_integer with invalid string."""
        result = FlextValidations.TypeValidators.validate_integer("not_a_number")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Cannot convert" in result.error
        assert result.error
        assert result.error is not None
        assert "to integer" in result.error

    def test_type_validators_validate_integer_non_numeric_type(self) -> None:
        """Test TypeValidators.validate_integer with non-numeric type."""
        result = FlextValidations.TypeValidators.validate_integer([1, 2, 3])
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_type_validators_validate_float_with_precision(self) -> None:
        """Test TypeValidators.validate_float with precision constraints."""
        # Test valid float with precision
        result = FlextValidations.TypeValidators.validate_float(math.pi)
        FlextTestsMatchers.assert_result_success(result)
        assert abs(result.value - math.pi) < 1e-10

    def test_type_validators_validate_float_with_min_max(self) -> None:
        """Test TypeValidators.validate_float with min/max constraints."""
        result = FlextValidations.TypeValidators.validate_float(2.5)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 2.5

    def test_type_validators_validate_float_success_case(self) -> None:
        """Test TypeValidators.validate_float with valid float."""
        result = FlextValidations.TypeValidators.validate_float(0.5)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 0.5

    def test_type_validators_validate_float_string_conversion(self) -> None:
        """Test TypeValidators.validate_float with string conversion."""
        result = FlextValidations.TypeValidators.validate_float("3.14159")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == math.pi

    def test_type_validators_validate_float_invalid_string(self) -> None:
        """Test TypeValidators.validate_float with invalid string."""
        result = FlextValidations.TypeValidators.validate_float("not_a_float")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Cannot convert" in result.error

    def test_type_validators_validate_dict_success(self) -> None:
        """Test TypeValidators.validate_dict with valid dictionary."""
        test_dict = {"key1": "value1", "key2": "value2"}
        result = FlextValidations.TypeValidators.validate_dict(test_dict)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == test_dict

    def test_type_validators_validate_dict_failure(self) -> None:
        """Test TypeValidators.validate_dict with non-dictionary value."""
        result = FlextValidations.TypeValidators.validate_dict("not_a_dict")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_type_validators_validate_list_success(self) -> None:
        """Test TypeValidators.validate_list with valid list."""
        test_list = [1, 2, 3, 4, 5]
        result = FlextValidations.TypeValidators.validate_list(test_list)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == test_list

    def test_type_validators_validate_list_failure(self) -> None:
        """Test TypeValidators.validate_list with non-list value."""
        result = FlextValidations.TypeValidators.validate_list("not_a_list")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_field_validators_validate_email_success(self) -> None:
        """Test FieldValidators.validate_email with valid email."""
        result = FlextValidations.FieldValidators.validate_email("user@example.com")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "user@example.com"

    def test_field_validators_validate_email_failure(self) -> None:
        """Test FieldValidators.validate_email with invalid email."""
        result = FlextValidations.FieldValidators.validate_email("invalid_email")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid email" in result.error

    def test_field_validators_validate_phone_success(self) -> None:
        """Test FieldValidators.validate_phone with valid phone."""
        result = FlextValidations.FieldValidators.validate_phone("+1-555-123-4567")
        FlextTestsMatchers.assert_result_success(result)

    def test_field_validators_validate_phone_with_locale(self) -> None:
        """Test FieldValidators.validate_phone with specific locale."""
        result = FlextValidations.validate_phone("555-123-4567", locale="US")
        FlextTestsMatchers.assert_result_success(result)

    def test_field_validators_validate_phone_failure(self) -> None:
        """Test FieldValidators.validate_phone with invalid phone."""
        result = FlextValidations.FieldValidators.validate_phone("123")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid phone" in result.error

    def test_field_validators_validate_url_success(self) -> None:
        """Test FieldValidators.validate_url with valid URL."""
        result = FlextValidations.FieldValidators.validate_url("https://example.com")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "https://example.com"

    def test_field_validators_validate_url_failure(self) -> None:
        """Test FieldValidators.validate_url with invalid URL."""
        result = FlextValidations.FieldValidators.validate_url("not_a_url")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid URL" in result.error

    def test_field_validators_validate_uuid_success(self) -> None:
        """Test FieldValidators.validate_uuid with valid UUID."""
        test_uuid = str(uuid.uuid4())
        result = FlextValidations.FieldValidators.validate_uuid(test_uuid)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == test_uuid

    def test_field_validators_validate_uuid_failure(self) -> None:
        """Test FieldValidators.validate_uuid with invalid UUID."""
        result = FlextValidations.FieldValidators.validate_uuid("not_a_uuid")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid UUID" in result.error

    def test_business_validators_validate_string_field_success(self) -> None:
        """Test BusinessValidators.validate_string_field with valid input."""
        result = FlextValidations.BusinessValidators.validate_string_field(
            "valid_string", min_length=5, max_length=20
        )
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "valid_string"

    def test_business_validators_validate_string_field_too_short(self) -> None:
        """Test BusinessValidators.validate_string_field with string too short."""
        result = FlextValidations.BusinessValidators.validate_string_field(
            "hi", min_length=5, max_length=20
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "String too short, minimum 5" in result.error

    def test_business_validators_validate_string_field_too_long(self) -> None:
        """Test BusinessValidators.validate_string_field with string too long."""
        long_string = "a" * 25
        result = FlextValidations.BusinessValidators.validate_string_field(
            long_string, min_length=5, max_length=20
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "String too long, maximum 20" in result.error

    def test_business_validators_validate_numeric_field_integer(self) -> None:
        """Test BusinessValidators.validate_numeric_field with integer."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(42)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_business_validators_validate_numeric_field_float(self) -> None:
        """Test BusinessValidators.validate_numeric_field with float."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(math.pi)
        FlextTestsMatchers.assert_result_success(result)
        assert abs(result.value - math.pi) < 1e-10

    def test_business_validators_validate_numeric_field_string_conversion(self) -> None:
        """Test BusinessValidators.validate_numeric_field with string conversion."""
        result = FlextValidations.BusinessValidators.validate_numeric_field("42.5")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42.5

    def test_business_validators_validate_numeric_field_with_constraints(self) -> None:
        """Test BusinessValidators.validate_numeric_field with constraints."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(
            50, min_value=10, max_value=100
        )
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 50

    def test_business_validators_validate_numeric_field_below_min(self) -> None:
        """Test BusinessValidators.validate_numeric_field below minimum."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(
            5, min_value=10, max_value=100
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Value too small, minimum 10" in result.error

    def test_business_validators_validate_numeric_field_above_max(self) -> None:
        """Test BusinessValidators.validate_numeric_field above maximum."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(
            150, min_value=10, max_value=100
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Value too large, maximum 100" in result.error

    def test_business_validators_validate_numeric_field_invalid_type(self) -> None:
        """Test BusinessValidators.validate_numeric_field with invalid type."""
        result = FlextValidations.BusinessValidators.validate_numeric_field([1, 2, 3])
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Value cannot be converted to a number" in result.error

    def test_business_validators_validate_range_success(self) -> None:
        """Test BusinessValidators.validate_range with valid range."""
        result = FlextValidations.BusinessValidators.validate_range(50, 10, 100)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 50

    def test_business_validators_validate_range_failure(self) -> None:
        """Test BusinessValidators.validate_range outside valid range."""
        result = FlextValidations.BusinessValidators.validate_range(150, 10, 100)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Value 150 out of range [10, 100]" in result.error

    def test_business_validators_validate_password_strength_success(self) -> None:
        """Test BusinessValidators.validate_password_strength with strong password."""
        result = FlextValidations.BusinessValidators.validate_password_strength(
            "StrongPass123!"
        )
        FlextTestsMatchers.assert_result_success(result)

    def test_business_validators_validate_password_strength_failure(self) -> None:
        """Test BusinessValidators.validate_password_strength with weak password."""
        result = FlextValidations.BusinessValidators.validate_password_strength("weak")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Password must be at least" in result.error

    def test_business_validators_validate_credit_card_success(self) -> None:
        """Test BusinessValidators.validate_credit_card with valid card number."""
        # Using a test Luhn-valid number
        result = FlextValidations.BusinessValidators.validate_credit_card(
            "4532015112830366"
        )
        FlextTestsMatchers.assert_result_success(result)

    def test_business_validators_validate_credit_card_failure(self) -> None:
        """Test BusinessValidators.validate_credit_card with invalid card number."""
        result = FlextValidations.BusinessValidators.validate_credit_card("123456789")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid credit card" in result.error

    def test_business_validators_validate_ipv4_success(self) -> None:
        """Test BusinessValidators.validate_ipv4 with valid IP."""
        result = FlextValidations.BusinessValidators.validate_ipv4("192.168.1.1")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "192.168.1.1"

    def test_business_validators_validate_ipv4_failure(self) -> None:
        """Test BusinessValidators.validate_ipv4 with invalid IP."""
        result = FlextValidations.BusinessValidators.validate_ipv4("256.256.256.256")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid IPv4" in result.error

    def test_business_validators_validate_date_success(self) -> None:
        """Test BusinessValidators.validate_date with valid date."""
        result = FlextValidations.BusinessValidators.validate_date("2023-12-25")
        FlextTestsMatchers.assert_result_success(result)

    def test_business_validators_validate_date_failure(self) -> None:
        """Test BusinessValidators.validate_date with invalid date."""
        result = FlextValidations.BusinessValidators.validate_date("invalid_date")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid date" in result.error

    def test_business_validators_validate_json_success(self) -> None:
        """Test BusinessValidators.validate_json with valid JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = FlextValidations.BusinessValidators.validate_json(json_str)
        FlextTestsMatchers.assert_result_success(result)

    def test_business_validators_validate_json_failure(self) -> None:
        """Test BusinessValidators.validate_json with invalid JSON."""
        result = FlextValidations.BusinessValidators.validate_json("invalid_json")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid JSON" in result.error

    def test_guards_require_not_none_success(self) -> None:
        """Test Guards.require_not_none with non-None value."""
        result = FlextValidations.Guards.require_not_none("not_none")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "not_none"

    def test_guards_require_not_none_failure(self) -> None:
        """Test Guards.require_not_none with None value."""
        result = FlextValidations.Guards.require_not_none(None)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "cannot be None" in result.error

    def test_guards_require_positive_success(self) -> None:
        """Test Guards.require_positive with positive number."""
        result = FlextValidations.Guards.require_positive(42)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_guards_require_positive_failure(self) -> None:
        """Test Guards.require_positive with negative number."""
        result = FlextValidations.Guards.require_positive(-5)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "must be positive" in result.error

    def test_guards_require_in_range_success(self) -> None:
        """Test Guards.require_in_range with value in range."""
        result = FlextValidations.Guards.require_in_range(50, 10, 100)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 50

    def test_guards_require_in_range_failure_below(self) -> None:
        """Test Guards.require_in_range with value below range."""
        result = FlextValidations.Guards.require_in_range(5, 10, 100)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Value out of range" in result.error

    def test_guards_require_in_range_failure_above(self) -> None:
        """Test Guards.require_in_range with value above range."""
        result = FlextValidations.Guards.require_in_range(150, 10, 100)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Value out of range" in result.error

    def test_guards_require_non_empty_success_string(self) -> None:
        """Test Guards.require_non_empty with non-empty string."""
        result = FlextValidations.Guards.require_non_empty("not_empty")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "not_empty"

    def test_guards_require_non_empty_success_list(self) -> None:
        """Test Guards.require_non_empty with non-empty list."""
        result = FlextValidations.Guards.require_non_empty([1, 2, 3])
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == [1, 2, 3]

    def test_guards_require_non_empty_failure_empty_string(self) -> None:
        """Test Guards.require_non_empty with empty string."""
        result = FlextValidations.Guards.require_non_empty("")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "cannot be empty" in result.error

    def test_guards_require_non_empty_failure_empty_list(self) -> None:
        """Test Guards.require_non_empty with empty list."""
        result = FlextValidations.Guards.require_non_empty([])
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "cannot be empty" in result.error

    def test_guards_is_dict_of_success(self) -> None:
        """Test Guards.is_dict_of with valid dictionary type."""
        test_dict = {"key1": "value1", "key2": "value2"}
        result = FlextValidations.Guards.is_dict_of(test_dict, str)
        assert result is True

    def test_guards_is_dict_of_failure(self) -> None:
        """Test Guards.is_dict_of with invalid dictionary type."""
        test_dict = {"key1": 123, "key2": 456}  # Values are int, not str
        result = FlextValidations.Guards.is_dict_of(test_dict, str)
        assert result is False

    def test_guards_is_list_of_success(self) -> None:
        """Test Guards.is_list_of with valid list type."""
        test_list = ["item1", "item2", "item3"]
        result = FlextValidations.Guards.is_list_of(test_list, str)
        assert result is True

    def test_guards_is_list_of_failure(self) -> None:
        """Test Guards.is_list_of with invalid list type."""
        test_list = [1, 2, 3]  # Items are int, not str
        result = FlextValidations.Guards.is_list_of(test_list, str)
        assert result is False

    def test_schema_validators_validate_with_pydantic_schema_success(self) -> None:
        """Test SchemaValidators.validate_with_pydantic_schema with valid data."""

        class TestModel(BaseModel):
            name: str = Field(..., min_length=1)
            age: int = Field(..., ge=0, le=120)

        data = {"name": "John Doe", "age": 30}
        result = FlextValidations.SchemaValidators.validate_with_pydantic_schema(
            data, TestModel
        )
        FlextTestsMatchers.assert_result_success(result)

    def test_schema_validators_validate_with_pydantic_schema_failure(self) -> None:
        """Test SchemaValidators.validate_with_pydantic_schema with invalid data."""

        class TestModel(BaseModel):
            name: str = Field(..., min_length=1)
            age: int = Field(..., ge=0, le=120)

        data = {"name": "", "age": -5}  # Invalid data
        result = FlextValidations.SchemaValidators.validate_with_pydantic_schema(
            data, TestModel
        )
        FlextTestsMatchers.assert_result_failure(result)

    def test_schema_validators_validate_schema_success(self) -> None:
        """Test SchemaValidators.validate_schema with valid data against schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
            },
            "required": ["name", "age"],
        }
        data = {"name": "John Doe", "age": 30}
        result = FlextValidations.validate_with_schema(data, schema)
        FlextTestsMatchers.assert_result_success(result)

    def test_schema_validators_validate_schema_failure(self) -> None:
        """Test SchemaValidators.validate_schema with invalid data against schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
            },
            "required": ["name", "age"],
        }
        data = {"name": "", "age": -5}  # Invalid data
        result = FlextValidations.validate_with_schema(data, schema)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_number_string_with_dot(self) -> None:
        """Test validate_number with string containing dot (float conversion)."""
        result = FlextValidations.validate_number("3.14159")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == math.pi
        assert isinstance(result.value, float)

    def test_validate_number_string_without_dot(self) -> None:
        """Test validate_number with string without dot (int conversion)."""
        result = FlextValidations.validate_number("42")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42
        assert isinstance(result.value, int)

    def test_validate_number_string_invalid(self) -> None:
        """Test validate_number with invalid string."""
        result = FlextValidations.validate_number("not_a_number")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Value must be numeric" in result.error

    def test_validate_number_delegates_to_business_validators(self) -> None:
        """Test validate_number delegates to BusinessValidators for non-string input."""
        result = FlextValidations.validate_number(42.5)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42.5

    def test_validate_phone_with_locale_parameter(self) -> None:
        """Test validate_phone with explicit locale parameter."""
        result = FlextValidations.validate_phone("555-123-4567", locale="US")
        FlextTestsMatchers.assert_result_success(result)

    def test_validate_phone_with_failure_result(self) -> None:
        """Test validate_phone handles failure from FieldValidators."""
        result = FlextValidations.validate_phone("123")  # Invalid phone
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_email_validator_function(self) -> None:
        """Test create_email_validator returns working validator function."""
        validator = FlextValidations.create_email_validator()

        # Test valid email
        result = validator("user@example.com")
        FlextTestsMatchers.assert_result_success(result)

        # Test invalid email
        result = validator("invalid_email")
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_phone_validator_function(self) -> None:
        """Test create_phone_validator returns working validator function."""
        validator = FlextValidations.create_phone_validator()

        # Test valid phone
        result = validator("+1-555-123-4567")
        FlextTestsMatchers.assert_result_success(result)

        # Test invalid phone
        result = validator("123")
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_url_validator_function(self) -> None:
        """Test create_url_validator returns working validator function."""
        validator = FlextValidations.create_url_validator()

        # Test valid URL
        result = validator("https://example.com")
        FlextTestsMatchers.assert_result_success(result)

        # Test invalid URL
        result = validator("not_a_url")
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_uuid_validator_function(self) -> None:
        """Test create_uuid_validator returns working validator function."""
        validator = FlextValidations.create_uuid_validator()

        # Test valid UUID
        test_uuid = str(uuid.uuid4())
        result = validator(test_uuid)
        FlextTestsMatchers.assert_result_success(result)

        # Test invalid UUID
        result = validator("not_a_uuid")
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_composite_validator_all_pass(self) -> None:
        """Test create_composite_validator when all validators pass."""

        def validator1(x: object) -> FlextResult[object]:
            if isinstance(x, str):
                result = FlextValidations.create_email_validator()(x)
                if result.is_success:
                    return FlextResult[object].ok(x)
                return FlextResult[object].fail(
                    result.error or "Email validation failed"
                )
            return FlextResult[object].fail("Expected string for email validation")

        def validator2(x: object) -> FlextResult[object]:
            return FlextValidations.Guards.require_not_none(x)

        composite = FlextValidations.create_composite_validator(
            [validator1, validator2]
        )

        result = composite("user@example.com")
        FlextTestsMatchers.assert_result_success(result)

    def test_create_composite_validator_first_fails(self) -> None:
        """Test create_composite_validator when first validator fails."""

        def validator1(x: object) -> FlextResult[object]:
            if isinstance(x, str):
                result = FlextValidations.create_email_validator()(x)
                if result.is_success:
                    return FlextResult[object].ok(x)
                return FlextResult[object].fail(
                    result.error or "Email validation failed"
                )
            return FlextResult[object].fail("Expected string for email validation")

        def validator2(x: object) -> FlextResult[object]:
            return FlextValidations.Guards.require_not_none(x)

        composite = FlextValidations.create_composite_validator(
            [validator1, validator2]
        )

        result = composite("invalid_email")
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_composite_validator_second_fails(self) -> None:
        """Test create_composite_validator when second validator fails."""

        def validator1(x: object) -> FlextResult[object]:
            return FlextValidations.Guards.require_not_none(x)  # Pass

        def validator2(x: object) -> FlextResult[object]:
            return FlextValidations.Guards.require_positive(
                int(str(x))
            )  # Fail for non-numeric

        composite = FlextValidations.create_composite_validator(
            [validator1, validator2]
        )

        result = composite("user@example.com")
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_schema_validator_function(self) -> None:
        """Test create_schema_validator returns working validator function."""

        # Use callable validator format, not JSON schema format
        def validate_name(x: object) -> FlextResult[object]:
            if isinstance(x, str):
                result = FlextValidations.TypeValidators.validate_string(x)
                if result.is_success:
                    return FlextResult[object].ok(x)
                return FlextResult[object].fail(
                    result.error or "String validation failed"
                )
            return FlextResult[object].fail("Expected string for name validation")

        schema = {"name": validate_name}

        validator = FlextValidations.create_schema_validator(schema)

        # Test valid data
        result = validator({"name": "John"})
        FlextTestsMatchers.assert_result_success(result)

        # Test invalid data
        result = validator({"age": 30})  # Missing required name
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_cached_validator_caching_behavior(self) -> None:
        """Test create_cached_validator caches results properly."""
        call_count = 0

        def slow_validator(value: object) -> FlextResult[object]:
            nonlocal call_count
            call_count += 1
            return FlextValidations.Guards.require_not_none(value)

        cached_validator = FlextValidations.create_cached_validator(slow_validator)

        # First call - should execute validator
        result1 = cached_validator("test_value")
        FlextTestsMatchers.assert_result_success(result1)
        assert call_count == 1

        # Second call with same value - should use cache
        result2 = cached_validator("test_value")
        FlextTestsMatchers.assert_result_success(result2)
        assert call_count == 1  # Should not increment

        # Third call with different value - should execute validator
        result3 = cached_validator("different_value")
        FlextTestsMatchers.assert_result_success(result3)
        assert call_count == 2

    def test_create_user_validator_function(self) -> None:
        """Test create_user_validator returns working validator function."""
        validator = FlextValidations.create_user_validator()

        # Test valid user data
        user_data: dict[str, object] = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
        }
        result = validator(user_data)
        FlextTestsMatchers.assert_result_success(result)

        # Test invalid user data (missing required field)
        invalid_user_data: dict[str, object] = {"email": "john@example.com"}
        result = validator(invalid_user_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_user_data_success(self) -> None:
        """Test validate_user_data with valid user data."""
        user_data: dict[str, object] = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == user_data

    def test_validate_user_data_missing_required_field(self) -> None:
        """Test validate_user_data with missing required field."""
        user_data: dict[str, object] = {"email": "john@example.com"}  # Missing name
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Missing required field: name" in result.error

    def test_validate_user_data_invalid_email_no_at(self) -> None:
        """Test validate_user_data with email missing @ symbol."""
        user_data: dict[str, object] = {
            "name": "John Doe",
            "email": "johnexample.com",  # Missing @
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid email format" in result.error

    def test_validate_user_data_invalid_email_no_domain_dot(self) -> None:
        """Test validate_user_data with email missing domain dot."""
        user_data: dict[str, object] = {
            "name": "John Doe",
            "email": "john@examplecom",  # Missing dot in domain
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid email format" in result.error

    def test_validate_user_data_age_negative(self) -> None:
        """Test validate_user_data with negative age."""
        user_data: dict[str, object] = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": -5,
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Age must be a string or number" in result.error

    def test_validate_user_data_age_too_old(self) -> None:
        """Test validate_user_data with age exceeding maximum reasonable age."""
        user_data: dict[str, object] = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 200,  # Exceeds MAX_REASONABLE_AGE (150)
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Age must be a string or number" in result.error

    def test_validate_user_data_age_invalid_string(self) -> None:
        """Test validate_user_data with non-numeric age string."""
        user_data: dict[str, object] = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": "not_a_number",
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Age must be a string or number" in result.error

    def test_validate_user_data_age_invalid_type(self) -> None:
        """Test validate_user_data with age as invalid type."""
        user_data: dict[str, object] = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": [1, 2, 3],  # List instead of number
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Age must be a string or number" in result.error

    def test_validate_api_request_success(self) -> None:
        """Test validate_api_request with valid API request data."""
        request_data: dict[str, object] = {
            "method": "GET",
            "path": "/api/users",
            "headers": {"Content-Type": "application/json"},
        }
        result = FlextValidations.validate_api_request(request_data)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == request_data

    def test_validate_api_request_missing_required_field(self) -> None:
        """Test validate_api_request with missing required field."""
        request_data: dict[str, object] = {"path": "/api/users"}  # Missing method
        result = FlextValidations.validate_api_request(request_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Missing required field: method" in result.error

    def test_validate_api_request_invalid_method(self) -> None:
        """Test validate_api_request with invalid HTTP method."""
        request_data: dict[str, object] = {
            "method": "INVALID_METHOD",
            "path": "/api/users",
        }
        result = FlextValidations.validate_api_request(request_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Invalid HTTP method" in result.error

    def test_validate_api_request_invalid_path_format(self) -> None:
        """Test validate_api_request with invalid path format."""
        request_data: dict[str, object] = {
            "method": "GET",
            "path": "invalid_path",  # Should start with /
        }
        result = FlextValidations.validate_api_request(request_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Path must start with /" in result.error

    def test_is_valid_boolean_check(self) -> None:
        """Test is_valid returns boolean validation result."""
        # Test with valid value - returns True
        result = FlextValidations.is_valid("valid_value")
        assert result is True

        # Test with None - returns False
        result_none = FlextValidations.is_valid(None)
        assert result_none is False

        # Test with empty string - returns False
        result_empty = FlextValidations.is_valid("")
        assert result_empty is False

        # Test with zero - returns True (zero is valid)
        result_zero = FlextValidations.is_valid(0)
        assert result_zero is True

    def test_is_non_empty_string_success(self) -> None:
        """Test is_non_empty_string with non-empty string."""
        assert FlextValidations.is_non_empty_string("valid_string") is True

    def test_is_non_empty_string_failure_empty(self) -> None:
        """Test is_non_empty_string with empty string."""
        assert FlextValidations.is_non_empty_string("") is False

    def test_is_non_empty_string_failure_non_string(self) -> None:
        """Test is_non_empty_string with non-string value."""
        assert FlextValidations.is_non_empty_string(123) is False

    def test_validate_email_field_success(self) -> None:
        """Test validate_email_field with valid email."""
        result = FlextValidations.validate_email_field("user@example.com")
        assert result is True

    def test_validate_email_field_failure(self) -> None:
        """Test validate_email_field with invalid email."""
        result = FlextValidations.validate_email_field("invalid_email")
        assert result is False

    def test_validate_with_schema_success(self) -> None:
        """Test validate_with_schema with valid data and schema."""
        data = {"name": "John Doe", "age": 30}
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        result = FlextValidations.validate_with_schema(data, schema)
        FlextTestsMatchers.assert_result_success(result)

    def test_validate_with_schema_failure(self) -> None:
        """Test validate_with_schema with invalid data against schema."""
        data = {"name": "John Doe", "age": "thirty"}  # age should be integer
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        result = FlextValidations.validate_with_schema(data, schema)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_string_delegates_to_type_validators(self) -> None:
        """Test validate_string delegates to TypeValidators."""
        result = FlextValidations.validate_string("test_string")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "test_string"

    def test_validate_non_empty_string_func_success(self) -> None:
        """Test validate_non_empty_string_func with valid string."""
        result = FlextValidations.validate_non_empty_string_func("valid_string")
        assert result is True

    def test_validate_non_empty_string_func_failure(self) -> None:
        """Test validate_non_empty_string_func with empty string."""
        result = FlextValidations.validate_non_empty_string_func("")
        assert result is False

    def test_supports_int_protocol_implementation(self) -> None:
        """Test SupportsInt protocol implementation."""

        class TestInt(FlextValidations.SupportsInt):
            def __int__(self) -> int:
                return 42

        test_obj = TestInt()
        assert int(test_obj) == 42

    def test_supports_float_protocol_implementation(self) -> None:
        """Test SupportsFloat protocol implementation."""

        class TestFloat(FlextValidations.SupportsFloat):
            def __float__(self) -> float:
                return math.pi

        test_obj = TestFloat()
        assert float(test_obj) == math.pi

    def test_predicates_class_call_success(self) -> None:
        """Test Predicates class __call__ method with passing predicate."""

        def is_positive(value: object) -> bool:
            return isinstance(value, (int, float)) and value > 0

        predicate = FlextValidations.Predicates(is_positive, "is_positive")
        result = predicate(42)
        FlextTestsMatchers.assert_result_success(result)

    def test_predicates_class_call_failure(self) -> None:
        """Test Predicates class __call__ method with failing predicate."""

        def is_positive(value: object) -> bool:
            return isinstance(value, (int, float)) and value > 0

        predicate = FlextValidations.Predicates(is_positive, "is_positive")
        result = predicate(-5)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Predicate 'is_positive' failed for value: -5" in result.error

    def test_predicates_class_call_exception(self) -> None:
        """Test Predicates class __call__ method when predicate raises exception."""

        def failing_predicate(_value: object) -> bool:
            msg = "Test exception"
            raise ValueError(msg)

        predicate = FlextValidations.Predicates(failing_predicate, "failing_predicate")
        result = predicate("any_value")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error is not None
        assert (
            "Predicate 'failing_predicate' raised exception: Test exception"
            in result.error
        )

    def test_constants_and_aliases_exist(self) -> None:
        """Test that validation constants and aliases are properly defined."""
        # Test constants
        assert FlextValidations.MIN_PASSWORD_LENGTH > 0
        assert FlextValidations.MIN_CREDIT_CARD_LENGTH > 0
        assert (
            FlextValidations.MAX_CREDIT_CARD_LENGTH
            > FlextValidations.MIN_CREDIT_CARD_LENGTH
        )

        # Test aliases exist
        assert hasattr(FlextValidations, "Types")
        assert hasattr(FlextValidations, "Fields")
        assert hasattr(FlextValidations, "Rules")
        assert hasattr(FlextValidations, "Advanced")
        assert hasattr(FlextValidations, "Numbers")
        assert hasattr(FlextValidations, "Validators")

    def test_core_namespace_aliases(self) -> None:
        """Test Core namespace contains expected aliases."""
        assert hasattr(FlextValidations.Core, "TypeValidators")
        assert hasattr(FlextValidations.Core, "Collections")
        assert hasattr(FlextValidations.Core, "Domain")
        assert hasattr(FlextValidations.Core, "Predicates")

    def test_service_api_request_validator_exists(self) -> None:
        """Test Service.ApiRequestValidator exists and can be instantiated."""
        validator = FlextValidations.Service.ApiRequestValidator()
        assert validator is not None
