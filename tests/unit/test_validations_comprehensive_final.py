"""FlextValidations comprehensive tests with fully corrected API usage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from flext_core import FlextResult, FlextValidations
from flext_tests import FlextTestsMatchers


class TestFlextValidationsComprehensive:
    """Comprehensive FlextValidations tests with fully corrected API usage targeting 33 uncovered lines."""

    def test_type_validators_validate_string_success(self) -> None:
        """Test TypeValidators.validate_string success case."""
        result = FlextValidations.TypeValidators.validate_string("hello")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "hello"

    def test_type_validators_validate_string_failure(self) -> None:
        """Test TypeValidators.validate_string failure case."""
        result = FlextValidations.TypeValidators.validate_string(123)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_type_validators_validate_integer_valid_input(self) -> None:
        """Test TypeValidators.validate_integer with valid input."""
        result = FlextValidations.TypeValidators.validate_integer(50)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 50

    def test_type_validators_validate_integer_invalid_string(self) -> None:
        """Test TypeValidators.validate_integer with invalid string."""
        result = FlextValidations.TypeValidators.validate_integer("not_a_number")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Cannot convert" in result.error

    def test_type_validators_validate_integer_non_numeric_type(self) -> None:
        """Test TypeValidators.validate_integer with non-numeric type."""
        result = FlextValidations.TypeValidators.validate_integer([1, 2, 3])
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_type_validators_validate_integer_string_conversion(self) -> None:
        """Test TypeValidators.validate_integer string conversion."""
        result = FlextValidations.TypeValidators.validate_integer("42")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_type_validators_validate_float_success(self) -> None:
        """Test TypeValidators.validate_float success case."""
        result = FlextValidations.TypeValidators.validate_float(math.pi)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == math.pi

    def test_type_validators_validate_float_string_conversion(self) -> None:
        """Test TypeValidators.validate_float string conversion."""
        result = FlextValidations.TypeValidators.validate_float("3.14")
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
        """Test TypeValidators.validate_dict success case."""
        test_dict = {"key": "value"}
        result = FlextValidations.TypeValidators.validate_dict(test_dict)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == test_dict

    def test_type_validators_validate_dict_failure(self) -> None:
        """Test TypeValidators.validate_dict failure case."""
        result = FlextValidations.TypeValidators.validate_dict("not_a_dict")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_type_validators_validate_list_success(self) -> None:
        """Test TypeValidators.validate_list success case."""
        test_list = [1, 2, 3]
        result = FlextValidations.TypeValidators.validate_list(test_list)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == test_list

    def test_type_validators_validate_list_failure(self) -> None:
        """Test TypeValidators.validate_list failure case."""
        result = FlextValidations.TypeValidators.validate_list("not_a_list")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Type mismatch" in result.error

    def test_field_validators_validate_email_success(self) -> None:
        """Test FieldValidators.validate_email success case."""
        result = FlextValidations.FieldValidators.validate_email("user@example.com")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "user@example.com"

    def test_field_validators_validate_email_failure(self) -> None:
        """Test FieldValidators.validate_email failure case."""
        result = FlextValidations.FieldValidators.validate_email("invalid-email")
        FlextTestsMatchers.assert_result_failure(result)

    def test_field_validators_validate_phone_success(self) -> None:
        """Test FieldValidators.validate_phone success case."""
        result = FlextValidations.FieldValidators.validate_phone("+1-555-123-4567")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "+1-555-123-4567"

    def test_field_validators_validate_phone_failure(self) -> None:
        """Test FieldValidators.validate_phone failure case."""
        result = FlextValidations.FieldValidators.validate_phone("123")
        FlextTestsMatchers.assert_result_failure(result)

    def test_field_validators_validate_url_success(self) -> None:
        """Test FieldValidators.validate_url success case."""
        result = FlextValidations.FieldValidators.validate_url("https://example.com")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "https://example.com"

    def test_field_validators_validate_url_failure(self) -> None:
        """Test FieldValidators.validate_url failure case."""
        result = FlextValidations.FieldValidators.validate_url("not_a_url")
        FlextTestsMatchers.assert_result_failure(result)

    def test_field_validators_validate_uuid_success(self) -> None:
        """Test FieldValidators.validate_uuid success case."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        result = FlextValidations.FieldValidators.validate_uuid(uuid_str)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == uuid_str

    def test_field_validators_validate_uuid_failure(self) -> None:
        """Test FieldValidators.validate_uuid failure case."""
        result = FlextValidations.FieldValidators.validate_uuid("not-a-uuid")
        FlextTestsMatchers.assert_result_failure(result)

    def test_business_validators_validate_string_field_success(self) -> None:
        """Test BusinessValidators.validate_string_field success case."""
        result = FlextValidations.BusinessValidators.validate_string_field(
            "hello", min_length=3, max_length=10
        )
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "hello"

    def test_business_validators_validate_string_field_too_short(self) -> None:
        """Test BusinessValidators.validate_string_field too short."""
        result = FlextValidations.BusinessValidators.validate_string_field(
            "hi", min_length=5, max_length=20
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "String too short" in result.error

    def test_business_validators_validate_string_field_too_long(self) -> None:
        """Test BusinessValidators.validate_string_field too long."""
        result = FlextValidations.BusinessValidators.validate_string_field(
            "this is a very long string", min_length=5, max_length=20
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "String too long" in result.error

    def test_business_validators_validate_numeric_field_integer(self) -> None:
        """Test BusinessValidators.validate_numeric_field with integer."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(42)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_business_validators_validate_numeric_field_float(self) -> None:
        """Test BusinessValidators.validate_numeric_field with float."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(math.pi)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == math.pi

    def test_business_validators_validate_numeric_field_string_conversion(self) -> None:
        """Test BusinessValidators.validate_numeric_field string conversion."""
        result = FlextValidations.BusinessValidators.validate_numeric_field("42")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

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
        assert "too small" in result.error

    def test_business_validators_validate_numeric_field_above_max(self) -> None:
        """Test BusinessValidators.validate_numeric_field above maximum."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(
            150, min_value=10, max_value=100
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "too large" in result.error

    def test_business_validators_validate_numeric_field_invalid_type(self) -> None:
        """Test BusinessValidators.validate_numeric_field invalid type."""
        result = FlextValidations.BusinessValidators.validate_numeric_field(
            "not_a_number"
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Value must be numeric" in result.error

    def test_business_validators_validate_range_success(self) -> None:
        """Test BusinessValidators.validate_range success case."""
        result = FlextValidations.BusinessValidators.validate_range(50, 10, 100)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 50

    def test_business_validators_validate_range_failure(self) -> None:
        """Test BusinessValidators.validate_range failure case."""
        result = FlextValidations.BusinessValidators.validate_range(150, 10, 100)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "out of range" in result.error

    def test_business_validators_validate_password_strength_success(self) -> None:
        """Test BusinessValidators.validate_password_strength success case."""
        result = FlextValidations.BusinessValidators.validate_password_strength(
            "StrongP@ssw0rd"
        )
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "StrongP@ssw0rd"

    def test_business_validators_validate_password_strength_failure(self) -> None:
        """Test BusinessValidators.validate_password_strength failure case."""
        result = FlextValidations.BusinessValidators.validate_password_strength("weak")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "at least 8 characters" in result.error

    def test_business_validators_validate_credit_card_success(self) -> None:
        """Test BusinessValidators.validate_credit_card success case."""
        result = FlextValidations.BusinessValidators.validate_credit_card(
            "4111111111111111"
        )
        FlextTestsMatchers.assert_result_success(result)

    def test_business_validators_validate_credit_card_failure(self) -> None:
        """Test BusinessValidators.validate_credit_card failure case."""
        result = FlextValidations.BusinessValidators.validate_credit_card("1234")
        FlextTestsMatchers.assert_result_failure(result)

    def test_business_validators_validate_ipv4_success(self) -> None:
        """Test BusinessValidators.validate_ipv4 success case."""
        result = FlextValidations.BusinessValidators.validate_ipv4("192.168.1.1")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "192.168.1.1"

    def test_business_validators_validate_ipv4_failure(self) -> None:
        """Test BusinessValidators.validate_ipv4 failure case."""
        result = FlextValidations.BusinessValidators.validate_ipv4("300.300.300.300")
        FlextTestsMatchers.assert_result_failure(result)

    def test_business_validators_validate_date_success(self) -> None:
        """Test BusinessValidators.validate_date success case."""
        result = FlextValidations.BusinessValidators.validate_date("2023-12-25")
        FlextTestsMatchers.assert_result_success(result)

    def test_business_validators_validate_date_failure(self) -> None:
        """Test BusinessValidators.validate_date failure case."""
        result = FlextValidations.BusinessValidators.validate_date("invalid-date")
        FlextTestsMatchers.assert_result_failure(result)

    def test_business_validators_validate_json_success(self) -> None:
        """Test BusinessValidators.validate_json success case."""
        json_str = '{"key": "value"}'
        result = FlextValidations.BusinessValidators.validate_json(json_str)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == {"key": "value"}

    def test_business_validators_validate_json_failure(self) -> None:
        """Test BusinessValidators.validate_json failure case."""
        result = FlextValidations.BusinessValidators.validate_json("invalid json")
        FlextTestsMatchers.assert_result_failure(result)

    def test_guards_require_not_none_success(self) -> None:
        """Test Guards.require_not_none success case."""
        result = FlextValidations.Guards.require_not_none("not_none")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "not_none"

    def test_guards_require_not_none_failure(self) -> None:
        """Test Guards.require_not_none failure case."""
        result = FlextValidations.Guards.require_not_none(None)
        FlextTestsMatchers.assert_result_failure(result)

    def test_guards_require_positive_success(self) -> None:
        """Test Guards.require_positive success case."""
        result = FlextValidations.Guards.require_positive(42)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_guards_require_positive_failure(self) -> None:
        """Test Guards.require_positive failure case."""
        result = FlextValidations.Guards.require_positive(-5)
        FlextTestsMatchers.assert_result_failure(result)

    def test_guards_require_in_range_success(self) -> None:
        """Test Guards.require_in_range success case."""
        result = FlextValidations.Guards.require_in_range(50, 10, 100)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 50

    def test_guards_require_in_range_failure_below(self) -> None:
        """Test Guards.require_in_range failure below range."""
        result = FlextValidations.Guards.require_in_range(5, 10, 100)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "out of range" in result.error

    def test_guards_require_in_range_failure_above(self) -> None:
        """Test Guards.require_in_range failure above range."""
        result = FlextValidations.Guards.require_in_range(150, 10, 100)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "out of range" in result.error

    def test_guards_require_non_empty_success_string(self) -> None:
        """Test Guards.require_non_empty success with string."""
        result = FlextValidations.Guards.require_non_empty("hello")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "hello"

    def test_guards_require_non_empty_success_list(self) -> None:
        """Test Guards.require_non_empty success with list."""
        result = FlextValidations.Guards.require_non_empty([1, 2, 3])
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == [1, 2, 3]

    def test_guards_require_non_empty_failure_empty_string(self) -> None:
        """Test Guards.require_non_empty failure with empty string."""
        result = FlextValidations.Guards.require_non_empty("")
        FlextTestsMatchers.assert_result_failure(result)

    def test_guards_require_non_empty_failure_empty_list(self) -> None:
        """Test Guards.require_non_empty failure with empty list."""
        result = FlextValidations.Guards.require_non_empty([])
        FlextTestsMatchers.assert_result_failure(result)

    def test_guards_is_list_of_success(self) -> None:
        """Test Guards.is_list_of success case."""
        result = FlextValidations.Guards.is_list_of([1, 2, 3], int)
        assert result is True

    def test_guards_is_list_of_failure(self) -> None:
        """Test Guards.is_list_of failure case."""
        result = FlextValidations.Guards.is_list_of([1, "2", 3], int)
        assert result is False

    def test_guards_is_dict_of_success(self) -> None:
        """Test Guards.is_dict_of success case."""
        test_dict = {"a": "value1", "b": "value2"}
        result = FlextValidations.Guards.is_dict_of(test_dict, str)
        assert result is True

    def test_guards_is_dict_of_failure(self) -> None:
        """Test Guards.is_dict_of failure case."""
        test_dict = {"a": 1, "b": "value2"}
        result = FlextValidations.Guards.is_dict_of(test_dict, str)
        assert result is False

    def test_schema_validators_validate_with_pydantic_schema_success(self) -> None:
        """Test SchemaValidators.validate_with_pydantic_schema success case."""

        class TestModel(BaseModel):
            name: str
            age: int

        data = {"name": "John", "age": 30}
        result = FlextValidations.SchemaValidators.validate_with_pydantic_schema(
            data, TestModel
        )
        FlextTestsMatchers.assert_result_success(result)
        assert result.value.name == "John"
        assert result.value.age == 30

    def test_schema_validators_validate_with_pydantic_schema_failure(self) -> None:
        """Test SchemaValidators.validate_with_pydantic_schema failure case."""

        class TestModel(BaseModel):
            name: str
            age: int

        data = {"name": "John"}  # Missing age
        result = FlextValidations.SchemaValidators.validate_with_pydantic_schema(
            data, TestModel
        )
        FlextTestsMatchers.assert_result_failure(result)

    def test_schema_validators_validate_schema_success(self) -> None:
        """Test SchemaValidators.validate_schema success case with function validators."""
        # Schema expects validator functions, not JSON schema
        schema = {
            "name": lambda x: FlextResult[str].ok(str(x))
            if isinstance(x, str)
            else FlextResult[str].fail("Must be string"),
            "age": lambda x: FlextResult[int].ok(int(x))
            if isinstance(x, int)
            else FlextResult[int].fail("Must be int"),
        }
        data = {"name": "John", "age": 30}
        result = FlextValidations.SchemaValidators.validate_schema(data, schema)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == data

    def test_schema_validators_validate_schema_failure(self) -> None:
        """Test SchemaValidators.validate_schema failure case."""
        schema = {
            "name": lambda x: FlextResult[str].ok(str(x))
            if isinstance(x, str)
            else FlextResult[str].fail("Must be string"),
            "age": lambda x: FlextResult[int].ok(int(x))
            if isinstance(x, int)
            else FlextResult[int].fail("Must be int"),
        }
        data = {"name": "John"}  # Missing age
        result = FlextValidations.SchemaValidators.validate_schema(data, schema)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_string_delegates_to_type_validators(self) -> None:
        """Test validate_string delegates to TypeValidators."""
        result = FlextValidations.validate_string("hello")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "hello"

    def test_validate_number_delegates_to_business_validators(self) -> None:
        """Test validate_number delegates to BusinessValidators."""
        result = FlextValidations.validate_number(42)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_validate_number_string_without_dot(self) -> None:
        """Test validate_number with string without dot."""
        result = FlextValidations.validate_number("42")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_validate_number_string_with_dot(self) -> None:
        """Test validate_number with string with dot."""
        result = FlextValidations.validate_number("3.14")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == math.pi

    def test_validate_number_string_invalid(self) -> None:
        """Test validate_number with invalid string."""
        result = FlextValidations.validate_number("not_a_number")
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_user_data_success(self) -> None:
        """Test validate_user_data success case."""
        user_data = {"name": "John Doe", "email": "john@example.com", "age": 30}
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == user_data

    def test_validate_user_data_missing_required_field(self) -> None:
        """Test validate_user_data missing required field."""
        user_data = {"name": "John Doe"}  # Missing email (required)
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "email" in result.error

    def test_validate_user_data_invalid_email_no_at(self) -> None:
        """Test validate_user_data with invalid email (no @)."""
        user_data = {
            "name": "John Doe",
            "email": "johnexample.com",  # No @
            "age": 30,
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_user_data_invalid_email_no_domain_dot(self) -> None:
        """Test validate_user_data with invalid email (no domain dot)."""
        user_data = {
            "name": "John Doe",
            "email": "john@example",  # No dot in domain
            "age": 30,
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_user_data_age_negative(self) -> None:
        """Test validate_user_data with negative age."""
        user_data = {"name": "John Doe", "email": "john@example.com", "age": -5}
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_user_data_age_too_old(self) -> None:
        """Test validate_user_data with age too old."""
        user_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 151,  # Above MAX_REASONABLE_AGE (150)
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_user_data_age_invalid_type(self) -> None:
        """Test validate_user_data with invalid age type."""
        user_data = {"name": "John Doe", "email": "john@example.com", "age": "thirty"}
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_user_data_age_invalid_string(self) -> None:
        """Test validate_user_data with age as invalid string."""
        user_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": "not_a_number",
        }
        result = FlextValidations.validate_user_data(user_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_api_request_success(self) -> None:
        """Test validate_api_request success case."""
        request_data = {
            "method": "GET",
            "path": "/api/users",
            "headers": {"Content-Type": "application/json"},
        }
        result = FlextValidations.validate_api_request(request_data)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == request_data

    def test_validate_api_request_missing_required_field(self) -> None:
        """Test validate_api_request missing required field."""
        request_data = {"method": "GET"}  # Missing path
        result = FlextValidations.validate_api_request(request_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_api_request_invalid_method(self) -> None:
        """Test validate_api_request with invalid method."""
        request_data = {"method": "INVALID", "path": "/api/users", "headers": {}}
        result = FlextValidations.validate_api_request(request_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_api_request_invalid_path_format(self) -> None:
        """Test validate_api_request with invalid path format."""
        request_data = {
            "method": "GET",
            "path": "invalid_path",  # Should start with /
            "headers": {},
        }
        result = FlextValidations.validate_api_request(request_data)
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_email_validator_function(self) -> None:
        """Test create_email_validator function."""
        validator = FlextValidations.create_email_validator()
        result = validator("user@example.com")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "user@example.com"

    def test_create_phone_validator_function(self) -> None:
        """Test create_phone_validator function."""
        validator = FlextValidations.create_phone_validator()
        result = validator("+1-555-123-4567")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "+1-555-123-4567"

    def test_create_url_validator_function(self) -> None:
        """Test create_url_validator function."""
        validator = FlextValidations.create_url_validator()
        result = validator("https://example.com")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "https://example.com"

    def test_create_uuid_validator_function(self) -> None:
        """Test create_uuid_validator function."""
        validator = FlextValidations.create_uuid_validator()
        result = validator("550e8400-e29b-41d4-a716-446655440000")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "550e8400-e29b-41d4-a716-446655440000"

    def test_create_composite_validator_all_pass(self) -> None:
        """Test create_composite_validator when all validators pass."""

        def validator1(x: object) -> FlextResult[object]:
            return FlextValidations.Guards.require_not_none(x)

        def validator2(x: object) -> FlextResult[str]:
            return FlextValidations.TypeValidators.validate_string(x)

        composite = FlextValidations.create_composite_validator(
            [validator1, validator2]
        )
        result = composite("test_string")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "test_string"

    def test_create_composite_validator_first_fails(self) -> None:
        """Test create_composite_validator when first validator fails."""

        def validator1(x: object) -> FlextResult[object]:
            return FlextValidations.Guards.require_not_none(x)

        def validator2(x: object) -> FlextResult[str]:
            return FlextValidations.TypeValidators.validate_string(x)

        composite = FlextValidations.create_composite_validator(
            [validator1, validator2]
        )
        result = composite(None)
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_composite_validator_second_fails(self) -> None:
        """Test create_composite_validator when second validator fails."""

        def validator1(x: object) -> FlextResult[object]:
            return FlextValidations.Guards.require_not_none(x)

        def validator2(x: object) -> FlextResult[str]:
            return FlextValidations.TypeValidators.validate_string(x)

        composite = FlextValidations.create_composite_validator(
            [validator1, validator2]
        )
        result = composite(123)  # Not None but not string
        FlextTestsMatchers.assert_result_failure(result)

    def test_create_schema_validator_function(self) -> None:
        """Test create_schema_validator function with validator functions."""
        schema = {
            "name": lambda x: FlextResult[str].ok(str(x))
            if isinstance(x, str)
            else FlextResult[str].fail("Must be string")
        }
        validator = FlextValidations.create_schema_validator(schema)
        result = validator({"name": "John"})
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == {"name": "John"}

    def test_create_cached_validator_caching_behavior(self) -> None:
        """Test create_cached_validator caches results properly."""
        call_count = 0

        def slow_validator(value: object) -> bool:
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

    def test_create_user_validator_function(self) -> None:
        """Test create_user_validator function."""
        validator = FlextValidations.create_user_validator()
        user_data = {"name": "John Doe", "email": "john@example.com", "age": 30}
        result = validator(user_data)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == user_data

    def test_validate_phone_with_locale_parameter(self) -> None:
        """Test validate_phone with locale functionality."""
        result = FlextValidations.validate_phone("+1-555-123-4567", locale="US")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "+1-555-123-4567"

    def test_validate_phone_with_failure_result(self) -> None:
        """Test validate_phone with failure case."""
        result = FlextValidations.validate_phone("invalid_phone", locale="US")
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_with_schema_success(self) -> None:
        """Test validate_with_schema success case using validator functions."""
        schema = {
            "value": lambda x: FlextResult[str].ok(str(x))
            if isinstance(x, str)
            else FlextResult[str].fail("Must be string")
        }
        result = FlextValidations.validate_with_schema({"value": "test"}, schema)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == {"value": "test"}

    def test_validate_with_schema_failure(self) -> None:
        """Test validate_with_schema failure case."""
        schema = {
            "value": lambda x: FlextResult[str].ok(str(x))
            if isinstance(x, str)
            else FlextResult[str].fail("Must be string")
        }
        result = FlextValidations.validate_with_schema({"value": 123}, schema)
        FlextTestsMatchers.assert_result_failure(result)

    def test_validate_email_field_success(self) -> None:
        """Test validate_email_field returns boolean success."""
        result = FlextValidations.validate_email_field("user@example.com")
        assert result is True

    def test_validate_email_field_failure(self) -> None:
        """Test validate_email_field returns boolean failure."""
        result = FlextValidations.validate_email_field("invalid-email")
        assert result is False

    def test_validate_non_empty_string_func_success(self) -> None:
        """Test validate_non_empty_string_func function returns boolean success."""
        result = FlextValidations.validate_non_empty_string_func("test")
        assert result is True

    def test_validate_non_empty_string_func_failure(self) -> None:
        """Test validate_non_empty_string_func function returns boolean failure."""
        result = FlextValidations.validate_non_empty_string_func("")
        assert result is False

    def test_is_non_empty_string_success(self) -> None:
        """Test is_non_empty_string success case."""
        result = FlextValidations.is_non_empty_string("test")
        assert result is True

    def test_is_non_empty_string_failure_empty(self) -> None:
        """Test is_non_empty_string failure with empty string."""
        result = FlextValidations.is_non_empty_string("")
        assert result is False

    def test_is_non_empty_string_failure_non_string(self) -> None:
        """Test is_non_empty_string failure with non-string."""
        result = FlextValidations.is_non_empty_string(123)
        assert result is False

    def test_is_valid_predicate_function(self) -> None:
        """Test is_valid predicate function behavior."""
        # is_valid with empty string returns False (empty strings are invalid)
        result = FlextValidations.is_valid("")
        assert result is False

    def test_predicates_class_instantiation(self) -> None:
        """Test Predicates class requires a function parameter."""

        # Predicates requires a func parameter
        def test_func(x: object) -> bool:
            return x is not None

        predicates = FlextValidations.Predicates(test_func)
        result = predicates("test_value")
        FlextTestsMatchers.assert_result_success(result)

    def test_predicates_class_call_failure(self) -> None:
        """Test Predicates class __call__ failure."""

        def always_fail(_x: object) -> bool:
            return False

        predicates = FlextValidations.Predicates(always_fail)
        result = predicates("test")
        FlextTestsMatchers.assert_result_failure(result)

    def test_predicates_class_call_exception(self) -> None:
        """Test Predicates class __call__ with exception handling."""

        def error_func(_x: object) -> None:
            msg = "Test error"
            raise ValueError(msg)

        predicates = FlextValidations.Predicates(error_func)
        result = predicates("test")
        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "raised exception" in result.error

    def test_protocol_classes_exist(self) -> None:
        """Test that Protocol classes exist but cannot be instantiated."""
        # Protocols exist as classes but can't be instantiated
        assert hasattr(FlextValidations, "SupportsInt")
        assert hasattr(FlextValidations, "SupportsFloat")

        # They should be Protocol classes
        assert hasattr(FlextValidations.SupportsInt, "__int__")
        assert hasattr(FlextValidations.SupportsFloat, "__float__")

    def test_constants_and_aliases_exist(self) -> None:
        """Test that constants and aliases are properly defined."""
        # Test validators exist
        assert hasattr(FlextValidations, "TypeValidators")
        assert hasattr(FlextValidations, "FieldValidators")
        assert hasattr(FlextValidations, "BusinessValidators")
        assert hasattr(FlextValidations, "Guards")
        assert hasattr(FlextValidations, "SchemaValidators")

    def test_core_namespace_implementation(self) -> None:
        """Test core namespace functionality."""
        # Test that Core namespace exists and provides access
        assert hasattr(FlextValidations, "Core")
        # Core might have different structure than expected
        assert hasattr(FlextValidations.Core, "TypeValidators")

    def test_service_api_request_validator_exists(self) -> None:
        """Test that service API request validator functionality exists."""
        # Test that the validate_api_request method exists and can be called
        request_data = {"method": "GET", "path": "/api/test", "headers": {}}
        result = FlextValidations.validate_api_request(request_data)
        # Should return a FlextResult
        assert hasattr(result, "is_success") or hasattr(result, "success")
