"""Comprehensive tests for FLEXT validation pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from flext_core import FlextResult, FlextValidations
from flext_core.typings import FlextTypes

# =============================================================================
# TEST VALIDATION UTILITIES
# =============================================================================


class TestFlextValidations:
    """Test FlextValidation functionality."""

    def test_validation_utilities_exist(self) -> None:
        """Test that validation utilities are available."""
        assert hasattr(FlextValidations, "validate_non_empty_string_func")
        assert hasattr(FlextValidations, "validate_email_field")
        assert hasattr(FlextValidations, "validate_numeric_field")

    def test_is_non_empty_string_valid(self) -> None:
        """Test non-empty string validation with valid input."""
        assert FlextValidations.validate_non_empty_string_func("test") is True
        assert FlextValidations.validate_non_empty_string_func("hello world") is True

    def test_is_non_empty_string_invalid(self) -> None:
        """Test non-empty string validation with invalid input."""
        assert FlextValidations.validate_non_empty_string_func("") is False
        assert FlextValidations.validate_non_empty_string_func(None) is False
        assert FlextValidations.validate_non_empty_string_func(123) is False

    def test_is_callable_valid(self) -> None:
        """Test callable validation with valid input."""

        def test_func() -> None:
            pass

        assert callable(test_func) is True
        assert callable(lambda x: x) is True
        assert callable(str) is True

    def test_is_callable_invalid(self) -> None:
        """Test callable validation with invalid input."""
        assert callable("not callable") is False
        assert callable(123) is False
        assert callable(None) is False

    def test_is_list_valid(self) -> None:
        """Test list validation with valid input."""
        result = FlextValidations.Core.TypeValidators.validate_list([])
        assert result.success is True

        result = FlextValidations.Core.TypeValidators.validate_list([1, 2, 3])
        assert result.success is True

        result = FlextValidations.Core.TypeValidators.validate_list(["a", "b"])
        assert result.success is True

    def test_is_list_invalid(self) -> None:
        """Test list validation with invalid input."""
        result = FlextValidations.Core.TypeValidators.validate_list("not a list")
        assert result.success is False

        result = FlextValidations.Core.TypeValidators.validate_list(123)
        assert result.success is False

        result = FlextValidations.Core.TypeValidators.validate_list(None)
        assert result.success is False

        result = FlextValidations.Core.TypeValidators.validate_list({"key": "value"})
        assert result.success is False

    def test_is_dict_valid(self) -> None:
        """Test dict validation with valid input."""
        result = FlextValidations.Core.TypeValidators.validate_dict({})
        assert result.success is True

        result = FlextValidations.Core.TypeValidators.validate_dict({"key": "value"})
        assert result.success is True

        result = FlextValidations.Core.TypeValidators.validate_dict({"a": 1, "b": 2})
        assert result.success is True

    def test_is_dict_invalid(self) -> None:
        """Test dict validation with invalid input."""
        result = FlextValidations.Core.TypeValidators.validate_dict("not a dict")
        assert result.success is False

        result = FlextValidations.Core.TypeValidators.validate_dict(123)
        assert result.success is False

        result = FlextValidations.Core.TypeValidators.validate_dict(None)
        assert result.success is False

        result = FlextValidations.Core.TypeValidators.validate_dict([1, 2, 3])
        assert result.success is False

    def test_none_handling_with_string_validation(self) -> None:
        """Test None handling using string validation."""
        # Test that None fails string validation
        result = FlextValidations.validate_non_empty_string_func(None)
        assert result is False

        # Test that non-None strings work as expected
        result = FlextValidations.validate_non_empty_string_func("test")
        assert result is True

        result = FlextValidations.validate_non_empty_string_func("")
        assert result is False

    def test_none_handling_with_type_validation(self) -> None:
        """Test None handling using type validation."""
        # Test that None fails various type validations
        string_result = FlextValidations.Core.TypeValidators.validate_string(None)
        assert string_result.success is False

        list_result = FlextValidations.Core.TypeValidators.validate_list(None)
        assert list_result.success is False

        dict_result = FlextValidations.Core.TypeValidators.validate_dict(None)
        assert dict_result.success is False

    def test_email_validation(self) -> None:
        """Test email validation using real FlextValidation API."""
        # Test valid email
        result = FlextValidations.Validators.validate_email("test@example.com")
        assert result.success is True

        # Test invalid email
        result = FlextValidations.Validators.validate_email("invalid")
        assert result.success is False

        # Test empty string
        result = FlextValidations.Validators.validate_email("")
        assert result.success is False

    def test_uuid_validation_with_string_check(self) -> None:
        """Test UUID validation using string validation."""
        # Test valid UUID format as string
        result = FlextValidations.validate_non_empty_string_func(
            "550e8400-e29b-41d4-a716-446655440000",
        )
        assert result is True

        # Test empty string
        result = FlextValidations.validate_non_empty_string_func("")
        assert result is False

        # Test with type validation
        string_result = FlextValidations.Core.TypeValidators.validate_string(
            "550e8400-e29b-41d4-a716-446655440000",
        )
        assert string_result.success is True

    def test_url_validation_with_string_check(self) -> None:
        """Test URL validation using string validation."""
        # Test valid URL format as string
        result = FlextValidations.validate_non_empty_string_func("https://example.com")
        assert result is True

        # Test empty string
        result = FlextValidations.validate_non_empty_string_func("")
        assert result is False

        # Test with type validation
        string_result = FlextValidations.Core.TypeValidators.validate_string(
            "https://example.com",
        )
        assert string_result.success is True


# =============================================================================
# TEST FLEXT RESULT FOR VALIDATION
# =============================================================================


class TestFlextResultValidation:
    """Test FlextResult usage for validation scenarios."""

    def test_success_result_creation(self) -> None:
        """Test creating successful validation result."""
        is_valid = True
        result = FlextResult[bool].ok(is_valid)

        if not result.success:
            raise AssertionError(f"Expected True, got {result.success}")
        assert result.value is True

    def test_failure_result_creation(self) -> None:
        """Test creating failed validation result."""
        result = FlextResult[object].fail("Validation failed")

        if result.success:
            raise AssertionError(f"Expected False, got {result.success}")
        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        # Em falha, `.value` raises TypeError
        with pytest.raises(
            TypeError,
            match="Attempted to access value on failed result",
        ):
            _ = result.value
        if result.error != "Validation failed":
            raise AssertionError(f"Expected {'Validation failed'}, got {result.error}")

    def test_validation_with_flext_result(self) -> None:
        """Test using FlextResult for validation scenarios."""

        def validate_email(email: str) -> FlextResult[str]:
            """Validate email and return result."""
            if not email:
                return FlextResult[str].fail("Email is required")
            if "@" not in email:
                return FlextResult[str].fail("Invalid email format")
            return FlextResult[str].ok(email)

        # Test successful validation
        result = validate_email("test@example.com")
        assert result.success
        if result.value != "test@example.com":
            raise AssertionError(f"Expected {'test@example.com'}, got {result.value}")

        # Test validation failure - empty
        result = validate_email("")
        assert result.is_failure
        assert result.error is not None
        if "required" not in (result.error or ""):
            raise AssertionError(f"Expected 'required' in {result.error}")

        # Test validation failure - invalid format
        result = validate_email("invalid")
        assert result.is_failure
        assert result.error is not None
        if "format" not in (result.error or ""):
            raise AssertionError(f"Expected 'format' in {result.error}")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestValidationIntegration:
    """Integration tests for validation system."""

    def test_complex_validation_scenario(self) -> None:
        """Test complex validation using available utilities."""

        def validate_user_data(
            data: object,
        ) -> FlextResult[FlextTypes.Core.Dict]:
            """Validate user data using FlextValidation utilities."""
            # Check if data is a dict using type validation
            dict_validation = FlextValidations.Core.TypeValidators.validate_dict(data)
            if not dict_validation.success:
                return FlextResult[FlextTypes.Core.Dict].fail(
                    "Data must be a dictionary"
                )

            # Now we know data is a dict, safe to cast and access
            data_dict = dict_validation.data  # Use validated data

            # Check required fields using string validation
            name = data_dict.get("name")
            if not FlextValidations.validate_non_empty_string_func(name):
                return FlextResult[FlextTypes.Core.Dict].fail(
                    "Name is required and must be a non-empty string",
                )

            # Check email format using real email validation
            email = data_dict.get("email")
            if email:
                email_validation = FlextValidations.Validators.validate_email(email)
                if not email_validation.success:
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        "Invalid email format"
                    )

            # Check if roles is a list when provided using type validation
            roles = data_dict.get("roles")
            if roles is not None:
                roles_validation = FlextValidations.Core.TypeValidators.validate_list(
                    roles,
                )
                if not roles_validation.success:
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        "Roles must be a list"
                    )

            return FlextResult[FlextTypes.Core.Dict].ok(data_dict)

        self._test_valid_user_data(validate_user_data)
        self._test_invalid_user_data(validate_user_data)

    def _test_valid_user_data(
        self,
        validate_func: Callable[[object], FlextResult[FlextTypes.Core.Dict]],
    ) -> None:
        """Test validation with valid user data."""
        valid_data: FlextTypes.Core.Dict = {
            "name": "John Doe",
            "email": "john@example.com",
            "roles": ["user", "REDACTED_LDAP_BIND_PASSWORD"],
        }
        result = validate_func(valid_data)
        assert result.success
        if result.value != valid_data:
            raise AssertionError(f"Expected {valid_data}, got {result.value}")

    def _test_invalid_user_data(
        self,
        validate_func: Callable[[object], FlextResult[FlextTypes.Core.Dict]],
    ) -> None:
        """Test validation with various invalid user data scenarios."""
        # Test invalid data - not a dict
        result = validate_func("not a dict")
        assert result.is_failure
        assert result.error is not None
        if "dictionary" not in (result.error or ""):
            raise AssertionError(f"Expected 'dictionary' in {result.error}")

        self._test_missing_name_scenario(validate_func)
        self._test_bad_email_scenario(validate_func)
        self._test_bad_roles_scenario(validate_func)

    def _test_missing_name_scenario(
        self,
        validate_func: Callable[[object], FlextResult[FlextTypes.Core.Dict]],
    ) -> None:
        """Test validation with missing name."""
        invalid_data: FlextTypes.Core.Dict = {"email": "test@example.com"}
        result = validate_func(invalid_data)
        assert result.is_failure
        assert result.error is not None
        if "name" not in (result.error or "").lower():
            raise AssertionError(f"Expected {'name'} in {(result.error or '').lower()}")

    def _test_bad_email_scenario(
        self,
        validate_func: Callable[[object], FlextResult[FlextTypes.Core.Dict]],
    ) -> None:
        """Test validation with bad email."""
        invalid_email_data: FlextTypes.Core.Dict = {"name": "John", "email": "invalid"}
        result = validate_func(invalid_email_data)
        assert result.is_failure
        assert result.error is not None
        if "email" not in (result.error or "").lower():
            raise AssertionError(
                f"Expected {'email'} in {(result.error or '').lower()}"
            )

    def _test_bad_roles_scenario(
        self,
        validate_func: Callable[[object], FlextResult[FlextTypes.Core.Dict]],
    ) -> None:
        """Test validation with bad roles."""
        invalid_roles_data: FlextTypes.Core.Dict = {
            "name": "John",
            "roles": "not a list",
        }
        result = validate_func(invalid_roles_data)
        assert result.is_failure
        assert result.error is not None
        if "list" not in (result.error or "").lower():
            raise AssertionError(f"Expected {'list'} in {(result.error or '').lower()}")

    def test_validation_chaining(self) -> None:
        """Test chaining multiple validations."""
        validation_chain = self._create_validation_chain()
        self._test_successful_validation(validation_chain)
        self._test_validation_failures(validation_chain)

    def _create_validation_chain(self) -> Callable[[str], FlextResult[str]]:
        """Create the validation chain for testing."""

        def validate_step1(value: str) -> FlextResult[str]:
            """First validation step."""
            if not FlextValidations.validate_non_empty_string_func(value):
                return FlextResult[str].fail("Step 1: Value must be a non-empty string")
            return FlextResult[str].ok(value)

        def validate_step2(value: str) -> FlextResult[str]:
            """Second validation step."""
            if len(value) < 3:
                return FlextResult[str].fail(
                    "Step 2: Value must be at least 3 characters",
                )
            return FlextResult[str].ok(value)

        def validate_step3(value: str) -> FlextResult[str]:
            """Third validation step."""
            if not value.isalnum():
                return FlextResult[str].fail("Step 3: Value must be alphanumeric")
            return FlextResult[str].ok(value.upper())

        def validate_all_steps(value: str) -> FlextResult[str]:
            """Chain all validation steps."""
            # Step 1
            result = validate_step1(value)
            if result.is_failure:
                return result

            # Step 2
            assert result.value is not None
            result = validate_step2(result.value)
            if result.is_failure:
                return result

            # Step 3
            assert result.value is not None
            return validate_step3(result.value)

        return validate_all_steps

    def _test_successful_validation(
        self,
        validation_chain: Callable[[str], FlextResult[str]],
    ) -> None:
        """Test successful validation scenario."""
        result = validation_chain("test123")
        assert result.success
        if result.value != "TEST123":
            raise AssertionError(f"Expected {'TEST123'}, got {result.value}")

    def _test_validation_failures(
        self,
        validation_chain: Callable[[str], FlextResult[str]],
    ) -> None:
        """Test validation failure scenarios."""
        # Test failure at step 1
        result = validation_chain("")
        assert result.is_failure
        assert result.error is not None
        if "Step 1" not in (result.error or ""):
            raise AssertionError(f"Expected 'Step 1' in {result.error}")

        # Test failure at step 2
        result = validation_chain("ab")
        assert result.is_failure
        assert result.error is not None
        if "Step 2" not in (result.error or ""):
            raise AssertionError(f"Expected 'Step 2' in {result.error}")

        # Test failure at step 3
        result = validation_chain("test@123")
        assert result.is_failure
        assert result.error is not None
        if "Step 3" not in (result.error or ""):
            raise AssertionError(f"Expected 'Step 3' in {result.error}")
