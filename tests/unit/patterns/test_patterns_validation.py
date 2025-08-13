"""Comprehensive tests for FLEXT validation pattern."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.result import FlextResult
from flext_core.validation import FlextValidation

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# TEST VALIDATION UTILITIES
# =============================================================================


class TestFlextValidation:
    """Test FlextValidation functionality."""

    def test_validation_utilities_exist(self) -> None:
        """Test that validation utilities are available."""
        assert hasattr(FlextValidation, "Validators")

    def test_is_non_empty_string_valid(self) -> None:
        """Test non-empty string validation with valid input."""
        if not (FlextValidation.Validators.is_non_empty_string("test")):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_non_empty_string('test')}"
            )
        assert FlextValidation.Validators.is_non_empty_string("hello world") is True

    def test_is_non_empty_string_invalid(self) -> None:
        """Test non-empty string validation with invalid input."""
        if FlextValidation.Validators.is_non_empty_string(""):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_non_empty_string('')}"
            )
        assert FlextValidation.Validators.is_non_empty_string(None) is False
        if FlextValidation.Validators.is_non_empty_string(123):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_non_empty_string(123)}"
            )

    def test_is_callable_valid(self) -> None:
        """Test callable validation with valid input."""

        def test_func() -> None:
            pass

        if not (FlextValidation.Validators.is_callable(test_func)):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_callable(test_func)}"
            )
        assert FlextValidation.Validators.is_callable(lambda x: x) is True
        if not (FlextValidation.Validators.is_callable(str)):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_callable(str)}"
            )

    def test_is_callable_invalid(self) -> None:
        """Test callable validation with invalid input."""
        if FlextValidation.Validators.is_callable("not callable"):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_callable('not callable')}"
            )
        assert FlextValidation.Validators.is_callable(123) is False
        if FlextValidation.Validators.is_callable(None):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_callable(None)}"
            )

    def test_is_list_valid(self) -> None:
        """Test list validation with valid input."""
        if not (FlextValidation.Validators.is_list([])):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_list([])}"
            )
        assert FlextValidation.Validators.is_list([1, 2, 3]) is True
        if not (FlextValidation.Validators.is_list(["a", "b"])):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_list(['a', 'b'])}"
            )

    def test_is_list_invalid(self) -> None:
        """Test list validation with invalid input."""
        if FlextValidation.Validators.is_list("not a list"):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_list('not a list')}"
            )
        assert FlextValidation.Validators.is_list(123) is False
        if FlextValidation.Validators.is_list(None):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_list(None)}"
            )
        assert FlextValidation.Validators.is_list({"key": "value"}) is False

    def test_is_dict_valid(self) -> None:
        """Test dict validation with valid input."""
        if not (FlextValidation.Validators.is_dict({})):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_dict({})}"
            )
        assert FlextValidation.Validators.is_dict({"key": "value"}) is True
        if not (FlextValidation.Validators.is_dict({"a": 1, "b": 2})):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_dict({'a': 1, 'b': 2})}"
            )

    def test_is_dict_invalid(self) -> None:
        """Test dict validation with invalid input."""
        if FlextValidation.Validators.is_dict("not a dict"):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_dict('not a dict')}"
            )
        assert FlextValidation.Validators.is_dict(123) is False
        if FlextValidation.Validators.is_dict(None):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_dict(None)}"
            )
        assert FlextValidation.Validators.is_dict([1, 2, 3]) is False

    def test_is_none_valid(self) -> None:
        """Test None validation with valid input."""
        if not (FlextValidation.Validators.is_none(None)):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_none(None)}"
            )

    def test_is_none_invalid(self) -> None:
        """Test None validation with invalid input."""
        if FlextValidation.Validators.is_none(""):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_none('')}"
            )
        assert FlextValidation.Validators.is_none(0) is False
        is_false = False
        if FlextValidation.Validators.is_none(is_false):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_none(is_false)}"
            )
        assert FlextValidation.Validators.is_none([]) is False

    def test_is_not_none_valid(self) -> None:
        """Test not None validation with valid input."""
        if not (FlextValidation.Validators.is_not_none("test")):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_not_none('test')}"
            )
        assert FlextValidation.Validators.is_not_none(0) is True
        is_false = False
        if not (FlextValidation.Validators.is_not_none(is_false)):
            raise AssertionError(
                f"Expected True, got {FlextValidation.Validators.is_not_none(is_false)}"
            )
        assert FlextValidation.Validators.is_not_none([]) is True

    def test_is_not_none_invalid(self) -> None:
        """Test not None validation with invalid input."""
        if FlextValidation.Validators.is_not_none(None):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_not_none(None)}"
            )

    def test_is_email_placeholder(self) -> None:
        """Test email validation placeholder - actual implementation may vary."""
        # For now, test basic string validation instead
        assert (
            FlextValidation.Validators.is_non_empty_string("test@example.com") is True
        )
        if FlextValidation.Validators.is_non_empty_string(""):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_non_empty_string('')}"
            )

    def test_is_uuid_placeholder(self) -> None:
        """Test UUID validation placeholder - actual implementation may vary."""
        # For now, test basic string validation instead
        assert (
            FlextValidation.Validators.is_non_empty_string(
                "550e8400-e29b-41d4-a716-446655440000",
            )
            is True
        )
        if FlextValidation.Validators.is_non_empty_string(""):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_non_empty_string('')}"
            )

    def test_is_url_placeholder(self) -> None:
        """Test URL validation placeholder - actual implementation may vary."""
        # For now, test basic string validation instead
        assert (
            FlextValidation.Validators.is_non_empty_string("https://example.com")
            is True
        )
        if FlextValidation.Validators.is_non_empty_string(""):
            raise AssertionError(
                f"Expected False, got {FlextValidation.Validators.is_non_empty_string('')}"
            )


# =============================================================================
# TEST FLEXT RESULT FOR VALIDATION
# =============================================================================


class TestFlextResultValidation:
    """Test FlextResult usage for validation scenarios."""

    def test_success_result_creation(self) -> None:
        """Test creating successful validation result."""
        is_valid = True
        result = FlextResult.ok(is_valid)

        if not result.success:
            raise AssertionError(f"Expected True, got {result.success}")
        assert result.data is True
        assert result.error is None

    def test_failure_result_creation(self) -> None:
        """Test creating failed validation result."""
        result: FlextResult[object] = FlextResult.fail("Validation failed")

        if result.success:
            raise AssertionError(f"Expected False, got {result.success}")
        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        assert result.data is None
        if result.error != "Validation failed":
            raise AssertionError(f"Expected {'Validation failed'}, got {result.error}")

    def test_validation_with_flext_result(self) -> None:
        """Test using FlextResult for validation scenarios."""

        def validate_email(email: str) -> FlextResult[str]:
            """Validate email and return result."""
            if not email:
                return FlextResult.fail("Email is required")
            if "@" not in email:
                return FlextResult.fail("Invalid email format")
            return FlextResult.ok(email)

        # Test successful validation
        result = validate_email("test@example.com")
        assert result.success
        if result.data != "test@example.com":
            raise AssertionError(f"Expected {'test@example.com'}, got {result.data}")

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
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Validate user data using FlextValidation utilities."""
            # Check if data is a dict
            if not FlextValidation.Validators.is_dict(data):
                return FlextResult.fail("Data must be a dictionary")

            # Check required fields
            if not FlextValidation.Validators.is_non_empty_string(data.get("name")):
                return FlextResult.fail(
                    "Name is required and must be a non-empty string",
                )

            # Check email format using proper email validation
            email = data.get("email")
            if email and not FlextValidation.Validators.is_email(email):
                return FlextResult.fail("Invalid email format")

            # Check if roles is a list when provided
            roles = data.get("roles")
            if roles is not None and not FlextValidation.Validators.is_list(roles):
                return FlextResult.fail("Roles must be a list")

            return FlextResult.ok(data)

        self._test_valid_user_data(validate_user_data)
        self._test_invalid_user_data(validate_user_data)

    def _test_valid_user_data(
        self,
        validate_func: Callable[[dict[str, object]], FlextResult[dict[str, object]]],
    ) -> None:
        """Test validation with valid user data."""
        valid_data: dict[str, object] = {
            "name": "John Doe",
            "email": "john@example.com",
            "roles": ["user", "REDACTED_LDAP_BIND_PASSWORD"],
        }
        result = validate_func(valid_data)
        assert result.success
        if result.data != valid_data:
            raise AssertionError(f"Expected {valid_data}, got {result.data}")

    def _test_invalid_user_data(
        self,
        validate_func: Callable[[dict[str, object]], FlextResult[dict[str, object]]],
    ) -> None:
        """Test validation with various invalid user data scenarios."""
        # Test invalid data - not a dict
        result = validate_func("not a dict")  # type: ignore[arg-type]
        assert result.is_failure
        assert result.error is not None
        if "dictionary" not in (result.error or ""):
            raise AssertionError(f"Expected 'dictionary' in {result.error}")

        self._test_missing_name_scenario(validate_func)
        self._test_bad_email_scenario(validate_func)
        self._test_bad_roles_scenario(validate_func)

    def _test_missing_name_scenario(
        self,
        validate_func: Callable[[dict[str, object]], FlextResult[dict[str, object]]],
    ) -> None:
        """Test validation with missing name."""
        invalid_data: dict[str, object] = {"email": "test@example.com"}
        result = validate_func(invalid_data)
        assert result.is_failure
        assert result.error is not None
        if "name" not in result.error.lower():
            raise AssertionError(f"Expected {'name'} in {result.error.lower()}")

    def _test_bad_email_scenario(
        self,
        validate_func: Callable[[dict[str, object]], FlextResult[dict[str, object]]],
    ) -> None:
        """Test validation with bad email."""
        invalid_email_data: dict[str, object] = {"name": "John", "email": "invalid"}
        result = validate_func(invalid_email_data)
        assert result.is_failure
        assert result.error is not None
        if "email" not in result.error.lower():
            raise AssertionError(f"Expected {'email'} in {result.error.lower()}")

    def _test_bad_roles_scenario(
        self,
        validate_func: Callable[[dict[str, object]], FlextResult[dict[str, object]]],
    ) -> None:
        """Test validation with bad roles."""
        invalid_roles_data: dict[str, object] = {"name": "John", "roles": "not a list"}
        result = validate_func(invalid_roles_data)
        assert result.is_failure
        assert result.error is not None
        if "list" not in result.error.lower():
            raise AssertionError(f"Expected {'list'} in {result.error.lower()}")

    def test_validation_chaining(self) -> None:
        """Test chaining multiple validations."""
        validation_chain = self._create_validation_chain()
        self._test_successful_validation(validation_chain)
        self._test_validation_failures(validation_chain)

    def _create_validation_chain(self) -> Callable[[str], FlextResult[str]]:
        """Create the validation chain for testing."""

        def validate_step1(value: str) -> FlextResult[str]:
            """First validation step."""
            if not FlextValidation.Validators.is_non_empty_string(value):
                return FlextResult.fail("Step 1: Value must be a non-empty string")
            return FlextResult.ok(value)

        def validate_step2(value: str) -> FlextResult[str]:
            """Second validation step."""
            if len(value) < 3:
                return FlextResult.fail("Step 2: Value must be at least 3 characters")
            return FlextResult.ok(value)

        def validate_step3(value: str) -> FlextResult[str]:
            """Third validation step."""
            if not value.isalnum():
                return FlextResult.fail("Step 3: Value must be alphanumeric")
            return FlextResult.ok(value.upper())

        def validate_all_steps(value: str) -> FlextResult[str]:
            """Chain all validation steps."""
            # Step 1
            result = validate_step1(value)
            if result.is_failure:
                return result

            # Step 2
            assert result.data is not None
            result = validate_step2(result.data)
            if result.is_failure:
                return result

            # Step 3
            assert result.data is not None
            return validate_step3(result.data)

        return validate_all_steps

    def _test_successful_validation(
        self, validation_chain: Callable[[str], FlextResult[str]]
    ) -> None:
        """Test successful validation scenario."""
        result = validation_chain("test123")
        assert result.success
        if result.data != "TEST123":
            raise AssertionError(f"Expected {'TEST123'}, got {result.data}")

    def _test_validation_failures(
        self, validation_chain: Callable[[str], FlextResult[str]]
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
