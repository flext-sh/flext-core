"""Simplified validation tests."""

from flext_core import FlextValidations


class TestValidationSimple:
    """Test simplified validation functionality."""

    def test_guards_exist(self) -> None:
        """Test that Guards class exists."""
        assert hasattr(FlextValidations, "Guards")

    def test_field_validators_exist(self) -> None:
        """Test that FieldValidators class exists."""
        assert hasattr(FlextValidations, "FieldValidators")

    def test_email_validation(self) -> None:
        """Test email validation works."""
        result = FlextValidations.FieldValidators.validate_email("test@example.com")
        assert result.is_success

        result = FlextValidations.FieldValidators.validate_email("invalid")
        assert result.is_failure

    def test_url_validation(self) -> None:
        """Test URL validation works."""
        result = FlextValidations.FieldValidators.validate_url("https://example.com")
        assert result.is_success

        result = FlextValidations.FieldValidators.validate_url("not a url")
        assert result.is_failure
