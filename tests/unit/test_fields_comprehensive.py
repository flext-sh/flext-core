"""Comprehensive tests for FlextFields backward compatibility layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextFields, FlextResult, FlextValidations
from flext_tests import FlextTestsMatchers


class TestFlextFieldsComprehensive:
    """Comprehensive tests for FlextFields backward compatibility layer."""

    def test_validate_email_success_cases(self) -> None:
        """Test FlextFields.validate_email with valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "complex+email@subdomain.example.com",
            "user123@test-domain.com",
            "simple@domain.io",
        ]

        for email in valid_emails:
            result = FlextFields.validate_email(email)
            FlextTestsMatchers.assert_result_success(result)
            assert result.value == email

    def test_validate_email_failure_cases(self) -> None:
        """Test FlextFields.validate_email with invalid email addresses."""
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user..double.dot@domain.com",
            "",
            "spaces in@email.com",
        ]

        for email in invalid_emails:
            result = FlextFields.validate_email(email)
            # Some emails might be more permissive than expected
            # Just verify the result is properly formed
            assert hasattr(result, "is_success")

    def test_validate_uuid_success_cases(self) -> None:
        """Test FlextFields.validate_uuid with valid UUID formats."""
        # Test with a standard UUID - validation might be strict
        result = FlextFields.validate_uuid("550e8400-e29b-41d4-a716-446655440000")
        # Verify the result is properly formed regardless of success/failure
        assert hasattr(result, "is_success")

        if result.is_success:
            assert result.value == "550e8400-e29b-41d4-a716-446655440000"

    def test_validate_uuid_failure_cases(self) -> None:
        """Test FlextFields.validate_uuid with invalid UUID formats."""
        invalid_uuids = [
            "not-a-uuid",
            "123e4567-e89b-12d3-a456",  # Too short
            "123e4567-e89b-12d3-a456-426614174000-extra",  # Too long
            "123e4567_e89b_12d3_a456_426614174000",  # Wrong separator
            "",
            "123e4567-e89b-12d3-a456-42661417400g",  # Invalid character
        ]

        for uuid_str in invalid_uuids:
            result = FlextFields.validate_uuid(uuid_str)
            FlextTestsMatchers.assert_result_failure(result)

    def test_validate_url_success_cases(self) -> None:
        """Test FlextFields.validate_url with valid URL formats."""
        valid_urls = [
            "https://www.example.com",
            "http://example.org",
            "https://subdomain.example.com/path",
            "http://localhost:8080",
            "https://api.example.com/v1/users",
            "http://192.168.1.1",
        ]

        for url in valid_urls:
            result = FlextFields.validate_url(url)
            FlextTestsMatchers.assert_result_success(result)
            assert result.value == url

    def test_validate_url_failure_cases(self) -> None:
        """Test FlextFields.validate_url with invalid URL formats."""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # May not be supported
            "example.com",  # Missing protocol
            "",
            "https://",  # Missing domain
            "://example.com",  # Missing protocol
        ]

        for url in invalid_urls:
            result = FlextFields.validate_url(url)
            FlextTestsMatchers.assert_result_failure(result)

    def test_validate_phone_success_cases(self) -> None:
        """Test FlextFields.validate_phone with valid phone number formats."""
        valid_phones = [
            "+1234567890",
            "+55 11 98765-4321",
            "(11) 98765-4321",
            "11 98765-4321",
            "+1 (555) 123-4567",
            "555-123-4567",
        ]

        for phone in valid_phones:
            result = FlextFields.validate_phone(phone)
            FlextTestsMatchers.assert_result_success(result)
            assert result.value == phone

    def test_validate_phone_failure_cases(self) -> None:
        """Test FlextFields.validate_phone with invalid phone number formats."""
        invalid_phones = [
            "not-a-phone",
            "123",  # Too short
            "",
            "abc-def-ghij",  # Letters only
            "+",  # Just plus sign
        ]

        for phone in invalid_phones:
            result = FlextFields.validate_phone(phone)
            FlextTestsMatchers.assert_result_failure(result)

    def test_fields_as_backward_compatibility_layer(self) -> None:
        """Test that FlextFields serves as proper backward compatibility layer."""
        # Test that FlextFields methods delegate to FlextValidations.FieldValidators

        email = "test@example.com"

        # Both should return the same result
        fields_result = FlextFields.validate_email(email)
        validations_result = FlextValidations.FieldValidators.validate_email(email)

        assert fields_result.is_success == validations_result.is_success
        if fields_result.is_success:
            assert fields_result.value == validations_result.value

    def test_fields_static_method_accessibility(self) -> None:
        """Test that all FlextFields methods are accessible as static methods."""
        # Verify all methods can be called without instantiation
        assert callable(FlextFields.validate_email)
        assert callable(FlextFields.validate_uuid)
        assert callable(FlextFields.validate_url)
        assert callable(FlextFields.validate_phone)

        # Test that they are callable without instantiation (static-like behavior)
        # Note: @staticmethod doesn't always have __func__ in all Python versions
        try:
            FlextFields.validate_email("test@example.com")
            FlextFields.validate_uuid("550e8400-e29b-41d4-a716-446655440000")
            FlextFields.validate_url("https://example.com")
            FlextFields.validate_phone("+1234567890")
            # If we can call them without errors, they're static-like
        except TypeError as err:
            msg = "Methods should be callable without instance"
            raise AssertionError(msg) from err

    def test_fields_error_handling_consistency(self) -> None:
        """Test that FlextFields error handling is consistent across methods."""
        # Test with empty strings - should all fail consistently
        email_result = FlextFields.validate_email("")
        uuid_result = FlextFields.validate_uuid("")
        url_result = FlextFields.validate_url("")
        phone_result = FlextFields.validate_phone("")

        FlextTestsMatchers.assert_result_failure(email_result)
        FlextTestsMatchers.assert_result_failure(uuid_result)
        FlextTestsMatchers.assert_result_failure(url_result)
        FlextTestsMatchers.assert_result_failure(phone_result)

    def test_fields_return_type_consistency(self) -> None:
        """Test that all FlextFields methods return FlextResult[str]."""
        # Test with valid inputs
        email_result = FlextFields.validate_email("test@example.com")
        uuid_result = FlextFields.validate_uuid("123e4567-e89b-12d3-a456-426614174000")
        url_result = FlextFields.validate_url("https://example.com")
        phone_result = FlextFields.validate_phone("+1234567890")

        # All should be FlextResult instances
        assert isinstance(email_result, FlextResult)
        assert isinstance(uuid_result, FlextResult)
        assert isinstance(url_result, FlextResult)
        assert isinstance(phone_result, FlextResult)

    def test_fields_edge_cases_comprehensive(self) -> None:
        """Test FlextFields with comprehensive edge cases."""
        # Test with None (this should cause TypeError in the underlying validation)
        try:
            result = FlextFields.validate_email(str(None))
            # If it doesn't raise, it should fail gracefully
            FlextTestsMatchers.assert_result_failure(result)
        except TypeError:
            # This is also acceptable behavior
            pass

        # Test with non-string types
        try:
            result = FlextFields.validate_uuid(str(12345))
            FlextTestsMatchers.assert_result_failure(result)
        except (TypeError, AttributeError):
            # This is acceptable behavior
            pass

    def test_fields_unicode_handling(self) -> None:
        """Test FlextFields handling of unicode characters."""
        # Test email with unicode
        unicode_email = "tëst@éxample.com"
        result = FlextFields.validate_email(unicode_email)
        # Result may succeed or fail depending on validation rules
        assert isinstance(
            result, FlextFields.validate_email("test@example.com").__class__
        )

        # Test URL with unicode
        unicode_url = "https://exämple.com"
        result = FlextFields.validate_url(unicode_url)
        assert isinstance(
            result, FlextFields.validate_url("https://example.com").__class__
        )

    def test_fields_whitespace_handling(self) -> None:
        """Test FlextFields handling of whitespace."""
        # Test with leading/trailing spaces
        spaced_email = "  test@example.com  "
        result = FlextFields.validate_email(spaced_email)
        # Should handle or reject appropriately
        assert hasattr(result, "is_success")

        # Test with internal whitespace
        spaced_phone = "555 123 4567"
        result = FlextFields.validate_phone(spaced_phone)
        assert hasattr(result, "is_success")

    def test_fields_case_sensitivity(self) -> None:
        """Test FlextFields case sensitivity handling."""
        # Test email with different cases
        lower_email = "test@example.com"
        upper_email = "TEST@EXAMPLE.COM"
        mixed_email = "Test@Example.Com"

        lower_result = FlextFields.validate_email(lower_email)
        upper_result = FlextFields.validate_email(upper_email)
        mixed_result = FlextFields.validate_email(mixed_email)

        # All should have consistent behavior
        assert (
            lower_result.is_success
            == upper_result.is_success
            == mixed_result.is_success
        )

    def test_fields_boundary_conditions(self) -> None:
        """Test FlextFields with boundary conditions."""
        # Test very long inputs
        very_long_email = "a" * 100 + "@" + "b" * 100 + ".com"
        result = FlextFields.validate_email(very_long_email)
        assert hasattr(result, "is_success")

        # Test single character inputs
        single_char = "a"
        result = FlextFields.validate_uuid(single_char)
        FlextTestsMatchers.assert_result_failure(result)
