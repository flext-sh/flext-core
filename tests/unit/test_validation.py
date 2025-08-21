"""Modern tests for FlextValidation - Advanced Validation Systems.

Refactored test suite using comprehensive testing libraries for validation functionality.
Demonstrates property-based testing, edge case validation, and extensive automation.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, strategies as st
from tests.support.domain_factories import UserDataFactory
from tests.support.factory_boy_factories import (
    EdgeCaseGenerators,
    create_validation_test_cases,
)
from tests.support.performance_utils import BenchmarkUtils, PerformanceProfiler

# Direct imports avoiding problematic paths
from flext_core.validation import FlextValidation

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# CORE VALIDATION TESTS
# ============================================================================


class TestFlextValidationCore:
    """Test core validation functionality with factory patterns."""

    def test_email_validation_with_factories(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test email validation using factory patterns."""
        # Generate users with valid emails
        users = [user_data_factory.build() for _ in range(50)]

        for user in users:
            email = user["email"]
            result = FlextValidation.validate_email(email)
            assert result.success, f"Valid email should pass: {email}"

    @pytest.mark.parametrize("valid_email", [
        "test@example.com",
        "user.name@domain.co.uk",
        "firstname+lastname@company.org",
        "test123@test-domain.com",
        "user_name@example-site.info"
    ])
    def test_valid_email_formats(self, valid_email: str) -> None:
        """Test various valid email formats."""
        result = FlextValidation.validate_email(valid_email)
        assert result.success, f"Email should be valid: {valid_email}"

    @pytest.mark.parametrize("invalid_email", [
        "invalid-email",
        "@domain.com",
        "user@",
        "user..name@domain.com",
        "user@domain",
        "",
        "user name@domain.com",
        "user@domain..com"
    ])
    def test_invalid_email_formats(self, invalid_email: str) -> None:
        """Test various invalid email formats."""
        result = FlextValidation.validate_email(invalid_email)
        assert result.is_failure, f"Email should be invalid: {invalid_email}"

    def test_string_validation_basic(self) -> None:
        """Test basic string validation."""
        # Valid strings
        assert FlextValidation.validate_non_empty_string("valid").success
        assert FlextValidation.validate_non_empty_string("test string").success

        # Invalid strings
        assert FlextValidation.validate_non_empty_string("").is_failure
        assert FlextValidation.validate_non_empty_string("   ").is_failure

    def test_numeric_validation_basic(self) -> None:
        """Test basic numeric validation."""
        # Valid numbers
        assert FlextValidation.validate_positive_number(5).success
        assert FlextValidation.validate_positive_number(0.1).success
        assert FlextValidation.validate_positive_number(1000).success

        # Invalid numbers
        assert FlextValidation.validate_positive_number(-1).is_failure
        assert FlextValidation.validate_positive_number(0).is_failure


# ============================================================================
# PROPERTY-BASED VALIDATION TESTS
# ============================================================================


class TestFlextValidationProperties:
    """Property-based tests using Hypothesis."""

    @given(st.emails())
    def test_email_validation_hypothesis(self, email: str) -> None:
        """Property-based test for email validation using Hypothesis emails."""
        result = FlextValidation.validate_email(email)
        # Hypothesis generates valid emails, so they should pass
        assert result.success, f"Hypothesis email should be valid: {email}"

    @given(st.text(min_size=1).filter(lambda x: "@" not in x))
    def test_non_email_strings(self, text: str) -> None:
        """Property-based test with strings that are definitely not emails."""
        assume("@" not in text)
        assume(len(text.strip()) > 0)

        result = FlextValidation.validate_email(text)
        assert result.is_failure, f"String without @ should not be valid email: {text}"

    @given(st.text(min_size=1, max_size=1000))
    def test_string_validation_properties(self, text: str) -> None:
        """Property-based test for string validation."""
        result = FlextValidation.validate_non_empty_string(text)

        # Property: non-empty trimmed strings should be valid
        if text.strip():
            assert result.success
        else:
            assert result.is_failure

    @given(st.floats(min_value=0.1, max_value=1000000))
    def test_positive_number_validation(self, number: float) -> None:
        """Property-based test for positive number validation."""
        assume(number > 0)

        result = FlextValidation.validate_positive_number(number)
        assert result.success, f"Positive number should be valid: {number}"

    @given(st.floats(max_value=0).filter(lambda x: not (x == 0.0 and str(x) == "0.0")))
    def test_non_positive_number_validation(self, number: float) -> None:
        """Property-based test for non-positive number validation."""
        assume(number <= 0)

        result = FlextValidation.validate_positive_number(number)
        assert result.is_failure, f"Non-positive number should be invalid: {number}"


# ============================================================================
# EDGE CASE VALIDATION TESTS
# ============================================================================


class TestFlextValidationEdgeCases:
    """Test validation with edge cases and boundary conditions."""

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.unicode_strings())
    def test_unicode_string_validation(self, edge_value: str) -> None:
        """Test string validation with Unicode edge cases."""
        result = FlextValidation.validate_non_empty_string(edge_value)

        # Should handle Unicode gracefully
        assert isinstance(result.success, bool)

        if edge_value.strip():
            assert result.success
        else:
            assert result.is_failure

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.boundary_numbers())
    def test_boundary_number_validation(self, edge_value: float) -> None:
        """Test number validation with boundary values."""
        result = FlextValidation.validate_positive_number(edge_value)

        # Should handle edge cases gracefully
        assert isinstance(result.success, bool)

        if edge_value > 0:
            assert result.success
        else:
            assert result.is_failure

    def test_very_long_string_validation(self) -> None:
        """Test validation with very long strings."""
        very_long_string = "a" * 100000

        result = FlextValidation.validate_non_empty_string(very_long_string)
        assert result.success

    def test_validation_with_special_characters(self) -> None:
        """Test validation with special characters."""
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"

        result = FlextValidation.validate_non_empty_string(special_chars)
        assert result.success

    def test_validation_with_newlines_and_tabs(self) -> None:
        """Test validation with whitespace characters."""
        whitespace_text = "text\nwith\tnewlines\rand\ttabs"

        result = FlextValidation.validate_non_empty_string(whitespace_text)
        assert result.success


# ============================================================================
# PERFORMANCE VALIDATION TESTS
# ============================================================================


class TestFlextValidationPerformance:
    """Test validation performance characteristics."""

    def test_email_validation_performance(self, benchmark: object) -> None:
        """Benchmark email validation performance."""
        emails = [
            "test1@example.com",
            "user.name@domain.co.uk",
            "firstname+lastname@company.org",
            "test123@test-domain.com",
            "user_name@example-site.info"
        ] * 200  # 1000 emails total

        def validate_emails() -> list[bool]:
            return [
                FlextValidation.validate_email(email).success
                for email in emails
            ]

        results = BenchmarkUtils.benchmark_with_warmup(
            benchmark, validate_emails, warmup_rounds=3
        )

        assert len(results) == 1000
        assert all(isinstance(r, bool) for r in results)

    def test_string_validation_performance(self, benchmark: object) -> None:
        """Benchmark string validation performance."""
        strings = [
            "valid string",
            "another valid string",
            "test with special chars !@#",
            "unicode string: café résumé",
            "long string: " + "x" * 1000
        ] * 200  # 1000 strings total

        def validate_strings() -> list[bool]:
            return [
                FlextValidation.validate_non_empty_string(string).success
                for string in strings
            ]

        results = BenchmarkUtils.benchmark_with_warmup(
            benchmark, validate_strings, warmup_rounds=3
        )

        assert len(results) == 1000
        assert all(isinstance(r, bool) for r in results)

    def test_validation_memory_efficiency(self) -> None:
        """Test memory efficiency of validation operations."""
        profiler = PerformanceProfiler()

        with profiler.profile_memory("validation_operations"):
            # Validate many emails
            emails = [f"user{i}@domain{i % 10}.com" for i in range(10000)]
            email_results = [
                FlextValidation.validate_email(email)
                for email in emails
            ]

            # Validate many strings
            strings = [f"test string {i}" for i in range(10000)]
            string_results = [
                FlextValidation.validate_non_empty_string(string)
                for string in strings
            ]

        profiler.assert_memory_efficient(
            max_memory_mb=30.0, operation_name="validation_operations"
        )

        # Verify results
        assert len(email_results) == 10000
        assert len(string_results) == 10000


# ============================================================================
# COMPLEX VALIDATION SCENARIOS
# ============================================================================


class TestFlextValidationScenarios:
    """Test complex validation scenarios and workflows."""

    def test_validation_chain(self, user_data_factory: UserDataFactory) -> None:
        """Test chaining multiple validations."""
        user_data = user_data_factory.build()

        # Chain validations
        email_valid = FlextValidation.validate_email(user_data["email"])
        name_valid = FlextValidation.validate_non_empty_string(user_data["name"])

        # Both should be valid for factory-generated data
        assert email_valid.success
        assert name_valid.success

    def test_validation_with_test_cases(self) -> None:
        """Test validation using comprehensive test cases."""
        test_cases = create_validation_test_cases()

        for case in test_cases:
            data = case["data"]
            expected = case["expected_valid"]

            if isinstance(data, str) and "@" in data:
                # Test as email
                result = FlextValidation.validate_email(data)
                if expected:
                    assert result.success, f"Expected valid email: {data}"
                else:
                    assert result.is_failure, f"Expected invalid email: {data}"

    def test_bulk_validation(self) -> None:
        """Test bulk validation operations."""
        # Generate bulk data
        emails = [f"user{i}@domain.com" for i in range(1000)]
        invalid_emails = [f"invalid{i}" for i in range(100)]

        # Validate bulk
        valid_results = [
            FlextValidation.validate_email(email).success
            for email in emails
        ]
        invalid_results = [
            FlextValidation.validate_email(email).success
            for email in invalid_emails
        ]

        # Check results
        assert all(valid_results), "All valid emails should pass"
        assert not any(invalid_results), "All invalid emails should fail"


# ============================================================================
# CUSTOM VALIDATION PATTERNS
# ============================================================================


class TestFlextCustomValidationPatterns:
    """Test custom validation patterns and extensibility."""

    def test_composite_validation(self) -> None:
        """Test composite validation patterns."""
        # Test data that should pass multiple validations
        test_data = {
            "email": "test@example.com",
            "name": "John Doe",
            "age": 25
        }

        # Validate each field
        email_result = FlextValidation.validate_email(test_data["email"])
        name_result = FlextValidation.validate_non_empty_string(test_data["name"])
        age_result = FlextValidation.validate_positive_number(test_data["age"])

        # All should pass
        assert email_result.success
        assert name_result.success
        assert age_result.success

    def test_validation_error_details(self) -> None:
        """Test that validation errors provide useful details."""
        # Test invalid email
        result = FlextValidation.validate_email("invalid-email")
        assert result.is_failure
        assert result.error is not None
        assert len(result.error) > 0

    def test_validation_with_none_values(self) -> None:
        """Test validation with None values."""
        # None should be handled gracefully
        email_result = FlextValidation.validate_email(None)
        string_result = FlextValidation.validate_non_empty_string(None)
        number_result = FlextValidation.validate_positive_number(None)

        # All should fail but not crash
        assert email_result.is_failure
        assert string_result.is_failure
        assert number_result.is_failure


# ============================================================================
# REGEX AND PATTERN VALIDATION
# ============================================================================


class TestFlextPatternValidation:
    """Test pattern-based validation if available."""

    def test_phone_number_validation(self) -> None:
        """Test phone number validation patterns."""
        valid_phones = [
            "+1-555-123-4567",
            "(555) 123-4567",
            "555.123.4567",
            "5551234567"
        ]

        invalid_phones = [
            "123",
            "abc-def-ghij",
            "555-123",
            "+1-555-123-456789"
        ]

        # Test if phone validation is available
        if hasattr(FlextValidation, "validate_phone_number"):
            for phone in valid_phones:
                result = FlextValidation.validate_phone_number(phone)
                assert result.success, f"Valid phone should pass: {phone}"

            for phone in invalid_phones:
                result = FlextValidation.validate_phone_number(phone)
                assert result.is_failure, f"Invalid phone should fail: {phone}"

    def test_url_validation(self) -> None:
        """Test URL validation patterns."""
        valid_urls = [
            "https://example.com",
            "http://www.example.com",
            "https://subdomain.example.com/path",
            "https://example.com:8080/path?query=value"
        ]

        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Depending on validation rules
            "https://",
            "example.com"  # No protocol
        ]

        # Test if URL validation is available
        if hasattr(FlextValidation, "validate_url"):
            for url in valid_urls:
                result = FlextValidation.validate_url(url)
                assert result.success, f"Valid URL should pass: {url}"

            for url in invalid_urls:
                result = FlextValidation.validate_url(url)
                assert result.is_failure, f"Invalid URL should fail: {url}"
