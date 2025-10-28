"""Unit tests for FlextValidations.

Tests consolidation of validation logic across the FLEXT ecosystem.
All validators should follow the railway-oriented programming pattern.
"""

from __future__ import annotations

from flext_core import FlextValidations


class TestValidateRequiredString:
    """Test required string validation."""

    def test_validate_required_string_with_value(self) -> None:
        """Test validation succeeds with non-empty string."""
        result = FlextValidations.validate_required_string("value")
        assert result.is_success
        assert result.unwrap() == "value"

    def test_validate_required_string_strips_whitespace(self) -> None:
        """Test validation strips leading/trailing whitespace."""
        result = FlextValidations.validate_required_string("  value  ")
        assert result.is_success
        assert result.unwrap() == "value"

    def test_validate_required_string_empty_fails(self) -> None:
        """Test validation fails with empty string."""
        result = FlextValidations.validate_required_string("")
        assert result.is_failure

    def test_validate_required_string_whitespace_fails(self) -> None:
        """Test validation fails with whitespace-only string."""
        result = FlextValidations.validate_required_string("   ")
        assert result.is_failure

    def test_validate_required_string_none_fails(self) -> None:
        """Test validation fails with None."""
        result = FlextValidations.validate_required_string(None)
        assert result.is_failure

    def test_validate_required_string_custom_context(self) -> None:
        """Test custom context in error message."""
        result = FlextValidations.validate_required_string(None, "Email")
        assert result.is_failure
        assert result.error is not None
        assert "Email cannot be empty" in result.error


class TestValidateChoice:
    """Test enum/choice validation."""

    def test_validate_choice_valid(self) -> None:
        """Test validation succeeds with valid choice."""
        result = FlextValidations.validate_choice(
            "base", {"base", "onelevel", "subtree"}
        )
        assert result.is_success
        assert result.unwrap() == "base"

    def test_validate_choice_invalid(self) -> None:
        """Test validation fails with invalid choice."""
        result = FlextValidations.validate_choice(
            "invalid", {"base", "onelevel", "subtree"}
        )
        assert result.is_failure

    def test_validate_choice_case_insensitive(self) -> None:
        """Test case-insensitive validation."""
        result = FlextValidations.validate_choice(
            "BASE", {"base", "onelevel", "subtree"}, case_sensitive=False
        )
        assert result.is_success

    def test_validate_choice_case_sensitive_fails(self) -> None:
        """Test case-sensitive validation fails on case mismatch."""
        result = FlextValidations.validate_choice(
            "BASE", {"base", "onelevel", "subtree"}, case_sensitive=True
        )
        assert result.is_failure

    def test_validate_choice_custom_context(self) -> None:
        """Test custom context in error message."""
        result = FlextValidations.validate_choice("invalid", {"valid"}, "Operation")
        assert result.is_failure
        assert result.error is not None
        assert "Invalid Operation" in result.error


class TestValidateLength:
    """Test string length validation."""

    def test_validate_length_min_valid(self) -> None:
        """Test validation succeeds when above minimum."""
        result = FlextValidations.validate_length("password", min_length=8)
        assert result.is_success

    def test_validate_length_min_fails(self) -> None:
        """Test validation fails when below minimum."""
        result = FlextValidations.validate_length("short", min_length=8)
        assert result.is_failure

    def test_validate_length_max_valid(self) -> None:
        """Test validation succeeds when below maximum."""
        result = FlextValidations.validate_length("value", max_length=10)
        assert result.is_success

    def test_validate_length_max_fails(self) -> None:
        """Test validation fails when above maximum."""
        result = FlextValidations.validate_length("very long string", max_length=10)
        assert result.is_failure

    def test_validate_length_range_valid(self) -> None:
        """Test validation succeeds within range."""
        result = FlextValidations.validate_length(
            "password", min_length=8, max_length=128
        )
        assert result.is_success

    def test_validate_length_custom_context(self) -> None:
        """Test custom context in error message."""
        result = FlextValidations.validate_length(
            "short", min_length=8, context="Password"
        )
        assert result.is_failure
        assert result.error is not None
        assert "Password must be at least 8" in result.error


class TestValidatePattern:
    """Test regex pattern validation."""

    def test_validate_pattern_match(self) -> None:
        """Test validation succeeds with matching pattern."""
        pattern = r"^[a-zA-Z0-9\-_]+$"
        result = FlextValidations.validate_pattern("my-attr-123", pattern)
        assert result.is_success

    def test_validate_pattern_no_match(self) -> None:
        """Test validation fails with non-matching pattern."""
        pattern = r"^[a-zA-Z0-9\-_]+$"
        result = FlextValidations.validate_pattern("invalid attr!", pattern)
        assert result.is_failure

    def test_validate_pattern_custom_context(self) -> None:
        """Test custom context in error message."""
        pattern = r"^[a-z]+$"
        result = FlextValidations.validate_pattern(
            "Invalid", pattern, "Attribute name"
        )
        assert result.is_failure
        assert result.error is not None
        assert "Attribute name format is invalid" in result.error


class TestValidateUri:
    """Test URI validation."""

    def test_validate_uri_valid(self) -> None:
        """Test validation succeeds with valid URI."""
        result = FlextValidations.validate_uri("ldap://localhost:389")
        assert result.is_success
        assert result.unwrap() == "ldap://localhost:389"

    def test_validate_uri_with_scheme_valid(self) -> None:
        """Test validation succeeds with allowed scheme."""
        result = FlextValidations.validate_uri(
            "ldap://localhost:389", ["ldap", "ldaps"]
        )
        assert result.is_success

    def test_validate_uri_with_scheme_invalid(self) -> None:
        """Test validation fails with disallowed scheme."""
        result = FlextValidations.validate_uri(
            "http://localhost", ["ldap", "ldaps"]
        )
        assert result.is_failure

    def test_validate_uri_empty_fails(self) -> None:
        """Test validation fails with empty URI."""
        result = FlextValidations.validate_uri("")
        assert result.is_failure

    def test_validate_uri_none_fails(self) -> None:
        """Test validation fails with None URI."""
        result = FlextValidations.validate_uri(None)
        assert result.is_failure

    def test_validate_uri_strips_whitespace(self) -> None:
        """Test validation strips whitespace."""
        result = FlextValidations.validate_uri("  ldap://localhost  ")
        assert result.is_success
        assert result.unwrap() == "ldap://localhost"


class TestValidatePortNumber:
    """Test port number validation."""

    def test_validate_port_number_valid_min(self) -> None:
        """Test validation succeeds with minimum port."""
        result = FlextValidations.validate_port_number(1)
        assert result.is_success

    def test_validate_port_number_valid_max(self) -> None:
        """Test validation succeeds with maximum port."""
        result = FlextValidations.validate_port_number(65535)
        assert result.is_success

    def test_validate_port_number_valid_common(self) -> None:
        """Test validation succeeds with common ports."""
        for port in [80, 389, 443, 3389, 5432, 8080]:
            result = FlextValidations.validate_port_number(port)
            assert result.is_success

    def test_validate_port_number_zero_fails(self) -> None:
        """Test validation fails with port 0."""
        result = FlextValidations.validate_port_number(0)
        assert result.is_failure

    def test_validate_port_number_too_high_fails(self) -> None:
        """Test validation fails with port > 65535."""
        result = FlextValidations.validate_port_number(65536)
        assert result.is_failure

    def test_validate_port_number_negative_fails(self) -> None:
        """Test validation fails with negative port."""
        result = FlextValidations.validate_port_number(-1)
        assert result.is_failure

    def test_validate_port_number_none_fails(self) -> None:
        """Test validation fails with None."""
        result = FlextValidations.validate_port_number(None)
        assert result.is_failure


class TestValidateNonNegative:
    """Test non-negative integer validation."""

    def test_validate_non_negative_zero(self) -> None:
        """Test validation succeeds with zero."""
        result = FlextValidations.validate_non_negative(0)
        assert result.is_success

    def test_validate_non_negative_positive(self) -> None:
        """Test validation succeeds with positive."""
        result = FlextValidations.validate_non_negative(100)
        assert result.is_success

    def test_validate_non_negative_negative_fails(self) -> None:
        """Test validation fails with negative."""
        result = FlextValidations.validate_non_negative(-1)
        assert result.is_failure

    def test_validate_non_negative_none_fails(self) -> None:
        """Test validation fails with None."""
        result = FlextValidations.validate_non_negative(None)
        assert result.is_failure


class TestValidatePositive:
    """Test positive integer validation."""

    def test_validate_positive_valid(self) -> None:
        """Test validation succeeds with positive."""
        result = FlextValidations.validate_positive(1)
        assert result.is_success

    def test_validate_positive_zero_fails(self) -> None:
        """Test validation fails with zero."""
        result = FlextValidations.validate_positive(0)
        assert result.is_failure

    def test_validate_positive_negative_fails(self) -> None:
        """Test validation fails with negative."""
        result = FlextValidations.validate_positive(-1)
        assert result.is_failure

    def test_validate_positive_none_fails(self) -> None:
        """Test validation fails with None."""
        result = FlextValidations.validate_positive(None)
        assert result.is_failure


class TestValidateRange:
    """Test range validation."""

    def test_validate_range_within_bounds(self) -> None:
        """Test validation succeeds within range."""
        result = FlextValidations.validate_range(50, min_value=1, max_value=100)
        assert result.is_success

    def test_validate_range_below_minimum(self) -> None:
        """Test validation fails below minimum."""
        result = FlextValidations.validate_range(0, min_value=1, max_value=100)
        assert result.is_failure

    def test_validate_range_above_maximum(self) -> None:
        """Test validation fails above maximum."""
        result = FlextValidations.validate_range(150, min_value=1, max_value=100)
        assert result.is_failure

    def test_validate_range_only_min(self) -> None:
        """Test validation with only minimum bound."""
        result = FlextValidations.validate_range(50, min_value=1)
        assert result.is_success

    def test_validate_range_only_max(self) -> None:
        """Test validation with only maximum bound."""
        result = FlextValidations.validate_range(50, max_value=100)
        assert result.is_success

    def test_validate_range_no_bounds(self) -> None:
        """Test validation with no bounds always succeeds."""
        result = FlextValidations.validate_range(999)
        assert result.is_success


__all__ = [
    "TestValidateChoice",
    "TestValidateLength",
    "TestValidateNonNegative",
    "TestValidatePattern",
    "TestValidatePortNumber",
    "TestValidatePositive",
    "TestValidateRange",
    "TestValidateRequiredString",
    "TestValidateUri",
]
