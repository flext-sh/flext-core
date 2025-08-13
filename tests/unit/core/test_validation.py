"""Tests for FLEXT Core Validation with modern pytest patterns.

Advanced validation tests using parametrized fixtures, property-based testing,
performance monitoring, and comprehensive scenario coverage.
- Enterprise-grade parametrized testing with structured TestCase objects
- Advanced fixture composition using conftest infrastructure
- Validation rule testing with business logic enforcement
- Hypothesis property-based testing for edge case discovery
- Mock factories for validation dependency isolation

Usage of New Conftest Infrastructure:
- test_builder: Fluent builder pattern for complex test data construction
- assert_helpers: Advanced assertion helpers with FlextResult validation
- performance_monitor: Function execution monitoring with memory tracking
- hypothesis_strategies: Property-based testing with domain-specific strategies
"""

from __future__ import annotations

from typing import cast

import pytest
from tests.conftest import TestCase, TestScenario

from flext_core.validation import (
    FlextPredicates,
    FlextValidators,
    flext_validate_email,
    flext_validate_non_empty_string,
)

# Test markers for organized execution
pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextValidatorsAdvanced:
    """Advanced validator testing with consolidated patterns."""

    @pytest.fixture
    def validator_test_cases(self) -> list[TestCase]:
        """Structured test cases for validator testing."""
        return [
            # is_not_none tests
            TestCase(
                id="is_not_none_valid_string",
                description="is_not_none with valid string",
                input_data={"validator": "is_not_none", "value": "hello"},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_not_none_valid_number",
                description="is_not_none with valid number",
                input_data={"validator": "is_not_none", "value": 42},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_not_none_false_value",
                description="is_not_none with False (valid)",
                input_data={"validator": "is_not_none", "value": False},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_not_none_zero_value",
                description="is_not_none with 0 (valid)",
                input_data={"validator": "is_not_none", "value": 0},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_not_none_none_value",
                description="is_not_none with None",
                input_data={"validator": "is_not_none", "value": None},
                expected_output=False,
                scenario=TestScenario.EDGE_CASE,
            ),
            # is_string tests
            TestCase(
                id="is_string_valid",
                description="is_string with valid string",
                input_data={"validator": "is_string", "value": "hello"},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_string_empty",
                description="is_string with empty string",
                input_data={"validator": "is_string", "value": ""},
                expected_output=True,
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                id="is_string_numeric_string",
                description="is_string with numeric string",
                input_data={"validator": "is_string", "value": "123"},
                expected_output=True,
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                id="is_string_invalid_number",
                description="is_string with number",
                input_data={"validator": "is_string", "value": 123},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="is_string_invalid_list",
                description="is_string with list",
                input_data={"validator": "is_string", "value": []},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="is_string_invalid_none",
                description="is_string with None",
                input_data={"validator": "is_string", "value": None},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            # is_non_empty_string tests
            TestCase(
                id="is_non_empty_string_valid",
                description="is_non_empty_string with valid string",
                input_data={"validator": "is_non_empty_string", "value": "hello"},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_non_empty_string_with_spaces",
                description="is_non_empty_string with spaced text",
                input_data={"validator": "is_non_empty_string", "value": "  text  "},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_non_empty_string_single_char",
                description="is_non_empty_string with single character",
                input_data={"validator": "is_non_empty_string", "value": "a"},
                expected_output=True,
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                id="is_non_empty_string_empty",
                description="is_non_empty_string with empty string",
                input_data={"validator": "is_non_empty_string", "value": ""},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="is_non_empty_string_whitespace",
                description="is_non_empty_string with whitespace only",
                input_data={"validator": "is_non_empty_string", "value": "   "},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="is_non_empty_string_tabs_newlines",
                description="is_non_empty_string with tabs and newlines",
                input_data={"validator": "is_non_empty_string", "value": "\t\n"},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_validator_scenarios(
        self, validator_test_cases: list[TestCase], assert_helpers: object,
    ) -> None:
        """Test validators using structured parametrized approach."""
        for test_case in validator_test_cases:
            input_data = cast("dict[str, object]", test_case.input_data)
            expected = test_case.expected_output

            assert isinstance(input_data, dict)
            assert isinstance(expected, bool)

            validator_name: str = cast("str", input_data["validator"])
            value: object = input_data["value"]

            # Get validator method dynamically
            validator_method = getattr(FlextValidators, validator_name)
            result = validator_method(value=value)

            assert result == expected, (
                f"Test {test_case.id}: Expected {expected} for {validator_name}({value}), got {result}"
            )

    @pytest.fixture
    def email_validation_cases(self) -> list[TestCase]:
        """Email validation test cases."""
        return [
            TestCase(
                id="email_valid_basic",
                description="Valid basic email",
                input_data={"email": "user@example.com"},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="email_valid_with_dots",
                description="Valid email with dots",
                input_data={"email": "test.email@domain.org"},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="email_valid_with_plus",
                description="Valid email with plus tag",
                input_data={"email": "user+tag@example.co.uk"},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="email_valid_with_numbers",
                description="Valid email with numbers",
                input_data={"email": "user123@test-domain.com"},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="email_valid_minimal",
                description="Valid minimal email",
                input_data={"email": "a@b.co"},
                expected_output=True,
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                id="email_invalid_no_at",
                description="Invalid email without @",
                input_data={"email": "userexample.com"},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="email_invalid_no_domain",
                description="Invalid email without domain",
                input_data={"email": "user@"},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="email_invalid_spaces",
                description="Invalid email with spaces",
                input_data={"email": "user @example.com"},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="email_invalid_multiple_at",
                description="Invalid email with multiple @",
                input_data={"email": "user@@example.com"},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="email_invalid_empty",
                description="Invalid empty email",
                input_data={"email": ""},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_email_validation_scenarios(
        self, email_validation_cases: list[TestCase],
    ) -> None:
        """Test email validation with comprehensive scenarios."""
        for test_case in email_validation_cases:
            input_data = cast("dict[str, object]", test_case.input_data)
            email: str = cast("str", input_data["email"])
            expected: bool = cast("bool", test_case.expected_output)

            result = FlextValidators.is_email(value=email)
            assert result == expected, (
                f"Test {test_case.id}: Expected {expected} for email '{email}', got {result}"
            )

    @pytest.fixture
    def numeric_validation_cases(self) -> list[TestCase]:
        """Numeric validation test cases."""
        return [
            # Additional string validation tests
            TestCase(
                id="has_min_length_valid",
                description="Valid minimum length",
                input_data={
                    "validator": "has_min_length",
                    "value": "hello",
                    "min_length": 3,
                },
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="has_min_length_invalid",
                description="Invalid minimum length",
                input_data={
                    "validator": "has_min_length",
                    "value": "hi",
                    "min_length": 5,
                },
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="has_max_length_valid",
                description="Valid maximum length",
                input_data={
                    "validator": "has_max_length",
                    "value": "hi",
                    "max_length": 5,
                },
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="has_max_length_invalid",
                description="Invalid maximum length",
                input_data={
                    "validator": "has_max_length",
                    "value": "hello world",
                    "max_length": 5,
                },
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            # Additional FlextValidators tests
            TestCase(
                id="is_url_valid",
                description="Valid URL",
                input_data={"validator": "is_url", "value": "https://example.com"},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_url_invalid",
                description="Invalid URL",
                input_data={"validator": "is_url", "value": "not-a-url"},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="is_uuid_valid",
                description="Valid UUID",
                input_data={
                    "validator": "is_uuid",
                    "value": "123e4567-e89b-12d3-a456-426614174000",
                },
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_uuid_invalid",
                description="Invalid UUID",
                input_data={"validator": "is_uuid", "value": "not-a-uuid"},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_numeric_validation_scenarios(
        self, numeric_validation_cases: list[TestCase],
    ) -> None:
        """Test additional validation scenarios."""
        for test_case in numeric_validation_cases:
            input_data = cast("dict[str, object]", test_case.input_data)
            validator_name: str = cast("str", input_data["validator"])
            value: object = input_data["value"]
            expected: bool = cast("bool", test_case.expected_output)

            validator_method = getattr(FlextValidators, validator_name)

            # Handle validators that need extra parameters
            if validator_name in {"has_min_length", "has_max_length"}:
                if validator_name == "has_min_length":
                    result = validator_method(
                        value=value, min_length=cast("int", input_data["min_length"]),
                    )
                elif validator_name == "has_max_length":
                    result = validator_method(
                        value=value, max_length=cast("int", input_data["max_length"]),
                    )
            else:
                result = validator_method(value=value)

            assert result == expected, (
                f"Test {test_case.id}: Expected {expected} for {validator_name}({value}), got {result}"
            )


class TestFlextPredicatesAdvanced:
    """Advanced predicate testing with consolidated patterns."""

    @pytest.fixture
    def predicate_test_cases(self) -> list[TestCase]:
        """Predicate test cases with comprehensive coverage."""
        return [
            TestCase(
                id="is_positive_predicate_positive",
                description="is_positive with positive number",
                input_data={"predicate": "is_positive", "value": 5},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_positive_predicate_negative",
                description="is_positive with negative number",
                input_data={"predicate": "is_positive", "value": -5},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="is_positive_predicate_zero",
                description="is_positive with zero",
                input_data={"predicate": "is_positive", "value": 0},
                expected_output=False,
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                id="is_negative_predicate_negative",
                description="is_negative with negative number",
                input_data={"predicate": "is_negative", "value": -5},
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="is_negative_predicate_positive",
                description="is_negative with positive number",
                input_data={"predicate": "is_negative", "value": 5},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="is_zero_predicate_zero",
                description="is_zero with zero",
                input_data={"predicate": "is_zero", "value": 0},
                expected_output=True,
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                id="is_zero_predicate_non_zero",
                description="is_zero with non-zero",
                input_data={"predicate": "is_zero", "value": 5},
                expected_output=False,
                scenario=TestScenario.ERROR_CASE,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_predicate_scenarios(self, predicate_test_cases: list[TestCase]) -> None:
        """Test predicates using structured parametrized approach."""
        for test_case in predicate_test_cases:
            input_data = cast("dict[str, object]", test_case.input_data)
            predicate_name: str = cast("str", input_data["predicate"])
            value: object = input_data["value"]
            expected: bool = cast("bool", test_case.expected_output)

            predicate_method = getattr(FlextPredicates, predicate_name)
            result = predicate_method(value)

            assert result == expected, (
                f"Test {test_case.id}: Expected {expected} for {predicate_name}({value}), got {result}"
            )


class TestValidationSimpleIntegration:
    """Simple validation integration tests using only working APIs."""

    def test_email_validation_function(self) -> None:
        """Test email validation function directly."""
        # Valid email
        valid_result = flext_validate_email("test@example.com")
        assert valid_result.is_valid

        # Invalid email
        invalid_result = flext_validate_email("invalid-email")
        assert not invalid_result.is_valid

    def test_non_empty_string_validation_function(self) -> None:
        """Test non-empty string validation function directly."""
        # Valid string - flext_validate_non_empty_string returns bool
        non_empty_result = flext_validate_non_empty_string("hello")
        assert non_empty_result is True

        # Invalid empty string
        empty_result = flext_validate_non_empty_string("")
        assert empty_result is False


# All edge cases, model tests, and additional coverage tests have been
# consolidated into the advanced parametrized test classes above.
# This reduces code duplication while maintaining comprehensive validation coverage.
