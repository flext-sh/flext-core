"""Centralized test scenarios for parametrized testing.

Provides shared scenario definitions for validation, parsing, and reliability
testing across the flext-core test suite. Eliminates duplication and serves
as single source of truth for test cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from flext_core import m, t

# =========================================================================
# Scenario Dataclasses
# =========================================================================


@dataclass(frozen=True, slots=True)
class ValidationScenario:
    """Single scenario for validation testing."""

    name: str
    validator_type: str  # "network", "string", "numeric"
    input_value: t.GeneralValueType
    input_params: dict[str, object] | None = None
    should_succeed: bool = True
    expected_value: t.GeneralValueType | None = None
    expected_error_contains: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ParserScenario:
    """Single scenario for parser testing."""

    name: str
    parser_method: str
    input_data: str
    expected_output: t.GeneralValueType | None = None
    should_succeed: bool = True
    error_contains: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ReliabilityScenario:
    """Single scenario for reliability testing (circuit breaker, retry)."""

    name: str
    strategy: str  # "retry", "circuit_breaker", "timeout"
    config: m.ConfigMap
    simulate_failures: int
    expected_state: str
    should_succeed: bool = True
    description: str | None = None


# =========================================================================
# Centralized Validation Scenarios
# =========================================================================


class ValidationScenarios:
    """Centralized validation scenarios - single source of truth."""

    # Network Validators
    URI_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="uri_valid_http",
            validator_type="network",
            input_value="http://example.com",
            should_succeed=True,
            expected_value="http://example.com",
            description="Valid HTTP URI",
        ),
        ValidationScenario(
            name="uri_valid_https",
            validator_type="network",
            input_value="https://example.com",
            should_succeed=True,
            expected_value="https://example.com",
            description="Valid HTTPS URI",
        ),
        ValidationScenario(
            name="uri_with_port",
            validator_type="network",
            input_value="https://example.com:8080",
            should_succeed=True,
            expected_value="https://example.com:8080",
            description="URI with custom port",
        ),
        ValidationScenario(
            name="uri_with_path",
            validator_type="network",
            input_value="https://example.com/path/to/resource",
            should_succeed=True,
            expected_value="https://example.com/path/to/resource",
            description="URI with path",
        ),
        ValidationScenario(
            name="uri_with_query",
            validator_type="network",
            input_value="https://example.com/path?key=value",
            should_succeed=True,
            expected_value="https://example.com/path?key=value",
            description="URI with query parameters",
        ),
        ValidationScenario(
            name="uri_with_fragment",
            validator_type="network",
            input_value="https://example.com/path#section",
            should_succeed=True,
            expected_value="https://example.com/path#section",
            description="URI with fragment",
        ),
        ValidationScenario(
            name="uri_none",
            validator_type="network",
            input_value=None,
            should_succeed=False,
            expected_error_contains="cannot be None",
            description="None URI rejection",
        ),
        ValidationScenario(
            name="uri_empty",
            validator_type="network",
            input_value="",
            should_succeed=False,
            expected_error_contains="cannot be empty",
            description="Empty URI rejection",
        ),
        ValidationScenario(
            name="uri_invalid_scheme",
            validator_type="network",
            input_value="ftp://example.com",
            should_succeed=False,
            expected_error_contains="not in allowed",
            description="Invalid URI scheme",
        ),
        ValidationScenario(
            name="uri_malformed",
            validator_type="network",
            input_value="not a valid uri",
            should_succeed=False,
            expected_error_contains="not a valid",
            description="Malformed URI",
        ),
        ValidationScenario(
            name="uri_whitespace",
            validator_type="network",
            input_value="   ",
            should_succeed=False,
            expected_error_contains="cannot be empty",
            description="Whitespace-only URI",
        ),
        ValidationScenario(
            name="uri_custom_scheme",
            validator_type="network",
            input_value="custom://example.com",
            input_params={"allowed_schemes": ["http", "https", "custom"]},
            should_succeed=True,
            expected_value="custom://example.com",
            description="Custom scheme with allowlist",
        ),
    ]

    PORT_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="port_valid_80",
            validator_type="network",
            input_value=80,
            should_succeed=True,
            expected_value=80,
            description="Valid port 80 (HTTP)",
        ),
        ValidationScenario(
            name="port_valid_443",
            validator_type="network",
            input_value=443,
            should_succeed=True,
            expected_value=443,
            description="Valid port 443 (HTTPS)",
        ),
        ValidationScenario(
            name="port_valid_8080",
            validator_type="network",
            input_value=8080,
            should_succeed=True,
            expected_value=8080,
            description="Valid port 8080",
        ),
        ValidationScenario(
            name="port_valid_1",
            validator_type="network",
            input_value=1,
            should_succeed=True,
            expected_value=1,
            description="Valid port 1 (minimum)",
        ),
        ValidationScenario(
            name="port_valid_65535",
            validator_type="network",
            input_value=65535,
            should_succeed=True,
            expected_value=65535,
            description="Valid port 65535 (maximum)",
        ),
        ValidationScenario(
            name="port_none",
            validator_type="network",
            input_value=None,
            should_succeed=False,
            expected_error_contains="cannot be None",
            description="None port rejection",
        ),
        ValidationScenario(
            name="port_zero",
            validator_type="network",
            input_value=0,
            should_succeed=False,
            expected_error_contains="must be between",
            description="Port zero rejection",
        ),
        ValidationScenario(
            name="port_negative",
            validator_type="network",
            input_value=-1,
            should_succeed=False,
            expected_error_contains="must be between",
            description="Negative port rejection",
        ),
        ValidationScenario(
            name="port_above_max",
            validator_type="network",
            input_value=65536,
            should_succeed=False,
            expected_error_contains="at most",
            description="Port above maximum",
        ),
    ]

    HOSTNAME_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="hostname_simple",
            validator_type="network",
            input_value="example",
            should_succeed=True,
            expected_value="example",
            description="Simple hostname",
        ),
        ValidationScenario(
            name="hostname_fqdn",
            validator_type="network",
            input_value="example.com",
            should_succeed=True,
            expected_value="example.com",
            description="Fully qualified domain name",
        ),
        ValidationScenario(
            name="hostname_subdomain",
            validator_type="network",
            input_value="sub.example.com",
            should_succeed=True,
            expected_value="sub.example.com",
            description="Hostname with subdomain",
        ),
        ValidationScenario(
            name="hostname_hyphen",
            validator_type="network",
            input_value="my-host.example.com",
            should_succeed=True,
            expected_value="my-host.example.com",
            description="Hostname with hyphen",
        ),
    ]

    # String Validators
    REQUIRED_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="required_valid",
            validator_type="string",
            input_value="non-empty",
            should_succeed=True,
            expected_value="non-empty",
            description="Valid non-empty string",
        ),
        ValidationScenario(
            name="required_unicode",
            validator_type="string",
            input_value="café",
            should_succeed=True,
            expected_value="café",
            description="Unicode characters",
        ),
        ValidationScenario(
            name="required_special",
            validator_type="string",
            input_value="test@#$%",
            should_succeed=True,
            expected_value="test@#$%",
            description="Special characters",
        ),
        ValidationScenario(
            name="required_none",
            validator_type="string",
            input_value=None,
            should_succeed=False,
            expected_error_contains="empty",
            description="None value rejection",
        ),
        ValidationScenario(
            name="required_empty",
            validator_type="string",
            input_value="",
            should_succeed=False,
            expected_error_contains="empty",
            description="Empty string rejection",
        ),
        ValidationScenario(
            name="required_whitespace",
            validator_type="string",
            input_value="   ",
            should_succeed=False,
            expected_error_contains="empty",
            description="Whitespace-only rejection",
        ),
        ValidationScenario(
            name="required_single_char",
            validator_type="string",
            input_value="a",
            should_succeed=True,
            expected_value="a",
            description="Single character string",
        ),
    ]

    CHOICE_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="choice_valid_single",
            validator_type="string",
            input_value="option1",
            input_params={"valid_choices": ["option1", "option2", "option3"]},
            should_succeed=True,
            expected_value="option1",
            description="Valid single choice",
        ),
        ValidationScenario(
            name="choice_valid_second",
            validator_type="string",
            input_value="option2",
            input_params={"valid_choices": ["option1", "option2", "option3"]},
            should_succeed=True,
            expected_value="option2",
            description="Valid second choice",
        ),
        ValidationScenario(
            name="choice_invalid",
            validator_type="string",
            input_value="invalid",
            input_params={"valid_choices": {"option1", "option2"}},
            should_succeed=False,
            expected_error_contains="Must be one of",
            description="Invalid choice",
        ),
        ValidationScenario(
            name="choice_case_sensitive",
            validator_type="string",
            input_value="OPTION1",
            input_params={
                "valid_choices": {"option1", "option2"},
                "case_sensitive": True,
            },
            should_succeed=False,
            expected_error_contains="Must be one of",
            description="Case-sensitive choice",
        ),
        ValidationScenario(
            name="choice_case_insensitive",
            validator_type="string",
            input_value="option1",
            input_params={
                "valid_choices": {"option1", "option2"},
                "case_sensitive": False,
            },
            should_succeed=True,
            expected_value="option1",
            description="Case-insensitive choice",
        ),
    ]

    LENGTH_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="length_exact",
            validator_type="string",
            input_value="12345",
            input_params={"min_length": 5, "max_length": 5},
            should_succeed=True,
            expected_value="12345",
            description="Exact length match",
        ),
        ValidationScenario(
            name="length_within_bounds",
            validator_type="string",
            input_value="hello",
            input_params={"min_length": 3, "max_length": 10},
            should_succeed=True,
            expected_value="hello",
            description="Length within bounds",
        ),
        ValidationScenario(
            name="length_below_min",
            validator_type="string",
            input_value="hi",
            input_params={"min_length": 3},
            should_succeed=False,
            expected_error_contains="at least",
            description="Length below minimum",
        ),
        ValidationScenario(
            name="length_above_max",
            validator_type="string",
            input_value="toolongstring",
            input_params={"max_length": 5},
            should_succeed=False,
            expected_error_contains="no more than",
            description="Length above maximum",
        ),
        ValidationScenario(
            name="length_zero_max",
            validator_type="string",
            input_value="",
            input_params={"min_length": 0, "max_length": 0},
            should_succeed=True,
            expected_value="",
            description="Zero-length string allowed",
        ),
    ]

    PATTERN_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="pattern_email_valid",
            validator_type="string",
            input_value="test@example.com",
            input_params={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
            should_succeed=True,
            expected_value="test@example.com",
            description="Valid email pattern",
        ),
        ValidationScenario(
            name="pattern_digits_only",
            validator_type="string",
            input_value="12345",
            input_params={"pattern": r"^\d+$"},
            should_succeed=True,
            expected_value="12345",
            description="Digits-only pattern",
        ),
        ValidationScenario(
            name="pattern_alphanumeric",
            validator_type="string",
            input_value="abc123",
            input_params={"pattern": r"^[a-zA-Z0-9]+$"},
            should_succeed=True,
            expected_value="abc123",
            description="Alphanumeric pattern",
        ),
        ValidationScenario(
            name="pattern_mismatch",
            validator_type="string",
            input_value="invalid@",
            input_params={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
            should_succeed=False,
            expected_error_contains="format is invalid",
            description="Pattern mismatch",
        ),
    ]

    # Numeric Validators
    NON_NEGATIVE_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="non_negative_zero",
            validator_type="numeric",
            input_value=0,
            should_succeed=True,
            expected_value=0,
            description="Zero is non-negative",
        ),
        ValidationScenario(
            name="non_negative_positive",
            validator_type="numeric",
            input_value=42,
            should_succeed=True,
            expected_value=42,
            description="Positive number",
        ),
        ValidationScenario(
            name="non_negative_large",
            validator_type="numeric",
            input_value=1000000,
            should_succeed=True,
            expected_value=1000000,
            description="Large positive number",
        ),
        ValidationScenario(
            name="non_negative_negative",
            validator_type="numeric",
            input_value=-1,
            should_succeed=False,
            expected_error_contains="non-negative",
            description="Negative rejection",
        ),
        ValidationScenario(
            name="non_negative_none",
            validator_type="numeric",
            input_value=None,
            should_succeed=False,
            expected_error_contains="cannot be None",
            description="None rejection",
        ),
    ]

    POSITIVE_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="positive_one",
            validator_type="numeric",
            input_value=1,
            should_succeed=True,
            expected_value=1,
            description="Positive value 1",
        ),
        ValidationScenario(
            name="positive_large",
            validator_type="numeric",
            input_value=999999,
            should_succeed=True,
            expected_value=999999,
            description="Large positive",
        ),
        ValidationScenario(
            name="positive_float",
            validator_type="numeric",
            input_value=0.1,
            should_succeed=True,
            expected_value=0.1,
            description="Positive float",
        ),
        ValidationScenario(
            name="positive_zero",
            validator_type="numeric",
            input_value=0,
            should_succeed=False,
            expected_error_contains="positive",
            description="Zero rejection",
        ),
        ValidationScenario(
            name="positive_negative",
            validator_type="numeric",
            input_value=-5,
            should_succeed=False,
            expected_error_contains="positive",
            description="Negative rejection",
        ),
        ValidationScenario(
            name="positive_none",
            validator_type="numeric",
            input_value=None,
            should_succeed=False,
            expected_error_contains="cannot be None",
            description="None rejection",
        ),
    ]

    RANGE_SCENARIOS: ClassVar[list[ValidationScenario]] = [
        ValidationScenario(
            name="range_within_bounds",
            validator_type="numeric",
            input_value=5,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=True,
            expected_value=5,
            description="Value within range",
        ),
        ValidationScenario(
            name="range_at_min",
            validator_type="numeric",
            input_value=1,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=True,
            expected_value=1,
            description="Value at minimum",
        ),
        ValidationScenario(
            name="range_at_max",
            validator_type="numeric",
            input_value=10,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=True,
            expected_value=10,
            description="Value at maximum",
        ),
        ValidationScenario(
            name="range_below_min",
            validator_type="numeric",
            input_value=0,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=False,
            expected_error_contains="at least",
            description="Value below minimum",
        ),
        ValidationScenario(
            name="range_above_max",
            validator_type="numeric",
            input_value=11,
            input_params={"min_value": 1, "max_value": 10},
            should_succeed=False,
            expected_error_contains="at most",
            description="Value above maximum",
        ),
        ValidationScenario(
            name="range_negative_range",
            validator_type="numeric",
            input_value=-5,
            input_params={"min_value": -10, "max_value": -1},
            should_succeed=True,
            expected_value=-5,
            description="Negative range",
        ),
        ValidationScenario(
            name="range_fractional",
            validator_type="numeric",
            input_value=2.5,
            input_params={"min_value": 0.5, "max_value": 5.5},
            should_succeed=True,
            expected_value=2.5,
            description="Fractional range",
        ),
        ValidationScenario(
            name="range_single_value",
            validator_type="numeric",
            input_value=5,
            input_params={"min_value": 5, "max_value": 5},
            should_succeed=True,
            expected_value=5,
            description="Single value range",
        ),
    ]


# =========================================================================
# Centralized Parser Scenarios
# =========================================================================


class ParserScenarios:
    """Centralized parser scenarios - single source of truth."""

    LDIF_PARSE_SCENARIOS: ClassVar[list[ParserScenario]] = [
        ParserScenario(
            name="parse_simple_dn",
            parser_method="parse",
            input_data="dn: cn=test,dc=example,dc=com",
            should_succeed=True,
            description="Simple DN parsing",
        ),
        ParserScenario(
            name="parse_with_attributes",
            parser_method="parse",
            input_data="dn: cn=test,dc=example,dc=com\nobjectClass: person\ncn: test",
            should_succeed=True,
            description="DN with attributes",
        ),
        ParserScenario(
            name="parse_invalid_dn",
            parser_method="parse",
            input_data="invalid",
            should_succeed=False,
            error_contains="invalid",
            description="Invalid DN format",
        ),
    ]


# =========================================================================
# Centralized Reliability Scenarios
# =========================================================================


class ReliabilityScenarios:
    """Centralized reliability scenarios - single source of truth."""

    RETRY_SCENARIOS: ClassVar[list[ReliabilityScenario]] = [
        ReliabilityScenario(
            name="retry_immediate_success",
            strategy="retry",
            config=m.ConfigMap(
                root={"max_retries": 3, "backoff_type": "constant", "backoff_ms": 10},
            ),
            simulate_failures=0,
            expected_state="success",
            should_succeed=True,
            description="Operation succeeds immediately",
        ),
        ReliabilityScenario(
            name="retry_after_one_failure",
            strategy="retry",
            config=m.ConfigMap(
                root={"max_retries": 3, "backoff_type": "constant", "backoff_ms": 10},
            ),
            simulate_failures=1,
            expected_state="success",
            should_succeed=True,
            description="Succeeds after one retry",
        ),
        ReliabilityScenario(
            name="retry_exhausted",
            strategy="retry",
            config=m.ConfigMap(
                root={"max_retries": 2, "backoff_type": "constant", "backoff_ms": 10},
            ),
            simulate_failures=5,
            expected_state="exhausted",
            should_succeed=False,
            description="All retries exhausted",
        ),
    ]

    CIRCUIT_BREAKER_SCENARIOS: ClassVar[list[ReliabilityScenario]] = [
        ReliabilityScenario(
            name="circuit_initial_closed",
            strategy="circuit_breaker",
            config=m.ConfigMap(root={"failure_threshold": 5, "timeout_ms": 1000}),
            simulate_failures=0,
            expected_state="closed",
            should_succeed=True,
            description="Circuit starts in closed state",
        ),
        ReliabilityScenario(
            name="circuit_open_on_threshold",
            strategy="circuit_breaker",
            config=m.ConfigMap(root={"failure_threshold": 2, "timeout_ms": 1000}),
            simulate_failures=3,
            expected_state="open",
            should_succeed=False,
            description="Circuit opens after threshold",
        ),
    ]
