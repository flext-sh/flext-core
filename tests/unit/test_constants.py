"""Comprehensive tests for FlextConstants - Foundation Constants.

Module: flext_core.constants
Scope: FlextConstants - core constants, validation patterns, error codes,
network settings, platform configs, and all nested constant classes

Tests FlextConstants functionality including:
- Core constant values and nested access
- Validation regex patterns
- Type safety and immutability
- Completeness checks
- Edge cases and integration

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar, cast

import pytest
from flext_tests import c, t, t as test_t, tm, u
from pydantic import BaseModel, ConfigDict, Field


class TestConstants:
    """Comprehensive test suite for FlextConstants using u."""

    class ConstantPathScenario(BaseModel):
        """Test scenario for constant path access."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        path: Annotated[str, Field(description="Constant access path")]
        expected: Annotated[str | int, Field(description="Expected constant value")]

    class PatternValidationScenario(BaseModel):
        """Test scenario for pattern validation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        pattern_attr: Annotated[str, Field(description="Pattern attribute path")]
        valid_cases: Annotated[
            t.StrSequence,
            Field(description="Inputs expected to match the pattern"),
        ]
        invalid_cases: Annotated[
            t.StrSequence,
            Field(description="Inputs expected to fail the pattern"),
        ]

    CORE_CONSTANT_PATHS: ClassVar[Sequence[ConstantPathScenario]] = [
        ConstantPathScenario(path="NAME", expected="FLEXT"),
        ConstantPathScenario(path="MIN_PORT", expected=1),
        ConstantPathScenario(path="MAX_PORT", expected=65535),
        ConstantPathScenario(path="MIN_NAME_LENGTH", expected=2),
        ConstantPathScenario(path="MAX_NAME_LENGTH", expected=100),
        ConstantPathScenario(path="MAX_EMAIL_LENGTH", expected=254),
        ConstantPathScenario(path="MIN_PHONE_DIGITS", expected=10),
        ConstantPathScenario(path="DEFAULT_TIMEOUT_SECONDS", expected=30),
        ConstantPathScenario(path="MAX_TIMEOUT_SECONDS", expected=3600),
        ConstantPathScenario(path="DEFAULT_LEVEL", expected="INFO"),
        ConstantPathScenario(path="FLEXT_API_PORT", expected=8000),
        ConstantPathScenario(path="DEFAULT_HOST", expected=c.LOCALHOST),
        ConstantPathScenario(path="MAX_TIMEOUT_SECONDS_PERFORMANCE", expected=600),
        ConstantPathScenario(path="DEFAULT_BATCH_SIZE", expected=1000),
        ConstantPathScenario(path="MAX_RETRY_ATTEMPTS", expected=3),
        ConstantPathScenario(path="JWT_DEFAULT_ALGORITHM", expected="HS256"),
        ConstantPathScenario(path="DEFAULT_HANDLER_TYPE", expected="command"),
        ConstantPathScenario(path="DEFAULT_WORKERS", expected=4),
        ConstantPathScenario(path="DEFAULT_HANDLER_MODE", expected="command"),
        ConstantPathScenario(path="FIELD_CREATED_AT", expected="created_at"),
        ConstantPathScenario(path="TYPE_MISMATCH", expected="Type mismatch"),
    ]
    PATTERN_VALIDATION_SCENARIOS: ClassVar[Sequence[PatternValidationScenario]] = [
        PatternValidationScenario(
            pattern_attr="PATTERN_EMAIL",
            valid_cases=[
                "test@example.com",
                "user.name+tag@example.co.uk",
                "valid_email@domain.com",
            ],
            invalid_cases=["invalid.email", "@example.com", "test@", "test@.com"],
        ),
        PatternValidationScenario(
            pattern_attr="PATTERN_URL",
            valid_cases=[
                "https://github.com",
                "http://FlextConstants.LOCALHOST:8000",
                "https://example.com/path?query=1",
            ],
            invalid_cases=[
                "not-a-url",
                "ftp://invalid.com",
                "://missing.protocol",
                "www.example.com",
            ],
        ),
        PatternValidationScenario(
            pattern_attr="PATTERN_PHONE_NUMBER",
            valid_cases=[
                "+5511987654321",
                "5511987654321",
                "+1234567890",
                "11987654321",
            ],
            invalid_cases=["123", "abc1234567890", "+abc1234567890", "123456789"],
        ),
        PatternValidationScenario(
            pattern_attr="PATTERN_UUID",
            valid_cases=[
                "550e8400-e29b-41d4-a716-446655440000",
                "550e8400e29b41d4a716446655440000",
            ],
            invalid_cases=[
                "invalid-uuid",
                "550e8400-e29b-41d4",
                "550e8400-e29b-41d4-a716-44665544000",
            ],
        ),
        PatternValidationScenario(
            pattern_attr="PATTERN_PATH",
            valid_cases=[
                "/home/user/file.txt",
                "C:\\Users\\file.txt",
                "relative/path/file.py",
            ],
            invalid_cases=[
                "path/with<invalid>chars",
                'path/with"quotes',
                "path/with|pipe",
            ],
        ),
    ]
    TYPE_CHECKS: ClassVar[Sequence[tuple[str | int, type]]] = [
        (c.NAME, str),
        (c.DEFAULT_RETRY_DELAY_SECONDS, int),
        (c.MAX_PORT, int),
        (c.MIN_NAME_LENGTH, int),
        (c.DEFAULT_LEVEL, str),
        (c.FLEXT_API_PORT, int),
    ]
    TYPE_CHECK_IDS: ClassVar[t.StrSequence] = [
        "name_str",
        "network_min_port_int",
        "network_max_port_int",
        "validation_min_name_length_int",
        "logging_default_level_str",
        "platform_flext_api_port_int",
    ]
    REQUIRED_ATTRIBUTES: ClassVar[t.StrSequence] = [
        "MIN_PORT",
        "MAX_PORT",
        "MIN_NAME_LENGTH",
        "MAX_EMAIL_LENGTH",
        "VALIDATION_ERROR",
        "TYPE_MISMATCH",
        "DEFAULT_TIMEOUT_SECONDS",
        "MAX_TIMEOUT_SECONDS",
        "LogLevel",
        "FLEXT_API_PORT",
        "MAX_TIMEOUT_SECONDS_PERFORMANCE",
        "MAX_RETRY_ATTEMPTS",
        "JWT_DEFAULT_ALGORITHM",
        "DEFAULT_HANDLER_TYPE",
        "DEFAULT_WORKERS",
        "DEFAULT_HANDLER_MODE",
        "FIELD_CREATED_AT",
        "DEFAULT_PAGE_SIZE",
        "ErrorType",
    ]
    LOG_LEVELS: ClassVar[t.StrSequence] = [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ]

    @pytest.mark.parametrize(
        "scenario",
        CORE_CONSTANT_PATHS,
        ids=lambda s: s.path,
    )
    def test_core_constant_values(self, scenario: ConstantPathScenario) -> None:
        """Test all core constant values using parametrized test cases."""
        actual = u.Tests.ConstantsHelpers.get_constant_by_path(scenario.path)
        tm.that(str(actual), eq=str(scenario.expected))

    @pytest.mark.parametrize("level", LOG_LEVELS)
    def test_core_logging_enum_levels(self, level: str) -> None:
        """Test logging level enum values."""
        actual = getattr(c.LogLevel, level)
        tm.that(actual, eq=level)

    @pytest.mark.parametrize(
        "scenario",
        PATTERN_VALIDATION_SCENARIOS,
        ids=lambda s: s.pattern_attr,
    )
    def test_validation_regex_patterns(
        self,
        scenario: PatternValidationScenario,
    ) -> None:
        """Test regex patterns with comprehensive valid and invalid cases."""
        compiled_pattern = u.Tests.ConstantsHelpers.compile_pattern(
            scenario.pattern_attr,
        )
        for valid_case in scenario.valid_cases:
            match_result = compiled_pattern.match(valid_case)
            tm.that(
                cast("test_t.NormalizedValue", match_result),
                none=False,
                msg=f"Expected '{valid_case}' to match pattern {scenario.pattern_attr}",
            )
        for invalid_case in scenario.invalid_cases:
            pattern_name = scenario.pattern_attr
            match_result = compiled_pattern.match(invalid_case)
            tm.that(
                cast("test_t.NormalizedValue", match_result),
                none=True,
                msg=f"Expected '{invalid_case}' to NOT match pattern {pattern_name}",
            )

    @pytest.mark.parametrize(
        ("value", "expected_type"),
        TYPE_CHECKS,
        ids=TYPE_CHECK_IDS,
    )
    def test_type_safety_constant_types(
        self,
        value: test_t.NormalizedValue,
        expected_type: type,
    ) -> None:
        """Test that constants have correct types."""
        tm.that(value, is_=expected_type, msg=f"Expected {value} to be {expected_type}")

    def test_type_safety_immutability(self) -> None:
        """Test that constants are effectively immutable."""
        tm.that(c.NAME, eq="FLEXT")
        tm.that(c.FLEXT_API_PORT, eq=8000)

    def test_type_safety_nested_access_patterns(self) -> None:
        """Test various nested access patterns work correctly."""
        tm.that(c.VALIDATION_ERROR, eq="VALIDATION_ERROR")
        tm.that(c.DEFAULT_TIMEOUT_SECONDS, eq=30)
        tm.that(c.LogLevel.ERROR, eq="ERROR")

    @pytest.mark.parametrize("attr", REQUIRED_ATTRIBUTES)
    def test_completeness_required_attributes_exist(self, attr: str) -> None:
        """Test that all required constant attributes exist."""
        tm.that(hasattr(c, attr), eq=True, msg=f"Missing attribute: {attr}")

    def test_completeness_documentation_exists(self) -> None:
        """Test that constants have proper documentation."""
        tm.that(c.__doc__, none=False)
        doc_lower = c.__doc__.lower() if c.__doc__ else ""
        tm.that(doc_lower, has="layer 0")

    def test_edge_cases_pattern_edge_cases(self) -> None:
        """Test regex patterns with edge cases."""
        email_pattern = u.Tests.ConstantsHelpers.compile_pattern(
            "PATTERN_EMAIL",
        )
        long_email = "a" * 64 + "@" + "b" * 63 + ".com"
        tm.that(len(long_email), lte=c.MAX_EMAIL_LENGTH)
        tm.that(
            cast("test_t.NormalizedValue", email_pattern.match(long_email)),
            none=False,
        )
        phone_pattern = u.Tests.ConstantsHelpers.compile_pattern(
            "PATTERN_PHONE_NUMBER",
        )
        tm.that(
            cast(
                "test_t.NormalizedValue",
                phone_pattern.match("+123456789012345"),
            ),
            none=False,
        )
        tm.that(
            cast("test_t.NormalizedValue", phone_pattern.match("+1234567890")),
            none=False,
        )

    def test_edge_cases_constant_ranges(self) -> None:
        """Test that numeric constants are in valid ranges."""
        tm.that(c.DEFAULT_RETRY_DELAY_SECONDS, gte=0)
        tm.that(c.MAX_PORT, lte=65535)
        tm.that(c.DEFAULT_RETRY_DELAY_SECONDS, lte=c.MAX_PORT)
        tm.that(c.DEFAULT_TIMEOUT_SECONDS, gt=0)
        tm.that(
            c.MAX_TIMEOUT_SECONDS,
            gt=c.DEFAULT_TIMEOUT_SECONDS,
        )
        tm.that(c.MIN_NAME_LENGTH, gt=0)
        tm.that(c.MIN_NAME_LENGTH, lt=c.HTTP_STATUS_MIN)

    def test_edge_cases_enum_completeness(self) -> None:
        """Test that enums contain all expected values."""
        for level in self.LOG_LEVELS:
            tm.that(hasattr(c.LogLevel, level), eq=True)
            tm.that(getattr(c.LogLevel, level), eq=level)

    def test_integration_cross_category_consistency(self) -> None:
        """Test consistency across related constant categories."""
        tm.that(
            c.DEFAULT_TIMEOUT_SECONDS,
            eq=c.DEFAULT_TIMEOUT_SECONDS,
        )
        tm.that(c.DEFAULT_HANDLER_TYPE, eq=c.DEFAULT_HANDLER_MODE)

    def test_integration_pattern_and_validation_consistency(self) -> None:
        """Test that patterns work with validation constants."""
        email_pattern = u.Tests.ConstantsHelpers.compile_pattern(
            "PATTERN_EMAIL",
        )
        max_length_email = "a" * (c.MAX_EMAIL_LENGTH - 9) + "@test.com"
        tm.that(len(max_length_email), lte=c.MAX_EMAIL_LENGTH)
        tm.that(
            cast("test_t.NormalizedValue", email_pattern.match(max_length_email)),
            none=False,
        )


__all__ = ["TestConstants"]
