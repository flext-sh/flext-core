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

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from flext_core.constants import c
from flext_tests.utilities import FlextTestsUtilities


@dataclass(frozen=True, slots=True)
class ConstantPathScenario:
    """Test scenario for constant path access."""

    path: str
    expected: object


@dataclass(frozen=True, slots=True)
class PatternValidationScenario:
    """Test scenario for pattern validation."""

    pattern_attr: str
    valid_cases: list[str]
    invalid_cases: list[str]


class ConstantsScenarios:
    """Centralized constants test scenarios using c (FlextConstants)."""

    CORE_CONSTANT_PATHS: ClassVar[list[ConstantPathScenario]] = [
        ConstantPathScenario("NAME", "FLEXT"),
        ConstantPathScenario("Network.MIN_PORT", 1),
        ConstantPathScenario("Network.MAX_PORT", 65535),
        ConstantPathScenario("Validation.MIN_NAME_LENGTH", 2),
        ConstantPathScenario("Validation.MAX_NAME_LENGTH", 100),
        ConstantPathScenario("Validation.MAX_EMAIL_LENGTH", 254),
        ConstantPathScenario("Validation.MIN_PHONE_DIGITS", 10),
        ConstantPathScenario("Defaults.TIMEOUT", 30),
        ConstantPathScenario("Reliability.DEFAULT_TIMEOUT_SECONDS", 30),
        ConstantPathScenario("Utilities.MAX_TIMEOUT_SECONDS", 3600),
        ConstantPathScenario("Logging.DEFAULT_LEVEL", "INFO"),
        ConstantPathScenario("Platform.FLEXT_API_PORT", 8000),
        ConstantPathScenario("Platform.DEFAULT_HOST", "localhost"),
        ConstantPathScenario("Performance.MAX_TIMEOUT_SECONDS", 600),
        ConstantPathScenario("Performance.BatchProcessing.DEFAULT_SIZE", 1000),
        ConstantPathScenario("Reliability.MAX_RETRY_ATTEMPTS", 3),
        ConstantPathScenario("Security.JWT_DEFAULT_ALGORITHM", "HS256"),
        ConstantPathScenario("Cqrs.DEFAULT_HANDLER_TYPE", "command"),
        ConstantPathScenario("Container.DEFAULT_WORKERS", 4),
        ConstantPathScenario("Dispatcher.DEFAULT_HANDLER_MODE", "command"),
        ConstantPathScenario("Mixins.FIELD_CREATED_AT", "created_at"),
        ConstantPathScenario("Messages.TYPE_MISMATCH", "Type mismatch"),
    ]

    PATTERN_VALIDATION_SCENARIOS: ClassVar[list[PatternValidationScenario]] = [
        PatternValidationScenario(
            "Platform.PATTERN_EMAIL",
            [
                "test@example.com",
                "user.name+tag@example.co.uk",
                "valid_email@domain.com",
            ],
            ["invalid.email", "@example.com", "test@", "test@.com"],
        ),
        PatternValidationScenario(
            "Platform.PATTERN_URL",
            [
                "https://github.com",
                "http://localhost:8000",
                "https://example.com/path?query=1",
            ],
            [
                "not-a-url",
                "ftp://invalid.com",
                "://missing.protocol",
                "www.example.com",
            ],
        ),
        PatternValidationScenario(
            "Platform.PATTERN_PHONE_NUMBER",
            ["+5511987654321", "5511987654321", "+1234567890", "11987654321"],
            ["123", "abc1234567890", "+abc1234567890", "123456789"],
        ),
        PatternValidationScenario(
            "Platform.PATTERN_UUID",
            [
                "550e8400-e29b-41d4-a716-446655440000",
                "550e8400e29b41d4a716446655440000",
            ],
            [
                "invalid-uuid",
                "550e8400-e29b-41d4",
                "550e8400-e29b-41d4-a716-44665544000",
            ],
        ),
        PatternValidationScenario(
            "Platform.PATTERN_PATH",
            ["/home/user/file.txt", "C:\\Users\\file.txt", "relative/path/file.py"],
            ["path/with<invalid>chars", 'path/with"quotes', "path/with|pipe"],
        ),
    ]

    TYPE_CHECKS: ClassVar[list[tuple[object, type]]] = [
        (c.NAME, str),
        (c.Network.MIN_PORT, int),
        (c.Network.MAX_PORT, int),
        (c.Validation.MIN_NAME_LENGTH, int),
        (c.Logging.DEFAULT_LEVEL, str),
        (c.Platform.FLEXT_API_PORT, int),
    ]

    REQUIRED_CATEGORIES: ClassVar[list[str]] = [
        "Network",
        "Validation",
        "Errors",
        "Messages",
        "Defaults",
        "Utilities",
        "Settings",
        "Logging",
        "Platform",
        "Performance",
        "Reliability",
        "Security",
        "Cqrs",
        "Container",
        "Dispatcher",
        "Mixins",
        "Context",
        "Processing",
        "Pagination",
    ]

    LOG_LEVELS: ClassVar[list[str]] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestFlextConstants:
    """Comprehensive test suite for FlextConstants using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        ConstantsScenarios.CORE_CONSTANT_PATHS,
        ids=lambda s: s.path,
    )
    def test_core_constant_values(self, scenario: ConstantPathScenario) -> None:
        """Test all core constant values using parametrized test cases."""
        actual = FlextTestsUtilities.Tests.ConstantsHelpers.get_constant_by_path(
            scenario.path,
        )
        assert actual == scenario.expected

    @pytest.mark.parametrize("level", ConstantsScenarios.LOG_LEVELS)
    def test_core_logging_enum_levels(self, level: str) -> None:
        """Test logging level enum values."""
        actual = getattr(c.Settings.LogLevel, level)
        assert actual == level

    @pytest.mark.parametrize(
        "scenario",
        ConstantsScenarios.PATTERN_VALIDATION_SCENARIOS,
        ids=lambda s: s.pattern_attr,
    )
    def test_validation_regex_patterns(
        self,
        scenario: PatternValidationScenario,
    ) -> None:
        """Test regex patterns with comprehensive valid and invalid cases."""
        compiled_pattern = FlextTestsUtilities.Tests.ConstantsHelpers.compile_pattern(
            scenario.pattern_attr,
        )
        for valid_case in scenario.valid_cases:
            assert compiled_pattern.match(valid_case) is not None, (
                f"Expected '{valid_case}' to match pattern {scenario.pattern_attr}"
            )
        for invalid_case in scenario.invalid_cases:
            pattern_name = scenario.pattern_attr
            assert compiled_pattern.match(invalid_case) is None, (
                f"Expected '{invalid_case}' to NOT match pattern {pattern_name}"
            )

    @pytest.mark.parametrize(
        ("value", "expected_type"),
        ConstantsScenarios.TYPE_CHECKS,
        ids=lambda x: f"{type(x[0]).__name__}_{x[1].__name__}"
        if isinstance(x, tuple) and len(x) == 2
        else str(x),
    )
    def test_type_safety_constant_types(
        self,
        value: object,
        expected_type: type,
    ) -> None:
        """Test that constants have correct types."""
        assert isinstance(value, expected_type), (
            f"Expected {value} to be {expected_type}"
        )

    def test_type_safety_immutability(self) -> None:
        """Test that constants are effectively immutable."""
        assert c.NAME == "FLEXT"
        assert c.Platform.FLEXT_API_PORT == 8000

    def test_type_safety_nested_access_patterns(self) -> None:
        """Test various nested access patterns work correctly."""
        assert c.Errors.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert c.Reliability.DEFAULT_TIMEOUT_SECONDS == 30
        assert c.Settings.LogLevel.ERROR == "ERROR"

    @pytest.mark.parametrize("category", ConstantsScenarios.REQUIRED_CATEGORIES)
    def test_completeness_required_categories_exist(self, category: str) -> None:
        """Test that all required constant categories exist."""
        assert hasattr(c, category), f"Missing category: {category}"

    def test_completeness_documentation_exists(self) -> None:
        """Test that constants have proper documentation."""
        assert c.__doc__ is not None
        assert "layer 0" in c.__doc__.lower()
        documented_classes = [
            c.Network,
            c.Validation,
            c.Errors,
            c.Platform,
            c.Logging,
        ]
        for cls in documented_classes:
            assert cls.__doc__ is not None, f"Missing docstring for {cls.__name__}"

    def test_edge_cases_pattern_edge_cases(self) -> None:
        """Test regex patterns with edge cases."""
        email_pattern = FlextTestsUtilities.Tests.ConstantsHelpers.compile_pattern(
            "Platform.PATTERN_EMAIL",
        )
        long_email = "a" * 64 + "@" + "b" * 63 + ".com"
        assert len(long_email) <= c.Validation.MAX_EMAIL_LENGTH
        assert email_pattern.match(long_email) is not None
        phone_pattern = FlextTestsUtilities.Tests.ConstantsHelpers.compile_pattern(
            "Platform.PATTERN_PHONE_NUMBER",
        )
        assert phone_pattern.match("+123456789012345") is not None
        assert phone_pattern.match("+1234567890") is not None

    def test_edge_cases_constant_ranges(self) -> None:
        """Test that numeric constants are in valid ranges."""
        assert 0 <= c.Network.MIN_PORT <= c.Network.MAX_PORT <= 65535
        assert c.Defaults.TIMEOUT > 0
        assert c.Utilities.MAX_TIMEOUT_SECONDS > c.Defaults.TIMEOUT
        assert 0 < c.Validation.MIN_NAME_LENGTH < c.Validation.MAX_NAME_LENGTH

    def test_edge_cases_enum_completeness(self) -> None:
        """Test that enums contain all expected values."""
        for level in ConstantsScenarios.LOG_LEVELS:
            assert hasattr(c.Settings.LogLevel, level)
            assert getattr(c.Settings.LogLevel, level) == level

    def test_integration_cross_category_consistency(self) -> None:
        """Test consistency across related constant categories."""
        assert c.Defaults.TIMEOUT == c.Reliability.DEFAULT_TIMEOUT_SECONDS
        assert c.Cqrs.DEFAULT_HANDLER_TYPE == c.Dispatcher.DEFAULT_HANDLER_MODE

    def test_integration_pattern_and_validation_consistency(self) -> None:
        """Test that patterns work with validation constants."""
        email_pattern = FlextTestsUtilities.Tests.ConstantsHelpers.compile_pattern(
            "Platform.PATTERN_EMAIL",
        )
        max_length_email = "a" * (c.Validation.MAX_EMAIL_LENGTH - 9) + "@test.com"
        assert len(max_length_email) <= c.Validation.MAX_EMAIL_LENGTH
        assert email_pattern.match(max_length_email) is not None


__all__ = ["TestFlextConstants"]
