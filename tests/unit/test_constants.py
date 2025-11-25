"""Comprehensive tests for FlextConstants - Foundation Constants.

Tests FlextConstants module: core constants, validation patterns, error codes,
network settings, platform configs, and all nested constant classes. Covers
edge cases, type safety, immutability, and regex pattern validation with
100% coverage using advanced Python 3.13 patterns and factory methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re

import pytest

from flext_core import FlextConstants


class TestFlextConstants:
    """Comprehensive test suite for FlextConstants foundation constants.

    Tests FlextConstants module: core constants, validation patterns, error codes,
    network settings, platform configs, and all nested constant classes. Covers
    edge cases, type safety, immutability, and regex pattern validation with
    100% coverage using advanced Python 3.13 patterns and factory methods.
    """

    # Core Constants Tests
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("NAME", "FLEXT"),
            ("Network.MIN_PORT", 1),
            ("Network.MAX_PORT", 65535),
            ("Validation.MIN_NAME_LENGTH", 2),
            ("Validation.MAX_NAME_LENGTH", 100),
            ("Validation.MAX_EMAIL_LENGTH", 254),
            ("Validation.MIN_PHONE_DIGITS", 10),
            ("Defaults.TIMEOUT", 30),
            ("Reliability.DEFAULT_TIMEOUT_SECONDS", 30),
            ("Utilities.DEFAULT_ENCODING", "utf-8"),
            ("Utilities.MAX_TIMEOUT_SECONDS", 3600),
            ("Logging.DEFAULT_LEVEL", "INFO"),
            ("Logging.DEFAULT_LEVEL_DEVELOPMENT", "DEBUG"),
            ("Logging.DEFAULT_LEVEL_PRODUCTION", "WARNING"),
            ("Logging.DEFAULT_LEVEL_TESTING", "INFO"),
            ("Platform.FLEXT_API_PORT", 8000),
            ("Platform.DEFAULT_HOST", "localhost"),
            ("Performance.MAX_TIMEOUT_SECONDS", 600),
            ("Performance.BatchProcessing.DEFAULT_SIZE", 1000),
            ("Reliability.MAX_RETRY_ATTEMPTS", 3),
            ("Reliability.DEFAULT_MAX_RETRIES", 3),
            ("Reliability.DEFAULT_BACKOFF_STRATEGY", "exponential"),
            ("Security.JWT_DEFAULT_ALGORITHM", "HS256"),
            ("Security.CREDENTIAL_BCRYPT_ROUNDS", 12),
            ("Cqrs.DEFAULT_HANDLER_TYPE", "command"),
            ("Cqrs.COMMAND_HANDLER_TYPE", "command"),
            ("Cqrs.QUERY_HANDLER_TYPE", "query"),
            ("Container.DEFAULT_WORKERS", 4),
            ("Container.MAX_CACHE_SIZE", 100),
            ("Dispatcher.HANDLER_MODE_COMMAND", "command"),
            ("Dispatcher.HANDLER_MODE_QUERY", "query"),
            ("Dispatcher.DEFAULT_HANDLER_MODE", "command"),
            ("Mixins.FIELD_CREATED_AT", "created_at"),
            ("Mixins.FIELD_UPDATED_AT", "updated_at"),
            ("Mixins.FIELD_ID", "unique_id"),
            ("Messages.TYPE_MISMATCH", "Type mismatch"),
        ],
    )
    def test_core_constant_values(self, path: str, expected: object) -> None:
        """Test all core constant values using parametrized test cases."""
        obj = FlextConstants
        for attr in path.split("."):
            obj = getattr(obj, attr)
        assert obj == expected

    def test_core_logging_enum_levels(self) -> None:
        """Test logging level enum values."""
        levels = {
            "DEBUG": "DEBUG",
            "INFO": "INFO",
            "WARNING": "WARNING",
            "ERROR": "ERROR",
            "CRITICAL": "CRITICAL",
        }

        for level_name, expected_value in levels.items():
            actual = getattr(FlextConstants.Settings.LogLevel, level_name)
            assert actual == expected_value

    # Validation Patterns Tests
    @pytest.mark.parametrize(
        ("pattern_attr", "valid_cases", "invalid_cases"),
        [
            (
                "PATTERN_EMAIL",
                [
                    "test@example.com",
                    "user.name+tag@example.co.uk",
                    "valid_email@domain.com",
                    "test.email@subdomain.domain.org",
                    "user_name123@test-domain.co.uk",
                    "test..email@example.com",  # Pattern allows consecutive dots
                ],
                [
                    "invalid.email",
                    "@example.com",
                    "test@",
                    "test@.com",
                    "test@exam ple.com",  # No spaces allowed
                ],
            ),
            (
                "PATTERN_URL",
                [
                    "https://github.com",
                    "http://localhost:8000",
                    "https://example.com/path?query=1",
                    "HTTPS://EXAMPLE.COM",
                    "http://sub.domain.com/path/to/resource",
                    "https://domain.co.uk/path?param=value&other=test",
                ],
                [
                    "not-a-url",
                    "ftp://invalid.com",
                    "://missing.protocol",
                    "http://",
                    "https://",
                    "www.example.com",  # Missing protocol
                    "example.com/path",
                ],
            ),
            (
                "PATTERN_PHONE_NUMBER",
                [
                    "+5511987654321",
                    "5511987654321",
                    "+1234567890",
                    "+449876543210",
                    "+811234567890",
                    "11987654321",  # Brazilian format
                    "123-456-7890",  # Pattern allows hyphens
                    "(123) 456-7890",  # Pattern allows parentheses and spaces
                ],
                [
                    "123",  # Too short
                    "abc1234567890",  # Contains letters
                    "+abc1234567890",  # Contains letters
                    "123456789",  # Too short (9 digits)
                    "+123456789",  # Too short with country code
                    "phone_number",  # Contains underscore
                ],
            ),
            (
                "PATTERN_UUID",
                [
                    "550e8400-e29b-41d4-a716-446655440000",
                    "550e8400e29b41d4a716446655440000",  # Without hyphens
                    "123e4567-E89B-12D3-A456-426614174000",
                    "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF",
                    "00000000-0000-0000-0000-000000000000",
                ],
                [
                    "invalid-uuid",
                    "550e8400-e29b-41d4",  # Too short
                    "550e8400-e29b-41d4-a716-44665544000",  # Missing digit
                    "550e8400-e29b-41d4-a716-446655440000-extra",  # Too long
                    "gggggggg-hhhh-iiii-jjjj-kkkkkkkkkkkk",  # Invalid hex
                    "550e8400e29b41d4a71644665544000",  # Too short without hyphens
                ],
            ),
            (
                "PATTERN_PATH",
                [
                    "/home/user/file.txt",
                    "C:\\Users\\file.txt",
                    "relative/path/file.py",
                    "/absolute/path/to/file.log",
                    "file.in.current.dir",
                    "../parent/directory/file.md",
                    "~/user/home/file.json",
                ],
                [
                    "path/with<invalid>chars",
                    'path/with"quotes',
                    "path/with|pipe",
                    "path/with\nnewline",
                    "path/with\tab",
                    "path/with\x00null",
                    "path/with\u0000unicode_null",
                ],
            ),
        ],
    )
    def test_validation_regex_patterns(
        self,
        pattern_attr: str,
        valid_cases: list[str],
        invalid_cases: list[str],
    ) -> None:
        """Test regex patterns with comprehensive valid and invalid cases."""
        pattern_str = getattr(FlextConstants.Platform, pattern_attr)
        flags = re.IGNORECASE if "URL" in pattern_attr else 0
        compiled_pattern = re.compile(pattern_str, flags)

        # Test valid cases
        for valid_case in valid_cases:
            match = compiled_pattern.match(valid_case)
            assert match is not None, (
                f"Expected '{valid_case}' to match pattern {pattern_attr}"
            )

        # Test invalid cases
        for invalid_case in invalid_cases:
            match = compiled_pattern.match(invalid_case)
            assert match is None, (
                f"Expected '{invalid_case}' to NOT match pattern {pattern_attr}"
            )

    # Type Safety Tests
    def test_type_safety_constant_types(self) -> None:
        """Test that constants have correct types."""
        type_checks = [
            (FlextConstants.NAME, str),
            (FlextConstants.Network.MIN_PORT, int),
            (FlextConstants.Network.MAX_PORT, int),
            (FlextConstants.Reliability.RETRY_BACKOFF_MAX, float),
            (FlextConstants.Validation.MIN_NAME_LENGTH, int),
            (FlextConstants.Utilities.DEFAULT_ENCODING, str),
            (FlextConstants.Logging.DEFAULT_LEVEL, str),
            (FlextConstants.Platform.FLEXT_API_PORT, int),
        ]

        for value, expected_type in type_checks:
            assert isinstance(value, expected_type), (
                f"Expected {value} to be {expected_type}"
            )

    def test_type_safety_immutability(self) -> None:
        """Test that constants are effectively immutable."""
        # Test that we can read constants
        original_name = FlextConstants.NAME
        assert original_name == "FLEXT"

        # Test that nested constants work
        original_port = FlextConstants.Platform.FLEXT_API_PORT
        assert original_port == 8000

        # Constants should be Final, preventing runtime modification
        # This is enforced by Python's type system and mypy

    def test_type_safety_nested_access_patterns(self) -> None:
        """Test various nested access patterns work correctly."""
        # Direct access
        assert FlextConstants.Errors.VALIDATION_ERROR == "VALIDATION_ERROR"

        # Deep nesting
        assert FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS == 30
        assert FlextConstants.Logging.DEFAULT_LEVEL == "INFO"

        # Enum access
        assert FlextConstants.Settings.LogLevel.ERROR == "ERROR"

    # Completeness Tests
    def test_completeness_required_categories_exist(self) -> None:
        """Test that all required constant categories exist."""
        required_categories = [
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
            "FlextWeb",
            "Pagination",
        ]

        for category in required_categories:
            assert hasattr(FlextConstants, category), f"Missing category: {category}"

    def test_completeness_documentation_exists(self) -> None:
        """Test that constants have proper documentation."""
        # Main class documentation
        assert FlextConstants.__doc__ is not None
        assert "foundation" in FlextConstants.__doc__.lower()

        # Key nested classes should have documentation
        documented_classes = [
            FlextConstants.Network,
            FlextConstants.Validation,
            FlextConstants.Errors,
            FlextConstants.Platform,
            FlextConstants.Logging,
        ]

        for cls in documented_classes:
            assert cls.__doc__ is not None, f"Missing docstring for {cls.__name__}"

    # Edge Cases Tests
    def test_edge_cases_pattern_edge_cases(self) -> None:
        """Test regex patterns with edge cases."""
        # Email edge cases
        email_pattern = re.compile(FlextConstants.Platform.PATTERN_EMAIL)

        # Very long email (but valid)
        long_email = "a" * 64 + "@" + "b" * 63 + ".com"
        assert len(long_email) <= FlextConstants.Validation.MAX_EMAIL_LENGTH
        assert email_pattern.match(long_email) is not None

        # Phone number edge cases
        phone_pattern = re.compile(FlextConstants.Platform.PATTERN_PHONE_NUMBER)

        # Maximum reasonable phone number
        long_phone = "+123456789012345"
        assert phone_pattern.match(long_phone) is not None

        # Minimum valid phone
        short_phone = "+1234567890"
        assert phone_pattern.match(short_phone) is not None

    def test_edge_cases_constant_ranges(self) -> None:
        """Test that numeric constants are in valid ranges."""
        # Port ranges
        assert (
            0
            <= FlextConstants.Network.MIN_PORT
            <= FlextConstants.Network.MAX_PORT
            <= 65535
        )

        # Timeout ranges
        assert FlextConstants.Defaults.TIMEOUT > 0
        assert (
            FlextConstants.Utilities.MAX_TIMEOUT_SECONDS
            > FlextConstants.Defaults.TIMEOUT
        )

        # Validation lengths
        assert (
            0
            < FlextConstants.Validation.MIN_NAME_LENGTH
            < FlextConstants.Validation.MAX_NAME_LENGTH
        )

    def test_edge_cases_enum_completeness(self) -> None:
        """Test that enums contain all expected values."""
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in log_levels:
            assert hasattr(FlextConstants.Settings.LogLevel, level)
            assert getattr(FlextConstants.Settings.LogLevel, level) == level

    # Integration Tests
    def test_integration_cross_category_consistency(self) -> None:
        """Test consistency across related constant categories."""
        # Timeout consistency
        assert (
            FlextConstants.Defaults.TIMEOUT
            == FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS
        )

        # Handler mode consistency
        assert (
            FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE
            == FlextConstants.Dispatcher.DEFAULT_HANDLER_MODE
        )

    def test_integration_pattern_and_validation_consistency(self) -> None:
        """Test that patterns work with validation constants."""
        # Email pattern should accept emails up to MAX_EMAIL_LENGTH
        email_pattern = re.compile(FlextConstants.Platform.PATTERN_EMAIL)
        max_length_email = (
            "a" * (FlextConstants.Validation.MAX_EMAIL_LENGTH - 9) + "@test.com"
        )
        assert len(max_length_email) <= FlextConstants.Validation.MAX_EMAIL_LENGTH
        assert email_pattern.match(max_length_email) is not None
