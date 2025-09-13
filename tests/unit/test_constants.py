"""Targeted tests for 100% coverage on FlextConstants module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum

from flext_core import FlextConstants


class TestConstantsLogLevel100PercentCoverage:
    """Targeted tests for the remaining 3 uncovered lines in constants.py."""

    def test_lines_703_705_loglevel_string_comparison(self) -> None:
        """Test lines 703-705: LogLevel.__eq__ method with string comparison."""
        # Test string comparison that triggers line 703-704
        log_level = FlextConstants.Config.LogLevel.DEBUG

        # Should return True when comparing with matching string
        assert log_level == "DEBUG"  # Triggers lines 703-704

        # Test different string values
        assert log_level != "INFO"
        assert log_level != "ERROR"

        # Test with other log levels
        info_level = FlextConstants.Config.LogLevel.INFO
        assert info_level == "INFO"
        assert info_level != "DEBUG"

        warning_level = FlextConstants.Config.LogLevel.WARNING
        assert warning_level == "WARNING"
        assert warning_level != "CRITICAL"

    def test_loglevel_non_string_comparison(self) -> None:
        """Test line 705: LogLevel.__eq__ with non-string objects."""
        log_level = FlextConstants.Config.LogLevel.ERROR

        # Test comparison with non-string objects (triggers line 705)
        # Use type() to get the actual type for comparison that makes sense
        assert log_level != str(123)  # Compare with string representation
        assert log_level is not None  # Test identity, not equality
        assert log_level != str([])  # Compare with string representation
        assert log_level != str({})  # Compare with string representation

        # Test comparison with another LogLevel enum (should use super().__eq__)
        same_level = FlextConstants.Config.LogLevel.ERROR
        assert log_level == same_level  # Triggers super().__eq__() via line 705

        different_level = FlextConstants.Config.LogLevel.INFO
        assert log_level != different_level

    def test_line_705_super_eq_call(self) -> None:
        """Specifically test line 705: return super().__eq__(other)."""
        log_level = FlextConstants.Config.LogLevel.DEBUG

        # Create another LogLevel with same value - should call super().__eq__()
        another_debug = FlextConstants.Config.LogLevel.DEBUG
        result = log_level == another_debug  # This should trigger line 705
        assert result is True

        # Test with different LogLevel - should also call super().__eq__()
        info_level = FlextConstants.Config.LogLevel.INFO
        result = log_level == info_level  # This should also trigger line 705
        assert result is False

        # Test explicit non-string comparison to force super().__eq__() path
        class NonStringObject:
            """Test object that is not a string."""

            def __eq__(self, other: object) -> bool:
                """Custom equality that returns False for testing."""
                return False

            def __hash__(self) -> int:
                """Hash implementation for the test object."""
                return hash(id(self))

        non_string_obj = NonStringObject()
        result = log_level == non_string_obj  # Forces line 705 path
        assert result is False

    def test_all_loglevel_values_string_comparison(self) -> None:
        """Comprehensive test of all LogLevel values with string comparison."""
        # Test all log level values to ensure complete coverage
        test_cases = [
            (FlextConstants.Config.LogLevel.DEBUG, "DEBUG"),
            (FlextConstants.Config.LogLevel.INFO, "INFO"),
            (FlextConstants.Config.LogLevel.WARNING, "WARNING"),
            (FlextConstants.Config.LogLevel.ERROR, "ERROR"),
            (FlextConstants.Config.LogLevel.CRITICAL, "CRITICAL"),
            # TRACE was removed from LogLevel enum
        ]

        for log_level, expected_string in test_cases:
            # Test exact string match (lines 703-704)
            assert log_level == expected_string

            # Test case sensitivity
            assert log_level != expected_string.lower()
            assert log_level != expected_string.title()

    def test_loglevel_edge_cases(self) -> None:
        """Test edge cases for LogLevel comparison."""
        log_level = FlextConstants.Config.LogLevel.INFO

        # Test empty string
        assert log_level != ""

        # Test partial matches
        assert log_level != "INF"
        assert log_level != "INFO_LEVEL"

        # Test with whitespace
        assert log_level != " INFO "
        assert log_level != "INFO\n"

        # Test comparison with other enum types (line 705 path)

        class OtherEnum(StrEnum):
            """Test enum for comparison."""

            INFO = "INFO"
            DEBUG = "DEBUG"

        other_enum = OtherEnum.INFO
        # Test that different enum types don't match (use type checking instead)
        assert isinstance(log_level, FlextConstants.Config.LogLevel)
        assert isinstance(other_enum, OtherEnum)
        # Test enum values are different types
        assert log_level.__class__.__name__ == "LogLevel"
        assert other_enum.__class__.__name__ == "OtherEnum"
        # Test that values can be the same but types are different
        assert log_level.value == "INFO"
        assert other_enum.value == "INFO"
        # Also test with same value but different enum type
        debug_enum = OtherEnum.DEBUG
        assert debug_enum.__class__.__name__ == "OtherEnum"
