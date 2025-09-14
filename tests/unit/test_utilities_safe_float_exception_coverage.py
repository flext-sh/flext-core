"""Targeted test for safe_float exception handling coverage - lines 125-126.

 This test file specifically targets the missing coverage lines 125-126 in
src/flext_core/utilities.py safe_float method exception handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities


class TestSafeFloatExceptionCoverage:
    """Targeted tests for safe_float exception handling coverage."""

    def test_safe_float_value_error_exception_path(self) -> None:
        """Test safe_float ValueError exception path - targets line 125-126."""
        # These should trigger ValueError in float() conversion
        test_cases = [
            ("invalid_string", 0.0, 0.0),
            ("abc123", 0.0, 0.0),
            ("not_a_number", 5.5, 5.5),
            ("", 0.0, 0.0),  # Empty string
            ("   ", 0.0, 0.0),  # Whitespace only
            ("inf++", 0.0, 0.0),  # Invalid infinity format
            ("12.34.56", 0.0, 0.0),  # Multiple decimal points
            ("12e", 0.0, 0.0),  # Incomplete scientific notation
            ("12e++", 0.0, 0.0),  # Invalid scientific notation
        ]

        for test_input, default_val, expected in test_cases:
            result = FlextUtilities.Conversions.safe_float(
                test_input, default=default_val
            )
            assert result == expected, (
                f"safe_float('{test_input}', default={default_val}) should return {expected}, got {result}"
            )

    def test_safe_float_type_error_exception_path(self) -> None:
        """Test safe_float TypeError exception path - targets line 125-126."""
        # These should trigger TypeError in float() conversion
        # Note: The method signature is str | float, but we need to test edge cases

        # Test with None (common edge case that causes TypeError)
        result = FlextUtilities.Conversions.safe_float(None, default=42.0)
        assert result == 42.0

        # Test with complex number (causes TypeError)
        result = FlextUtilities.Conversions.safe_float(complex(1, 2), default=99.9)
        assert result == 99.9

        # Test with list (causes TypeError)
        result = FlextUtilities.Conversions.safe_float([1, 2, 3], default=33.3)
        assert result == 33.3

        # Test with dict (causes TypeError)
        result = FlextUtilities.Conversions.safe_float({"key": "value"}, default=77.7)
        assert result == 77.7

    def test_safe_float_both_exception_types_comprehensive(self) -> None:
        """Comprehensive test covering both ValueError and TypeError paths."""
        # Mix of inputs that should trigger different exception types
        test_cases = [
            # ValueError cases (invalid string representations)
            ("invalid", 10.0),
            ("not_float", 20.0),
            ("12.34.56", 30.0),  # Multiple decimals
            # TypeError cases (wrong types)
            (None, 40.0),
            (complex(1, 1), 50.0),
            ([1, 2], 60.0),
            ({"a": 1}, 70.0),
            (object(), 80.0),
        ]

        for test_input, expected_default in test_cases:
            result = FlextUtilities.Conversions.safe_float(
                test_input, default=expected_default
            )
            assert result == expected_default, (
                f"safe_float({test_input!r}, default={expected_default}) should return {expected_default}"
            )

    def test_safe_float_exception_handling_without_default(self) -> None:
        """Test exception handling when using default parameter value (0.0)."""
        # Test ValueError path with default 0.0
        result = FlextUtilities.Conversions.safe_float("invalid_string")
        assert result == 0.0

        # Test TypeError path with default 0.0
        result = FlextUtilities.Conversions.safe_float(None)
        assert result == 0.0

        result = FlextUtilities.Conversions.safe_float(complex(1, 2))
        assert result == 0.0

    def test_safe_float_successful_conversions_still_work(self) -> None:
        """Verify that normal successful conversions still work properly."""
        # Valid string conversions
        assert FlextUtilities.Conversions.safe_float("123.45") == 123.45
        assert FlextUtilities.Conversions.safe_float("0.0") == 0.0
        assert FlextUtilities.Conversions.safe_float("-456.78") == -456.78

        # Valid float inputs
        assert FlextUtilities.Conversions.safe_float(789.12) == 789.12
        assert FlextUtilities.Conversions.safe_float(0.0) == 0.0
        assert FlextUtilities.Conversions.safe_float(-123.45) == -123.45
