"""Debug coverage test for utilities.py safe_float method lines 125-126.

This test specifically targets the exact missing lines in safe_float exception handling
using a trace-based approach to ensure coverage detection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities


class TestUtilitiesDebugCoverage:
    """Debug tests for utilities coverage using direct exception forcing."""

    def test_safe_float_exception_paths_real_inputs(self) -> None:
        """Test safe_float exception paths using real invalid inputs (no mocks)."""
        # ValueError: invalid string for float conversion
        result1 = FlextUtilities.Conversions.safe_float("not_a_float", default=111.0)
        assert result1 == 111.0

        # TypeError: non-convertible object
        result2 = FlextUtilities.Conversions.safe_float(object(), default=222.0)
        assert result2 == 222.0

    def test_safe_float_comprehensive_edge_cases(self) -> None:
        """Test comprehensive edge cases to ensure all exception paths are covered."""
        # Edge cases that might not be covered by regular tests
        edge_cases = [
            # String cases that should cause ValueError
            ("", 1.1),
            ("   ", 2.2),  # Whitespace
            ("1.2.3.4", 5.5),  # Multiple dots
            ("12e", 6.6),  # Incomplete scientific notation
            ("e12", 7.7),  # Invalid scientific notation
            ("12e+-3", 8.8),  # Invalid scientific notation with conflicting signs
            ("++12", 9.9),  # Multiple signs
            ("12++", 10.1),  # Trailing invalid characters
            # Type cases that should cause TypeError
            (None, 11.1),
            (complex(0, 1), 12.2),
            ([], 13.3),
            ({}, 14.4),
            (set(), 15.5),
            (frozenset(), 16.6),
            (object(), 17.7),
            (type, 19.9),
            (Exception(), 20.0),
        ]

        for test_input, expected_default in edge_cases:
            result = FlextUtilities.Conversions.safe_float(
                test_input, default=expected_default
            )
            assert result == expected_default, (
                f"safe_float({test_input!r}) should return {expected_default}, got {result}"
            )
