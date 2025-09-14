"""Debug coverage test for utilities.py safe_float method lines 125-126.

This test specifically targets the exact missing lines in safe_float exception handling
using a trace-based approach to ensure coverage detection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import builtins

from flext_core import FlextUtilities


class TestUtilitiesDebugCoverage:
    """Debug tests for utilities coverage using direct exception forcing."""

    def test_safe_float_exception_paths_with_monkey_patch(self) -> None:
        """Test safe_float by temporarily patching float() to force exceptions."""
        original_float = builtins.float
        exception_counter = 0

        def counting_float(value: object) -> float:
            nonlocal exception_counter
            # Force exceptions for certain values to ensure coverage
            if value == "force_value_error":
                exception_counter += 1
                msg = "Forced ValueError for coverage"
                raise ValueError(msg)
            if value == "force_type_error":
                exception_counter += 1
                msg = "Forced TypeError for coverage"
                raise TypeError(msg)
            return original_float(value)

        try:
            # Temporarily replace float()
            builtins.float = counting_float

            # Test forced ValueError path
            result1 = FlextUtilities.Conversions.safe_float(
                "force_value_error", default=111.0
            )
            assert result1 == 111.0, (
                f"Forced ValueError should return default, got {result1}"
            )

            # Test forced TypeError path
            result2 = FlextUtilities.Conversions.safe_float(
                "force_type_error", default=222.0
            )
            assert result2 == 222.0, (
                f"Forced TypeError should return default, got {result2}"
            )

            # Verify exceptions were actually raised and caught
            assert exception_counter == 2, (
                f"Expected 2 exceptions, got {exception_counter}"
            )

        finally:
            # Restore original float()
            builtins.float = original_float

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
