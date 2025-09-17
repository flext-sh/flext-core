"""Comprehensive tests to achieve 100% coverage for FlextUtilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities


class TestFlextUtilities100Percent:
    """Tests targeting 100% coverage for FlextUtilities."""

    def test_safe_bool_exception_handling(self) -> None:
        """Test safe_bool exception handling - lines 129-130."""

        # Test ValueError exception
        class BadValue:
            def __bool__(self) -> bool:
                error_msg = "Bad value"
                raise ValueError(error_msg)

        result = FlextUtilities.Conversions.safe_bool(BadValue(), default=True)
        assert result is True

        # Test TypeError exception
        class BadType:
            def __bool__(self) -> bool:
                error_msg = "Bad type"
                raise TypeError(error_msg)

        result = FlextUtilities.Conversions.safe_bool(BadType(), default=False)
        assert result is False
