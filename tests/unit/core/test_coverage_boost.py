"""Coverage boost tests for uncovered lines.

This module contains targeted tests to increase code coverage
for specific uncovered lines in the codebase.
"""

from __future__ import annotations

import pytest

from flext_core.utilities import flext_safe_int_conversion
from flext_core.version import get_version_info

pytestmark = [pytest.mark.unit]


class TestCoverageBoost:
    """Tests to boost coverage for specific uncovered lines."""

    def test_safe_int_conversion_coverage(self) -> None:
        """Test safe int conversion function edge cases."""
        # Test edge cases to increase coverage
        result = flext_safe_int_conversion("not_a_number", 42)
        assert result == 42

        result = flext_safe_int_conversion(None, None)
        assert result is None

    def test_version_info_coverage(self) -> None:
        """Test version info function."""
        # Simple test to increase coverage
        version_info = get_version_info()
        assert version_info.major >= 0
        assert version_info.minor >= 0
        assert version_info.patch >= 0
