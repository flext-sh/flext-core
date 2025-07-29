"""Tests for FLEXT Core constants module."""

from __future__ import annotations

import pytest

from flext_core.constants import FlextLogLevel

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextLogLevel:
    """Test FlextLogLevel functionality."""

    def test_hash_functionality(self) -> None:
        """Test __hash__ method of FlextLogLevel."""
        level = FlextLogLevel.INFO

        # Hash should be consistent
        hash1 = hash(level)
        hash2 = hash(level)
        if hash1 != hash2:
            msg = f"Expected {hash2}, got {hash1}"
            raise AssertionError(msg)

        # Hash should be based on value
        if hash(level) != hash(level.value):
            msg = f"Expected {hash(level.value)}, got {hash(level)}"
            raise AssertionError(msg)

    def test_get_numeric_value(self) -> None:
        """Test get_numeric_value method."""
        # Test all log levels have numeric values
        debug_value = FlextLogLevel.DEBUG.get_numeric_value()
        info_value = FlextLogLevel.INFO.get_numeric_value()
        warning_value = FlextLogLevel.WARNING.get_numeric_value()
        error_value = FlextLogLevel.ERROR.get_numeric_value()
        critical_value = FlextLogLevel.CRITICAL.get_numeric_value()

        # Verify values are integers
        assert isinstance(debug_value, int)
        assert isinstance(info_value, int)
        assert isinstance(warning_value, int)
        assert isinstance(error_value, int)
        assert isinstance(critical_value, int)

        # Verify value order (higher level = higher value)
        assert debug_value < info_value
        assert info_value < warning_value
        assert warning_value < error_value
        assert error_value < critical_value
