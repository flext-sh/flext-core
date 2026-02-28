"""Tests for flext_infra.check module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.
"""

from __future__ import annotations

import flext_infra.check
import pytest


class TestFlextInfraCheck:
    """Tests for flext_infra.check module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        with pytest.raises(AttributeError):
            _ = flext_infra.check.nonexistent_symbol_xyz

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        exports = dir(flext_infra.check)
        assert isinstance(exports, list)
        assert len(exports) > 0
