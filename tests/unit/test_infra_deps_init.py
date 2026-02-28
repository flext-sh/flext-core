"""Tests for flext_infra.deps module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.
"""

from __future__ import annotations

import flext_infra.deps
import pytest


class TestFlextInfraDeps:
    """Tests for flext_infra.deps module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        with pytest.raises(AttributeError):
            _ = flext_infra.deps.nonexistent_symbol_xyz

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        exports = dir(flext_infra.deps)
        assert isinstance(exports, list)
        assert len(exports) > 0
