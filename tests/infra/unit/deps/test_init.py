"""Tests for flext_infra.deps module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.
"""

from __future__ import annotations

import pytest

import flext_infra.deps as deps_mod
from flext_tests import tm


class TestFlextInfraDeps:
    """Tests for flext_infra.deps module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        with pytest.raises(AttributeError):
            _ = deps_mod.nonexistent_symbol_xyz

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        exports = dir(deps_mod)
        tm.that(len(exports) > 0, eq=True)
