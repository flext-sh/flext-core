"""Tests for flext_infra.maintenance module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.
"""

from __future__ import annotations

import flext_infra.maintenance
import pytest
from flext_infra.maintenance import FlextInfraPythonVersionEnforcer


class TestFlextInfraMaintenance:
    """Tests for flext_infra.maintenance module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        with pytest.raises(AttributeError):
            _ = flext_infra.maintenance.nonexistent_symbol_xyz

    def test_lazy_import_python_version_enforcer(self) -> None:
        """Test lazy import of FlextInfraPythonVersionEnforcer."""
        assert FlextInfraPythonVersionEnforcer is not None

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        exports = dir(flext_infra.maintenance)
        assert "FlextInfraPythonVersionEnforcer" in exports
