"""Tests for flext_infra.basemk module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.
"""

from __future__ import annotations

import flext_infra.basemk
import pytest
from flext_infra.basemk import FlextInfraBaseMkGenerator, FlextInfraBaseMkTemplateEngine


class TestFlextInfraBaseMk:
    """Tests for flext_infra.basemk module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        with pytest.raises(AttributeError):
            _ = flext_infra.basemk.nonexistent_symbol_xyz

    def test_lazy_import_template_engine(self) -> None:
        """Test lazy import of FlextInfraBaseMkTemplateEngine."""
        assert FlextInfraBaseMkTemplateEngine is not None

    def test_lazy_import_generator(self) -> None:
        """Test lazy import of FlextInfraBaseMkGenerator."""
        assert FlextInfraBaseMkGenerator is not None

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        exports = dir(flext_infra.basemk)
        assert "FlextInfraBaseMkTemplateEngine" in exports
        assert "FlextInfraBaseMkGenerator" in exports
