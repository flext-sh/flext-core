"""Tests for flext_infra.docs module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.
"""

from __future__ import annotations

import flext_infra.docs
import pytest
from flext_infra.docs import (
    FlextInfraDocAuditor,
    FlextInfraDocBuilder,
    FlextInfraDocFixer,
    FlextInfraDocGenerator,
    FlextInfraDocValidator,
)


class TestFlextInfraDocs:
    """Tests for flext_infra.docs module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        with pytest.raises(AttributeError):
            _ = flext_infra.docs.nonexistent_symbol_xyz

    def test_lazy_import_builder(self) -> None:
        """Test lazy import of FlextInfraDocBuilder."""
        assert FlextInfraDocBuilder is not None

    def test_lazy_import_fixer(self) -> None:
        """Test lazy import of FlextInfraDocFixer."""
        assert FlextInfraDocFixer is not None

    def test_lazy_import_generator(self) -> None:
        """Test lazy import of FlextInfraDocGenerator."""
        assert FlextInfraDocGenerator is not None

    def test_lazy_import_validator(self) -> None:
        """Test lazy import of FlextInfraDocValidator."""
        assert FlextInfraDocValidator is not None

    def test_lazy_import_auditor(self) -> None:
        """Test lazy import of FlextInfraDocAuditor."""
        assert FlextInfraDocAuditor is not None

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        exports = dir(flext_infra.docs)
        assert "FlextInfraDocBuilder" in exports
        assert "FlextInfraDocFixer" in exports
        assert "FlextInfraDocGenerator" in exports
        assert "FlextInfraDocValidator" in exports
        assert "FlextInfraDocAuditor" in exports
