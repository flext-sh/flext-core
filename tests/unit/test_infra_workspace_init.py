"""Tests for flext_infra.workspace module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.
"""

from __future__ import annotations

import pytest


class TestFlextInfraWorkspace:
    """Tests for flext_infra.workspace module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        import flext_infra.workspace  # noqa: PLC0415

        with pytest.raises(AttributeError):
            _ = flext_infra.workspace.nonexistent_symbol_xyz

    def test_lazy_import_orchestrator_service(self) -> None:
        """Test lazy import of FlextInfraOrchestratorService."""
        from flext_infra.workspace import FlextInfraOrchestratorService  # noqa: PLC0415

        assert FlextInfraOrchestratorService is not None

    def test_lazy_import_sync_service(self) -> None:
        """Test lazy import of FlextInfraSyncService."""
        from flext_infra.workspace import FlextInfraSyncService  # noqa: PLC0415

        assert FlextInfraSyncService is not None

    def test_lazy_import_migrator(self) -> None:
        """Test lazy import of FlextInfraProjectMigrator."""
        from flext_infra.workspace import FlextInfraProjectMigrator  # noqa: PLC0415

        assert FlextInfraProjectMigrator is not None

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        import flext_infra.workspace  # noqa: PLC0415

        exports = dir(flext_infra.workspace)
        assert "FlextInfraOrchestratorService" in exports
        assert "FlextInfraSyncService" in exports
        assert "FlextInfraProjectMigrator" in exports
