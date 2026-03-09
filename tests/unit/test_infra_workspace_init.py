"""Tests for flext_infra.workspace module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.
"""

from __future__ import annotations

import importlib

import pytest


class TestFlextInfraWorkspace:
    """Tests for flext_infra.workspace module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        workspace_module = importlib.import_module("flext_infra.workspace")

        with pytest.raises(AttributeError):
            _ = workspace_module.nonexistent_symbol_xyz

    def test_lazy_import_orchestrator_service(self) -> None:
        """Test lazy import of FlextInfraOrchestratorService."""
        workspace_module = importlib.import_module("flext_infra.workspace")
        orchestrator_cls = getattr(
            workspace_module,
            "FlextInfraOrchestratorService",
        )

        assert orchestrator_cls is not None

    def test_lazy_import_sync_service(self) -> None:
        """Test lazy import of FlextInfraSyncService."""
        workspace_module = importlib.import_module("flext_infra.workspace")
        sync_service_cls = getattr(workspace_module, "FlextInfraSyncService")

        assert sync_service_cls is not None

    def test_lazy_import_migrator(self) -> None:
        """Test lazy import of FlextInfraProjectMigrator."""
        workspace_module = importlib.import_module("flext_infra.workspace")
        migrator_cls = getattr(
            workspace_module,
            "FlextInfraProjectMigrator",
        )

        assert migrator_cls is not None

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        workspace_module = importlib.import_module("flext_infra.workspace")

        exports = dir(workspace_module)
        assert "FlextInfraOrchestratorService" in exports
        assert "FlextInfraSyncService" in exports
        assert "FlextInfraProjectMigrator" in exports
