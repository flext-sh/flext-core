"""Workspace management services.

Provides services for workspace detection, synchronization, and orchestration
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.workspace.detector import (
        FlextInfraWorkspaceDetector,
        WorkspaceMode,
    )
    from flext_infra.workspace.migrator import FlextInfraProjectMigrator
    from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService
    from flext_infra.workspace.sync import FlextInfraSyncService

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraOrchestratorService": ("flext_infra.workspace.orchestrator", "FlextInfraOrchestratorService"),
    "FlextInfraProjectMigrator": ("flext_infra.workspace.migrator", "FlextInfraProjectMigrator"),
    "FlextInfraSyncService": ("flext_infra.workspace.sync", "FlextInfraSyncService"),
    "FlextInfraWorkspaceDetector": ("flext_infra.workspace.detector", "FlextInfraWorkspaceDetector"),
    "WorkspaceMode": ("flext_infra.workspace.detector", "WorkspaceMode"),
}

__all__ = [
    "FlextInfraOrchestratorService",
    "FlextInfraProjectMigrator",
    "FlextInfraSyncService",
    "FlextInfraWorkspaceDetector",
    "WorkspaceMode",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
