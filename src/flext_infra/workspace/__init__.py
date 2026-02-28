"""Workspace management services.

Provides services for workspace detection, synchronization, and orchestration
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.workspace.detector import WorkspaceDetector, WorkspaceMode
    from flext_infra.workspace.migrator import ProjectMigrator
    from flext_infra.workspace.orchestrator import OrchestratorService
    from flext_infra.workspace.sync import SyncService

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "OrchestratorService": (
        "flext_infra.workspace.orchestrator",
        "OrchestratorService",
    ),
    "ProjectMigrator": ("flext_infra.workspace.migrator", "ProjectMigrator"),
    "SyncService": ("flext_infra.workspace.sync", "SyncService"),
    "WorkspaceDetector": ("flext_infra.workspace.detector", "WorkspaceDetector"),
    "WorkspaceMode": ("flext_infra.workspace.detector", "WorkspaceMode"),
}

__all__ = [
    "OrchestratorService",
    "ProjectMigrator",
    "SyncService",
    "WorkspaceDetector",
    "WorkspaceMode",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
