"""Workspace management services.

Provides services for workspace detection, synchronization, and orchestration
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detector import WorkspaceDetector, WorkspaceMode
    from .migrator import ProjectMigrator
    from .orchestrator import OrchestratorService
    from .sync import SyncService

__all__ = [
    "OrchestratorService",
    "ProjectMigrator",
    "SyncService",
    "WorkspaceDetector",
    "WorkspaceMode",
]


def __getattr__(name: str) -> object:
    if name in {"WorkspaceDetector", "WorkspaceMode"}:
        from .detector import (
            WorkspaceDetector as _WorkspaceDetector,
            WorkspaceMode as _WorkspaceMode,
        )

        exports: dict[str, object] = {
            "WorkspaceDetector": _WorkspaceDetector,
            "WorkspaceMode": _WorkspaceMode,
        }
        return exports[name]

    if name == "ProjectMigrator":
        from .migrator import ProjectMigrator as _ProjectMigrator

        return _ProjectMigrator

    if name == "OrchestratorService":
        from .orchestrator import (
            OrchestratorService as _OrchestratorService,
        )

        return _OrchestratorService

    if name == "SyncService":
        from .sync import SyncService as _SyncService

        return _SyncService

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
