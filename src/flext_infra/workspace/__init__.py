# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Workspace management services.

Provides services for workspace detection, synchronization, and orchestration
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.workspace.detector import (
        FlextInfraWorkspaceDetector,
        WorkspaceMode,
    )
    from flext_infra.workspace.migrator import FlextInfraProjectMigrator
    from flext_infra.workspace.orchestrator import (
        FlextInfraOrchestratorService,
        FlextInfraOrchestratorService as s,
    )
    from flext_infra.workspace.sync import FlextInfraSyncService, main

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraOrchestratorService": (
        "flext_infra.workspace.orchestrator",
        "FlextInfraOrchestratorService",
    ),
    "FlextInfraProjectMigrator": (
        "flext_infra.workspace.migrator",
        "FlextInfraProjectMigrator",
    ),
    "FlextInfraSyncService": ("flext_infra.workspace.sync", "FlextInfraSyncService"),
    "FlextInfraWorkspaceDetector": (
        "flext_infra.workspace.detector",
        "FlextInfraWorkspaceDetector",
    ),
    "WorkspaceMode": ("flext_infra.workspace.detector", "WorkspaceMode"),
    "main": ("flext_infra.workspace.sync", "main"),
    "s": ("flext_infra.workspace.orchestrator", "FlextInfraOrchestratorService"),
}

__all__ = [
    "FlextInfraOrchestratorService",
    "FlextInfraProjectMigrator",
    "FlextInfraSyncService",
    "FlextInfraWorkspaceDetector",
    "WorkspaceMode",
    "main",
    "s",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
