# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""GitHub integration services.

Provides services for GitHub API interactions, workflow management, and
repository operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.github.linter import FlextInfraWorkflowLinter
    from flext_infra.github.pr import FlextInfraPrManager, main
    from flext_infra.github.pr_workspace import FlextInfraPrWorkspaceManager, u
    from flext_infra.github.workflows import FlextInfraWorkflowSyncer, SyncOperation

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraPrManager": ("flext_infra.github.pr", "FlextInfraPrManager"),
    "FlextInfraPrWorkspaceManager": ("flext_infra.github.pr_workspace", "FlextInfraPrWorkspaceManager"),
    "FlextInfraWorkflowLinter": ("flext_infra.github.linter", "FlextInfraWorkflowLinter"),
    "FlextInfraWorkflowSyncer": ("flext_infra.github.workflows", "FlextInfraWorkflowSyncer"),
    "SyncOperation": ("flext_infra.github.workflows", "SyncOperation"),
    "main": ("flext_infra.github.pr", "main"),
    "u": ("flext_infra.github.pr_workspace", "u"),
}

__all__ = [
    "FlextInfraPrManager",
    "FlextInfraPrWorkspaceManager",
    "FlextInfraWorkflowLinter",
    "FlextInfraWorkflowSyncer",
    "SyncOperation",
    "main",
    "u",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
