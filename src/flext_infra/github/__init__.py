"""GitHub integration services.

Provides services for GitHub API interactions, workflow management, and
repository operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.github.linter import WorkflowLinter
    from flext_infra.github.pr import PrManager
    from flext_infra.github.pr_workspace import PrWorkspaceManager
    from flext_infra.github.workflows import WorkflowSyncer

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "PrManager": ("flext_infra.github.pr", "PrManager"),
    "PrWorkspaceManager": ("flext_infra.github.pr_workspace", "PrWorkspaceManager"),
    "WorkflowLinter": ("flext_infra.github.linter", "WorkflowLinter"),
    "WorkflowSyncer": ("flext_infra.github.workflows", "WorkflowSyncer"),
}

__all__ = [
    "PrManager",
    "PrWorkspaceManager",
    "WorkflowLinter",
    "WorkflowSyncer",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
