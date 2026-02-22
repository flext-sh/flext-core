"""GitHub integration services.

Provides services for GitHub API interactions, workflow management, and
repository operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "WorkflowLinter":
        from flext_infra.github.linter import WorkflowLinter

        return WorkflowLinter
    if name == "PrManager":
        from flext_infra.github.pr import PrManager

        return PrManager
    if name == "PrWorkspaceManager":
        from flext_infra.github.pr_workspace import PrWorkspaceManager

        return PrWorkspaceManager
    if name == "WorkflowSyncer":
        from flext_infra.github.workflows import WorkflowSyncer

        return WorkflowSyncer
    raise AttributeError(name)


__all__ = [
    "PrManager",
    "PrWorkspaceManager",
    "WorkflowLinter",
    "WorkflowSyncer",
]
