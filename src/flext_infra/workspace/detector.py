"""Workspace mode detection service.

Detects whether a project runs in standalone or workspace mode by inspecting
parent repository origin URL. Migrated from scripts/mode.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import override
from urllib.parse import urlparse

from flext_core import r, s
from flext_infra import c, output, u


class WorkspaceMode(StrEnum):
    """Workspace execution mode enumeration."""

    WORKSPACE = c.Infra.ReportKeys.WORKSPACE
    STANDALONE = "standalone"


class FlextInfraWorkspaceDetector(s[WorkspaceMode]):
    """Infrastructure service for detecting workspace mode.

    Inspects parent repository origin URL to determine if a project
    runs in workspace (flext) or standalone mode.

    """

    def __init__(self) -> None:
        """Initialize the workspace detector."""
        super().__init__(
            config_type=None,
            config_overrides=None,
            initial_context=None,
            subproject=None,
            services=None,
            factories=None,
            resources=None,
            container_overrides=None,
            wire_modules=None,
            wire_packages=None,
            wire_classes=None,
        )

    @staticmethod
    def _repo_name_from_url(url: str) -> str:
        """Extract repository name from Git URL.

        Args:
            url: Git repository URL (SSH or HTTPS).

        Returns:
            Repository name without .git suffix.

        """
        parsed = urlparse(url)
        path = parsed.path or url
        name = path.rsplit("/", 1)[-1]
        return name.removesuffix(c.Infra.Git.DIR)

    def detect(self, project_root: Path) -> r[WorkspaceMode]:
        """Detect workspace mode by inspecting parent repository origin URL.

        Args:
            project_root: Path to the project directory.

        Returns:
            r with WorkspaceMode.WORKSPACE if parent repo is 'flext',
            WorkspaceMode.STANDALONE otherwise.

        """
        try:
            parent = project_root.resolve().parent
            git_marker = parent / c.Infra.Git.DIR
            if not git_marker.exists():
                output.info("Running in standalone mode (no parent workspace detected)")
                return r[WorkspaceMode].ok(WorkspaceMode.STANDALONE)
            result = u.Infra.git_run(
                ["config", "--get", "remote.origin.url"], cwd=parent
            )
            if result.is_failure:
                output.info("Running in standalone mode (unable to detect workspace)")
                return r[WorkspaceMode].ok(WorkspaceMode.STANDALONE)
            origin = result.value.strip()
            if not origin:
                output.info("Running in standalone mode (no remote origin found)")
                return r[WorkspaceMode].ok(WorkspaceMode.STANDALONE)
            repo_name = self._repo_name_from_url(origin)
            mode = (
                WorkspaceMode.WORKSPACE
                if repo_name == c.Infra.Packages.ROOT
                else WorkspaceMode.STANDALONE
            )
            if mode == WorkspaceMode.STANDALONE:
                output.info(f"Running in standalone mode (parent repo: {repo_name})")
            return r[WorkspaceMode].ok(mode)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            output.info(f"Running in standalone mode (detection error: {exc})")
            return r[WorkspaceMode].fail(f"Detection failed: {exc}")

    @override
    def execute(self) -> r[WorkspaceMode]:
        """Not used; call detect() directly instead."""
        return r[WorkspaceMode].fail("Use detect() method directly")


__all__ = ["FlextInfraWorkspaceDetector", "WorkspaceMode"]
