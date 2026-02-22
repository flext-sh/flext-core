"""Workspace mode detection service.

Detects whether a project runs in standalone or workspace mode by inspecting
parent repository origin URL. Migrated from scripts/mode.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
from enum import Enum
from pathlib import Path
from typing import override
from urllib.parse import urlparse

from flext_core.result import FlextResult as r
from flext_core.service import FlextService


class WorkspaceMode(str, Enum):
    """Workspace execution mode enumeration."""

    WORKSPACE = "workspace"
    STANDALONE = "standalone"


class WorkspaceDetector(FlextService[WorkspaceMode]):
    """Infrastructure service for detecting workspace mode.

    Inspects parent repository origin URL to determine if a project
    runs in workspace (flext) or standalone mode.

    Example:
        detector = WorkspaceDetector()
        result = detector.detect(Path("flext-core"))
        if result.is_success:
            mode = result.value
            print(f"Mode: {mode}")

    """

    @override
    def execute(self) -> r[WorkspaceMode]:
        """Not used; call detect() directly instead."""
        return r[WorkspaceMode].fail("Use detect() method directly")

    def detect(self, project_root: Path) -> r[WorkspaceMode]:
        """Detect workspace mode by inspecting parent repository origin URL.

        Args:
            project_root: Path to the project directory.

        Returns:
            FlextResult with WorkspaceMode.WORKSPACE if parent repo is 'flext',
            WorkspaceMode.STANDALONE otherwise.

        """
        try:
            parent = project_root.resolve().parent
            git_marker = parent / ".git"

            if not git_marker.exists():
                return r[WorkspaceMode].ok(WorkspaceMode.STANDALONE)

            result = subprocess.run(
                ["git", "-C", str(parent), "config", "--get", "remote.origin.url"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return r[WorkspaceMode].ok(WorkspaceMode.STANDALONE)

            origin = result.stdout.strip()
            if not origin:
                return r[WorkspaceMode].ok(WorkspaceMode.STANDALONE)

            repo_name = self._repo_name_from_url(origin)
            mode = (
                WorkspaceMode.WORKSPACE
                if repo_name == "flext"
                else WorkspaceMode.STANDALONE
            )
            return r[WorkspaceMode].ok(mode)

        except Exception as exc:
            return r[WorkspaceMode].fail(f"Detection failed: {exc}")

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
        return name.removesuffix(".git")


__all__ = [
    "WorkspaceDetector",
    "WorkspaceMode",
]
