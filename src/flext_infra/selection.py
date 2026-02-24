"""Project selection and filtering service.

Wraps project selection logic with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core.result import FlextResult, r

from flext_infra.constants import ic
from flext_infra.discovery import DiscoveryService
from flext_infra.models import im


class ProjectSelector:
    """Infrastructure service for project selection and filtering.

    Combines project discovery with filtering and resolution capabilities.
    """

    def __init__(
        self,
        discovery: DiscoveryService | None = None,
    ) -> None:
        """Initialize the project selector."""
        self._discovery = discovery or DiscoveryService()

    def filter_projects(
        self,
        projects: list[im.ProjectInfo],
        kind: str,
    ) -> FlextResult[list[im.ProjectInfo]]:
        """Filter a list of projects by their kind.

        Kind is encoded in the ``stack`` field as ``{tech}/{kind}``.

        Args:
            projects: The list of projects to filter.
            kind: The kind to include ("submodule", "external", or "all").

        Returns:
            FlextResult with filtered project list.

        """
        if kind == "all":
            return r[list[im.ProjectInfo]].ok(list(projects))
        filtered = [p for p in projects if p.stack.endswith(f"/{kind}")]
        return r[list[im.ProjectInfo]].ok(filtered)

    def resolve_projects(
        self,
        workspace_root: Path,
        names: list[str],
    ) -> FlextResult[list[im.ProjectInfo]]:
        """Resolve project names into ProjectInfo structures.

        Args:
            workspace_root: The root directory of the workspace.
            names: Project names to resolve. If empty, returns all.

        Returns:
            FlextResult with sorted list of resolved projects.

        """
        discover_result = self._discovery.discover_projects(workspace_root)
        if discover_result.is_failure:
            return r[list[im.ProjectInfo]].fail(
                discover_result.error or "discovery failed",
            )

        projects = discover_result.value
        if not names:
            return r[list[im.ProjectInfo]].ok(
                sorted(projects, key=lambda p: p.name),
            )

        by_name = {p.name: p for p in projects}
        missing = [name for name in names if name not in by_name]
        if missing:
            missing_text = ", ".join(sorted(missing))
            return r[list[im.ProjectInfo]].fail(
                f"unknown projects: {missing_text}",
            )

        resolved = [by_name[name] for name in names]
        return r[list[im.ProjectInfo]].ok(
            sorted(resolved, key=lambda p: p.name),
        )

    def python_projects(
        self,
        workspace_root: Path,
        names: list[str] | None = None,
    ) -> FlextResult[list[im.ProjectInfo]]:
        """Resolve projects that have pyproject.toml (Python only).

        Args:
            workspace_root: The root directory of the workspace.
            names: Optional project names to filter.

        Returns:
            FlextResult with Python project list.

        """
        result = self.resolve_projects(workspace_root, names or [])
        if result.is_failure:
            return result

        python_only = [
            p for p in result.value if (p.path / ic.Files.PYPROJECT_FILENAME).exists()
        ]
        return r[list[im.ProjectInfo]].ok(python_only)


__all__ = ["ProjectSelector"]
