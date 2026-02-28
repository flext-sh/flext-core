"""Project selection and filtering service.

Wraps project selection logic with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import FlextResult, r

from flext_infra import DiscoveryService, m


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

    def resolve_projects(
        self,
        workspace_root: Path,
        names: list[str],
    ) -> FlextResult[list[m.ProjectInfo]]:
        """Resolve project names into ProjectInfo structures.

        Args:
            workspace_root: The root directory of the workspace.
            names: Project names to resolve. If empty, returns all.

        Returns:
            FlextResult with sorted list of resolved projects.

        """
        discover_result = self._discovery.discover_projects(workspace_root)
        if discover_result.is_failure:
            return r[list[m.ProjectInfo]].fail(
                discover_result.error or "discovery failed",
            )

        projects = discover_result.value
        if not names:
            return r[list[m.ProjectInfo]].ok(
                sorted(projects, key=lambda p: p.name),
            )

        by_name = {p.name: p for p in projects}
        missing = [name for name in names if name not in by_name]
        if missing:
            missing_text = ", ".join(sorted(missing))
            return r[list[m.ProjectInfo]].fail(
                f"unknown projects: {missing_text}",
            )

        resolved = [by_name[name] for name in names]
        return r[list[m.ProjectInfo]].ok(
            sorted(resolved, key=lambda p: p.name),
        )


__all__ = ["ProjectSelector"]
