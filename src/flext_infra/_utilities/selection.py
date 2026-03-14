"""Project selection and filtering utilities.

All methods are static — exposed via u.Infra.resolve_projects() through MRO.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
from flext_infra.models import FlextInfraModels as m


class FlextInfraUtilitiesSelection:
    """Static project selection and filtering utilities.

    All methods are ``@staticmethod`` — no instantiation required.
    Delegates discovery to ``FlextInfraUtilitiesDiscovery`` directly.
    """

    @staticmethod
    def resolve_projects(
        workspace_root: Path,
        names: list[str],
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        """Resolve project names into ProjectInfo structures.

        Args:
            workspace_root: The root directory of the workspace.
            names: Project names to resolve. If empty, returns all.

        Returns:
            r with sorted list of resolved projects.

        """
        discover_result = FlextInfraUtilitiesDiscovery.discover_projects(
            workspace_root,
        )
        if discover_result.is_failure:
            return r[list[m.Infra.Workspace.ProjectInfo]].fail(
                discover_result.error or "discovery failed",
            )
        projects = discover_result.value
        if not names:
            return r[list[m.Infra.Workspace.ProjectInfo]].ok(
                sorted(projects, key=lambda proj: proj.name),
            )
        by_name = {proj.name: proj for proj in projects}
        missing = [name for name in names if name not in by_name]
        if missing:
            missing_text = ", ".join(sorted(missing))
            return r[list[m.Infra.Workspace.ProjectInfo]].fail(
                f"unknown projects: {missing_text}",
            )
        resolved = [by_name[name] for name in names]
        return r[list[m.Infra.Workspace.ProjectInfo]].ok(
            sorted(resolved, key=lambda proj: proj.name),
        )


__all__ = ["FlextInfraUtilitiesSelection"]
