"""Utilities for selecting and filtering workspace projects."""

from __future__ import annotations

from pathlib import Path

from .discovery import ProjectInfo, discover_projects


def filter_projects(projects: list[ProjectInfo], kind: str) -> list[ProjectInfo]:
    """Filter a list of projects by their kind.

    Args:
        projects: The list of project information to filter.
        kind: The kind of project to include ("submodule", "external", or "all").

    Returns:
        A list of projects matching the specified kind.

    """
    if kind == "all":
        return list(projects)
    return [project for project in projects if project.kind == kind]


def resolve_projects(workspace_root: Path, names: list[str]) -> list[ProjectInfo]:
    """Resolve project names into their corresponding ProjectInfo structures.

    Args:
        workspace_root: The root directory of the workspace.
        names: A list of project names to resolve. If empty, all projects are returned.

    Returns:
        A list of resolved ProjectInfo structures, sorted by name.

    Raises:
        RuntimeError: If any of the requested project names are unknown.

    """
    projects = discover_projects(workspace_root)
    if not names:
        return sorted(projects, key=lambda project: project.name)

    by_name = {project.name: project for project in projects}
    missing = [name for name in names if name not in by_name]
    if missing:
        missing_text = ", ".join(sorted(missing))
        msg = f"unknown projects: {missing_text}"
        raise RuntimeError(msg)

    resolved = [by_name[name] for name in names]
    return sorted(resolved, key=lambda project: project.name)
