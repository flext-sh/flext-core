"""Utilities for discovering subprojects within the workspace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectInfo:
    """Information about a discovered project."""

    path: Path
    name: str
    kind: str


def _is_git_project(path: Path) -> bool:
    """Check if a directory is a Git repository."""
    return (path / ".git").exists()


def _submodule_names(workspace_root: Path) -> set[str]:
    """Retrieve the names of all submodules defined in .gitmodules."""
    gitmodules = workspace_root / ".gitmodules"
    if not gitmodules.exists():
        return set()
    try:
        content = gitmodules.read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(re.findall(r"^\s*path\s*=\s*(.+?)\s*$", content, re.MULTILINE))


def discover_projects(workspace_root: Path) -> list[ProjectInfo]:
    """Discover all subprojects in the workspace.

    Args:
        workspace_root: The root directory of the workspace.

    Returns:
        A list of ProjectInfo structures for each discovered project.

    """
    projects: list[ProjectInfo] = []
    submodules = _submodule_names(workspace_root)
    for entry in sorted(workspace_root.iterdir(), key=lambda value: value.name):
        if not entry.is_dir() or entry.name == "cmd" or entry.name.startswith("."):
            continue
        if not _is_git_project(entry):
            continue
        if not (entry / "Makefile").exists():
            continue
        if not (entry / "pyproject.toml").exists() and not (entry / "go.mod").exists():
            continue
        kind = "submodule" if entry.name in submodules else "external"
        projects.append(ProjectInfo(path=entry, name=entry.name, kind=kind))
    return projects
