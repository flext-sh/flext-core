"""Utilities for discovering subprojects within the workspace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_ENCODING,
    MAKEFILE_FILENAME,
    PYPROJECT_FILENAME,
    PYPROJECT_SKIP_DIRS,
)


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
        content = gitmodules.read_text(encoding=DEFAULT_ENCODING)
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
        if not (entry / MAKEFILE_FILENAME).exists():
            continue
        if (
            not (entry / PYPROJECT_FILENAME).exists()
            and not (entry / "go.mod").exists()
        ):
            continue
        kind = "submodule" if entry.name in submodules else "external"
        projects.append(ProjectInfo(path=entry, name=entry.name, kind=kind))
    return projects


def find_all_pyproject_files(
    workspace_root: Path,
    *,
    skip_dirs: frozenset[str] | None = None,
    project_paths: list[Path] | None = None,
) -> list[Path]:
    """Find every ``pyproject.toml`` under *workspace_root* recursively.

    Unlike :func:`discover_projects` (which returns top-level project
    directories), this finds **all** ``pyproject.toml`` files including
    nested ones inside projects.

    Args:
        workspace_root: Root of the workspace tree.
        skip_dirs: Directory names to exclude from traversal.
            Defaults to :data:`~scripts.libs.config.PYPROJECT_SKIP_DIRS`.
        project_paths: If given, only return ``pyproject.toml`` files
            belonging to these project directories (used by
            ``fix_pyrefly_config.py`` for targeted runs).

    """
    if project_paths:
        selected: list[Path] = []
        for p in project_paths:
            target = p if p.name == PYPROJECT_FILENAME else p / PYPROJECT_FILENAME
            if target.exists() and target.is_file():
                selected.append(target)
        return sorted(set(selected))

    effective_skip = skip_dirs if skip_dirs is not None else PYPROJECT_SKIP_DIRS
    return [
        p
        for p in sorted(workspace_root.rglob(PYPROJECT_FILENAME))
        if not any(skip in p.parts for skip in effective_skip)
    ]
