#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Shared utilities for documentation maintenance scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.libs.config import DEFAULT_ENCODING, DOC_EXCLUDED_DIRS, PYPROJECT_FILENAME
from scripts.libs.discovery import discover_projects
from scripts.libs.json_io import write_json

__all__ = [
    "Scope",
    "build_scopes",
    "iter_markdown_files",
    "selected_project_names",
    "write_json",
    "write_markdown",
]


@dataclass(frozen=True)
class Scope:
    """Documentation scope targeting a project or workspace root."""

    name: str
    path: Path
    report_dir: Path


def selected_project_names(
    root: Path, project: str | None, projects: str | None
) -> list[str]:
    """Resolve CLI project flags to a concrete name list."""
    all_names = [p.name for p in discover_projects(root)]
    if project:
        return [project]
    if projects:
        requested = [part.strip() for part in projects.split(",") if part.strip()]
        if len(requested) == 1 and " " in requested[0]:
            requested = [
                part.strip() for part in requested[0].split(" ") if part.strip()
            ]
        return requested
    return all_names


def build_scopes(
    root: Path, project: str | None, projects: str | None, output_dir: str
) -> list[Scope]:
    """Build Scope objects for workspace root and each selected project."""
    scopes: list[Scope] = [
        Scope(name="root", path=root, report_dir=(root / output_dir).resolve()),
    ]
    for name in selected_project_names(root, project, projects):
        path = (root / name).resolve()
        if not path.exists() or not (path / PYPROJECT_FILENAME).exists():
            continue
        scopes.append(
            Scope(name=name, path=path, report_dir=(path / output_dir).resolve())
        )
    return scopes


def write_markdown(path: Path, lines: list[str]) -> None:
    """Write markdown lines to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("\n".join(lines).rstrip() + "\n", encoding=DEFAULT_ENCODING)


def iter_markdown_files(root: Path) -> list[Path]:
    """Recursively collect markdown files under the docs scope."""
    docs_root = root / "docs"
    search_root = docs_root if docs_root.is_dir() else root
    return sorted(
        path
        for path in search_root.rglob("*.md")
        if not any(
            part in DOC_EXCLUDED_DIRS or part.startswith(".") for part in path.parts
        )
    )
