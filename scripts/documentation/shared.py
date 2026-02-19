#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Shared utilities for documentation maintenance scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".reports",
    "site",
}


@dataclass(frozen=True)
class Scope:
    """Documentation scope targeting a project or workspace root."""

    name: str
    path: Path
    report_dir: Path


def workspace_projects(root: Path) -> list[str]:
    """Return sorted project names that contain a pyproject.toml."""
    return [
        item.name
        for item in sorted(root.iterdir())
        if item.is_dir() and (item / "pyproject.toml").exists()
    ]


def selected_project_names(
    root: Path, project: str | None, projects: str | None
) -> list[str]:
    """Resolve CLI project flags to a concrete name list."""
    all_names = workspace_projects(root)
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
        if not path.exists() or not (path / "pyproject.toml").exists():
            continue
        scopes.append(
            Scope(name=name, path=path, report_dir=(path / output_dir).resolve())
        )
    return scopes


def write_json(path: Path, payload: object) -> None:
    """Write a JSON payload to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def write_markdown(path: Path, lines: list[str]) -> None:
    """Write markdown lines to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def iter_markdown_files(root: Path) -> list[Path]:
    """Recursively collect markdown files under the docs scope."""

    docs_root = root / "docs"
    search_root = docs_root if docs_root.is_dir() else root
    return sorted(
        path
        for path in search_root.rglob("*.md")
        if not any(part in EXCLUDED_DIRS or part.startswith(".") for part in path.parts)
    )
