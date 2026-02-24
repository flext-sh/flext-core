"""Shared utilities for documentation services.

Provides common models, scope resolution, and markdown helpers
used across documentation auditor, fixer, builder, generator,
and validator services.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from flext_core.result import FlextResult, r
from flext_core.typings import t
from pydantic import BaseModel, ConfigDict, Field

from flext_infra.constants import ic
from flext_infra.discovery import DiscoveryService
from flext_infra.json_io import JsonService
from flext_infra.reporting import ReportingService


class DocScope(BaseModel):
    """Documentation scope targeting a project or workspace root."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1, description="Scope name (project or 'root')")
    path: Path = Field(description="Absolute path to the scope root")
    report_dir: Path = Field(description="Report output directory for this scope")


_discovery = DiscoveryService()
_json_svc = JsonService()
_reporting = ReportingService()


def build_scopes(
    root: Path,
    project: str | None,
    projects: str | None,
    output_dir: str,
) -> FlextResult[list[DocScope]]:
    """Build DocScope objects for workspace root and each selected project.

    Args:
        root: Workspace root directory.
        project: Single project name filter.
        projects: Comma-separated project name filter.
        output_dir: Report output directory relative to scope root.

    Returns:
        FlextResult with list of DocScope objects.

    """
    try:
        scopes: list[DocScope] = [
            DocScope(
                name="root",
                path=root,
                report_dir=(root / output_dir).resolve(),
            ),
        ]
        names = _selected_project_names(root, project, projects)
        for name in names:
            path = (root / name).resolve()
            if not path.exists() or not (path / ic.Files.PYPROJECT_FILENAME).exists():
                continue
            scopes.append(
                DocScope(
                    name=name,
                    path=path,
                    report_dir=(path / output_dir).resolve(),
                ),
            )
        return r[list[DocScope]].ok(scopes)
    except (OSError, TypeError, ValueError) as exc:
        return r[list[DocScope]].fail(f"scope resolution failed: {exc}")


def _selected_project_names(
    root: Path,
    project: str | None,
    projects: str | None,
) -> list[str]:
    """Resolve CLI project flags to a concrete name list."""
    if project:
        return [project]
    if projects:
        requested = [part.strip() for part in projects.split(",") if part.strip()]
        if len(requested) == 1 and " " in requested[0]:
            requested = [
                part.strip() for part in requested[0].split(" ") if part.strip()
            ]
        return requested
    result = _discovery.discover_projects(root)
    if result.is_success:
        return [p.name for p in result.value]
    return []


def iter_markdown_files(root: Path) -> list[Path]:
    """Recursively collect markdown files under the docs scope.

    Args:
        root: Root directory to scan.

    Returns:
        Sorted list of markdown file paths.

    """
    docs_root = root / "docs"
    search_root = docs_root if docs_root.is_dir() else root
    return sorted(
        path
        for path in search_root.rglob("*.md")
        if not any(
            part in ic.Excluded.DOC_EXCLUDED_DIRS or part.startswith(".")
            for part in path.parts
        )
    )


def write_json(
    path: Path,
    payload: BaseModel | Mapping[str, t.ConfigMapValue],
) -> FlextResult[bool]:
    """Write JSON payload to path.

    Args:
        path: Destination file path.
        payload: Data to serialize.

    Returns:
        FlextResult[bool] with True on success.

    """
    return _json_svc.write(path, payload)


def write_markdown(path: Path, lines: list[str]) -> FlextResult[bool]:
    """Write markdown lines to path, creating parent dirs as needed.

    Args:
        path: Destination file path.
        lines: Lines to write.

    Returns:
        FlextResult[bool] with True on success.

    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(
            "\n".join(lines).rstrip() + "\n",
            encoding=ic.Encoding.DEFAULT,
        )
        return r[bool].ok(True)
    except OSError as exc:
        return r[bool].fail(f"markdown write error: {exc}")


__all__ = [
    "DocScope",
    "build_scopes",
    "iter_markdown_files",
    "write_json",
    "write_markdown",
]
