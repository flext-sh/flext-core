"""Shared utilities for documentation services.

Provides common models, scope resolution, and markdown helpers
used across documentation auditor, fixer, builder, generator,
and validator services.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, JsonValue

from flext_core import r
from flext_infra import c, m, u


class FlextInfraDocsShared:
    """Single class for shared documentation helpers (scope, markdown, json)."""

    @staticmethod
    def _selected_project_names(
        workspace_root: Path,
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
        result: r[list[m.Infra.Workspace.ProjectInfo]] = u.Infra.discover_projects(
            workspace_root,
        )
        return result.fold(
            on_failure=lambda _: [],
            on_success=lambda v: [p.name for p in v],
        )

    @staticmethod
    def build_scopes(
        workspace_root: Path,
        project: str | None,
        projects: str | None,
        output_dir: str,
    ) -> r[list[m.Infra.Docs.FlextInfraDocScope]]:
        """Build DocScope objects for workspace root and each selected project."""
        try:
            scopes: list[m.Infra.Docs.FlextInfraDocScope] = [
                m.Infra.Docs.FlextInfraDocScope(
                    name=c.Infra.ReportKeys.ROOT,
                    path=workspace_root,
                    report_dir=(workspace_root / output_dir).resolve(),
                ),
            ]
            names = FlextInfraDocsShared._selected_project_names(
                workspace_root,
                project,
                projects,
            )
            for name in names:
                path = (workspace_root / name).resolve()
                if (
                    not path.exists()
                    or not (path / c.Infra.Files.PYPROJECT_FILENAME).exists()
                ):
                    continue
                scopes.append(
                    m.Infra.Docs.FlextInfraDocScope(
                        name=name,
                        path=path,
                        report_dir=(path / output_dir).resolve(),
                    ),
                )
            return r[list[m.Infra.Docs.FlextInfraDocScope]].ok(scopes)
        except (OSError, TypeError, ValueError) as exc:
            return r[list[m.Infra.Docs.FlextInfraDocScope]].fail(
                f"scope resolution failed: {exc}",
            )

    @staticmethod
    def iter_markdown_files(workspace_root: Path) -> list[Path]:
        """Recursively collect markdown files under the docs scope."""
        docs_root = workspace_root / c.Infra.Directories.DOCS
        search_root = docs_root if docs_root.is_dir() else workspace_root
        return sorted(
            path
            for path in search_root.rglob("*.md")
            if not any(
                part in c.Infra.Excluded.DOC_EXCLUDED_DIRS or part.startswith(".")
                for part in path.parts
            )
        )

    @staticmethod
    def write_json(
        path: Path,
        payload: JsonValue | BaseModel | Mapping[str, JsonValue] | Sequence[JsonValue],
    ) -> r[bool]:
        """Write JSON payload to path."""
        return u.Infra.write_json(path, payload)

    @staticmethod
    def write_markdown(path: Path, lines: list[str]) -> r[bool]:
        """Write markdown lines to path, creating parent dirs as needed."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(
                "\n".join(lines).rstrip() + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"markdown write error: {exc}")


__all__ = ["FlextInfraDocsShared"]
