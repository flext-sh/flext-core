"""Shared utilities for documentation services.

Provides common models, scope resolution, and markdown helpers
used across documentation auditor, fixer, builder, generator,
and validator services.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from flext_core import r
from flext_infra import (
    FlextInfraDiscoveryService,
    FlextInfraJsonService,
    FlextInfraReportingService,
    c,
    m,
    t,
)

_discovery = FlextInfraDiscoveryService()
_json_svc = FlextInfraJsonService()
_reporting = FlextInfraReportingService()


class FlextInfraDocsShared:
    """Single class for shared documentation helpers (scope, markdown, json)."""

    @staticmethod
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
        result: r[list[m.Infra.Workspace.ProjectInfo]] = _discovery.discover_projects(
            root,
        )
        if result.is_success:
            return [p.name for p in result.value]
        return []

    @staticmethod
    def build_scopes(
        root: Path,
        project: str | None,
        projects: str | None,
        output_dir: str,
    ) -> r[list[m.Infra.Docs.FlextInfraDocScope]]:
        """Build DocScope objects for workspace root and each selected project."""
        try:
            scopes: list[m.Infra.Docs.FlextInfraDocScope] = [
                m.Infra.Docs.FlextInfraDocScope(
                    name=c.Infra.ReportKeys.ROOT,
                    path=root,
                    report_dir=(root / output_dir).resolve(),
                ),
            ]
            names = FlextInfraDocsShared._selected_project_names(
                root,
                project,
                projects,
            )
            for name in names:
                path = (root / name).resolve()
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
    def iter_markdown_files(root: Path) -> list[Path]:
        """Recursively collect markdown files under the docs scope."""
        docs_root = root / c.Infra.Directories.DOCS
        search_root = docs_root if docs_root.is_dir() else root
        return sorted(
            path
            for path in search_root.rglob("*.md")
            if not any(
                part in c.Infra.Excluded.DOC_EXCLUDED_DIRS or part.startswith(".")
                for part in path.parts
            )
        )

    @staticmethod
    def write_json(path: Path, payload: BaseModel | t.ConfigurationMapping) -> r[bool]:
        """Write JSON payload to path."""
        return _json_svc.write(path, payload)

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
