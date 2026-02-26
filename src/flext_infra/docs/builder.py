"""Documentation builder service.

Builds MkDocs sites for workspace projects, returning structured
FlextResult reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
from flext_core.result import FlextResult, r

from flext_infra.constants import c
from flext_infra.docs.shared import (
    DocScope,
    FlextInfraDocsShared,
)
from flext_infra.subprocess import CommandRunner

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BuildReport:
    """Outcome of a single MkDocs build attempt."""

    scope: str
    result: str
    reason: str
    site_dir: str


class DocBuilder:
    """Infrastructure service for documentation building.

    Runs MkDocs build for workspace projects and returns
    structured FlextResult reports.
    """

    def __init__(self) -> None:
        """Initialize the documentation builder."""
        self._runner = CommandRunner()

    def build(
        self,
        root: Path,
        *,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = ".reports/docs",
    ) -> FlextResult[list[BuildReport]]:
        """Build MkDocs sites across project scopes.

        Args:
            root: Workspace root directory.
            project: Single project name filter.
            projects: Comma-separated project names.
            output_dir: Report output directory.

        Returns:
            FlextResult with list of BuildReport objects.

        """
        scopes_result = FlextInfraDocsShared.build_scopes(
            root=root,
            project=project,
            projects=projects,
            output_dir=output_dir,
        )
        if scopes_result.is_failure:
            return r[list[BuildReport]].fail(scopes_result.error or "scope error")

        reports: list[BuildReport] = []
        for scope in scopes_result.value:
            report = self._build_scope(scope)
            reports.append(report)

        return r[list[BuildReport]].ok(reports)

    def _build_scope(self, scope: DocScope) -> BuildReport:
        """Run mkdocs build --strict for a single scope."""
        report = self._run_mkdocs(scope)
        self._write_reports(scope, report)
        logger.info(
            "docs_build_scope_completed",
            project=scope.name,
            phase="build",
            result=report.result,
            reason=report.reason,
        )
        return report

    def _run_mkdocs(self, scope: DocScope) -> BuildReport:
        """Run mkdocs build --strict and return the result."""
        config = scope.path / "mkdocs.yml"
        if not config.exists():
            return BuildReport(
                scope=scope.name,
                result="SKIP",
                reason="mkdocs.yml not found",
                site_dir="",
            )

        site_dir = (scope.path / ".reports/docs/site").resolve()
        site_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "mkdocs",
            "build",
            "--strict",
            "-f",
            str(config),
            "-d",
            str(site_dir),
        ]
        completed = self._runner.run_raw(cmd, cwd=scope.path)
        if completed.is_failure:
            return BuildReport(
                scope=scope.name,
                result=c.Status.FAIL,
                reason=completed.error or "mkdocs build failed",
                site_dir=site_dir.as_posix(),
            )

        output = completed.value
        if output.exit_code == 0:
            return BuildReport(
                scope=scope.name,
                result=c.Status.OK,
                reason="build succeeded",
                site_dir=site_dir.as_posix(),
            )
        reason = (output.stderr or output.stdout).strip().splitlines()
        tail = reason[-1] if reason else f"mkdocs exited {output.exit_code}"
        return BuildReport(
            scope=scope.name,
            result=c.Status.FAIL,
            reason=tail,
            site_dir=site_dir.as_posix(),
        )

    @staticmethod
    def _write_reports(scope: DocScope, report: BuildReport) -> None:
        """Persist build JSON summary and markdown report."""
        _ = FlextInfraDocsShared.write_json(
            scope.report_dir / "build-summary.json",
            {"summary": asdict(report)},
        )
        _ = FlextInfraDocsShared.write_markdown(
            scope.report_dir / "build-report.md",
            [
                "# Docs Build Report",
                "",
                f"Scope: {report.scope}",
                f"Result: {report.result}",
                f"Reason: {report.reason}",
                f"Site dir: {report.site_dir}",
            ],
        )


__all__ = ["BuildReport", "DocBuilder"]
