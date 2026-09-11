"""Census service for namespace violation counting and reporting.

Read-only service that counts and classifies namespace violations
across all workspace projects using FlextInfraNamespaceValidator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.base import s
from flext_infra.validate.namespace_validator import FlextInfraNamespaceValidator

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraCodegenCensus(s[str]):
    """Read-only census service for namespace violation counting."""

    @override
    def execute(self) -> p.Result[str]:
        """Execute the census directly from the validated CLI service model."""
        if self.apply_changes:
            return r[str].fail(
                "census is read-only; use flext-infra codegen auto-fix --apply"
            )
        try:
            reports = self.run()
        except c.EXC_OS_RUNTIME_TYPE as exc:
            return r[str].fail_op("census", exc)
        total_violations = sum(report.total for report in reports)
        total_fixable = sum(report.fixable for report in reports)
        if self.output_format == c.Cli.OutputFormats.JSON:
            payload: t.Infra.MutableInfraMapping = {
                c.Infra.RK_PROJECTS: [report.model_dump() for report in reports],
                "total_violations": total_violations,
                "total_fixable": total_fixable,
            }
            return r[str].ok(t.Infra.INFRA_MAPPING_ADAPTER.dump_json(payload).decode())
        lines: t.MutableSequenceOf[str] = [
            (
                f"  {report.project}: {report.total} violations"
                f" ({report.fixable} fixable)"
            )
            for report in reports
            if report.total > 0
        ]
        lines.append(
            f"Total: {total_violations} violations ({total_fixable} fixable)"
            f" across {len(reports)} projects"
        )
        return r[str].ok("\n".join(lines))

    def run(
        self,
        repository_root: Path | None = None,
        *,
        output_format: str = c.Cli.OutputFormats.JSON,
        projects: t.SequenceOf[p.Infra.ProjectInfo] | None = None,
    ) -> t.SequenceOf[m.Infra.CensusReport]:
        """Run census on all projects in workspace.

        Args:
            repository_root: Override repository root (defaults to self.repository_root).
            output_format: Unused, kept for API compat.
            projects: Pre-discovered projects to skip redundant discovery.

        Returns:
            List of CensusReport models, one per scanned project.

        """
        _ = output_format
        workspace = repository_root or self.repository_root
        return self._run_project_census(workspace, projects=projects)

    def _run_project_census(
        self,
        workspace: Path,
        *,
        projects: t.SequenceOf[p.Infra.ProjectInfo] | None = None,
    ) -> t.SequenceOf[m.Infra.CensusReport]:
        """Census all projects in workspace using the standard path."""
        if projects is not None:
            selected_projects = tuple(projects)
        else:
            projects_result = u.Infra.projects(workspace)
            selected_projects = (
                tuple(projects_result.unwrap()) if projects_result.success else ()
            )
        return [self._census_project(project) for project in selected_projects]

    def _census_project(self, project: p.Infra.ProjectInfo) -> m.Infra.CensusReport:
        """Run census on a single project."""
        violations_result = u.Infra.parse_namespace_validation(
            FlextInfraNamespaceValidator().validate_project(project.path)
        )
        violations = (
            list(violations_result.unwrap()) if violations_result.success else []
        )
        return m.Infra.CensusReport(
            project=project.name,
            violations=list(violations),
            total=len(violations),
            fixable=u.count(violations, lambda v: v.fixable),
        )


__all__: list[str] = ["FlextInfraCodegenCensus"]
