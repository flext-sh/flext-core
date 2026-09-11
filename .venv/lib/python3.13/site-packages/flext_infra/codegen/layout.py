"""Project-layout engine command service (flext-0wuz, epic flext-hzox).

Check mode reports layout violations from the declarative SSOT in
``config/codegen.yaml``; apply mode performs the reorganization idempotently
(git mv / archive-not-delete / canonical gitignore) and requires a verified
fixed point, mirroring the conform service contract.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, override

from flext_infra import c, m, p, r, t, u
from flext_infra.base import s
from flext_infra.codegen._layout_apply import FlextInfraCodegenLayoutApplyMixin
from flext_infra.codegen._layout_plan import FlextInfraCodegenLayoutPlanMixin


class FlextInfraCodegenLayout(
    FlextInfraCodegenLayoutApplyMixin, FlextInfraCodegenLayoutPlanMixin, s[str]
):
    """Check or apply the canonical project layout from the layout SSOT."""

    project_name: Annotated[
        str | None, m.Field(alias="project", description="Single project to conform")
    ] = None

    @override
    def execute(self) -> p.Result[str]:
        """Run check (default) or apply across the selected projects."""
        selected = self._project_dirs()
        if selected.failure:
            return r[str].fail(selected.error or "project selection failed")
        spec = self._layout_spec
        reports: list[m.Infra.LayoutProjectReport] = []
        for project_dir in selected.value:
            planned = self.plan_project(project_dir)
            if self.effective_dry_run:
                reports.append(planned)
                continue
            applied = self.apply_project(project_dir, planned)
            if applied.failure:
                return r[str].fail(
                    applied.error or f"layout apply failed: {project_dir.name}"
                )
            residual = self.plan_project(project_dir)
            actionable: tuple[m.Infra.LayoutFinding, ...] = residual.actionable
            unresolved = tuple(
                finding.path for finding in actionable if finding.status == "planned"
            )
            if unresolved:
                paths = ", ".join(unresolved)
                return r[str].fail(
                    f"layout apply did not reach a fixed point in "
                    f"{project_dir.name}: {paths}"
                )
            reports.append(applied.value)
        output = self._render_output(reports)
        if self.effective_dry_run and spec.severity == "error":
            blocking: list[str] = []
            for report in reports:
                report_actionable: tuple[m.Infra.LayoutFinding, ...] = report.actionable
                blocking.extend(finding.message for finding in report_actionable)
            if blocking:
                return r[str].fail(f"layout violations: {'; '.join(blocking)}")
        return r[str].ok(output)

    def check_project(self, project_dir: Path) -> m.Infra.LayoutProjectReport:
        """Plan one project directory (gate seam — pure, never writes)."""
        return self.plan_project(project_dir)

    def _project_dirs(self) -> p.Result[t.SequenceOf[Path]]:
        """Resolve the selected project directories, degrading to plain dirs."""
        if self.project_name is not None:
            candidate = self.repository_root / self.project_name
            if candidate.is_dir():
                return r[t.SequenceOf[Path]].ok((candidate,))
            if (
                self.repository_root.name == self.project_name
                and self.repository_root.is_dir()
            ):
                return r[t.SequenceOf[Path]].ok((self.repository_root,))
            return r[t.SequenceOf[Path]].fail(
                f"project not found in workspace: {self.project_name}"
            )
        discovered = u.Infra.projects(self.repository_root)
        if discovered.success and discovered.value:
            return r[t.SequenceOf[Path]].ok(
                tuple(project.path for project in discovered.value)
            )
        if self.repository_root.is_dir():
            return r[t.SequenceOf[Path]].ok((self.repository_root,))
        return r[t.SequenceOf[Path]].fail("no projects discovered")

    def _render_output(self, reports: t.SequenceOf[m.Infra.LayoutProjectReport]) -> str:
        """Render the text or JSON summary for the collected reports."""
        if self.output_format == c.Cli.OutputFormats.JSON:
            return m.Infra.LayoutRunReport(reports=tuple(reports)).model_dump_json()
        lines: t.MutableSequenceOf[str] = []
        mode = "CHECK" if self.effective_dry_run else "APPLY"
        for report in reports:
            lines.append(f"[{mode}] {report.project}")
            for finding in report.findings:
                lines.append(f"  {finding.rule}: {finding.message}")
            if not report.findings:
                lines.append("  layout canonical: no findings")
        actionable = sum(len(report.actionable) for report in reports)
        applied = sum(report.applied_count for report in reports)
        lines.append(
            f"{len(reports)} project(s), {actionable} actionable finding(s), "
            f"{applied} applied"
        )
        return "\n".join(lines)


__all__: list[str] = ["FlextInfraCodegenLayout"]
