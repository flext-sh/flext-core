"""Validate silent failure sentinels across governed Python projects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, m, u
from flext_infra.base import s
from flext_infra.detectors.silent_failure_detector import (
    FlextInfraSilentFailureDetector,
)

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraSilentFailureValidator(s[bool]):
    """Validate that failure paths do not collapse into sentinel returns."""

    project_filter: Annotated[
        str | None, m.Field(description="Project filter (comma-separated)")
    ] = None

    def _selected_projects(
        self, projects: t.SequenceOf[p.Infra.ProjectInfo]
    ) -> t.SequenceOf[p.Infra.ProjectInfo]:
        """Return the selected projects."""
        if self.project_filter is None:
            return projects
        selected = {
            item.strip() for item in self.project_filter.split(",") if item.strip()
        }
        return [project for project in projects if project.name in selected]

    def build_report(self) -> p.Result[m.Infra.ValidationReport]:
        """Build one validation report for the selected workspace projects."""
        issues: t.MutableSequenceOf[str] = []
        projects_result = u.Infra.projects(self.repository_root)
        projects = self._selected_projects(
            tuple(projects_result.unwrap()) if projects_result.success else ()
        )
        for project in projects:
            iter_result = u.Infra.iter_python_files(
                m.Infra.SourceScanRequest(project_roots=(project.path,))
            )
            if iter_result.failure:
                return r[m.Infra.ValidationReport].fail(
                    iter_result.error
                    or f"python file iteration failed for {project.name}"
                )
            rope_project = u.Infra.init_rope_project(project.path)
            try:
                for file_path in iter_result.value:
                    issues.extend(
                        issue.formatted
                        for issue in FlextInfraSilentFailureDetector.detect_file(
                            m.Infra.DetectorContext(
                                file_path=file_path,
                                project_root=project.path,
                                rope_project=rope_project,
                            )
                        )
                    )
            finally:
                rope_project.close()
        passed = len(issues) == 0
        summary = (
            "silent failure validation passed"
            if passed
            else f"silent failure validation found {len(issues)} issue(s)"
        )
        return r[m.Infra.ValidationReport].ok(
            m.Infra.ValidationReport(
                passed=passed, violations=list(issues), summary=summary
            )
        )

    @override
    def execute(self) -> p.Result[bool]:
        """Execute silent-failure validation and collapse the report to `r[bool]`.

        Failure details honor ``--output-format`` (json emits the full report
        model) and always carry ALL findings — no display truncation.
        """
        report_result = self.build_report()
        if report_result.failure:
            return r[bool].fail(
                report_result.error or "silent failure validation failed"
            )
        report = report_result.value
        if report.passed:
            return r[bool].ok(True)
        details = (
            report.model_dump_json()
            if self.output_format == c.Cli.OutputFormats.JSON
            else "\n".join([report.summary, *report.violations])
        )
        return r[bool].fail(details)


__all__: list[str] = ["FlextInfraSilentFailureValidator"]
