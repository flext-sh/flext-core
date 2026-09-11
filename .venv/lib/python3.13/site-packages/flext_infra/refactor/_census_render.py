"""Census workspace-report text rendering — extracted concern."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_infra import m


class FlextInfraRefactorCensusRenderMixin:
    """Plain-text rendering of the census WorkspaceReport.

    Composed into FlextInfraRefactorCensus via inheritance; pure presentation
    over the report model (no census state).
    """

    @staticmethod
    def _render_workspace_report(report: m.Infra.Census.WorkspaceReport) -> str:
        """Render workspace census report from typed model fields."""
        lines = [
            "Workspace Census Report",
            f"Objects: {report.total_objects}",
            f"Violations: {report.total_violations}",
            f"Fixable: {report.total_fixable}",
            f"Fixes: {report.fixes_total}",
            f"Unused: {report.unused_count}",
            f"Removal candidates: {report.removal_candidate_count}",
            f"Duplicate groups: {len(report.duplicates)}",
            f"Duration: {report.scan_duration_seconds:.2f}s",
            "",
        ]
        lines.extend(
            "- "
            f"{project.project}: "
            f"objects={project.objects_total} "
            f"violations={project.violations_total} "
            f"unused={project.unused_count} "
            f"candidates={project.removal_candidate_count}"
            for project in report.projects
        )
        if report.removal_candidates:
            lines.extend(("", "Candidate preview:"))
            for candidate in report.removal_candidates[:10]:
                reference_groups = (
                    candidate.runtime_reference_sites,
                    candidate.script_reference_sites,
                )
                reference_preview = next(
                    (
                        ", ".join(f"{site.file_path}:{site.line}" for site in group[:3])
                        for group in reference_groups
                        if group
                    ),
                    "",
                )
                lines.append(
                    "- "
                    f"{candidate.reason} "
                    f"{candidate.object_name} "
                    f"@ {candidate.file_path}:{candidate.line}"
                    + (f" refs={reference_preview}" if reference_preview else "")
                )
        return "\n".join(lines)

    @staticmethod
    def render_text(report: m.Infra.Census.WorkspaceReport) -> str:
        """Render the canonical workspace census report."""
        return FlextInfraRefactorCensusRenderMixin._render_workspace_report(report)


__all__: list[str] = ["FlextInfraRefactorCensusRenderMixin"]
