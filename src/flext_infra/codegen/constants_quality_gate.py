"""Workspace-wide quality gate for constants refactor outcomes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from flext_infra import FlextInfraCodegenCensus, c, t, u


class FlextInfraCodegenConstantsQualityGate:
    """Run final constants migration checks with before/after comparison."""

    _REPORT_DIR: Path = Path(c.Infra.Codegen.QualityGate.REPORT_DIR)
    _PASS_VERDICTS: tuple[str, ...] = c.Infra.Codegen.QualityGate.PASS_VERDICTS

    def __init__(
        self,
        *,
        workspace_root: Path,
        before_report: Path | None = None,
        baseline_file: Path | None = None,
    ) -> None:
        """Initialize quality gate with workspace and optional baseline source."""
        super().__init__()
        self._workspace_root = workspace_root.resolve()
        self._before_report = before_report
        self._baseline_file = baseline_file

    def run(self) -> dict[str, t.Infra.InfraValue]:
        """Execute quality gate and return structured report payload."""
        before_payload, before_source, before_load_error = (
            u.Infra.quality_gate_load_before_payload(
                workspace_root=self._workspace_root,
                before_report=self._before_report,
                baseline_file=self._baseline_file,
            )
        )
        census_reports = FlextInfraCodegenCensus(
            workspace_root=self._workspace_root,
        ).run()
        duplicate_groups = u.Infra.quality_gate_count_duplicate_constant_groups(
            self._workspace_root,
        )
        modified_files = u.Infra.quality_gate_modified_python_files(
            self._workspace_root,
        )
        pyrefly_check = u.Infra.quality_gate_run_pyrefly_check(
            self._workspace_root,
            modified_files,
        )
        ruff_check = u.Infra.quality_gate_run_ruff_check(
            self._workspace_root,
            modified_files,
        )
        import_scan = u.Infra.quality_gate_scan_import_nodes(
            self._workspace_root,
            modified_files,
        )
        before_metrics = u.Infra.quality_gate_before_metrics(before_payload)
        after_metrics = u.Infra.quality_gate_after_metrics(
            census_reports=census_reports,
            duplicate_groups=duplicate_groups,
            import_scan=import_scan,
            modified_files=modified_files,
        )
        improvement = u.Infra.quality_gate_improvement(
            before_metrics,
            after_metrics,
        )
        checks = u.Infra.quality_gate_build_checks(
            after_metrics=after_metrics,
            improvement=improvement,
            pyrefly_check=pyrefly_check,
            ruff_check=ruff_check,
            before_available=before_payload is not None,
            before_load_error=before_load_error,
        )
        verdict = u.Infra.quality_gate_compute_verdict(checks, improvement)
        checks_infra: list[t.Infra.InfraValue] = list(checks)
        projects_infra: list[t.Infra.InfraValue] = list(
            u.Infra.quality_gate_project_findings(census_reports)
        )
        report: dict[str, t.Infra.InfraValue] = {
            "workspace": str(self._workspace_root),
            "generated_at": datetime.now(UTC).isoformat(),
            "verdict": verdict,
            "checks": checks_infra,
            "baseline": {
                "source": before_source,
                "load_error": before_load_error,
                "provided": before_payload is not None,
            },
            "before": before_metrics,
            "after": after_metrics,
            "improvement": improvement,
            "projects": projects_infra,
        }
        report["artifacts"] = u.Infra.quality_gate_write_artifacts(
            workspace_root=self._workspace_root,
            report=report,
            render_text=self.render_text(report),
            census_reports=census_reports,
            duplicate_groups=duplicate_groups,
            before_payload=before_payload,
        )
        return report

    @classmethod
    def render_text(cls, report: dict[str, t.Infra.InfraValue]) -> str:
        """Render compact human-readable summary."""
        checks = u.Infra.dict_list(report.get("checks"))
        before = u.Infra.dict_or_empty(report.get("before"))
        after = u.Infra.dict_or_empty(report.get("after"))
        improvement = u.Infra.dict_or_empty(report.get("improvement"))
        lines: list[str] = [
            f"Workspace: {report.get('workspace', '')}",
            f"Verdict: {report.get('verdict', 'FAIL')}",
            "",
            "Checks:",
        ]
        for check in checks:
            status = "PASS" if bool(check.get("passed", False)) else "FAIL"
            lines.append(f"- [{status}] {check.get('name', 'unknown')}")
            detail = str(check.get("detail", "")).strip()
            if detail:
                lines.append(f"  {detail}")
        lines.extend([
            "",
            "Before/After:",
            f"- violations: {before.get('total_violations', 'n/a')} -> {after.get('total_violations', 'n/a')}",
            f"- duplicates: {before.get('duplicate_groups', 'n/a')} -> {after.get('duplicate_groups', 'n/a')}",
            f"- projects: {after.get('projects_total', 0)} total, {after.get('projects_passed', 0)} passed, {after.get('projects_failed', 0)} failed",
            "",
            "Improvement:",
            f"- violations_delta: {improvement.get('violations_delta', 0)}",
            f"- duplicates_delta: {improvement.get('duplicates_delta', 0)}",
            f"- violations_reduced: {improvement.get('violations_reduced', 0)}",
            f"- duplicates_eliminated: {improvement.get('duplicates_eliminated', 0)}",
        ])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def is_success_verdict(verdict: str) -> bool:
        """Return True for verdicts that should exit with status 0."""
        return verdict in FlextInfraCodegenConstantsQualityGate._PASS_VERDICTS


__all__ = ["FlextInfraCodegenConstantsQualityGate"]
