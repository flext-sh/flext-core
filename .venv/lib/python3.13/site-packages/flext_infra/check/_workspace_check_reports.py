"""Workspace check report rendering: markdown + SARIF + summary — extracted concern."""

from __future__ import annotations

import operator
from pathlib import Path

from flext_infra import c, m, p, r, t, u


class FlextInfraWorkspaceCheckReportsMixin:
    """Render markdown/SARIF reports and print the run summary.

    Composed into FlextInfraWorkspaceChecker via inheritance; pure rendering
    over gate results (no checker state).
    """

    @staticmethod
    def _generate_markdown(
        results: t.SequenceOf[m.Infra.ProjectResult],
        gates: t.StrSequence,
        timestamp: str,
    ) -> str:
        """Render markdown check report from project gate results."""
        lines: list[str] = [
            "# Workspace Check Report",
            "",
            f"Generated: {timestamp}",
            f"Projects: {len(results)}",
            "",
            "## Summary",
            "",
            "| Project | Status | Errors |",
            "|---|---:|---:|",
        ]
        for project in results:
            status = "PASS" if project.passed else "FAIL"
            lines.append(f"| {project.project} | {status} | {project.total_errors} |")
        lines.extend(["", "## Details", ""])
        for project in results:
            lines.append(f"### {project.project}")
            for gate in gates:
                execution = project.gates.get(gate)
                if execution is None:
                    continue
                gate_status = "PASS" if execution.result.passed else "FAIL"
                lines.append(
                    f"- {gate}: {gate_status} ({len(execution.issues)} issues)"
                )
                lines.extend(f"  - {issue.formatted}" for issue in execution.issues)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _generate_sarif(
        results: t.SequenceOf[m.Infra.ProjectResult], gates: t.StrSequence
    ) -> m.Infra.SarifReport:
        """Build the SARIF 2.1.0 report model from workspace gate results."""
        rules_by_id: dict[str, m.Infra.SarifRule] = {}
        sarif_results: list[m.Infra.SarifResult] = []
        for project in results:
            for gate in gates:
                execution = project.gates.get(gate)
                if execution is None:
                    continue
                for issue in execution.issues:
                    rule_id = issue.code or gate
                    rules_by_id.setdefault(
                        rule_id,
                        m.Infra.SarifRule(
                            id=rule_id, short_description=f"{gate} issue"
                        ),
                    )
                    sarif_results.append(
                        m.Infra.SarifResult(
                            rule_id=rule_id,
                            level="warning"
                            if issue.severity.lower() == c.Infra.SeverityLevel.WARNING
                            else "error",
                            message=issue.message,
                            locations=[
                                m.Infra.SarifLocation(
                                    uri=issue.file,
                                    start_line=issue.line,
                                    start_column=issue.column,
                                )
                            ],
                        )
                    )
        return m.Infra.SarifReport(
            runs=(
                m.Infra.SarifRun(
                    tool_name="flext-infra-check",
                    information_uri="https://github.com/flext-sh/flext-infra",
                    rules=tuple(rules_by_id.values()),
                    results=tuple(sarif_results),
                ),
            )
        )

    @staticmethod
    def _write_reports_and_summary(
        resolved_gates: t.StrSequence,
        report_base: Path,
        outcome: p.Infra.WorkspaceLoopOutcome,
    ) -> p.Result[t.SequenceOf[m.Infra.ProjectResult]]:
        """Write markdown/SARIF reports and print summary to output."""
        results = outcome.results
        timestamp = u.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        md_path = report_base / "check-report.md"
        md_write_result = u.Cli.atomic_write_text_file(
            md_path,
            FlextInfraWorkspaceCheckReportsMixin._generate_markdown(
                results, resolved_gates, timestamp
            ),
        )
        if md_write_result.failure:
            return r[t.SequenceOf[m.Infra.ProjectResult]].fail(
                md_write_result.error or "failed to write markdown report"
            )
        sarif_path = report_base / "check-report.sarif"
        sarif_report = FlextInfraWorkspaceCheckReportsMixin._generate_sarif(
            results, resolved_gates
        )
        try:
            u.Infra.export_pydantic_json(sarif_report, sarif_path)
        except OSError as exc:
            return r[t.SequenceOf[m.Infra.ProjectResult]].fail(
                f"failed to write sarif report: {exc}"
            )
        total_errors = sum(project.total_errors for project in results)
        success = len(results) - outcome.failed
        u.Cli.summary(
            m.Infra.SummaryStats(
                verb=c.Infra.VERB_CHECK,
                total=len(results),
                success=success,
                failed=outcome.failed,
                skipped=outcome.skipped,
                elapsed=outcome.total_elapsed,
            )
        )
        u.Cli.info(f"Reports: {md_path}")
        u.Cli.info(f"         {sarif_path}")
        if total_errors > 0:
            u.Cli.info("Errors by project:")
            for project in sorted(
                results, key=operator.attrgetter("total_errors"), reverse=True
            ):
                if project.total_errors == 0:
                    continue
                breakdown = ", ".join(
                    f"{gate}={project.gates[gate].error_count}"
                    for gate in resolved_gates
                    if gate in project.gates and project.gates[gate].error_count
                )
                u.Cli.error(
                    f"{project.project:30s} {project.total_errors:6d}  ({breakdown})"
                )
        return r[t.SequenceOf[m.Infra.ProjectResult]].ok(results)


__all__: list[str] = ["FlextInfraWorkspaceCheckReportsMixin"]
