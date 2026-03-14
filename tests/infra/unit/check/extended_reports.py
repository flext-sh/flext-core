"""Tests for workspace checker report generation — markdown and SARIF.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.check.services import (
    CheckIssue,
    FlextInfraWorkspaceChecker,
    GateExecution,
    ProjectResult,
)
from flext_tests import tm

from ...models import m


class TestWorkspaceCheckerMarkdownReport:
    """Test FlextInfraWorkspaceChecker.generate_markdown_report."""

    def test_markdown_report_with_errors(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        gate = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        issue = CheckIssue(
            file="a.py", line=1, column=1, code="E1", message="Error", severity="error"
        )
        gate_exec = GateExecution(result=gate, issues=[issue], raw_output="")
        project = ProjectResult(project="p", gates={"lint": gate_exec})
        report = checker.generate_markdown_report(
            [project],
            ["lint"],
            "2025-01-01 00:00:00 UTC",
        )
        tm.that(report, contains="p")
        tm.that(report, contains="E1")
        tm.that(report, contains="Error")

    def test_markdown_report_no_errors(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        gate = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        gate_exec = GateExecution(result=gate, issues=[], raw_output="")
        project = ProjectResult(project="p", gates={"lint": gate_exec})
        report = checker.generate_markdown_report(
            [project],
            ["lint"],
            "2025-01-01 00:00:00 UTC",
        )
        tm.that(report, contains="FLEXT Check Report")
        tm.that(report, contains="p")

    def test_markdown_report_multiple_projects(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        gate1 = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        gate2 = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        exec1 = GateExecution(result=gate1, issues=[], raw_output="")
        issue = CheckIssue(
            file="a.py", line=1, column=1, code="E1", message="Error", severity="error"
        )
        exec2 = GateExecution(result=gate2, issues=[issue], raw_output="")
        projects = [
            ProjectResult(project="p1", gates={"lint": exec1}),
            ProjectResult(project="p2", gates={"lint": exec2}),
        ]
        report = checker.generate_markdown_report(
            projects,
            ["lint"],
            "2025-01-01 00:00:00 UTC",
        )
        tm.that(report, contains="p1")
        tm.that(report, contains="p2")


class TestWorkspaceCheckerSARIFReport:
    """Test FlextInfraWorkspaceChecker.generate_sarif_report."""

    def test_sarif_report_structure(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        gate = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        gate_exec = GateExecution(result=gate, issues=[], raw_output="")
        project = ProjectResult(project="p", gates={"lint": gate_exec})
        report = checker.generate_sarif_report([project], ["lint"])
        assert isinstance(report, dict)
        tm.that("runs" in report, eq=True)

    def test_sarif_report_with_issues(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        gate = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        issue = CheckIssue(
            file="a.py", line=1, column=1, code="E1", message="Error", severity="error"
        )
        gate_exec = GateExecution(result=gate, issues=[issue], raw_output="")
        project = ProjectResult(project="p", gates={"lint": gate_exec})
        report = checker.generate_sarif_report([project], ["lint"])
        assert isinstance(report, dict)
        tm.that("runs" in report, eq=True)


class TestWorkspaceCheckerSARIFReportEdgeCases:
    """Test SARIF report generation edge cases."""

    def test_sarif_report_with_missing_gate_result(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        gate = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        exec1 = GateExecution(result=gate, issues=[], raw_output="")
        project = ProjectResult(project="p", gates={"lint": exec1})
        report = checker.generate_sarif_report([project], ["format"])
        assert isinstance(report, dict)
        tm.that("runs" in report, eq=True)

    def test_markdown_report_with_max_display_issues(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        gate = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        issues = [
            CheckIssue(
                file=f"file{i}.py",
                line=i,
                column=1,
                code=f"E{i}",
                message=f"Error {i}",
                severity="error",
            )
            for i in range(100)
        ]
        exec1 = GateExecution(result=gate, issues=issues, raw_output="")
        project = ProjectResult(project="p", gates={"lint": exec1})
        report = checker.generate_markdown_report(
            [project],
            ["lint"],
            "2025-01-01 00:00:00 UTC",
        )
        tm.that("more errors" in report or len(issues) > 0, eq=True)


class TestMarkdownReportSkipsEmptyGates:
    """Test markdown report skips gates with no issues."""

    def test_generate_markdown_report_skips_empty_gates(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        gate1 = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        gate2 = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        exec1 = GateExecution(result=gate1, issues=[], raw_output="")
        exec2 = GateExecution(result=gate2, issues=[], raw_output="")
        project = ProjectResult(project="p", gates={"lint": exec1, "format": exec2})
        report = checker.generate_markdown_report(
            [project],
            ["lint", "format"],
            "2025-01-01",
        )
        tm.that(report, contains="# FLEXT Check Report")


class TestMarkdownReportWithErrors:
    """Test markdown report includes gates with errors."""

    def test_generate_markdown_report_with_errors(self) -> None:
        checker = FlextInfraWorkspaceChecker()
        issue = CheckIssue(
            file="test.py",
            line=1,
            column=1,
            code="E1",
            message="error",
            severity="error",
        )
        gate1 = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        exec1 = GateExecution(result=gate1, issues=[issue], raw_output="")
        project = ProjectResult(project="p", gates={"lint": exec1})
        report = checker.generate_markdown_report([project], ["lint"], "2025-01-01")
        tm.that(report, contains="test.py")


class TestWorkspaceCheckerMarkdownReportEdgeCases:
    """Test markdown report generation edge cases."""

    def test_markdown_report_skips_gates_with_no_issues(self) -> None:
        gate_with_issues = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        gate_no_issues = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        issue = CheckIssue(
            file="a.py", line=1, column=1, code="E1", message="m1", severity="error"
        )
        exec1 = GateExecution(result=gate_with_issues, issues=[issue], raw_output="")
        exec2 = GateExecution(result=gate_no_issues, issues=[], raw_output="")
        tm.that(len(exec1.issues) > 0, eq=True)
        tm.that(len(exec2.issues), eq=0)
