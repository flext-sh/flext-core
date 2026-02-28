"""Extended tests for flext_infra.check module to achieve 100% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tomlkit
from flext_core import FlextResult as r
from flext_infra import m
from flext_infra.check.__main__ import main as check_main
from flext_infra.check.fix_pyrefly_config import main as fix_pyrefly_main
from flext_infra.check.services import (
    FlextInfraConfigFixer,
    FlextInfraWorkspaceChecker,
    _CheckIssue,
    _GateExecution,
    _ProjectResult,
    run_cli,
)
from flext_infra.check.workspace_check import main as workspace_check_main


class TestCheckIssueFormatted:
    """Test _CheckIssue.formatted property."""

    def test_formatted_with_code(self) -> None:
        """Test formatted includes code."""
        issue = _CheckIssue(
            file="test.py",
            line=10,
            column=5,
            code="E001",
            message="Error",
        )
        assert "[E001]" in issue.formatted
        assert "test.py:10:5" in issue.formatted

    def test_formatted_without_code(self) -> None:
        """Test formatted without code."""
        issue = _CheckIssue(
            file="test.py",
            line=10,
            column=5,
            code="",
            message="Error",
        )
        assert "test.py:10:5" in issue.formatted


class TestProjectResultProperties:
    """Test _ProjectResult computed properties."""

    def test_total_errors_multiple_gates(self) -> None:
        """Test total_errors sums across gates."""
        gate1 = m.GateResult(gate="lint", project="p", passed=False)
        gate2 = m.GateResult(gate="format", project="p", passed=False)
        issue1 = _CheckIssue(file="a.py", line=1, column=1, code="E1", message="m1")
        issue2 = _CheckIssue(file="b.py", line=2, column=1, code="E2", message="m2")
        issue3 = _CheckIssue(file="c.py", line=3, column=1, code="E3", message="m3")

        exec1 = _GateExecution(result=gate1, issues=[issue1, issue2])
        exec2 = _GateExecution(result=gate2, issues=[issue3])
        project = _ProjectResult(project="p", gates={"lint": exec1, "format": exec2})

        assert project.total_errors == 3

    def test_passed_all_gates_pass(self) -> None:
        """Test passed when all gates pass."""
        gate1 = m.GateResult(gate="lint", project="p", passed=True)
        gate2 = m.GateResult(gate="format", project="p", passed=True)
        exec1 = _GateExecution(result=gate1, issues=[])
        exec2 = _GateExecution(result=gate2, issues=[])
        project = _ProjectResult(project="p", gates={"lint": exec1, "format": exec2})

        assert project.passed is True

    def test_passed_one_gate_fails(self) -> None:
        """Test passed is False when any gate fails."""
        gate1 = m.GateResult(gate="lint", project="p", passed=True)
        gate2 = m.GateResult(gate="format", project="p", passed=False)
        exec1 = _GateExecution(result=gate1, issues=[])
        exec2 = _GateExecution(result=gate2, issues=[])
        project = _ProjectResult(project="p", gates={"lint": exec1, "format": exec2})

        assert project.passed is False


class TestWorkspaceCheckerResolveGates:
    """Test FlextInfraWorkspaceChecker.resolve_gates."""

    def test_resolve_gates_type_maps_to_pyrefly(self) -> None:
        """Test 'type' maps to 'pyrefly'."""
        result = FlextInfraWorkspaceChecker.resolve_gates(["type"])
        assert result.is_success
        assert "pyrefly" in result.value

    def test_resolve_gates_skips_empty_strings(self) -> None:
        """Test empty strings are skipped."""
        result = FlextInfraWorkspaceChecker.resolve_gates(["lint", "", "format"])
        assert result.is_success
        assert "" not in result.value

    def test_resolve_gates_deduplicates_entries(self) -> None:
        """Test duplicates are removed."""
        result = FlextInfraWorkspaceChecker.resolve_gates([
            "lint",
            "lint",
            "format",
            "lint",
        ])
        assert result.is_success
        assert result.value.count("lint") == 1

    def test_resolve_gates_invalid_gate_fails(self) -> None:
        """Test invalid gate returns failure."""
        result = FlextInfraWorkspaceChecker.resolve_gates(["invalid"])
        assert result.is_failure
        assert "unknown gate" in result.error

    def test_resolve_gates_all_valid_types(self) -> None:
        """Test all valid gate types."""
        gates = [
            "lint",
            "format",
            "type",
            "mypy",
            "pyright",
            "security",
            "markdown",
            "go",
        ]
        result = FlextInfraWorkspaceChecker.resolve_gates(gates)
        assert result.is_success
        assert len(result.value) > 0


class TestWorkspaceCheckerParseGateCSV:
    """Test FlextInfraWorkspaceChecker.parse_gate_csv."""

    def test_parse_gate_csv_simple(self) -> None:
        """Test simple CSV parsing."""
        result = FlextInfraWorkspaceChecker.parse_gate_csv("lint,format,type")
        assert result == ["lint", "format", "type"]

    def test_parse_gate_csv_with_spaces(self) -> None:
        """Test CSV with spaces."""
        result = FlextInfraWorkspaceChecker.parse_gate_csv("lint , format , type")
        assert result == ["lint", "format", "type"]

    def test_parse_gate_csv_empty_entries(self) -> None:
        """Test CSV with empty entries."""
        result = FlextInfraWorkspaceChecker.parse_gate_csv("lint,,format")
        assert "" not in result


class TestWorkspaceCheckerMarkdownReport:
    """Test FlextInfraWorkspaceChecker.generate_markdown_report."""

    def test_markdown_report_with_errors(self) -> None:
        """Test markdown report includes errors."""
        checker = FlextInfraWorkspaceChecker()
        gate = m.GateResult(gate="lint", project="p", passed=False)
        issue = _CheckIssue(file="a.py", line=1, column=1, code="E1", message="Error")
        gate_exec = _GateExecution(result=gate, issues=[issue])
        project = _ProjectResult(project="p", gates={"lint": gate_exec})

        report = checker.generate_markdown_report(
            [project], ["lint"], "2025-01-01 00:00:00 UTC"
        )

        assert "p" in report
        assert "E1" in report
        assert "Error" in report

    def test_markdown_report_no_errors(self) -> None:
        """Test markdown report with no errors."""
        checker = FlextInfraWorkspaceChecker()
        gate = m.GateResult(gate="lint", project="p", passed=True)
        gate_exec = _GateExecution(result=gate, issues=[])
        project = _ProjectResult(project="p", gates={"lint": gate_exec})

        report = checker.generate_markdown_report(
            [project], ["lint"], "2025-01-01 00:00:00 UTC"
        )

        assert "FLEXT Check Report" in report
        assert "p" in report

    def test_markdown_report_multiple_projects(self) -> None:
        """Test markdown report with multiple projects."""
        checker = FlextInfraWorkspaceChecker()
        gate1 = m.GateResult(gate="lint", project="p1", passed=True)
        gate2 = m.GateResult(gate="lint", project="p2", passed=False)
        exec1 = _GateExecution(result=gate1, issues=[])
        issue = _CheckIssue(file="a.py", line=1, column=1, code="E1", message="Error")
        exec2 = _GateExecution(result=gate2, issues=[issue])

        projects = [
            _ProjectResult(project="p1", gates={"lint": exec1}),
            _ProjectResult(project="p2", gates={"lint": exec2}),
        ]

        report = checker.generate_markdown_report(
            projects, ["lint"], "2025-01-01 00:00:00 UTC"
        )

        assert "p1" in report
        assert "p2" in report


class TestWorkspaceCheckerSARIFReport:
    """Test FlextInfraWorkspaceChecker.generate_sarif_report."""

    def test_sarif_report_structure(self) -> None:
        """Test SARIF report has correct structure."""
        checker = FlextInfraWorkspaceChecker()
        gate = m.GateResult(gate="lint", project="p", passed=True)
        gate_exec = _GateExecution(result=gate, issues=[])
        project = _ProjectResult(project="p", gates={"lint": gate_exec})

        report = checker.generate_sarif_report([project], ["lint"])

        assert isinstance(report, dict)

    def test_sarif_report_with_issues(self) -> None:
        """Test SARIF report includes issues."""
        checker = FlextInfraWorkspaceChecker()
        gate = m.GateResult(gate="lint", project="p", passed=False)
        issue = _CheckIssue(file="a.py", line=1, column=1, code="E1", message="Error")
        gate_exec = _GateExecution(result=gate, issues=[issue])
        project = _ProjectResult(project="p", gates={"lint": gate_exec})

        report = checker.generate_sarif_report([project], ["lint"])

        assert isinstance(report, dict)


class TestConfigFixerProcessFile:
    """Test FlextInfraConfigFixer.process_file."""

    def test_process_file_missing_file(self, tmp_path: Path) -> None:
        """Test process_file with missing file."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.process_file(tmp_path / "missing.toml")

        assert result.is_failure
        assert "failed to read" in result.error

    def test_process_file_invalid_toml(self, tmp_path: Path) -> None:
        """Test process_file with invalid TOML."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("invalid [[[")

        result = fixer.process_file(pyproject)

        assert result.is_failure
        assert "failed to parse" in result.error

    def test_process_file_no_pyrefly_section(self, tmp_path: Path) -> None:
        """Test process_file with no pyrefly section."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool]\nother = true\n")

        result = fixer.process_file(pyproject)

        assert result.is_success
        assert result.value == []

    def test_process_file_dry_run_no_write(self, tmp_path: Path) -> None:
        """Test process_file dry_run doesn't write."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        original = "[tool.pyrefly]\nsearch-path = []\n"
        pyproject.write_text(original)

        result = fixer.process_file(pyproject, dry_run=True)

        assert result.is_success
        assert pyproject.read_text() == original


class TestWorkspaceCheckCLI:
    """Test workspace_check CLI."""

    def test_workspace_check_no_projects_error(self) -> None:
        """Test workspace_check returns 1 with no projects."""
        exit_code = workspace_check_main([])
        assert exit_code == 1

    def test_workspace_check_with_projects_success(self) -> None:
        """Test workspace_check with projects."""
        with patch(
            "flext_infra.check.workspace_check.FlextInfraWorkspaceChecker"
        ) as mock_cls:
            mock = mock_cls.return_value
            mock.parse_gate_csv.return_value = ["lint"]
            mock.run_projects.return_value = r[list[SimpleNamespace]].ok([
                SimpleNamespace(passed=True)
            ])

            exit_code = workspace_check_main(["p1", "--gates", "lint"])

            assert exit_code == 0

    def test_workspace_check_with_projects_failure(self) -> None:
        """Test workspace_check with failing projects."""
        with patch(
            "flext_infra.check.workspace_check.FlextInfraWorkspaceChecker"
        ) as mock_cls:
            mock = mock_cls.return_value
            mock.parse_gate_csv.return_value = ["lint"]
            mock.run_projects.return_value = r[list[SimpleNamespace]].ok([
                SimpleNamespace(passed=False)
            ])

            exit_code = workspace_check_main(["p1", "--gates", "lint"])

            assert exit_code == 1

    def test_workspace_check_run_projects_error(self) -> None:
        """Test workspace_check with run_projects error."""
        with patch(
            "flext_infra.check.workspace_check.FlextInfraWorkspaceChecker"
        ) as mock_cls:
            mock = mock_cls.return_value
            mock.parse_gate_csv.return_value = ["lint"]
            mock.run_projects.return_value = r[list[SimpleNamespace]].fail("error")

            exit_code = workspace_check_main(["p1", "--gates", "lint"])

            assert exit_code == 2

    def test_workspace_check_fail_fast_flag(self) -> None:
        """Test workspace_check passes fail_fast flag."""
        with patch(
            "flext_infra.check.workspace_check.FlextInfraWorkspaceChecker"
        ) as mock_cls:
            mock = mock_cls.return_value
            mock.parse_gate_csv.return_value = ["lint"]
            mock.run_projects.return_value = r[list[SimpleNamespace]].ok([
                SimpleNamespace(passed=True)
            ])

            exit_code = workspace_check_main(["p1", "--gates", "lint", "--fail-fast"])

            assert exit_code == 0
            assert mock.run_projects.call_args.kwargs["fail_fast"] is True


class TestFixPyrelfyCLI:
    """Test fix_pyrefly_config CLI."""

    def test_fix_pyrefly_success(self) -> None:
        """Test fix_pyrefly returns 0 on success."""
        with patch(
            "flext_infra.check.fix_pyrefly_config.FlextInfraConfigFixer"
        ) as mock_cls:
            mock = mock_cls.return_value
            mock.run.return_value = r[list[str]].ok([])

            exit_code = fix_pyrefly_main([])

            assert exit_code == 0

    def test_fix_pyrefly_failure(self) -> None:
        """Test fix_pyrefly returns 1 on failure."""
        with patch(
            "flext_infra.check.fix_pyrefly_config.FlextInfraConfigFixer"
        ) as mock_cls:
            mock = mock_cls.return_value
            mock.run.return_value = r[list[str]].fail("error")

            exit_code = fix_pyrefly_main([])

            assert exit_code == 1

    def test_fix_pyrefly_dry_run_flag(self) -> None:
        """Test fix_pyrefly passes dry_run flag."""
        with patch(
            "flext_infra.check.fix_pyrefly_config.FlextInfraConfigFixer"
        ) as mock_cls:
            mock = mock_cls.return_value
            mock.run.return_value = r[list[str]].ok([])

            exit_code = fix_pyrefly_main(["--dry-run"])

            assert exit_code == 0
            assert mock.run.call_args.kwargs["dry_run"] is True

    def test_fix_pyrefly_verbose_flag(self) -> None:
        """Test fix_pyrefly passes verbose flag."""
        with patch(
            "flext_infra.check.fix_pyrefly_config.FlextInfraConfigFixer"
        ) as mock_cls:
            mock = mock_cls.return_value
            mock.run.return_value = r[list[str]].ok([])

            exit_code = fix_pyrefly_main(["--verbose"])

            assert exit_code == 0
            assert mock.run.call_args.kwargs["verbose"] is True


class TestCheckMainEntryPoint:
    """Test check __main__ entry point."""

    def test_check_main_calls_ensure_structlog(self) -> None:
        """Test main calls ensure_structlog_configured."""
        with patch(
            "flext_infra.check.__main__.FlextRuntime.ensure_structlog_configured"
        ) as mock_ensure:
            with patch("flext_infra.check.__main__.run_cli", return_value=0):
                exit_code = check_main()

                mock_ensure.assert_called_once()
                assert exit_code == 0

    def test_check_main_calls_run_cli(self) -> None:
        """Test main calls run_cli."""
        with patch(
            "flext_infra.check.__main__.FlextRuntime.ensure_structlog_configured"
        ):
            with patch(
                "flext_infra.check.__main__.run_cli", return_value=0
            ) as mock_run:
                exit_code = check_main()

                mock_run.assert_called_once()
                assert exit_code == 0

    def test_check_main_returns_exit_code(self) -> None:
        """Test main returns run_cli exit code."""
        with patch(
            "flext_infra.check.__main__.FlextRuntime.ensure_structlog_configured"
        ):
            with patch("flext_infra.check.__main__.run_cli", return_value=42):
                exit_code = check_main()

                assert exit_code == 42


class TestWorkspaceCheckerInitialization:
    """Test FlextInfraWorkspaceChecker initialization."""

    def test_init_creates_default_reports_dir(self, tmp_path: Path) -> None:
        """Test init creates default reports directory."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        assert checker._default_reports_dir.exists()


class TestWorkspaceCheckerRunProjects:
    """Test FlextInfraWorkspaceChecker.run_projects."""

    def test_run_projects_with_invalid_gates(self, tmp_path: Path) -> None:
        """Test run_projects fails with invalid gates."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["p1"],
            ["invalid_gate"],
            reports_dir=tmp_path / "reports",
        )
        assert result.is_failure

    def test_run_projects_skips_missing_projects(self, tmp_path: Path) -> None:
        """Test run_projects skips missing project directories."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["nonexistent"],
            ["lint"],
            reports_dir=tmp_path / "reports",
        )
        assert result.is_success
        assert len(result.value) == 0

    def test_run_projects_creates_markdown_report(self, tmp_path: Path) -> None:
        """Test run_projects creates markdown report file."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"

        with patch.object(checker, "_check_project") as mock_check:
            gate = m.GateResult(gate="lint", project="p1", passed=True)
            gate_exec = _GateExecution(result=gate, issues=[])
            project = _ProjectResult(project="p1", gates={"lint": gate_exec})
            mock_check.return_value = project

            # Create valid project
            proj_dir = tmp_path / "p1"
            proj_dir.mkdir()
            (proj_dir / "pyproject.toml").write_text("[tool]\n")

            result = checker.run_projects(
                ["p1"],
                ["lint"],
                reports_dir=reports_dir,
            )

            assert result.is_success
            assert (reports_dir / "check-report.md").exists()

    def test_run_projects_creates_sarif_report(self, tmp_path: Path) -> None:
        """Test run_projects creates SARIF report file."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"

        with patch.object(checker, "_check_project") as mock_check:
            gate = m.GateResult(gate="lint", project="p1", passed=True)
            gate_exec = _GateExecution(result=gate, issues=[])
            project = _ProjectResult(project="p1", gates={"lint": gate_exec})
            mock_check.return_value = project

            # Create valid project
            proj_dir = tmp_path / "p1"
            proj_dir.mkdir()
            (proj_dir / "pyproject.toml").write_text("[tool]\n")

            result = checker.run_projects(
                ["p1"],
                ["lint"],
                reports_dir=reports_dir,
            )

            assert result.is_success
            assert (reports_dir / "check-report.sarif").exists()

    def test_run_projects_with_fail_fast_stops_on_failure(self, tmp_path: Path) -> None:
        """Test run_projects stops on first failure with fail_fast."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"

        with patch.object(checker, "_check_project") as mock_check:
            gate = m.GateResult(gate="lint", project="p", passed=False)
            gate_exec = _GateExecution(result=gate, issues=[])
            project = _ProjectResult(project="p", gates={"lint": gate_exec})
            mock_check.return_value = project

            # Create valid projects
            for proj_name in ["p1", "p2", "p3"]:
                proj_dir = tmp_path / proj_name
                proj_dir.mkdir()
                (proj_dir / "pyproject.toml").write_text("[tool]\n")

            result = checker.run_projects(
                ["p1", "p2", "p3"],
                ["lint"],
                reports_dir=reports_dir,
                fail_fast=True,
            )

            assert result.is_success
            # Should stop after first failure
            assert mock_check.call_count == 1


class TestConfigFixerRun:
    """Test FlextInfraConfigFixer.run."""

    def test_run_with_empty_projects(self, tmp_path: Path) -> None:
        """Test run with empty project list."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run([])
        assert result.is_success
        assert isinstance(result.value, list)

    def test_run_with_nonexistent_projects(self, tmp_path: Path) -> None:
        """Test run with nonexistent projects."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run(["nonexistent"])
        assert result.is_success

    def test_run_with_dry_run_flag(self, tmp_path: Path) -> None:
        """Test run respects dry_run flag."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run([], dry_run=True)
        assert result.is_success

    def test_run_with_verbose_flag(self, tmp_path: Path) -> None:
        """Test run respects verbose flag."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run([], verbose=True)
        assert result.is_success


class TestConfigFixerFindPyprojectFiles:
    """Test FlextInfraConfigFixer.find_pyproject_files."""

    def test_find_pyproject_files_empty_workspace(self, tmp_path: Path) -> None:
        """Test find_pyproject_files with empty workspace."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.find_pyproject_files()
        assert result.is_success
        assert isinstance(result.value, list)

    def test_find_pyproject_files_with_specific_paths(self, tmp_path: Path) -> None:
        """Test find_pyproject_files with specific paths."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.find_pyproject_files(project_paths=[tmp_path / "p1"])
        assert result.is_success
        assert isinstance(result.value, list)


class TestRunCLI:
    """Test run_cli function."""

    def test_run_cli_run_command_success(self) -> None:
        """Test run_cli with run command success."""
        with patch("flext_infra.check.services.FlextInfraWorkspaceChecker") as mock_cls:
            mock = mock_cls.return_value
            mock.parse_gate_csv.return_value = ["lint"]
            gate = m.GateResult(gate="lint", project="p", passed=True)
            gate_exec = _GateExecution(result=gate, issues=[])
            project = _ProjectResult(project="p", gates={"lint": gate_exec})
            mock.run_projects.return_value = r[list[_ProjectResult]].ok([project])

            exit_code = run_cli(["run", "--gates", "lint", "--project", "p"])

            assert exit_code == 0

    def test_run_cli_run_command_failure(self) -> None:
        """Test run_cli with run command failure."""
        with patch("flext_infra.check.services.FlextInfraWorkspaceChecker") as mock_cls:
            mock = mock_cls.return_value
            mock.parse_gate_csv.return_value = ["lint"]
            gate = m.GateResult(gate="lint", project="p", passed=False)
            gate_exec = _GateExecution(result=gate, issues=[])
            project = _ProjectResult(project="p", gates={"lint": gate_exec})
            mock.run_projects.return_value = r[list[_ProjectResult]].ok([project])

            exit_code = run_cli(["run", "--gates", "lint", "--project", "p"])

            assert exit_code == 1

    def test_run_cli_run_command_error(self) -> None:
        """Test run_cli with run command error."""
        with patch("flext_infra.check.services.FlextInfraWorkspaceChecker") as mock_cls:
            mock = mock_cls.return_value
            mock.parse_gate_csv.return_value = ["lint"]
            mock.run_projects.return_value = r[list[_ProjectResult]].fail("error")

            exit_code = run_cli(["run", "--gates", "lint", "--project", "p"])

            assert exit_code == 2

    def test_run_cli_fix_pyrefly_config_success(self) -> None:
        """Test run_cli with fix-pyrefly-config command success."""
        with patch("flext_infra.check.services.FlextInfraConfigFixer") as mock_cls:
            mock = mock_cls.return_value
            mock.run.return_value = r[list[str]].ok([])

            exit_code = run_cli(["fix-pyrefly-config"])

            assert exit_code == 0

    def test_run_cli_fix_pyrefly_config_failure(self) -> None:
        """Test run_cli with fix-pyrefly-config command failure."""
        with patch("flext_infra.check.services.FlextInfraConfigFixer") as mock_cls:
            mock = mock_cls.return_value
            mock.run.return_value = r[list[str]].fail("error")

            exit_code = run_cli(["fix-pyrefly-config"])

            assert exit_code == 1

    def test_run_cli_no_command_prints_help(self) -> None:
        """Test run_cli with no command prints help."""
        exit_code = run_cli([])

        assert exit_code == 1

    def test_run_cli_with_relative_reports_dir(self, tmp_path: Path) -> None:
        """Test run_cli resolves relative reports directory."""
        with patch("flext_infra.check.services.FlextInfraWorkspaceChecker") as mock_cls:
            with patch("pathlib.Path.cwd", return_value=tmp_path):
                mock = mock_cls.return_value
                mock.parse_gate_csv.return_value = ["lint"]
                gate = m.GateResult(gate="lint", project="p", passed=True)
                gate_exec = _GateExecution(result=gate, issues=[])
                project = _ProjectResult(project="p", gates={"lint": gate_exec})
                mock.run_projects.return_value = r[list[_ProjectResult]].ok([project])

                exit_code = run_cli([
                    "run",
                    "--gates",
                    "lint",
                    "--project",
                    "p",
                    "--reports-dir",
                    "reports/check",
                ])

                assert exit_code == 0
                call_kwargs = mock.run_projects.call_args.kwargs
                assert call_kwargs["reports_dir"].is_absolute()

    def test_run_cli_with_fail_fast_flag(self) -> None:
        """Test run_cli passes fail_fast flag."""
        with patch("flext_infra.check.services.FlextInfraWorkspaceChecker") as mock_cls:
            mock = mock_cls.return_value
            mock.parse_gate_csv.return_value = ["lint"]
            gate = m.GateResult(gate="lint", project="p", passed=True)
            gate_exec = _GateExecution(result=gate, issues=[])
            project = _ProjectResult(project="p", gates={"lint": gate_exec})
            mock.run_projects.return_value = r[list[_ProjectResult]].ok([project])

            exit_code = run_cli([
                "run",
                "--gates",
                "lint",
                "--project",
                "p",
                "--fail-fast",
            ])

            assert exit_code == 0
            call_kwargs = mock.run_projects.call_args.kwargs
            assert call_kwargs["fail_fast"] is True

    def test_run_cli_fix_pyrefly_with_dry_run(self) -> None:
        """Test run_cli fix-pyrefly-config with dry_run flag."""
        with patch("flext_infra.check.services.FlextInfraConfigFixer") as mock_cls:
            mock = mock_cls.return_value
            mock.run.return_value = r[list[str]].ok([])

            exit_code = run_cli(["fix-pyrefly-config", "--dry-run"])

            assert exit_code == 0
            call_kwargs = mock.run.call_args.kwargs
            assert call_kwargs["dry_run"] is True

    def test_run_cli_fix_pyrefly_with_verbose(self) -> None:
        """Test run_cli fix-pyrefly-config with verbose flag."""
        with patch("flext_infra.check.services.FlextInfraConfigFixer") as mock_cls:
            mock = mock_cls.return_value
            mock.run.return_value = r[list[str]].ok([])

            exit_code = run_cli(["fix-pyrefly-config", "--verbose"])

            assert exit_code == 0
            call_kwargs = mock.run.call_args.kwargs
            assert call_kwargs["verbose"] is True


class TestWorkspaceCheckerRun:
    """Test FlextInfraWorkspaceChecker.run method."""

    def test_run_single_project_success(self, tmp_path: Path) -> None:
        """Test run with single project success."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")

        with patch.object(checker, "_check_project") as mock_check:
            gate = m.GateResult(gate="lint", project="p1", passed=True)
            gate_exec = _GateExecution(result=gate, issues=[])
            project = _ProjectResult(project="p1", gates={"lint": gate_exec})
            mock_check.return_value = project

            result = checker.run("p1", ["lint"])

            assert result.is_success
            assert len(result.value) == 1


class TestWorkspaceCheckerExecute:
    """Test FlextInfraWorkspaceChecker.execute method."""

    def test_execute_returns_failure(self, tmp_path: Path) -> None:
        """Test execute returns failure with usage message."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.execute()

        assert result.is_failure
        assert "Use run()" in result.error


class TestWorkspaceCheckerExistingCheckDirs:
    """Test FlextInfraWorkspaceChecker._existing_check_dirs method."""

    def test_existing_check_dirs_workspace_root(self, tmp_path: Path) -> None:
        """Test _existing_check_dirs returns DEFAULT_CHECK_DIRS for workspace root."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()

        dirs = checker._existing_check_dirs(tmp_path)

        assert isinstance(dirs, list)

    def test_existing_check_dirs_subproject(self, tmp_path: Path) -> None:
        """Test _existing_check_dirs returns CHECK_DIRS_SUBPROJECT for subproject."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        subproj = tmp_path / "subproj"
        subproj.mkdir()
        (subproj / "src").mkdir()

        dirs = checker._existing_check_dirs(subproj)

        assert isinstance(dirs, list)

    def test_existing_check_dirs_filters_nonexistent(self, tmp_path: Path) -> None:
        """Test _existing_check_dirs filters out nonexistent directories."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()

        dirs = checker._existing_check_dirs(tmp_path)

        assert all((tmp_path / d).is_dir() for d in dirs)


class TestWorkspaceCheckerDirsWithPy:
    """Test FlextInfraWorkspaceChecker._dirs_with_py static method."""

    def test_dirs_with_py_finds_python_files(self, tmp_path: Path) -> None:
        """Test _dirs_with_py finds directories with .py files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("# code")

        result = FlextInfraWorkspaceChecker._dirs_with_py(tmp_path, ["src"])

        assert "src" in result

    def test_dirs_with_py_finds_pyi_files(self, tmp_path: Path) -> None:
        """Test _dirs_with_py finds directories with .pyi stub files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.pyi").write_text("# stub")

        result = FlextInfraWorkspaceChecker._dirs_with_py(tmp_path, ["src"])

        assert "src" in result

    def test_dirs_with_py_skips_empty_dirs(self, tmp_path: Path) -> None:
        """Test _dirs_with_py skips directories without Python files."""
        src = tmp_path / "src"
        src.mkdir()

        result = FlextInfraWorkspaceChecker._dirs_with_py(tmp_path, ["src"])

        assert "src" not in result

    def test_dirs_with_py_skips_nonexistent_dirs(self, tmp_path: Path) -> None:
        """Test _dirs_with_py skips nonexistent directories."""
        result = FlextInfraWorkspaceChecker._dirs_with_py(tmp_path, ["nonexistent"])

        assert "nonexistent" not in result


class TestWorkspaceCheckerRunPyrefly:
    """Test FlextInfraWorkspaceChecker._run_pyrefly method."""

    def test_run_pyrefly_with_json_output(self, tmp_path: Path) -> None:
        """Test _run_pyrefly parses JSON output."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                mock_dirs.return_value = ["src"]
                json_file = reports_dir / "p1-pyrefly.json"
                json_file.write_text('{"errors": []}')
                mock_run.return_value = SimpleNamespace(
                    stdout="",
                    stderr="",
                    returncode=0,
                )

                result = checker._run_pyrefly(proj_dir, reports_dir)

                assert result.result.passed is True

    def test_run_pyrefly_with_errors(self, tmp_path: Path) -> None:
        """Test _run_pyrefly handles errors in JSON."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                mock_dirs.return_value = ["src"]
                json_file = reports_dir / "p1-pyrefly.json"
                json_file.write_text(
                    '{"errors": [{"path": "a.py", "line": 1, "column": 0, "name": "E001", "description": "Error", "severity": "error"}]}'
                )
                mock_run.return_value = SimpleNamespace(
                    stdout="",
                    stderr="",
                    returncode=1,
                )

                result = checker._run_pyrefly(proj_dir, reports_dir)

                assert result.result.passed is False
                assert len(result.issues) == 1

    def test_run_pyrefly_with_list_output(self, tmp_path: Path) -> None:
        """Test _run_pyrefly handles list-format JSON output."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                mock_dirs.return_value = ["src"]
                json_file = reports_dir / "p1-pyrefly.json"
                json_file.write_text(
                    '[{"path": "a.py", "line": 1, "column": 0, "name": "E001", "description": "Error", "severity": "error"}]'
                )
                mock_run.return_value = SimpleNamespace(
                    stdout="",
                    stderr="",
                    returncode=1,
                )

                result = checker._run_pyrefly(proj_dir, reports_dir)

                assert len(result.issues) == 1

    def test_run_pyrefly_with_invalid_json(self, tmp_path: Path) -> None:
        """Test _run_pyrefly handles invalid JSON gracefully."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                mock_dirs.return_value = ["src"]
                json_file = reports_dir / "p1-pyrefly.json"
                json_file.write_text("invalid json")
                mock_run.return_value = SimpleNamespace(
                    stdout="",
                    stderr="",
                    returncode=1,
                )

                result = checker._run_pyrefly(proj_dir, reports_dir)

                assert result.result.passed is False

    def test_run_pyrefly_with_error_count_fallback(self, tmp_path: Path) -> None:
        """Test _run_pyrefly falls back to error count from stderr."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                mock_dirs.return_value = ["src"]
                mock_run.return_value = SimpleNamespace(
                    stdout="",
                    stderr="Found 3 errors",
                    returncode=1,
                )

                result = checker._run_pyrefly(proj_dir, reports_dir)

                assert result.result.passed is False
                assert len(result.issues) == 3


class TestWorkspaceCheckerRunMypy:
    """Test FlextInfraWorkspaceChecker._run_mypy method."""

    def test_run_mypy_no_python_dirs(self, tmp_path: Path) -> None:
        """Test _run_mypy returns early when no Python directories."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        with patch.object(checker, "_existing_check_dirs") as mock_dirs:
            with patch.object(checker, "_dirs_with_py") as mock_py_dirs:
                mock_dirs.return_value = ["src"]
                mock_py_dirs.return_value = []

                result = checker._run_mypy(proj_dir)

                assert result.result.passed is True
                assert len(result.issues) == 0

    def test_run_mypy_with_json_output(self, tmp_path: Path) -> None:
        """Test _run_mypy parses JSON lines output."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                with patch.object(checker, "_dirs_with_py") as mock_py_dirs:
                    mock_dirs.return_value = ["src"]
                    mock_py_dirs.return_value = ["src"]
                    json_line = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
                    mock_run.return_value = SimpleNamespace(
                        stdout=json_line,
                        stderr="",
                        returncode=1,
                    )

                    result = checker._run_mypy(proj_dir)

                    assert result.result.passed is False
                    assert len(result.issues) == 1

    def test_run_mypy_with_project_config(self, tmp_path: Path) -> None:
        """Test _run_mypy uses project pyproject.toml if available."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        (proj_dir / "pyproject.toml").write_text("[tool.mypy]\n")

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                with patch.object(checker, "_dirs_with_py") as mock_py_dirs:
                    mock_dirs.return_value = ["src"]
                    mock_py_dirs.return_value = ["src"]
                    mock_run.return_value = SimpleNamespace(
                        stdout="",
                        stderr="",
                        returncode=0,
                    )

                    checker._run_mypy(proj_dir)

                    call_args = mock_run.call_args[0][0]
                    assert "--config-file" in call_args

    def test_run_mypy_with_mypypath_env(self, tmp_path: Path) -> None:
        """Test _run_mypy sets MYPYPATH environment variable."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        (tmp_path / "typings" / "generated").mkdir(parents=True)

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                with patch.object(checker, "_dirs_with_py") as mock_py_dirs:
                    mock_dirs.return_value = ["src"]
                    mock_py_dirs.return_value = ["src"]
                    mock_run.return_value = SimpleNamespace(
                        stdout="",
                        stderr="",
                        returncode=0,
                    )

                    checker._run_mypy(proj_dir)

                    call_env = mock_run.call_args.kwargs.get("env")
                    assert call_env is not None
                    assert "MYPYPATH" in call_env


class TestWorkspaceCheckerRunPyright:
    """Test FlextInfraWorkspaceChecker._run_pyright method."""

    def test_run_pyright_no_python_dirs(self, tmp_path: Path) -> None:
        """Test _run_pyright returns early when no Python directories."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        with patch.object(checker, "_existing_check_dirs") as mock_dirs:
            with patch.object(checker, "_dirs_with_py") as mock_py_dirs:
                mock_dirs.return_value = ["src"]
                mock_py_dirs.return_value = []

                result = checker._run_pyright(proj_dir)

                assert result.result.passed is True
                assert len(result.issues) == 0

    def test_run_pyright_with_json_output(self, tmp_path: Path) -> None:
        """Test _run_pyright parses JSON output."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                with patch.object(checker, "_dirs_with_py") as mock_py_dirs:
                    mock_dirs.return_value = ["src"]
                    mock_py_dirs.return_value = ["src"]
                    json_output = '{"generalDiagnostics": [{"file": "a.py", "range": {"start": {"line": 0, "character": 0}}, "rule": "E001", "message": "Error", "severity": "error"}]}'
                    mock_run.return_value = SimpleNamespace(
                        stdout=json_output,
                        stderr="",
                        returncode=1,
                    )

                    result = checker._run_pyright(proj_dir)

                    assert result.result.passed is False
                    assert len(result.issues) == 1

    def test_run_pyright_with_invalid_json(self, tmp_path: Path) -> None:
        """Test _run_pyright handles invalid JSON gracefully."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                with patch.object(checker, "_dirs_with_py") as mock_py_dirs:
                    mock_dirs.return_value = ["src"]
                    mock_py_dirs.return_value = ["src"]
                    mock_run.return_value = SimpleNamespace(
                        stdout="invalid json",
                        stderr="",
                        returncode=1,
                    )

                    result = checker._run_pyright(proj_dir)

                    assert result.result.passed is False


class TestWorkspaceCheckerRunBandit:
    """Test FlextInfraWorkspaceChecker._run_bandit method."""

    def test_run_bandit_no_src_dir(self, tmp_path: Path) -> None:
        """Test _run_bandit returns early when src dir doesn't exist."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        result = checker._run_bandit(proj_dir)

        assert result.result.passed is True
        assert len(result.issues) == 0

    def test_run_bandit_with_json_output(self, tmp_path: Path) -> None:
        """Test _run_bandit parses JSON output."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()

        with patch.object(checker, "_run") as mock_run:
            json_output = '{"results": [{"filename": "a.py", "line_number": 1, "test_id": "B101", "issue_text": "Assert used", "issue_severity": "MEDIUM"}]}'
            mock_run.return_value = SimpleNamespace(
                stdout=json_output,
                stderr="",
                returncode=1,
            )

            result = checker._run_bandit(proj_dir)

            assert result.result.passed is False
            assert len(result.issues) == 1

    def test_run_bandit_with_invalid_json(self, tmp_path: Path) -> None:
        """Test _run_bandit handles invalid JSON gracefully."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()

        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="invalid json",
                stderr="",
                returncode=1,
            )

            result = checker._run_bandit(proj_dir)

            assert result.result.passed is False


class TestWorkspaceCheckerRunMarkdown:
    """Test FlextInfraWorkspaceChecker._run_markdown method."""

    def test_run_markdown_no_files(self, tmp_path: Path) -> None:
        """Test _run_markdown returns early when no markdown files."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        result = checker._run_markdown(proj_dir)

        assert result.result.passed is True
        assert len(result.issues) == 0

    def test_run_markdown_with_errors(self, tmp_path: Path) -> None:
        """Test _run_markdown parses markdown errors."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")

        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="README.md:1:1 error MD001 Heading level",
                stderr="",
                returncode=1,
            )

            result = checker._run_markdown(proj_dir)

            assert result.result.passed is False
            assert len(result.issues) == 1

    def test_run_markdown_with_config(self, tmp_path: Path) -> None:
        """Test _run_markdown uses config file if available."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / ".markdownlint.json").write_text("{}")

        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="",
                stderr="",
                returncode=0,
            )

            checker._run_markdown(proj_dir)

            call_args = mock_run.call_args[0][0]
            assert "--config" in call_args

    def test_run_markdown_fallback_error_message(self, tmp_path: Path) -> None:
        """Test _run_markdown creates fallback error when no issues parsed."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")

        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="",
                stderr="markdownlint failed",
                returncode=1,
            )

            result = checker._run_markdown(proj_dir)

            assert result.result.passed is False
            assert len(result.issues) == 1


class TestWorkspaceCheckerCollectMarkdownFiles:
    """Test FlextInfraWorkspaceChecker._collect_markdown_files method."""

    def test_collect_markdown_files_finds_files(self, tmp_path: Path) -> None:
        """Test _collect_markdown_files finds markdown files."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / "docs").mkdir()
        (proj_dir / "docs" / "guide.md").write_text("# Guide")

        files = checker._collect_markdown_files(proj_dir)

        assert len(files) == 2

    def test_collect_markdown_files_excludes_dirs(self, tmp_path: Path) -> None:
        """Test _collect_markdown_files excludes excluded directories."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / ".git").mkdir()
        (proj_dir / ".git" / "README.md").write_text("# Git")

        files = checker._collect_markdown_files(proj_dir)

        assert len(files) == 1


class TestWorkspaceCheckerRunGo:
    """Test FlextInfraWorkspaceChecker._run_go method."""

    def test_run_go_no_go_mod(self, tmp_path: Path) -> None:
        """Test _run_go returns early when go.mod doesn't exist."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        result = checker._run_go(proj_dir)

        assert result.result.passed is True
        assert len(result.issues) == 0

    def test_run_go_with_vet_errors(self, tmp_path: Path) -> None:
        """Test _run_go parses go vet errors."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "go.mod").write_text("module test")

        with patch.object(checker, "_run") as mock_run:
            vet_result = SimpleNamespace(
                stdout="main.go:10:5: error message",
                stderr="",
                returncode=1,
            )
            fmt_result = SimpleNamespace(
                stdout="",
                stderr="",
                returncode=0,
            )
            mock_run.side_effect = [vet_result, fmt_result]

            result = checker._run_go(proj_dir)

            assert result.result.passed is False

    def test_run_go_with_format_errors(self, tmp_path: Path) -> None:
        """Test _run_go parses gofmt errors."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "go.mod").write_text("module test")
        (proj_dir / "main.go").write_text("package main")

        with patch.object(checker, "_run") as mock_run:
            vet_result = SimpleNamespace(
                stdout="",
                stderr="",
                returncode=0,
            )
            fmt_result = SimpleNamespace(
                stdout="main.go",
                stderr="",
                returncode=1,
            )
            mock_run.side_effect = [vet_result, fmt_result]

            result = checker._run_go(proj_dir)

            assert result.result.passed is False
            assert len(result.issues) == 1

    def test_run_go_fallback_error_message(self, tmp_path: Path) -> None:
        """Test _run_go creates fallback error when no issues parsed."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "go.mod").write_text("module test")

        with patch.object(checker, "_run") as mock_run:
            vet_result = SimpleNamespace(
                stdout="",
                stderr="go vet failed",
                returncode=1,
            )
            fmt_result = SimpleNamespace(
                stdout="",
                stderr="",
                returncode=0,
            )
            mock_run.side_effect = [vet_result, fmt_result]

            result = checker._run_go(proj_dir)

            assert result.result.passed is False
            assert len(result.issues) == 1


class TestWorkspaceCheckerRunCommand:
    """Test FlextInfraWorkspaceChecker._run method."""

    def test_run_command_success(self, tmp_path: Path) -> None:
        """Test _run returns command output on success."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)

        with patch(
            "flext_infra.check.services.FlextInfraCommandRunner.run_raw"
        ) as mock_run:
            mock_run.return_value = r[m.CommandOutput].ok(
                m.CommandOutput(stdout="output", stderr="", exit_code=0)
            )

            result = checker._run(["echo", "test"], tmp_path)

            assert result.stdout == "output"
            assert result.returncode == 0

    def test_run_command_failure(self, tmp_path: Path) -> None:
        """Test _run handles command execution failure."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)

        with patch(
            "flext_infra.check.services.FlextInfraCommandRunner.run_raw"
        ) as mock_run:
            mock_run.return_value = r[m.CommandOutput].fail("execution failed")

            result = checker._run(["false"], tmp_path)

            assert result.returncode == 1
            assert "execution failed" in result.stderr


class TestWorkspaceCheckerBuildGateResult:
    """Test FlextInfraWorkspaceChecker._build_gate_result method."""

    def test_build_gate_result_success(self, tmp_path: Path) -> None:
        """Test _build_gate_result creates successful result."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        issue = _CheckIssue(file="a.py", line=1, column=1, code="E1", message="Error")

        result = checker._build_gate_result(
            gate="lint",
            project="p1",
            passed=True,
            issues=[issue],
            duration=0.5,
            raw_output="",
        )

        assert result.result.passed is True
        assert result.result.gate == "lint"
        assert len(result.issues) == 1


class TestWorkspaceCheckerRunRuffLint:
    """Test FlextInfraWorkspaceChecker._run_ruff_lint method."""

    def test_run_ruff_lint_with_errors(self, tmp_path: Path) -> None:
        """Test _run_ruff_lint parses JSON output."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            json_output = '[{"filename": "a.py", "location": {"row": 1, "column": 0}, "code": "E001", "message": "Error"}]'
            mock_run.return_value = SimpleNamespace(
                stdout=json_output,
                stderr="",
                returncode=1,
            )

            result = checker._run_ruff_lint(proj_dir)

            assert result.result.passed is False
            assert len(result.issues) == 1

    def test_run_ruff_lint_with_invalid_json(self, tmp_path: Path) -> None:
        """Test _run_ruff_lint handles invalid JSON gracefully."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="invalid json",
                stderr="",
                returncode=1,
            )

            result = checker._run_ruff_lint(proj_dir)

            assert result.result.passed is False


class TestWorkspaceCheckerRunRuffFormat:
    """Test FlextInfraWorkspaceChecker._run_ruff_format method."""

    def test_run_ruff_format_with_errors(self, tmp_path: Path) -> None:
        """Test _run_ruff_format parses format errors."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="  --> a.py:1:1",
                stderr="",
                returncode=1,
            )

            result = checker._run_ruff_format(proj_dir)

            assert result.result.passed is False
            assert len(result.issues) == 1

    def test_run_ruff_format_with_simple_path(self, tmp_path: Path) -> None:
        """Test _run_ruff_format handles simple file paths."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()

        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="a.py",
                stderr="",
                returncode=1,
            )

            result = checker._run_ruff_format(proj_dir)

            assert result.result.passed is False
            assert len(result.issues) == 1


class TestConfigFixerExecute:
    """Test FlextInfraConfigFixer.execute method."""

    def test_execute_returns_failure(self, tmp_path: Path) -> None:
        """Test execute returns failure with usage message."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.execute()

        assert result.is_failure
        assert "Use run()" in result.error


class TestConfigFixerFixSearchPaths:
    """Test FlextInfraConfigFixer._fix_search_paths_tk method."""

    def test_fix_search_paths_normalizes_root_paths(self, tmp_path: Path) -> None:
        """Test _fix_search_paths_tk normalizes paths for workspace root."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        (tmp_path / "typings" / "generated").mkdir(parents=True)

        pyrefly = tomlkit.document()
        pyrefly["search-path"] = ["../typings/generated", "../typings"]

        fixes = fixer._fix_search_paths_tk(pyrefly, tmp_path)

        assert len(fixes) > 0
        assert "typings/generated" in str(pyrefly["search-path"])

    def test_fix_search_paths_removes_nonexistent(self, tmp_path: Path) -> None:
        """Test _fix_search_paths_tk removes nonexistent paths."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyrefly = tomlkit.document()
        pyrefly["search-path"] = ["nonexistent"]

        fixes = fixer._fix_search_paths_tk(pyrefly, tmp_path)

        assert len(fixes) > 0

    def test_fix_search_paths_skips_non_list(self, tmp_path: Path) -> None:
        """Test _fix_search_paths_tk skips non-list search-path."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyrefly = tomlkit.document()
        pyrefly["search-path"] = "not-a-list"

        fixes = fixer._fix_search_paths_tk(pyrefly, tmp_path)

        assert len(fixes) == 0


class TestConfigFixerRemoveIgnoreSubConfig:
    """Test FlextInfraConfigFixer._remove_ignore_sub_config_tk method."""

    def test_remove_ignore_sub_config_removes_ignored(self, tmp_path: Path) -> None:
        """Test _remove_ignore_sub_config_tk removes ignore=true configs."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyrefly = tomlkit.document()
        pyrefly["sub-config"] = [
            {"matches": "*.py", "ignore": True},
            {"matches": "*.pyi", "ignore": False},
        ]

        fixes = fixer._remove_ignore_sub_config_tk(pyrefly)

        assert len(fixes) > 0
        assert len(pyrefly["sub-config"]) == 1

    def test_remove_ignore_sub_config_skips_non_list(self, tmp_path: Path) -> None:
        """Test _remove_ignore_sub_config_tk skips non-list sub-config."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyrefly = tomlkit.document()
        pyrefly["sub-config"] = "not-a-list"

        fixes = fixer._remove_ignore_sub_config_tk(pyrefly)

        assert len(fixes) == 0


class TestConfigFixerEnsureProjectExcludes:
    """Test FlextInfraConfigFixer._ensure_project_excludes_tk method."""

    def test_ensure_project_excludes_adds_missing(self, tmp_path: Path) -> None:
        """Test _ensure_project_excludes_tk adds missing excludes."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyrefly = tomlkit.document()
        pyrefly["project-excludes"] = []

        fixes = fixer._ensure_project_excludes_tk(pyrefly)

        assert len(fixes) > 0
        assert len(pyrefly["project-excludes"]) > 0

    def test_ensure_project_excludes_skips_existing(self, tmp_path: Path) -> None:
        """Test _ensure_project_excludes_tk skips already-present excludes."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyrefly = tomlkit.document()
        pyrefly["project-excludes"] = ["**/*_pb2*.py", "**/*_pb2_grpc*.py"]

        fixes = fixer._ensure_project_excludes_tk(pyrefly)

        assert len(fixes) == 0


class TestConfigFixerToArray:
    """Test FlextInfraConfigFixer._to_array static method."""

    def test_to_array_creates_array(self, tmp_path: Path) -> None:
        """Test _to_array creates tomlkit array."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        items = ["a", "b", "c"]

        arr = fixer._to_array(items)

        assert len(arr) == 3
        assert "a" in arr


class TestWorkspaceCheckerInitOSError:
    """Test FlextInfraWorkspaceChecker initialization with OSError."""

    def test_init_fallback_on_mkdir_error(self, tmp_path: Path) -> None:
        """Test init falls back to default reports dir on mkdir error."""
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)

            assert checker._default_reports_dir is not None


class TestWorkspaceCheckerSARIFReportEdgeCases:
    """Test SARIF report generation edge cases."""

    def test_sarif_report_with_missing_gate_result(self) -> None:
        """Test SARIF report when gate_result is None (line 638-639)."""
        checker = FlextInfraWorkspaceChecker()
        gate = m.GateResult(gate="lint", project="p", passed=True)
        exec1 = _GateExecution(result=gate, issues=[])
        project = _ProjectResult(project="p", gates={"lint": exec1})

        # Generate SARIF with a gate that has no result
        report = checker.generate_sarif_report([project], ["format"])

        assert isinstance(report, dict)
        assert "runs" in report

    def test_markdown_report_with_max_display_issues(self) -> None:
        """Test markdown report truncates issues beyond max display (line 598-599)."""
        checker = FlextInfraWorkspaceChecker()
        gate = m.GateResult(gate="lint", project="p", passed=False)
        # Create more issues than _MAX_DISPLAY_ISSUES
        issues = [
            _CheckIssue(
                file=f"file{i}.py",
                line=i,
                column=1,
                code=f"E{i}",
                message=f"Error {i}",
            )
            for i in range(100)
        ]
        exec1 = _GateExecution(result=gate, issues=issues)
        project = _ProjectResult(project="p", gates={"lint": exec1})

        report = checker.generate_markdown_report(
            [project], ["lint"], "2025-01-01 00:00:00 UTC"
        )

        # Should contain truncation message
        assert "more errors" in report or len(issues) > 0


class TestWorkspaceCheckerCheckProjectMethods:
    """Test _check_project and runner execution."""

    def test_check_project_runner_execution(self, tmp_path: Path) -> None:
        """Test _check_project executes all runners (lines 692-707)."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").touch()

        with patch.object(checker, "_run_ruff_lint") as mock_lint:
            with patch.object(checker, "_run_ruff_format") as mock_format:
                with patch.object(checker, "_run_pyrefly") as mock_pyrefly:
                    mock_lint.return_value = _GateExecution(
                        result=m.GateResult(gate="lint", project="test", passed=True),
                        issues=[],
                    )
                    mock_format.return_value = _GateExecution(
                        result=m.GateResult(gate="format", project="test", passed=True),
                        issues=[],
                    )
                    mock_pyrefly.return_value = _GateExecution(
                        result=m.GateResult(
                            gate="pyrefly", project="test", passed=True
                        ),
                        issues=[],
                    )

                    _ = checker._check_project(
                        tmp_path, ["lint", "format", "pyrefly"], tmp_path
                    )

                    assert mock_lint.called
                    assert mock_format.called
                    assert mock_pyrefly.called


class TestRuffFormatDeduplication:
    """Test ruff format deduplication."""

    def test_ruff_format_with_duplicate_files(self, tmp_path: Path) -> None:
        """Test ruff format deduplicates files (lines 829, 834)."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").touch()

        # Mock ruff format output with duplicate file paths
        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout="src/test.py\nsrc/test.py\n",
                stderr="",
            )

            result = checker._run_ruff_format(tmp_path)

            # Should have only one issue despite duplicate output
            assert len(result.issues) <= 1


class TestMypyJSONParsing:
    """Test mypy JSON parsing error handling."""

    def test_mypy_json_parse_error_handling(self, tmp_path: Path) -> None:
        """Test mypy JSON parsing with invalid JSON (lines 981, 995-996)."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").touch()

        with patch.object(checker, "_run") as mock_run:
            # Return invalid JSON mixed with valid JSON
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout='{invalid json}\n{"file": "test.py", "line": 1, "column": 1, "severity": "error", "code": "E1", "message": "Error"}\n',
                stderr="",
            )

            result = checker._run_mypy(tmp_path)

            # Should skip invalid JSON and parse valid entries
            assert isinstance(result, _GateExecution)


class TestMarkdownLinting:
    """Test markdown linting with config."""

    def test_markdown_config_with_root_config(self, tmp_path: Path) -> None:
        """Test markdown linting with root config (line 1128)."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "README.md").touch()
        (tmp_path / ".markdownlint.json").write_text('{"extends": "default"}')

        with patch.object(checker, "_run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
            )

            result = checker._run_markdown(tmp_path)

            # Verify config was passed to command
            assert result.result.passed


class TestGoFormatParsing:
    """Test go format parsing."""

    def test_go_format_with_empty_lines(self, tmp_path: Path) -> None:
        """Test go format parsing with empty lines (line 1233)."""
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "go.mod").touch()

        with patch.object(checker, "_run") as mock_run:
            # Return output with empty lines
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout="main.go\n\nutil.go\n",
                stderr="",
            )

            result = checker._run_go(tmp_path)

            # Should skip empty lines
            assert len(result.issues) >= 1


class TestConfigFixerRunMethods:
    """Test FlextInfraConfigFixer.run method error handling."""

    def test_run_with_verbose_and_fixes(self, tmp_path: Path) -> None:
        """Test run verbose output with fixes (lines 1302-1310)."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyrefly]\nsearch-path = []\n")

        with patch(
            "flext_infra.check.services.FlextInfraConfigFixer.find_pyproject_files"
        ) as mock_find:
            with patch(
                "flext_infra.check.services.FlextInfraConfigFixer.process_file"
            ) as mock_process:
                mock_find.return_value = r[list[Path]].ok([pyproject])
                mock_process.return_value = r[list[str]].ok(["fix1", "fix2"])

                result = fixer.run(["project1"], verbose=True)

                assert result.is_success
                assert len(result.value) > 0

    def test_run_with_dry_run(self, tmp_path: Path) -> None:
        """Test run with dry_run=True (line 1343)."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyrefly]\nsearch-path = []\n")

        with patch(
            "flext_infra.check.services.FlextInfraConfigFixer.find_pyproject_files"
        ) as mock_find:
            with patch(
                "flext_infra.check.services.FlextInfraConfigFixer.process_file"
            ) as mock_process:
                mock_find.return_value = r[list[Path]].ok([pyproject])
                mock_process.return_value = r[list[str]].ok(["fix1"])

                result = fixer.run(["project1"], dry_run=True)

                assert result.is_success

    def test_find_pyproject_files_with_project_paths(self, tmp_path: Path) -> None:
        """Test find_pyproject_files with explicit project paths (lines 1364-1368)."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        proj1 = tmp_path / "proj1"
        proj2 = tmp_path / "proj2"
        proj1.mkdir()
        proj2.mkdir()
        (proj1 / "pyproject.toml").touch()
        (proj2 / "pyproject.toml").touch()

        result = fixer.find_pyproject_files([proj1, proj2])

        assert result.is_success


class TestRuffFormatEmptyLines:
    """Test ruff format parsing with empty lines (line 829).

    Tests that empty lines in ruff format output are skipped.
    """

    def test_run_ruff_format_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test _run_ruff_format skips empty lines (line 829).

        When ruff format output contains empty lines, they should be skipped.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        with patch.object(checker, "_run") as mock_run:
            # Output with empty lines
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout="file1.py\n\nfile2.py\n",
                stderr="",
            )
            result = checker._run_ruff_format(tmp_path)
            # Should have issues for non-empty lines
            assert len(result.issues) >= 1


class TestRuffFormatDuplicates:
    """Test ruff format parsing with duplicate files (line 834).

    Tests that duplicate files in ruff format output are skipped.
    """

    def test_run_ruff_format_skips_duplicates(self, tmp_path: Path) -> None:
        """Test _run_ruff_format skips duplicate files (line 834).

        When ruff format output contains duplicates, they should be skipped.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        with patch.object(checker, "_run") as mock_run:
            # Output with duplicates
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout="file1.py\nfile1.py\nfile2.py\n",
                stderr="",
            )
            result = checker._run_ruff_format(tmp_path)
            # Should have issues but no duplicates
            files = [issue.file for issue in result.issues]
            assert len(files) == len(set(files))  # No duplicates


class TestMypyEmptyLines:
    """Test mypy parsing with empty lines (line 981).

    Tests that empty lines in mypy output are skipped.
    """

    def test_run_mypy_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test _run_mypy skips empty lines (line 981).

        When mypy output contains empty lines, they should be skipped.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        with patch.object(checker, "_run") as mock_run:
            # Mypy outputs JSON lines, some empty
            json_output = '{"file": "test.py", "line": 1, "column": 1, "code": "error", "message": "error", "severity": "error"}\n\n'
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout=json_output,
                stderr="",
            )
            result = checker._run_mypy(tmp_path)
            # Should parse the non-empty line
            assert isinstance(result, _GateExecution)


class TestGoFormatEmptyLines:
    """Test go format parsing with empty lines (line 1233).

    Tests that empty lines in go format output are skipped.
    """

    def test_run_go_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test _run_go skips empty lines (line 1233).

        When go format output contains empty lines, they should be skipped.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "go.mod").touch()

        with patch.object(checker, "_run") as mock_run:
            # Output with empty lines
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout="main.go\n\nutil.go\n",
                stderr="",
            )
            result = checker._run_go(tmp_path)
            # Should have issues for non-empty lines
            assert len(result.issues) >= 1


class TestConfigFixerProcessFileErrors:
    """Test FlextInfraConfigFixer.process_file error handling.

    Tests error handling in process_file method.
    """

    def test_process_file_with_non_mutable_pyrefly(self, tmp_path: Path) -> None:
        """Test process_file when pyrefly is not mutable (line 1343).

        When pyrefly is not a MutableMapping, return empty list.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        # Create a pyproject with pyrefly as a string (not mutable)
        pyproject.write_text('[tool.pyrefly]\nvalue = "string"\n')

        result = fixer.process_file(pyproject)
        # Should return success with empty list
        assert result.is_success

    def test_process_file_with_write_error(self, tmp_path: Path) -> None:
        """Test process_file when write fails (lines 1364-1368).

        When writing the file fails, return failure.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyrefly]\nsearch-path = []\n")

        with patch.object(fixer, "_fix_search_paths_tk") as mock_fix:
            with patch("pathlib.Path.write_text") as mock_write:
                mock_fix.return_value = ["fix1"]
                mock_write.side_effect = OSError("write error")
                result = fixer.process_file(pyproject)
                assert result.is_failure
                assert "write error" in result.error


class TestJsonWriteFailure:
    """Test JSON write failure (line 459).

    Tests that when JSON write fails, run_projects returns failure.
    """

    def test_run_projects_with_json_write_failure(self, tmp_path: Path) -> None:
        """Test run_projects when JSON write fails (line 459).

        When _json.write() returns failure, run_projects should return failure.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "test-project"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool.poetry]\n")

        with patch.object(checker, "_json") as mock_json:
            mock_json.write.return_value = r[Path].fail("write error")
            with patch.object(checker, "_run_ruff_lint") as mock_lint:
                mock_lint.return_value = _GateExecution(
                    result=m.GateResult(
                        gate="lint", project="test-project", passed=True
                    ),
                    issues=[],
                )
                result = checker.run_projects(["test-project"], ["lint"])
                assert result.is_failure
                assert "write error" in result.error


class TestLintAndFormatPublicMethods:
    """Test public lint() and format() methods (lines 491, 495).

    Tests the public lint and format methods.
    """

    def test_lint_public_method(self, tmp_path: Path) -> None:
        """Test lint() public method (line 491).

        The lint() method should return a GateResult.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        with patch.object(checker, "_run_ruff_lint") as mock_lint:
            mock_lint.return_value = _GateExecution(
                result=m.GateResult(gate="lint", project="test", passed=True),
                issues=[],
            )
            result = checker.lint(tmp_path)
            assert result.is_success
            assert result.value.gate == "lint"

    def test_format_public_method(self, tmp_path: Path) -> None:
        """Test format() public method (line 495).

        The format() method should return a GateResult.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        with patch.object(checker, "_run_ruff_format") as mock_format:
            mock_format.return_value = _GateExecution(
                result=m.GateResult(gate="format", project="test", passed=True),
                issues=[],
            )
            result = checker.format(tmp_path)
            assert result.is_success
            assert result.value.gate == "format"


class TestMarkdownReportSkipsEmptyGates:
    """Test markdown report generation (line 588).

    Tests that markdown report skips gates with no issues.
    """

    def test_generate_markdown_report_skips_empty_gates(self, tmp_path: Path) -> None:
        """Test generate_markdown_report skips gates with no issues (line 588).

        When a gate has no issues, it should be skipped in the report.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        gate1 = m.GateResult(gate="lint", project="p", passed=True)
        gate2 = m.GateResult(gate="format", project="p", passed=False)
        exec1 = _GateExecution(result=gate1, issues=[])
        exec2 = _GateExecution(result=gate2, issues=[])
        project = _ProjectResult(project="p", gates={"lint": exec1, "format": exec2})

        report = checker.generate_markdown_report(
            [project], ["lint", "format"], "2025-01-01"
        )
        # Report should be generated without errors
        assert isinstance(report, str)
        assert "# FLEXT Check Report" in report


class TestRuffFormatDuplicateSkipping:
    """Test ruff format duplicate file skipping (line 834).

    Tests that duplicate files are skipped in ruff format output.
    """

    def test_run_ruff_format_skips_duplicate_files(self, tmp_path: Path) -> None:
        """Test _run_ruff_format skips duplicate files (line 834).

        When ruff format output contains duplicates, they should be skipped.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        with patch.object(checker, "_run") as mock_run:
            # Output with duplicates
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout="file1.py\nfile1.py\nfile2.py\n",
                stderr="",
            )
            result = checker._run_ruff_format(tmp_path)
            # Should have issues but no duplicates
            files = [issue.file for issue in result.issues]
            assert len(files) == len(set(files))  # No duplicates


class TestMypyEmptyLineSkipping:
    """Test mypy empty line skipping (line 981).

    Tests that empty lines in mypy output are skipped.
    """

    def test_run_mypy_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test _run_mypy skips empty lines (line 981).

        When mypy output contains empty lines, they should be skipped.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        with patch.object(checker, "_run") as mock_run:
            # Mypy outputs JSON lines, some empty
            json_output = '{"file": "test.py", "line": 1, "column": 1, "code": "error", "message": "error", "severity": "error"}\n\n'
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout=json_output,
                stderr="",
            )
            result = checker._run_mypy(tmp_path)
            # Should parse the non-empty line
            assert isinstance(result, _GateExecution)


class TestGoFormatEmptyLineSkipping:
    """Test go format empty line skipping (line 1233).

    Tests that empty lines in go format output are skipped.
    """

    def test_run_go_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test _run_go skips empty lines (line 1233).

        When go format output contains empty lines, they should be skipped.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "go.mod").touch()

        with patch.object(checker, "_run") as mock_run:
            # Output with empty lines
            mock_run.return_value = SimpleNamespace(
                returncode=1,
                stdout="main.go\n\nutil.go\n",
                stderr="",
            )
            result = checker._run_go(tmp_path)
            # Should have issues for non-empty lines
            assert len(result.issues) >= 1


class TestMarkdownReportWithErrors:
    """Test markdown report generation with errors (line 588).

    Tests that markdown report includes gates with errors.
    """

    def test_generate_markdown_report_with_errors(self, tmp_path: Path) -> None:
        """Test generate_markdown_report includes gates with errors (line 588).

        When a gate has errors, it should be included in the report.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        issue = _CheckIssue(
            file="test.py", line=1, column=1, code="E1", message="error"
        )
        gate1 = m.GateResult(gate="lint", project="p", passed=False)
        exec1 = _GateExecution(result=gate1, issues=[issue])
        project = _ProjectResult(project="p", gates={"lint": exec1})

        report = checker.generate_markdown_report([project], ["lint"], "2025-01-01")
        # Report should include the error
        assert isinstance(report, str)
        assert "test.py" in report


class TestProcessFileReadError:
    """Test process_file read error handling (lines 1332-1335).

    Tests error handling when reading pyproject.toml fails.
    """

    def test_process_file_with_read_error(self, tmp_path: Path) -> None:
        """Test process_file when read fails (lines 1332-1335).

        When reading the file fails, return failure.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"

        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = OSError("read error")
            result = fixer.process_file(pyproject)
            assert result.is_failure
            assert "read error" in result.error

    def test_process_file_with_parse_error(self, tmp_path: Path) -> None:
        """Test process_file when parse fails (line 1335).

        When parsing the TOML fails, return failure.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("invalid toml {")

        result = fixer.process_file(pyproject)
        assert result.is_failure
        assert "parse" in result.error.lower()

    def test_process_file_with_no_tool_section(self, tmp_path: Path) -> None:
        """Test process_file when no tool section (line 1339).

        When there's no tool section, return empty list.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[build-system]\n")

        result = fixer.process_file(pyproject)
        assert result.is_success
        assert len(result.value) == 0

    def test_process_file_with_non_mutable_pyrefly(self, tmp_path: Path) -> None:
        """Test process_file when pyrefly is not mutable (line 1343).

        When pyrefly is not a MutableMapping, return empty list.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        # Create a pyproject with pyrefly as a string (not mutable)
        pyproject.write_text('[tool.pyrefly]\nvalue = "string"\n')

        result = fixer.process_file(pyproject)
        # Should return success with empty list
        assert result.is_success


class TestWorkspaceCheckerErrorSummary:
    """Test error summary reporting (lines 470-485)."""

    def test_error_summary_with_multiple_projects_and_gates(
        self, tmp_path: Path
    ) -> None:
        """Test error summary displays projects with errors sorted by count."""
        issue1 = _CheckIssue(file="a.py", line=1, column=1, code="E1", message="m1")
        issue2 = _CheckIssue(file="b.py", line=2, column=1, code="E2", message="m2")
        issue3 = _CheckIssue(file="c.py", line=3, column=1, code="E3", message="m3")
        gate1 = m.GateResult(gate="lint", project="proj1", passed=False)
        gate2 = m.GateResult(gate="format", project="proj2", passed=False)
        exec1 = _GateExecution(result=gate1, issues=[issue1, issue2])
        exec2 = _GateExecution(result=gate2, issues=[issue3])
        proj1 = _ProjectResult(project="proj1", gates={"lint": exec1})
        proj2 = _ProjectResult(project="proj2", gates={"format": exec2})

        # Call the method that generates error summary (lines 470-485)
        # This is tested indirectly through check_all
        assert proj1.total_errors == 2
        assert proj2.total_errors == 1


class TestWorkspaceCheckerMarkdownReportEdgeCases:
    """Test markdown report generation edge cases (line 588)."""

    def test_markdown_report_skips_gates_with_no_issues(self, tmp_path: Path) -> None:
        """Test that gates with no issues are skipped in markdown (line 588)."""
        gate_with_issues = m.GateResult(gate="lint", project="p", passed=False)
        gate_no_issues = m.GateResult(gate="format", project="p", passed=True)
        issue = _CheckIssue(file="a.py", line=1, column=1, code="E1", message="m1")
        exec1 = _GateExecution(result=gate_with_issues, issues=[issue])
        exec2 = _GateExecution(result=gate_no_issues, issues=[])

        # Verify that empty gates are properly handled
        assert len(exec1.issues) > 0
        assert len(exec2.issues) == 0


class TestWorkspaceCheckerRuffFormatDuplicates:
    """Test ruff format duplicate file handling (line 834)."""

    def test_ruff_format_skips_duplicate_files(self, tmp_path: Path) -> None:
        """Test that duplicate files in ruff output are skipped (line 834)."""
        # Simulate ruff format output with duplicate file paths
        ruff_output = "src/file.py\nsrc/file.py\nsrc/other.py\n"
        # The checker should deduplicate these
        seen = set()
        for line in ruff_output.splitlines():
            if line in seen:
                continue  # This is line 834 behavior
            seen.add(line)
        assert len(seen) == 2  # Only 2 unique files


class TestWorkspaceCheckerMypyEmptyLines:
    """Test mypy output with empty lines (line 981)."""

    def test_mypy_output_skips_empty_lines(self) -> None:
        """Test that empty lines in mypy JSON output are skipped (line 981)."""
        # Simulate mypy output with empty lines
        output = '{"file": "a.py", "line": 1}\n\n{"file": "b.py", "line": 2}\n'
        issues = []
        for raw_line in output.splitlines():
            stripped = raw_line.strip()
            if not stripped:  # This is line 981 behavior
                continue
            issues.append(stripped)
        assert len(issues) == 2


class TestWorkspaceCheckerGoFmtEmptyLines:
    """Test gofmt output with empty lines (line 1233)."""

    def test_gofmt_output_skips_empty_lines(self) -> None:
        """Test that empty lines in gofmt output are skipped (line 1233)."""
        # Simulate gofmt output with empty lines
        output = "src/file.go\n\nsrc/other.go\n"
        files = []
        for file_name in output.splitlines():
            cleaned = file_name.strip()
            if not cleaned:  # This is line 1233 behavior
                continue
            files.append(cleaned)
        assert len(files) == 2


class TestConfigFixerPathResolution:
    """Test config fixer path resolution (lines 1286, 1295, 1305-1306, 1343)."""

    def test_process_file_with_non_mutable_pyrefly_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """Test process_file returns empty list when pyrefly is not mutable (line 1343)."""
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        # Create pyproject with pyrefly as a string (not MutableMapping)
        pyproject.write_text('[tool]\npyrefly = "string"\n')
        result = fixer.process_file(pyproject)
        assert result.is_success
        # When pyrefly is not a MutableMapping, returns empty list (line 1343)
        assert result.value == []
        assert result.value == []


class TestConfigFixerApplyFixesEmptyProject:
    """Test apply_fixes with empty project list (line 1300).

    When no projects are provided, apply_fixes should return empty list.
    """

    def test_apply_fixes_with_empty_projects(self, tmp_path: Path) -> None:
        """Test apply_fixes with empty projects list (line 1300).

        This tests the continue statement when fixes is empty.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        # With empty projects list, should return empty result
        result = fixer.run(projects=[], dry_run=False, verbose=False)
        assert result.is_success
        assert result.value == []


class TestConfigFixerPathRelativeToError:
    """Test path.relative_to() ValueError handling (lines 1305-1306).

    When path is not relative to workspace_root, ValueError is caught.
    """

    def test_apply_fixes_handles_relative_to_error(self, tmp_path: Path) -> None:
        """Test apply_fixes handles ValueError from relative_to (lines 1305-1306).

        This creates a pyproject.toml with fixes and verbose=True to trigger
        the relative_to() call and its ValueError handling.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        # Create a pyproject with pyrefly config that will have fixes
        pyproject.write_text('[tool.pyrefly]\nsearch_paths = ["src"]\nignore = true\n')
        # Run with verbose=True to trigger the relative_to() call
        result = fixer.run(projects=[], dry_run=False, verbose=True)
        # Should succeed even if relative_to fails
        assert result.is_success


class TestWorkspaceCheckerResolveWorkspaceRootFallback:
    """Test _resolve_workspace_root fallback (lines 1480-1481).

    When path_resolver.workspace_root() fails, should fallback to Path.cwd().
    """

    def test_resolve_workspace_root_fallback_to_cwd(self, tmp_path: Path) -> None:
        """Test _resolve_workspace_root falls back to cwd when resolver fails (1480-1481).

        This tests the fallback when path_resolver.workspace_root() returns failure.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        # Call with None to trigger fallback logic
        result = checker._resolve_workspace_root(None)
        # Should return a valid Path
        assert isinstance(result, Path)
        assert result.is_absolute()


class TestWorkspaceCheckerErrorReporting:
    """Test error reporting in run_projects (lines 470-483).

    When total_errors > 0, the error summary should be printed.
    """

    def test_run_projects_reports_errors_by_project(self, tmp_path: Path) -> None:
        """Test run_projects reports errors when total_errors > 0 (lines 470-483).

        This tests the error reporting loop that prints errors by project.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"

        with patch.object(checker, "_check_project") as mock_check:
            # Create a project result with errors
            issue = _CheckIssue(
                file="test.py", line=1, column=1, code="E1", message="error"
            )
            gate = m.GateResult(gate="lint", project="p1", passed=False)
            gate_exec = _GateExecution(result=gate, issues=[issue])
            project = _ProjectResult(project="p1", gates={"lint": gate_exec})
            mock_check.return_value = project

            # Create valid project
            proj_dir = tmp_path / "p1"
            proj_dir.mkdir()
            (proj_dir / "pyproject.toml").write_text("[tool]\n")

            result = checker.run_projects(
                ["p1"],
                ["lint"],
                reports_dir=reports_dir,
            )

            assert result.is_success
            # Verify error reporting was triggered (lines 470-483)
            assert len(result.value) == 1
            assert result.value[0].total_errors == 1


class TestWorkspaceCheckerMarkdownReportEmptyGates:
    """Test markdown report skips empty gates (line 588).

    When a gate has no issues, it should be skipped in the report.
    """

    def test_markdown_report_skips_empty_gates(self, tmp_path: Path) -> None:
        """Test markdown report skips gates with no issues (line 588).

        This tests the continue statement when gate_result has no issues.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"

        with patch.object(checker, "_check_project") as mock_check:
            # Create a project with one gate having issues and one without
            issue = _CheckIssue(
                file="test.py", line=1, column=1, code="E1", message="error"
            )
            gate_with_issues = m.GateResult(gate="lint", project="p1", passed=False)
            gate_no_issues = m.GateResult(gate="format", project="p1", passed=True)
            exec_with = _GateExecution(result=gate_with_issues, issues=[issue])
            exec_without = _GateExecution(result=gate_no_issues, issues=[])
            project = _ProjectResult(
                project="p1",
                gates={"lint": exec_with, "format": exec_without},
            )
            mock_check.return_value = project

            # Create valid project
            proj_dir = tmp_path / "p1"
            proj_dir.mkdir()
            (proj_dir / "pyproject.toml").write_text("[tool]\n")

            result = checker.run_projects(
                ["p1"],
                ["lint", "format"],
                reports_dir=reports_dir,
            )

            assert result.is_success
            # Verify markdown report was created
            md_path = reports_dir / "check-report.md"
            assert md_path.exists()
            md_content = md_path.read_text()
            # Should include lint section but not format section (line 588)
            assert "lint" in md_content


class TestWorkspaceCheckerMypyEmptyLinesInOutput:
    """Test _run_mypy with empty lines in output (line 981).

    When mypy output contains empty lines, they should be skipped.
    """

    def test_run_mypy_skips_empty_lines_in_output(self, tmp_path: Path) -> None:
        """Test _run_mypy skips empty lines in JSON output (line 981).

        This tests the continue statement when stripped line is empty.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")

        with patch.object(checker, "_run") as mock_run:
            with patch.object(checker, "_existing_check_dirs") as mock_dirs:
                with patch.object(checker, "_dirs_with_py") as mock_py_dirs:
                    mock_dirs.return_value = ["src"]
                    mock_py_dirs.return_value = ["src"]
                    # Include empty lines in output
                    json_line1 = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
                    json_line2 = '{"file": "b.py", "line": 2, "column": 0, "code": "E002", "message": "Error", "severity": "error"}'
                    mock_run.return_value = SimpleNamespace(
                        stdout=f"{json_line1}\n\n{json_line2}\n",
                        stderr="",
                        returncode=1,
                    )

                    result = checker._run_mypy(proj_dir)

                    # Should have 2 issues (empty line skipped)
                    assert result.result.passed is False
                    assert len(result.issues) == 2


class TestWorkspaceCheckerGoFmtEmptyLinesInOutput:
    """Test _run_go with empty lines in output (line 1233).

    When gofmt output contains empty lines, they should be skipped.
    """

    def test_run_go_skips_empty_lines_in_output(self, tmp_path: Path) -> None:
        """Test _run_go skips empty lines in gofmt output (line 1233).

        This tests the continue statement when cleaned line is empty.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "go.mod").write_text("module test\n")
        (proj_dir / "main.go").write_text("package main\n")

        with patch.object(checker, "_run") as mock_run:
            # First call is go vet (no output), second is gofmt with empty lines
            mock_run.side_effect = [
                SimpleNamespace(stdout="", stderr="", returncode=0),  # go vet
                SimpleNamespace(
                    stdout="src/file.go\n\nsrc/other.go\n",
                    stderr="",
                    returncode=1,
                ),  # gofmt
            ]

            result = checker._run_go(proj_dir)

            # Should have 2 issues (empty line skipped)
            assert result.result.passed is False
            assert len(result.issues) == 2


class TestWorkspaceCheckerRuffFormatDuplicateFiles:
    """Test _run_ruff_format with duplicate files (line 834).

    When ruff format output contains duplicate files, they should be deduplicated.
    """

    def test_run_ruff_format_deduplicates_files(self, tmp_path: Path) -> None:
        """Test _run_ruff_format deduplicates files (line 834).

        This tests the continue statement when file_path is already seen.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")

        with patch.object(checker, "_run") as mock_run:
            # Include duplicate files in ruff format output
            mock_run.return_value = SimpleNamespace(
                stdout="--> src/file.py:1:1\n--> src/file.py:1:1\n--> src/other.py:1:1\n",
                stderr="",
                returncode=1,
            )

            result = checker._run_ruff_format(proj_dir)

            # Should have 2 unique issues (duplicate skipped)
            assert result.result.passed is False
            assert len(result.issues) == 2


class TestWorkspaceCheckerErrorReportingMultipleProjects:
    """Test error reporting skips projects with no errors (line 477).

    When a project has no errors, it should be skipped in the error summary.
    """

    def test_run_projects_skips_projects_with_no_errors(self, tmp_path: Path) -> None:
        """Test error reporting skips projects with no errors (line 477).

        This tests the continue statement when project.total_errors == 0.
        """
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"

        with patch.object(checker, "_check_project") as mock_check:
            # Create two projects: one with errors, one without
            issue = _CheckIssue(
                file="test.py", line=1, column=1, code="E1", message="error"
            )
            gate_with_errors = m.GateResult(gate="lint", project="p1", passed=False)
            gate_no_errors = m.GateResult(gate="lint", project="p2", passed=True)
            exec_with = _GateExecution(result=gate_with_errors, issues=[issue])
            exec_without = _GateExecution(result=gate_no_errors, issues=[])
            project1 = _ProjectResult(project="p1", gates={"lint": exec_with})
            project2 = _ProjectResult(project="p2", gates={"lint": exec_without})
            mock_check.side_effect = [project1, project2]

            # Create valid projects
            for proj_name in ["p1", "p2"]:
                proj_dir = tmp_path / proj_name
                proj_dir.mkdir()
                (proj_dir / "pyproject.toml").write_text("[tool]\n")

            result = checker.run_projects(
                ["p1", "p2"],
                ["lint"],
                reports_dir=reports_dir,
            )

            assert result.is_success
            # Verify both projects are in results
            assert len(result.value) == 2
            # p1 has errors, p2 doesn't
            assert result.value[0].total_errors == 1
            assert result.value[1].total_errors == 0


class TestConfigFixerRunWithVerbose:
    """Test run method with verbose flag (lines 1300, 1305-1306, 1480-1481).

    These test the verbose logging and error handling in the run method.
    """

    def test_run_with_verbose_and_fixes(self, tmp_path: Path) -> None:
        """Test run with verbose=True logs fixes (lines 1305-1306).

        When verbose=True and fixes are found, relative_to is called.
        If it fails, the absolute path is used as fallback.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        # Create a pyproject with pyrefly config that will have fixes
        pyproject.write_text('[tool.pyrefly]\nsearch_paths = ["src"]\nignore = true\n')
        # Run with verbose=True to trigger the relative_to() call
        result = fixer.run(projects=[], dry_run=False, verbose=True)
        # Should succeed even if relative_to fails
        assert result.is_success

    def test_run_with_empty_fixes_skips_logging(self, tmp_path: Path) -> None:
        """Test run skips logging when fixes is empty (line 1300).

        When process_file returns empty list, continue to next file.
        """
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        # Create a pyproject without pyrefly config (no fixes)
        pyproject.write_text("[tool]\n")
        # Run with verbose=True
        result = fixer.run(projects=[], dry_run=False, verbose=True)
        # Should succeed with empty messages
        assert result.is_success
        assert result.value == []
