"""Tests for workspace checker error reporting and integration-level scenarios.

Uses monkeypatch to inject controlled behavior instead of unittest.mock.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_infra import m
from flext_infra.check.services import (
    FlextInfraWorkspaceChecker,
    _CheckIssue,
    _GateExecution,
    _ProjectResult,
)
from flext_tests import tm

from ._stubs import make_gate_exec, make_issue


class TestErrorReporting:
    """Test error reporting in run_projects."""

    def test_reports_errors_by_project(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        issue = make_issue(file="test.py")
        gate_exec = make_gate_exec(issues=[issue])
        project = _ProjectResult(project="p1", gates={"lint": gate_exec})

        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")

        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that(len(result.value), eq=1)
        tm.that(result.value[0].total_errors, eq=1)

    def test_skips_projects_with_no_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        issue = make_issue(file="test.py")
        exec_with = make_gate_exec(issues=[issue])
        exec_without = make_gate_exec(issues=[])
        project1 = _ProjectResult(project="p1", gates={"lint": exec_with})
        project2 = _ProjectResult(project="p2", gates={"lint": exec_without})
        call_idx = [0]
        projects = [project1, project2]

        def _fake_check(*_a: object, **_kw: object) -> _ProjectResult:
            idx = call_idx[0]
            call_idx[0] += 1
            return projects[idx]

        monkeypatch.setattr(checker, "_check_project", _fake_check)
        for name in ["p1", "p2"]:
            d = tmp_path / name
            d.mkdir()
            (d / "pyproject.toml").write_text("[tool]\n")

        result = checker.run_projects(["p1", "p2"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that(len(result.value), eq=2)
        tm.that(result.value[0].total_errors, eq=1)
        tm.that(result.value[1].total_errors, eq=0)


class TestMarkdownReportEmptyGates:
    """Test markdown report skips empty gates in run_projects."""

    def test_skips_empty_gates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        issue = make_issue(file="test.py")
        exec_with = make_gate_exec(issues=[issue])
        exec_without = make_gate_exec(issues=[])
        project = _ProjectResult(
            project="p1",
            gates={"lint": exec_with, "format": exec_without},
        )
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")

        result = checker.run_projects(
            ["p1"],
            ["lint", "format"],
            reports_dir=reports_dir,
        )
        tm.ok(result)
        md_path = reports_dir / "check-report.md"
        assert md_path.exists()
        tm.that(md_path.read_text(), contains="lint")


class TestMypyEmptyLinesInOutput:
    """Test _run_mypy with empty lines in output."""

    def test_skips_empty_lines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        line1 = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
        line2 = '{"file": "b.py", "line": 2, "column": 0, "code": "E002", "message": "Error", "severity": "error"}'

        def _stub_run(_cmd: list[str], _cwd: Path, **_kw: object) -> SimpleNamespace:
            return SimpleNamespace(
                stdout=f"{line1}\n\n{line2}\n", stderr="", returncode=1
            )

        monkeypatch.setattr(checker, "_run", _stub_run)
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        monkeypatch.setattr(
            checker, "_dirs_with_py", staticmethod(lambda _r, _d: ["src"])
        )
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)


class TestGoFmtEmptyLinesInOutput:
    """Test _run_go with empty lines in output."""

    def test_skips_empty_lines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "go.mod").write_text("module test\n")
        (proj_dir / "main.go").write_text("package main\n")
        call_idx = [0]
        results = [
            SimpleNamespace(stdout="", stderr="", returncode=0),
            SimpleNamespace(
                stdout="src/file.go\n\nsrc/other.go\n", stderr="", returncode=1
            ),
        ]

        def _stub_run(_cmd: list[str], _cwd: Path, **_kw: object) -> SimpleNamespace:
            idx = call_idx[0]
            call_idx[0] += 1
            return results[idx] if idx < len(results) else results[-1]

        monkeypatch.setattr(checker, "_run", _stub_run)
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)


class TestRuffFormatDuplicateFiles:
    """Test _run_ruff_format with duplicate files."""

    def test_deduplicates_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")

        def _stub_run(_cmd: list[str], _cwd: Path, **_kw: object) -> SimpleNamespace:
            return SimpleNamespace(
                stdout="--> src/file.py:1:1\n--> src/file.py:1:1\n--> src/other.py:1:1\n",
                stderr="",
                returncode=1,
            )

        monkeypatch.setattr(checker, "_run", _stub_run)
        result = checker._run_ruff_format(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)
