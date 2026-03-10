"""Tests for workspace checker error reporting and integration scenarios.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.check.services import (
    FlextInfraWorkspaceChecker,
    _ProjectResult,
)
from flext_tests import tm
from tests.infra import h

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
        h.mk_project(tmp_path, "p1")

        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that(len(result.value), eq=1)
        tm.that(result.value[0].total_errors, eq=1)

    def test_skips_projects_with_no_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        issue = make_issue(file="test.py")
        exec_with = make_gate_exec(issues=[issue])
        exec_without = make_gate_exec(issues=[])
        project1 = _ProjectResult(project="p1", gates={"lint": exec_with})
        project2 = _ProjectResult(project="p2", gates={"lint": exec_without})
        project_iter = iter([project1, project2])
        monkeypatch.setattr(
            checker, "_check_project", lambda *_a, **_kw: next(project_iter)
        )
        h.mk_project(tmp_path, "p1")
        h.mk_project(tmp_path, "p2")
        result = checker.run_projects(["p1", "p2"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that(len(result.value), eq=2)
        tm.that(result.value[0].total_errors, eq=1)
        tm.that(result.value[1].total_errors, eq=0)


class TestMarkdownReportEmptyGates:
    """Test markdown report skips empty gates in run_projects."""

    def test_skips_empty_gates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        issue = make_issue(file="test.py")
        exec_with = make_gate_exec(issues=[issue])
        exec_without = make_gate_exec(issues=[])
        project = _ProjectResult(
            project="p1", gates={"lint": exec_with, "format": exec_without}
        )
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        h.mk_project(tmp_path, "p1")
        result = checker.run_projects(
            ["p1"], ["lint", "format"], reports_dir=reports_dir
        )
        tm.ok(result)
        md_path = reports_dir / "check-report.md"
        tm.that(md_path.exists(), eq=True)
        tm.that(md_path.read_text(), contains="lint")


class TestMypyEmptyLinesInOutput:
    """Test _run_mypy with empty lines in output."""

    def test_skips_empty_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        line1 = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
        line2 = '{"file": "b.py", "line": 2, "column": 0, "code": "E002", "message": "Error", "severity": "error"}'

        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: h.stub_run(stdout=f"{line1}\n\n{line2}\n", returncode=1),
        )
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "go.mod").write_text("module test\n")
        (proj_dir / "main.go").write_text("package main\n")
        call_idx = [0]
        results = [
            h.stub_run(),
            h.stub_run(stdout="src/file.go\n\nsrc/other.go\n", returncode=1),
        ]

        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: results[
                min(
                    call_idx.__setitem__(0, call_idx[0] + 1) or call_idx[0] - 1,
                    len(results) - 1,
                )
            ],
        )
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)


class TestRuffFormatDuplicateFiles:
    """Test _run_ruff_format with duplicate files."""

    def test_deduplicates_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: h.stub_run(
                stdout="--> src/file.py:1:1\n--> src/file.py:1:1\n--> src/other.py:1:1\n",
                returncode=1,
            ),
        )
        result = checker._run_ruff_format(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)
