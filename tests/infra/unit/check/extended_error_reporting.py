"""Tests for workspace checker error reporting and integration scenarios.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from flext_infra.check.services import (
    FlextInfraWorkspaceChecker,
    ProjectResult,
)
from flext_tests import tm

from ... import h
from ...models import m
from ._stubs import make_gate_exec, make_issue

CheckProjectStub = Callable[[Path, list[str], Path], ProjectResult]
RunStub = Callable[
    [list[str], Path, int, dict[str, str] | None], m.Infra.Core.CommandOutput
]


def _check_project_stub(project: ProjectResult) -> CheckProjectStub:
    def _fake_check(
        _project_dir: Path, _gates: list[str], _reports_dir: Path
    ) -> ProjectResult:
        return project

    return _fake_check


def _iter_check_project_stub(projects: list[ProjectResult]) -> CheckProjectStub:
    project_iter = iter(projects)

    def _fake_check(
        _project_dir: Path, _gates: list[str], _reports_dir: Path
    ) -> ProjectResult:
        return next(project_iter)

    return _fake_check


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
        project = ProjectResult(project="p1", gates={"lint": gate_exec})

        monkeypatch.setattr(checker, "_check_project", _check_project_stub(project))
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
        project1 = ProjectResult(project="p1", gates={"lint": exec_with})
        project2 = ProjectResult(project="p2", gates={"lint": exec_without})
        monkeypatch.setattr(
            checker,
            "_check_project",
            _iter_check_project_stub([project1, project2]),
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
        project = ProjectResult(
            project="p1", gates={"lint": exec_with, "format": exec_without}
        )
        monkeypatch.setattr(checker, "_check_project", _check_project_stub(project))
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

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return m.Infra.Core.CommandOutput(
                stdout=f"{line1}\n\n{line2}\n",
                stderr="",
                exit_code=1,
            )

        def _fake_existing_dirs(_project_dir: Path) -> list[str]:
            del _project_dir
            return ["src"]

        def _fake_dirs_with_py(_project_dir: Path, _dirs: list[str]) -> list[str]:
            del _project_dir, _dirs
            return ["src"]

        monkeypatch.setattr(checker, "_run", _fake_run)
        monkeypatch.setattr(checker, "_existing_check_dirs", _fake_existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(_fake_dirs_with_py))
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
            m.Infra.Core.CommandOutput(stdout="", stderr="", exit_code=0),
            m.Infra.Core.CommandOutput(
                stdout="src/file.go\n\nsrc/other.go\n",
                stderr="",
                exit_code=1,
            ),
        ]

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            index = min(call_idx[0], len(results) - 1)
            call_idx[0] += 1
            return results[index]

        monkeypatch.setattr(checker, "_run", _fake_run)
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

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return m.Infra.Core.CommandOutput(
                stdout="--> src/file.py:1:1\n--> src/file.py:1:1\n--> src/other.py:1:1\n",
                stderr="",
                exit_code=1,
            )

        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_ruff_format(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)
