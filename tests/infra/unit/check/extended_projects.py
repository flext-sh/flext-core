"""Tests for workspace checker project-level operations — run_projects, run, lint, format.

Uses monkeypatch to inject controlled results instead of unittest.mock.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import m
from flext_infra.check.services import (
    FlextInfraWorkspaceChecker,
    _CheckIssue,
    _GateExecution,
    _ProjectResult,
)
from flext_tests import tm


def _make_gate_exec(
    gate: str = "lint",
    project: str = "p",
    passed: bool = True,
    issues: list[_CheckIssue] | None = None,
) -> _GateExecution:
    """Create a _GateExecution with sensible defaults."""
    return _GateExecution(
        result=m.Infra.Check.GateResult(gate=gate, project=project, passed=passed),
        issues=issues or [],
    )


class TestRunProjects:
    """Test FlextInfraWorkspaceChecker.run_projects."""

    def test_run_projects_with_invalid_gates(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["p1"], ["invalid_gate"], reports_dir=tmp_path / "reports"
        )
        tm.fail(result)

    def test_run_projects_skips_missing_projects(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["nonexistent"], ["lint"], reports_dir=tmp_path / "reports"
        )
        tm.ok(result, eq=[])

    def test_run_projects_creates_markdown_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        gate_exec = _make_gate_exec(passed=False)
        project = _ProjectResult(project="p1", gates={"lint": gate_exec})
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")
        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that((reports_dir / "check-report.md").exists(), eq=True)

    def test_run_projects_creates_sarif_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        gate_exec = _make_gate_exec(passed=True)
        project = _ProjectResult(project="p1", gates={"lint": gate_exec})
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")
        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that((reports_dir / "check-report.sarif").exists(), eq=True)

    def test_run_projects_with_fail_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        gate_exec = _make_gate_exec(passed=False)
        project = _ProjectResult(project="p", gates={"lint": gate_exec})
        call_count = [0]

        def _fake_check(*_a: object, **_kw: object) -> _ProjectResult:
            call_count[0] += 1
            return project

        monkeypatch.setattr(checker, "_check_project", _fake_check)
        for name in ["p1", "p2", "p3"]:
            d = tmp_path / name
            d.mkdir()
            (d / "pyproject.toml").write_text("[tool]\n")
        result = checker.run_projects(
            ["p1", "p2", "p3"], ["lint"], reports_dir=reports_dir, fail_fast=True
        )
        tm.ok(result)
        tm.that(call_count[0], eq=1)

    def test_run_projects_reports_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        issue = _CheckIssue(
            file="test.py", line=1, column=1, code="E1", message="error"
        )
        gate_exec = _GateExecution(
            result=m.Infra.Check.GateResult(gate="lint", project="p", passed=True),
            issues=[issue],
        )
        project = _ProjectResult(project="p1", gates={"lint": gate_exec})
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")
        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that(len(result.value), eq=1)
        tm.that(result.value[0].total_errors, eq=1)

    def test_run_projects_multiple_with_mixed_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        issue = _CheckIssue(
            file="test.py", line=1, column=1, code="E1", message="error"
        )
        exec_with = _GateExecution(
            result=m.Infra.Check.GateResult(gate="lint", project="p", passed=True),
            issues=[issue],
        )
        exec_without = _make_gate_exec(passed=True)
        project1 = _ProjectResult(project="p1", gates={"lint": exec_with})
        project2 = _ProjectResult(project="p2", gates={"lint": exec_without})
        results_iter = iter([project1, project2])
        monkeypatch.setattr(
            checker, "_check_project", lambda *_a, **_kw: next(results_iter)
        )
        for name in ["p1", "p2"]:
            d = tmp_path / name
            d.mkdir()
            (d / "pyproject.toml").write_text("[tool]\n")
        result = checker.run_projects(["p1", "p2"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that(len(result.value), eq=2)
        tm.that(result.value[0].total_errors, eq=1)
        tm.that(result.value[1].total_errors, eq=0)


class TestRunSingleProject:
    """Test FlextInfraWorkspaceChecker.run method."""

    def test_run_single_project_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")
        gate_exec = _make_gate_exec(passed=True)
        project = _ProjectResult(project="p1", gates={"lint": gate_exec})
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        result = checker.run("p1", ["lint"])
        tm.ok(result)
        tm.that(len(result.value), eq=1)


class TestLintAndFormatPublicMethods:
    """Test public lint() and format() methods."""

    def test_lint_public_method(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()
        gate_exec = _make_gate_exec(gate="lint", passed=True)
        monkeypatch.setattr(checker, "_run_ruff_lint", lambda _p: gate_exec)
        result = checker.lint(tmp_path)
        tm.ok(result)
        tm.that(result.value.gate, eq="lint")

    def test_format_public_method(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()
        gate_exec = _make_gate_exec(gate="format", passed=True)
        monkeypatch.setattr(checker, "_run_ruff_format", lambda _p: gate_exec)
        result = checker.format(tmp_path)
        tm.ok(result)
        tm.that(result.value.gate, eq="format")


class TestCheckProjectRunners:
    """Test _check_project executes all runners."""

    def test_check_project_runner_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").touch()
        called: dict[str, bool] = {"lint": False, "format": False, "pyrefly": False}

        def _fake_lint(_p: Path) -> _GateExecution:
            called["lint"] = True
            return _make_gate_exec(gate="lint")

        def _fake_format(_p: Path) -> _GateExecution:
            called["format"] = True
            return _make_gate_exec(gate="format")

        def _fake_pyrefly(_p: Path, _r: Path | None = None) -> _GateExecution:
            called["pyrefly"] = True
            return _make_gate_exec(gate="pyrefly")

        monkeypatch.setattr(checker, "_run_ruff_lint", _fake_lint)
        monkeypatch.setattr(checker, "_run_ruff_format", _fake_format)
        monkeypatch.setattr(checker, "_run_pyrefly", _fake_pyrefly)
        _ = checker._check_project(tmp_path, ["lint", "format", "pyrefly"], tmp_path)
        tm.that(called["lint"], eq=True)
        tm.that(called["format"], eq=True)
        tm.that(called["pyrefly"], eq=True)
