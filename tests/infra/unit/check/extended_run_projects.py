"""Tests for workspace checker run_projects and run methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
    *,
    passed: bool = True,
    issues: list[_CheckIssue] | None = None,
) -> _GateExecution:
    """Helper to create a _GateExecution."""
    return _GateExecution(
        result=m.Infra.Check.GateResult(gate=gate, project=project, passed=passed),
        issues=issues or [],
    )


def _setup_project(tmp_path: Path, name: str) -> Path:
    """Create a minimal project directory with pyproject.toml."""
    proj_dir = tmp_path / name
    proj_dir.mkdir()
    (proj_dir / "pyproject.toml").write_text("[tool]\n")
    return proj_dir


class TestRunProjectsValidation:
    """Test run_projects input validation and edge cases."""

    def test_invalid_gates(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["p1"], ["invalid_gate"], reports_dir=tmp_path / "reports"
        )
        tm.fail(result)

    def test_skips_missing_projects(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["nonexistent"], ["lint"], reports_dir=tmp_path / "reports"
        )
        tm.ok(result)
        tm.that(len(result.value), eq=0)


class TestRunProjectsReports:
    """Test run_projects report generation."""

    def test_creates_markdown_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        project = _ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec(passed=False)},
        )
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        _setup_project(tmp_path, "p1")
        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that((reports_dir / "check-report.md").exists(), eq=True)

    def test_creates_sarif_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        project = _ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec(passed=True)},
        )
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        _setup_project(tmp_path, "p1")
        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that((reports_dir / "check-report.sarif").exists(), eq=True)


class TestRunProjectsBehavior:
    """Test run_projects fail_fast and error reporting."""

    def test_fail_fast_stops_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        call_count = [0]

        def _fake_check(*_a: object, **_kw: object) -> _ProjectResult:
            call_count[0] += 1
            return _ProjectResult(
                project="p",
                gates={"lint": _make_gate_exec(passed=False)},
            )

        monkeypatch.setattr(checker, "_check_project", _fake_check)
        for name in ["p1", "p2", "p3"]:
            _setup_project(tmp_path, name)
        result = checker.run_projects(
            ["p1", "p2", "p3"],
            ["lint"],
            reports_dir=tmp_path / "reports",
            fail_fast=True,
        )
        tm.ok(result)
        tm.that(call_count[0], eq=1)

    def test_reports_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        issue = _CheckIssue(
            file="test.py", line=1, column=1, code="E1", message="error"
        )
        gate_exec = _make_gate_exec(passed=True, issues=[issue])
        project = _ProjectResult(project="p1", gates={"lint": gate_exec})
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        _setup_project(tmp_path, "p1")
        result = checker.run_projects(
            ["p1"], ["lint"], reports_dir=tmp_path / "reports"
        )
        tm.ok(result)
        tm.that(len(result.value), eq=1)
        tm.that(result.value[0].total_errors, eq=1)

    def test_multiple_with_mixed_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        issue = _CheckIssue(
            file="test.py", line=1, column=1, code="E1", message="error"
        )
        exec_with = _make_gate_exec(passed=True, issues=[issue])
        exec_without = _make_gate_exec(passed=True)
        project1 = _ProjectResult(project="p1", gates={"lint": exec_with})
        project2 = _ProjectResult(project="p2", gates={"lint": exec_without})
        results_iter = iter([project1, project2])
        monkeypatch.setattr(
            checker, "_check_project", lambda *_a, **_kw: next(results_iter)
        )
        for name in ["p1", "p2"]:
            _setup_project(tmp_path, name)
        result = checker.run_projects(
            ["p1", "p2"], ["lint"], reports_dir=tmp_path / "reports"
        )
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
        _setup_project(tmp_path, "p1")
        project = _ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec(passed=True)},
        )
        monkeypatch.setattr(checker, "_check_project", lambda *_a, **_kw: project)
        result = checker.run("p1", ["lint"])
        tm.ok(result)
        tm.that(len(result.value), eq=1)
