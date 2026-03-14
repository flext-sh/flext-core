"""Tests for workspace checker run_projects and run methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from flext_infra import m
from flext_infra.check.services import (
    CheckIssue,
    FlextInfraWorkspaceChecker,
    GateExecution,
    ProjectResult,
)
from flext_tests import tm

CheckProjectStub = Callable[[Path, list[str], Path], ProjectResult]


def _make_gate_exec(
    gate: str = "lint",
    project: str = "p",
    *,
    passed: bool = True,
    issues: list[CheckIssue] | None = None,
) -> GateExecution:
    """Helper to create a _GateExecution."""
    return GateExecution(
        result=m.Infra.Check.GateResult(
            gate=gate, project=project, passed=passed, errors=[], duration=0.0
        ),
        issues=issues or [],
        raw_output="",
    )


def _setup_project(tmp_path: Path, name: str) -> Path:
    """Create a minimal project directory with pyproject.toml."""
    proj_dir = tmp_path / name
    proj_dir.mkdir()
    (proj_dir / "pyproject.toml").write_text("[tool]\n")
    return proj_dir


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
        project = ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec(passed=False)},
        )
        monkeypatch.setattr(checker, "_check_project", _check_project_stub(project))
        _setup_project(tmp_path, "p1")
        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that((reports_dir / "check-report.md").exists(), eq=True)

    def test_creates_sarif_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        project = ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec(passed=True)},
        )
        monkeypatch.setattr(checker, "_check_project", _check_project_stub(project))
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

        def _fake_check(
            _project_dir: Path,
            _gates: list[str],
            _reports_dir: Path,
        ) -> ProjectResult:
            del _project_dir, _gates, _reports_dir
            call_count[0] += 1
            return ProjectResult(
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
        issue = CheckIssue(
            file="test.py",
            line=1,
            column=1,
            code="E1",
            message="error",
            severity="error",
        )
        gate_exec = _make_gate_exec(passed=True, issues=[issue])
        project = ProjectResult(project="p1", gates={"lint": gate_exec})
        monkeypatch.setattr(checker, "_check_project", _check_project_stub(project))
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
        issue = CheckIssue(
            file="test.py",
            line=1,
            column=1,
            code="E1",
            message="error",
            severity="error",
        )
        exec_with = _make_gate_exec(passed=True, issues=[issue])
        exec_without = _make_gate_exec(passed=True)
        project1 = ProjectResult(project="p1", gates={"lint": exec_with})
        project2 = ProjectResult(project="p2", gates={"lint": exec_without})
        monkeypatch.setattr(
            checker,
            "_check_project",
            _iter_check_project_stub([project1, project2]),
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
        project = ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec(passed=True)},
        )
        monkeypatch.setattr(checker, "_check_project", _check_project_stub(project))
        result = checker.run("p1", ["lint"])
        tm.ok(result)
        tm.that(len(result.value), eq=1)
