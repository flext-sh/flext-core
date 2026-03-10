"""Tests for workspace checker run_projects and run methods.

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
    gate: str,
    project: str,
    *,
    passed: bool,
    issues: list[_CheckIssue] | None = None,
) -> _GateExecution:
    """Helper to create a _GateExecution."""
    return _GateExecution(
        result=m.Infra.Check.GateResult(gate=gate, project=project, passed=passed),
        issues=issues or [],
    )


class TestWorkspaceCheckerRunProjects:
    """Test FlextInfraWorkspaceChecker.run_projects."""

    def test_run_projects_with_invalid_gates(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["p1"],
            ["invalid_gate"],
            reports_dir=tmp_path / "reports",
        )
        tm.fail(result)

    def test_run_projects_skips_missing_projects(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.run_projects(
            ["nonexistent"],
            ["lint"],
            reports_dir=tmp_path / "reports",
        )
        tm.ok(result)
        tm.that(len(result.value), eq=0)

    def test_run_projects_creates_markdown_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        project = _ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec("lint", "p", passed=False)},
        )
        monkeypatch.setattr(checker, "_check_project", lambda *_a: project)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")
        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that((reports_dir / "check-report.md").exists(), eq=True)

    def test_run_projects_creates_sarif_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        project = _ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec("lint", "p", passed=True)},
        )
        monkeypatch.setattr(checker, "_check_project", lambda *_a: project)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")
        result = checker.run_projects(["p1"], ["lint"], reports_dir=reports_dir)
        tm.ok(result)
        tm.that((reports_dir / "check-report.sarif").exists(), eq=True)

    def test_run_projects_with_fail_fast_stops_on_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        reports_dir = tmp_path / "reports"
        call_count = [0]

        def _fake_check(*_a: object) -> _ProjectResult:
            call_count[0] += 1
            return _ProjectResult(
                project="p",
                gates={"lint": _make_gate_exec("lint", "p", passed=False)},
            )

        monkeypatch.setattr(checker, "_check_project", _fake_check)
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
        tm.ok(result)
        tm.that(call_count[0], eq=1)


class TestWorkspaceCheckerRun:
    """Test FlextInfraWorkspaceChecker.run method."""

    def test_run_single_project_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool]\n")
        project = _ProjectResult(
            project="p1",
            gates={"lint": _make_gate_exec("lint", "p", passed=True)},
        )
        monkeypatch.setattr(checker, "_check_project", lambda *_a: project)
        result = checker.run("p1", ["lint"])
        tm.ok(result)
        tm.that(len(result.value), eq=1)


class TestWorkspaceCheckerCheckProjectMethods:
    """Test _check_project and runner execution."""

    def test_check_project_runner_execution(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").touch()
        call_log: list[str] = []

        def _fake_lint(*_a: object) -> _GateExecution:
            call_log.append("lint")
            return _make_gate_exec("lint", "p", passed=True)

        def _fake_format(*_a: object) -> _GateExecution:
            call_log.append("format")
            return _make_gate_exec("format", "p", passed=True)

        def _fake_pyrefly(*_a: object, **_kw: object) -> _GateExecution:
            call_log.append("pyrefly")
            return _make_gate_exec("pyrefly", "p", passed=True)

        monkeypatch.setattr(checker, "_run_ruff_lint", _fake_lint)
        monkeypatch.setattr(checker, "_run_ruff_format", _fake_format)
        monkeypatch.setattr(checker, "_run_pyrefly", _fake_pyrefly)
        _ = checker._check_project(tmp_path, ["lint", "format", "pyrefly"], tmp_path)
        tm.that("lint" in call_log, eq=True)
        tm.that("format" in call_log, eq=True)
        tm.that("pyrefly" in call_log, eq=True)


class TestJsonWriteFailure:
    """Test JSON write failure."""

    def test_run_projects_with_json_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "test-project"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool.poetry]\n")

        class _FakeJson:
            def write_json(self, *_a: object, **_kw: object) -> r[bool]:
                return r[bool].fail("write error")

        monkeypatch.setattr(checker, "_json", _FakeJson())
        monkeypatch.setattr(
            checker,
            "_run_ruff_lint",
            lambda *_a: _make_gate_exec("lint", "p", passed=True),
        )
        result = checker.run_projects(["test-project"], ["lint"])
        tm.fail(result, has="write error")


class TestLintAndFormatPublicMethods:
    """Test public lint() and format() methods."""

    def test_lint_public_method(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.setattr(
            checker,
            "_run_ruff_lint",
            lambda *_a: _make_gate_exec("lint", "p", passed=True),
        )
        result = checker.lint(tmp_path)
        tm.ok(result)
        tm.that(result.value.gate, eq="lint")

    def test_format_public_method(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.setattr(
            checker,
            "_run_ruff_format",
            lambda *_a: _make_gate_exec("format", "p", passed=True),
        )
        result = checker.format(tmp_path)
        tm.ok(result)
        tm.that(result.value.gate, eq="format")
