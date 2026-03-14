"""Tests for workspace checker public methods and runner execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.check.services import (
    CheckIssue,
    FlextInfraWorkspaceChecker,
    GateExecution,
)
from flext_tests import tm

from ...models import m


def _make_gate_exec(
    gate: str = "lint",
    project: str = "p",
    *,
    passed: bool = True,
    issues: list[CheckIssue] | None = None,
) -> GateExecution:
    return GateExecution(
        result=m.Infra.Check.GateResult(
            gate=gate, project=project, passed=passed, errors=[], duration=0.0
        ),
        issues=issues or [],
        raw_output="",
    )


class TestLintAndFormatPublicMethods:
    def test_lint_public_method(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        def _fake_lint(_project_dir: Path) -> GateExecution:
            del _project_dir
            return _make_gate_exec(gate="lint", passed=True)

        monkeypatch.setattr(
            checker,
            "_run_ruff_lint",
            _fake_lint,
        )
        result = checker.lint(tmp_path)
        tm.ok(result)
        tm.that(result.value.gate, eq="lint")

    def test_format_public_method(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        def _fake_format(_project_dir: Path) -> GateExecution:
            del _project_dir
            return _make_gate_exec(gate="format", passed=True)

        monkeypatch.setattr(
            checker,
            "_run_ruff_format",
            _fake_format,
        )
        result = checker.format(tmp_path)
        tm.ok(result)
        tm.that(result.value.gate, eq="format")


class TestCheckProjectRunners:
    def test_check_project_runner_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").touch()
        call_log: list[str] = []

        def _fake_lint(_project_dir: Path) -> GateExecution:
            del _project_dir
            call_log.append("lint")
            return _make_gate_exec(gate="lint", passed=True)

        def _fake_format(_project_dir: Path) -> GateExecution:
            del _project_dir
            call_log.append("format")
            return _make_gate_exec(gate="format", passed=True)

        def _fake_pyrefly(
            _project_dir: Path, _reports_dir: Path | None = None
        ) -> GateExecution:
            del _project_dir, _reports_dir
            call_log.append("pyrefly")
            return _make_gate_exec(gate="pyrefly", passed=True)

        monkeypatch.setattr(checker, "_run_ruff_lint", _fake_lint)
        monkeypatch.setattr(checker, "_run_ruff_format", _fake_format)
        monkeypatch.setattr(checker, "_run_pyrefly", _fake_pyrefly)
        _ = checker._check_project(tmp_path, ["lint", "format", "pyrefly"], tmp_path)
        tm.that("lint" in call_log, eq=True)
        tm.that("format" in call_log, eq=True)
        tm.that("pyrefly" in call_log, eq=True)
