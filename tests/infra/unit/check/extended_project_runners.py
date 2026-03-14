"""Tests for workspace checker project runner execution and public methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r, t
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


class TestCheckProjectRunners:
    def test_check_project_runner_execution(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").touch()
        called: dict[str, bool] = {"lint": False, "format": False, "pyrefly": False}

        def _fake_lint(_p: Path) -> GateExecution:
            called["lint"] = True
            return _make_gate_exec(gate="lint")

        def _fake_format(_p: Path) -> GateExecution:
            called["format"] = True
            return _make_gate_exec(gate="format")

        def _fake_pyrefly(_p: Path, _r: Path | None = None) -> GateExecution:
            called["pyrefly"] = True
            return _make_gate_exec(gate="pyrefly")

        monkeypatch.setattr(checker, "_run_ruff_lint", _fake_lint)
        monkeypatch.setattr(checker, "_run_ruff_format", _fake_format)
        monkeypatch.setattr(checker, "_run_pyrefly", _fake_pyrefly)
        _ = checker._check_project(tmp_path, ["lint", "format", "pyrefly"], tmp_path)
        tm.that(called["lint"], eq=True)
        tm.that(called["format"], eq=True)
        tm.that(called["pyrefly"], eq=True)


class TestJsonWriteFailure:
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
            def write_json(self, *_a: t.Scalar, **_kw: t.Scalar) -> r[bool]:
                return r[bool].fail("write error")

        monkeypatch.setattr(checker, "_json", _FakeJson())

        def _fake_lint(_project_dir: Path) -> GateExecution:
            del _project_dir
            return _make_gate_exec("lint", "p", passed=True)

        monkeypatch.setattr(
            checker,
            "_run_ruff_lint",
            _fake_lint,
        )
        result = checker.run_projects(["test-project"], ["lint"])
        tm.fail(result, has="write error")


class TestLintAndFormatPublicMethods:
    def test_lint_public_method(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()

        def _fake_lint(_project_dir: Path) -> GateExecution:
            del _project_dir
            return _make_gate_exec("lint", "p", passed=True)

        monkeypatch.setattr(
            checker,
            "_run_ruff_lint",
            _fake_lint,
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

        def _fake_format(_project_dir: Path) -> GateExecution:
            del _project_dir
            return _make_gate_exec("format", "p", passed=True)

        monkeypatch.setattr(
            checker,
            "_run_ruff_format",
            _fake_format,
        )
        result = checker.format(tmp_path)
        tm.ok(result)
        tm.that(result.value.gate, eq="format")
