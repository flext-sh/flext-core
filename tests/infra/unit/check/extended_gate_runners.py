"""Tests for workspace checker gate runner methods — ruff, pyrefly, mypy.

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
    _GateExecution,
)
from flext_tests import tm


class TestWorkspaceCheckerRunRuffLint:
    """Test FlextInfraWorkspaceChecker._run_ruff_lint method."""

    def test_run_ruff_lint_with_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        json_output = '[{"filename": "a.py", "location": {"row": 1, "column": 0}, "code": "E001", "message": "Error"}]'
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout=json_output,
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_ruff_lint(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_ruff_lint_with_invalid_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="invalid json",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_ruff_lint(proj_dir)
        tm.that(result.result.passed, eq=False)


class TestWorkspaceCheckerRunRuffFormat:
    """Test FlextInfraWorkspaceChecker._run_ruff_format method."""

    def test_run_ruff_format_with_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="  --> a.py:1:1",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_ruff_format(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_ruff_format_with_simple_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="a.py",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_ruff_format(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)


class TestWorkspaceCheckerRunPyrefly:
    """Test FlextInfraWorkspaceChecker._run_pyrefly method."""

    def test_run_pyrefly_with_json_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text('{"errors": []}')
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="",
                stderr="",
                returncode=0,
            ),
        )
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=True)

    def test_run_pyrefly_with_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text(
            '{"errors": [{"path": "a.py", "line": 1, "column": 0, '
            '"name": "E001", "description": "Error", "severity": "error"}]}',
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_pyrefly_with_invalid_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text("invalid json")
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)

    def test_run_pyrefly_with_error_count_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="",
                stderr="Found 3 errors",
                returncode=1,
            ),
        )
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=3)

    def test_run_pyrefly_with_list_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text(
            '[{"path": "a.py", "line": 1, "column": 0, '
            '"name": "E001", "description": "Error", "severity": "error"}]',
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(len(result.issues), eq=1)
