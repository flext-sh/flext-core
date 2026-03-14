"""Tests for workspace checker gate runners — mypy and pyright.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm

from ...models import m


def _existing_dirs(_project_dir: Path) -> list[str]:
    del _project_dir
    return ["src"]


def _no_python_dirs(_project_dir: Path, _dirs: list[str]) -> list[str]:
    del _project_dir, _dirs
    return []


def _src_python_dirs(_project_dir: Path, _dirs: list[str]) -> list[str]:
    del _project_dir, _dirs
    return ["src"]


def _run_result(stdout: str, returncode: int) -> m.Infra.Core.CommandOutput:
    return m.Infra.Core.CommandOutput(stdout=stdout, stderr="", exit_code=returncode)


class TestWorkspaceCheckerRunMypy:
    """Test FlextInfraWorkspaceChecker._run_mypy method."""

    def test_run_mypy_no_python_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", _no_python_dirs)
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_mypy_with_json_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        json_line = (
            '{"file": "a.py", "line": 1, "column": 0,'
            ' "code": "E001", "message": "Error", "severity": "error"}'
        )

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return _run_result(json_line, 1)

        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", _src_python_dirs)
        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)


class TestWorkspaceCheckerRunPyright:
    """Test FlextInfraWorkspaceChecker._run_pyright method."""

    def test_run_pyright_no_python_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", _no_python_dirs)
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_pyright_with_json_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        json_output = (
            '{"generalDiagnostics": [{"file": "a.py",'
            ' "range": {"start": {"line": 0, "character": 0}},'
            ' "rule": "E001", "message": "Error", "severity": "error"}]}'
        )

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return _run_result(json_output, 1)

        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", _src_python_dirs)
        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_pyright_with_invalid_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return _run_result("invalid json", 1)

        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", _src_python_dirs)
        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=False)
