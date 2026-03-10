"""Tests for workspace checker gate runners — mypy and pyright.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm


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
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(checker, "_dirs_with_py", lambda *_a: [])
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
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(checker, "_dirs_with_py", lambda *_a: ["src"])
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout=json_line,
                stderr="",
                returncode=1,
            ),
        )
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
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(checker, "_dirs_with_py", lambda *_a: [])
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
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(checker, "_dirs_with_py", lambda *_a: ["src"])
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout=json_output,
                stderr="",
                returncode=1,
            ),
        )
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
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda *_a: ["src"])
        monkeypatch.setattr(checker, "_dirs_with_py", lambda *_a: ["src"])
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="invalid json",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=False)
