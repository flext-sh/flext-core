"""Tests for workspace checker gate runners — mypy, pyright, bandit, markdown, go.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_core import r
from flext_infra import m
from flext_infra.check.services import (
    FlextInfraWorkspaceChecker,
    _GateExecution,
)
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
        json_line = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
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
        json_output = '{"generalDiagnostics": [{"file": "a.py", "range": {"start": {"line": 0, "character": 0}}, "rule": "E001", "message": "Error", "severity": "error"}]}'
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


class TestWorkspaceCheckerRunBandit:
    """Test FlextInfraWorkspaceChecker._run_bandit method."""

    def test_run_bandit_no_src_dir(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_bandit_with_json_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        json_output = '{"results": [{"filename": "a.py", "line_number": 1, "test_id": "B101", "issue_text": "Assert used", "issue_severity": "MEDIUM"}]}'
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout=json_output,
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_bandit_with_invalid_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="invalid json",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=False)


class TestWorkspaceCheckerRunMarkdown:
    """Test FlextInfraWorkspaceChecker._run_markdown method."""

    def test_run_markdown_no_files(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_markdown_with_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="README.md:1:1 error MD001 Heading level",
                stderr="",
                returncode=1,
            ),
        )
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_markdown_with_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / ".markdownlint.json").write_text("{}")
        captured_args: list[list[str]] = []

        def _fake_run(cmd: list[str], *_a: object, **_kw: object) -> SimpleNamespace:
            captured_args.append(cmd)
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        monkeypatch.setattr(checker, "_run", _fake_run)
        checker._run_markdown(proj_dir)
        tm.that("--config" in captured_args[0], eq=True)

    def test_run_markdown_fallback_error_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        monkeypatch.setattr(
            checker,
            "_run",
            lambda *_a, **_kw: SimpleNamespace(
                stdout="",
                stderr="markdownlint failed",
                returncode=1,
            ),
        )
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)
