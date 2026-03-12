"""Tests for workspace checker runners — pyright, bandit, markdown, go, ruff.

Uses monkeypatch to inject controlled subprocess output.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm
from tests.infra import h

from ...models import m

RunCallable = Callable[
    [list[str], Path, int, dict[str, str] | None], m.Infra.Core.CommandOutput
]


def _as_command_output(
    result: m.Infra.Core.CommandOutput | SimpleNamespace,
) -> m.Infra.Core.CommandOutput:
    if isinstance(result, m.Infra.Core.CommandOutput):
        return result
    return m.Infra.Core.CommandOutput(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.returncode,
    )


def _stub_run(result: m.Infra.Core.CommandOutput | SimpleNamespace) -> RunCallable:
    """Create a stub _run method returning a fixed result."""

    def _run(
        _cmd: list[str],
        _cwd: Path,
        _timeout: int = 120,
        _env: dict[str, str] | None = None,
    ) -> m.Infra.Core.CommandOutput:
        del _cmd, _cwd, _timeout, _env
        return _as_command_output(result)

    return _run


def _existing_dirs(_project_dir: Path) -> list[str]:
    del _project_dir
    return ["src"]


def _no_dirs_with_py(_project_dir: Path, _dirs: list[str]) -> list[str]:
    del _project_dir, _dirs
    return []


def _src_dirs_with_py(_project_dir: Path, _dirs: list[str]) -> list[str]:
    del _project_dir, _dirs
    return ["src"]


class TestRunPyright:
    """Test FlextInfraWorkspaceChecker._run_pyright method."""

    def test_run_pyright_no_python_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(_no_dirs_with_py))
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_pyright_with_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_src=True)
        (proj_dir / "src" / "main.py").write_text("# code")
        json_output = '{"generalDiagnostics": [{"file": "a.py", "range": {"start": {"line": 0, "character": 0}}, "rule": "E001", "message": "Error", "severity": "error"}]}'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(h.stub_run(stdout=json_output, returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(_src_dirs_with_py))
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_pyright_with_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_src=True)
        (proj_dir / "src" / "main.py").write_text("# code")
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(h.stub_run(stdout="invalid json", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(_src_dirs_with_py))
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=False)


class TestRunBandit:
    """Test FlextInfraWorkspaceChecker._run_bandit method."""

    def test_run_bandit_no_src_dir(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_bandit_with_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_src=True)
        json_output = '{"results": [{"filename": "a.py", "line_number": 1, "test_id": "B101", "issue_text": "Assert used", "issue_severity": "MEDIUM"}]}'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(h.stub_run(stdout=json_output, returncode=1)),
        )
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_bandit_with_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_src=True)
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(h.stub_run(stdout="invalid json", returncode=1)),
        )
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=False)


class TestRunMarkdown:
    """Test FlextInfraWorkspaceChecker._run_markdown method."""

    def test_run_markdown_no_files(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_markdown_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "README.md").write_text("# Test")
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(
                h.stub_run(
                    stdout="README.md:1:1 error MD001 Heading level",
                    returncode=1,
                )
            ),
        )
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_markdown_fallback_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "README.md").write_text("# Test")
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(h.stub_run(stderr="markdownlint failed", returncode=1)),
        )
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)
