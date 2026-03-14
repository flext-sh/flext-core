"""Tests for workspace checker runner methods (pyrefly, mypy).

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

from ...helpers import h
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


class TestRunPyrefly:
    """Test FlextInfraWorkspaceChecker._run_pyrefly method."""

    def test_run_pyrefly_with_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text('{"errors": []}')
        monkeypatch.setattr(checker, "_run", _stub_run(h.stub_run()))
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=True)

    def test_run_pyrefly_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text(
            '{"errors": [{"path": "a.py", "line": 1, "column": 0, "name": "E001", "description": "Error", "severity": "error"}]}'
        )
        monkeypatch.setattr(checker, "_run", _stub_run(h.stub_run(returncode=1)))
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_pyrefly_with_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text("invalid json")
        monkeypatch.setattr(checker, "_run", _stub_run(h.stub_run(returncode=1)))
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)

    def test_run_pyrefly_with_error_count_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(h.stub_run(stderr="Found 3 errors", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=3)

    def test_run_pyrefly_with_list_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text(
            '[{"path": "a.py", "line": 1, "column": 0, "name": "E001", "description": "Error", "severity": "error"}]'
        )
        monkeypatch.setattr(checker, "_run", _stub_run(h.stub_run(returncode=1)))
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(len(result.issues), eq=1)


class TestRunMypy:
    """Test FlextInfraWorkspaceChecker._run_mypy method."""

    def test_run_mypy_no_python_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(_no_dirs_with_py))
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_mypy_with_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_src=True)
        (proj_dir / "src" / "main.py").write_text("# code")
        json_line = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(h.stub_run(stdout=json_line, returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(_src_dirs_with_py))
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_mypy_skips_empty_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_src=True)
        (proj_dir / "src" / "main.py").write_text("# code")
        line1 = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
        line2 = '{"file": "b.py", "line": 2, "column": 0, "code": "E002", "message": "Error", "severity": "error"}'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(h.stub_run(stdout=f"{line1}\n\n{line2}\n", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", _existing_dirs)
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(_src_dirs_with_py))
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)
