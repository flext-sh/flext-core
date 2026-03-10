"""Tests for workspace checker runner methods — pyrefly, mypy, pyright, bandit, etc.

Uses monkeypatch to inject controlled subprocess output instead of unittest.mock.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_infra.check.services import (
    FlextInfraWorkspaceChecker,
    _GateExecution,
)
from flext_tests import tm


def _stub_run(
    result: SimpleNamespace,
) -> object:
    """Create a stub _run method returning a fixed result."""

    def _run(
        _cmd: list[str],
        _cwd: Path,
        **_kw: object,
    ) -> SimpleNamespace:
        return result

    return _run


def _stub_run_sequence(
    results: list[SimpleNamespace],
) -> object:
    """Create a stub _run method returning results in sequence."""
    call_idx = [0]

    def _run(
        _cmd: list[str],
        _cwd: Path,
        **_kw: object,
    ) -> SimpleNamespace:
        idx = call_idx[0]
        call_idx[0] += 1
        return results[idx] if idx < len(results) else results[-1]

    return _run


class TestRunPyrefly:
    """Test FlextInfraWorkspaceChecker._run_pyrefly method."""

    def test_run_pyrefly_with_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text('{"errors": []}')
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout="", stderr="", returncode=0)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=True)

    def test_run_pyrefly_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text(
            '{"errors": [{"path": "a.py", "line": 1, "column": 0, "name": "E001", "description": "Error", "severity": "error"}]}'
        )
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout="", stderr="", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_pyrefly_with_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text("invalid json")
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout="", stderr="", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)

    def test_run_pyrefly_with_error_count_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(
                SimpleNamespace(stdout="", stderr="Found 3 errors", returncode=1)
            ),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=3)

    def test_run_pyrefly_with_list_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        json_file = reports_dir / "p1-pyrefly.json"
        json_file.write_text(
            '[{"path": "a.py", "line": 1, "column": 0, "name": "E001", "description": "Error", "severity": "error"}]'
        )
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout="", stderr="", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        result = checker._run_pyrefly(proj_dir, reports_dir)
        tm.that(len(result.issues), eq=1)


class TestRunMypy:
    """Test FlextInfraWorkspaceChecker._run_mypy method."""

    def test_run_mypy_no_python_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(lambda _r, _d: []))
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_mypy_with_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        json_line = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout=json_line, stderr="", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        monkeypatch.setattr(
            checker, "_dirs_with_py", staticmethod(lambda _r, _d: ["src"])
        )
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_mypy_skips_empty_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        line1 = '{"file": "a.py", "line": 1, "column": 0, "code": "E001", "message": "Error", "severity": "error"}'
        line2 = '{"file": "b.py", "line": 2, "column": 0, "code": "E002", "message": "Error", "severity": "error"}'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(
                SimpleNamespace(stdout=f"{line1}\n\n{line2}\n", stderr="", returncode=1)
            ),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        monkeypatch.setattr(
            checker, "_dirs_with_py", staticmethod(lambda _r, _d: ["src"])
        )
        result = checker._run_mypy(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)
