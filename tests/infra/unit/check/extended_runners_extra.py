"""Tests for workspace checker runners — pyright, bandit, markdown, go, ruff.

Uses monkeypatch to inject controlled subprocess output instead of unittest.mock.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm


def _stub_run(result: SimpleNamespace) -> object:
    """Create a stub _run method returning a fixed result."""

    def _run(_cmd: list[str], _cwd: Path, **_kw: object) -> SimpleNamespace:
        return result

    return _run


def _stub_run_seq(results: list[SimpleNamespace]) -> object:
    """Create a stub _run returning results in sequence."""
    idx = [0]

    def _run(_cmd: list[str], _cwd: Path, **_kw: object) -> SimpleNamespace:
        i = idx[0]
        idx[0] += 1
        return results[i] if i < len(results) else results[-1]

    return _run


class TestRunPyright:
    """Test FlextInfraWorkspaceChecker._run_pyright method."""

    def test_run_pyright_no_python_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        monkeypatch.setattr(checker, "_dirs_with_py", staticmethod(lambda _r, _d: []))
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_pyright_with_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        json_output = '{"generalDiagnostics": [{"file": "a.py", "range": {"start": {"line": 0, "character": 0}}, "rule": "E001", "message": "Error", "severity": "error"}]}'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout=json_output, stderr="", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        monkeypatch.setattr(
            checker, "_dirs_with_py", staticmethod(lambda _r, _d: ["src"])
        )
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_pyright_with_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        (proj_dir / "src" / "main.py").write_text("# code")
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout="invalid json", stderr="", returncode=1)),
        )
        monkeypatch.setattr(checker, "_existing_check_dirs", lambda _p: ["src"])
        monkeypatch.setattr(
            checker, "_dirs_with_py", staticmethod(lambda _r, _d: ["src"])
        )
        result = checker._run_pyright(proj_dir)
        tm.that(result.result.passed, eq=False)


class TestRunBandit:
    """Test FlextInfraWorkspaceChecker._run_bandit method."""

    def test_run_bandit_no_src_dir(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_bandit_with_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        json_output = '{"results": [{"filename": "a.py", "line_number": 1, "test_id": "B101", "issue_text": "Assert used", "issue_severity": "MEDIUM"}]}'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout=json_output, stderr="", returncode=1)),
        )
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_bandit_with_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "src").mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout="invalid json", stderr="", returncode=1)),
        )
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=False)


class TestRunMarkdown:
    """Test FlextInfraWorkspaceChecker._run_markdown method."""

    def test_run_markdown_no_files(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_markdown_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(
                SimpleNamespace(
                    stdout="README.md:1:1 error MD001 Heading level",
                    stderr="",
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
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(
                SimpleNamespace(stdout="", stderr="markdownlint failed", returncode=1)
            ),
        )
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)


class TestRunGo:
    """Test FlextInfraWorkspaceChecker._run_go method."""

    def test_run_go_no_go_mod(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_go_with_vet_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "go.mod").write_text("module test")
        vet = SimpleNamespace(
            stdout="main.go:10:5: error message", stderr="", returncode=1
        )
        fmt = SimpleNamespace(stdout="", stderr="", returncode=0)
        monkeypatch.setattr(checker, "_run", _stub_run_seq([vet, fmt]))
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)

    def test_run_go_with_format_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "go.mod").write_text("module test")
        (proj_dir / "main.go").write_text("package main")
        vet = SimpleNamespace(stdout="", stderr="", returncode=0)
        fmt = SimpleNamespace(stdout="main.go", stderr="", returncode=1)
        monkeypatch.setattr(checker, "_run", _stub_run_seq([vet, fmt]))
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_go_skips_empty_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "go.mod").write_text("module test\n")
        (proj_dir / "main.go").write_text("package main\n")
        vet = SimpleNamespace(stdout="", stderr="", returncode=0)
        fmt = SimpleNamespace(
            stdout="src/file.go\n\nsrc/other.go\n", stderr="", returncode=1
        )
        monkeypatch.setattr(checker, "_run", _stub_run_seq([vet, fmt]))
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)
