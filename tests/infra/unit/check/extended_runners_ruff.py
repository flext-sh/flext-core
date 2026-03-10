"""Tests for workspace checker ruff lint/format and command runner.

Uses monkeypatch to inject controlled subprocess output instead of unittest.mock.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_core import r
from flext_infra import m
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm


def _stub_run(result: SimpleNamespace) -> object:
    """Create a stub _run method returning a fixed result."""

    def _run(_cmd: list[str], _cwd: Path, **_kw: object) -> SimpleNamespace:
        return result

    return _run


class TestRunRuffLint:
    """Test FlextInfraWorkspaceChecker._run_ruff_lint method."""

    def test_run_ruff_lint_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        json_output = '[{"filename": "a.py", "location": {"row": 1, "column": 0}, "code": "E001", "message": "Error"}]'
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout=json_output, stderr="", returncode=1)),
        )
        result = checker._run_ruff_lint(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_ruff_lint_with_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout="invalid json", stderr="", returncode=1)),
        )
        result = checker._run_ruff_lint(proj_dir)
        tm.that(result.result.passed, eq=False)


class TestRunRuffFormat:
    """Test FlextInfraWorkspaceChecker._run_ruff_format method."""

    def test_run_ruff_format_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(
                SimpleNamespace(stdout="  --> a.py:1:1", stderr="", returncode=1)
            ),
        )
        result = checker._run_ruff_format(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_ruff_format_with_simple_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(SimpleNamespace(stdout="a.py", stderr="", returncode=1)),
        )
        result = checker._run_ruff_format(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_ruff_format_deduplicates_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(
                SimpleNamespace(
                    stdout="--> src/file.py:1:1\n--> src/file.py:1:1\n--> src/other.py:1:1\n",
                    stderr="",
                    returncode=1,
                )
            ),
        )
        result = checker._run_ruff_format(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)

    def test_run_ruff_format_skips_empty_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.setattr(
            checker,
            "_run",
            _stub_run(
                SimpleNamespace(
                    returncode=1, stdout="file1.py\n\nfile2.py\n", stderr=""
                )
            ),
        )
        result = checker._run_ruff_format(tmp_path)
        tm.that(len(result.issues) >= 1, eq=True)


class TestRunCommand:
    """Test FlextInfraWorkspaceChecker._run method."""

    def test_run_command_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)

        def _fake_run_raw(
            _self: object, _cmd: list[str], **_kw: object
        ) -> r[m.Infra.Core.CommandOutput]:
            return r[m.Infra.Core.CommandOutput].ok(
                m.Infra.Core.CommandOutput(stdout="output", stderr="", exit_code=0)
            )

        monkeypatch.setattr(FlextInfraUtilitiesSubprocess, "run_raw", _fake_run_raw)
        result = checker._run(["echo", "test"], tmp_path)
        tm.that(result.stdout, eq="output")
        tm.that(result.exit_code, eq=0)

    def test_run_command_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)

        def _fake_run_raw(
            _self: object, _cmd: list[str], **_kw: object
        ) -> r[m.Infra.Core.CommandOutput]:
            return r[m.Infra.Core.CommandOutput].fail("execution failed")

        monkeypatch.setattr(FlextInfraUtilitiesSubprocess, "run_raw", _fake_run_raw)
        result = checker._run(["false"], tmp_path)
        tm.that(result.exit_code, eq=1)
        tm.that(result.stderr, contains="execution failed")


class TestCollectMarkdownFiles:
    """Test FlextInfraWorkspaceChecker._collect_markdown_files method."""

    def test_collect_markdown_files_finds_files(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / "docs").mkdir()
        (proj_dir / "docs" / "guide.md").write_text("# Guide")
        files = checker._collect_markdown_files(proj_dir)
        tm.that(len(files), eq=2)

    def test_collect_markdown_files_excludes_dirs(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = tmp_path / "p1"
        proj_dir.mkdir()
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / ".git").mkdir()
        (proj_dir / ".git" / "README.md").write_text("# Git")
        files = checker._collect_markdown_files(proj_dir)
        tm.that(len(files), eq=1)
