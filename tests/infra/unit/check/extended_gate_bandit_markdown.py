"""Tests for workspace checker gate runners — bandit and markdown.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm

from ...helpers import h
from ...models import m


def _run_stub(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> m.Infra.Core.CommandOutput:
    return m.Infra.Core.CommandOutput(
        stdout=stdout, stderr=stderr, exit_code=returncode
    )


class TestWorkspaceCheckerRunBandit:
    def test_run_bandit_no_src_dir(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_bandit_with_json_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_src=True)
        json_output = (
            '{"results": [{"filename": "a.py", "line_number": 1,'
            ' "test_id": "B101", "issue_text": "Assert used",'
            ' "issue_severity": "MEDIUM"}]}'
        )

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return _run_stub(stdout=json_output, returncode=1)

        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_bandit_with_invalid_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_src=True)

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return _run_stub(stdout="invalid json", returncode=1)

        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_bandit(proj_dir)
        tm.that(result.result.passed, eq=False)


class TestWorkspaceCheckerRunMarkdown:
    def test_run_markdown_no_files(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_markdown_with_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "README.md").write_text("# Test")

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return _run_stub(
                stdout="README.md:1:1 error MD001 Heading level",
                returncode=1,
            )

        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_markdown_with_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / ".markdownlint.json").write_text("{}")
        captured_args: list[list[str]] = []

        def _fake_run(
            cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cwd, _timeout, _env
            captured_args.append(cmd)
            return _run_stub()

        monkeypatch.setattr(checker, "_run", _fake_run)
        checker._run_markdown(proj_dir)
        tm.that("--config" in captured_args[0], eq=True)

    def test_run_markdown_fallback_error_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "README.md").write_text("# Test")

        def _fake_run(
            _cmd: list[str],
            _cwd: Path,
            _timeout: int = 120,
            _env: dict[str, str] | None = None,
        ) -> m.Infra.Core.CommandOutput:
            del _cmd, _cwd, _timeout, _env
            return _run_stub(stderr="markdownlint failed", returncode=1)

        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_markdown(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)
