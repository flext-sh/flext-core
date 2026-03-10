"""Tests for workspace checker gate runners — bandit and markdown.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm


class TestWorkspaceCheckerRunBandit:
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
        json_output = (
            '{"results": [{"filename": "a.py", "line_number": 1,'
            ' "test_id": "B101", "issue_text": "Assert used",'
            ' "issue_severity": "MEDIUM"}]}'
        )
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
