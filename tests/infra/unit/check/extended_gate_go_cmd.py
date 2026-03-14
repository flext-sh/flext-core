"""Tests for workspace checker — go, command, collect, and run_command methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_core import r, t
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm

from ...helpers import h
from ...models import m


class TestWorkspaceCheckerRunGo:
    """Test FlextInfraWorkspaceChecker._run_go method."""

    def test_run_go_no_go_mod(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=True)
        tm.that(len(result.issues), eq=0)

    def test_run_go_with_vet_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "go.mod").write_text("module test")
        call_count = [0]

        def _fake_run(*_a: t.Scalar, **_kw: t.Scalar) -> SimpleNamespace:
            call_count[0] += 1
            if call_count[0] == 1:
                return h.stub_run(stdout="main.go:10:5: error message", returncode=1)
            return h.stub_run()

        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)

    def test_run_go_with_format_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "go.mod").write_text("module test")
        (proj_dir / "main.go").write_text("package main")
        call_count = [0]

        def _fake_run(*_a: t.Scalar, **_kw: t.Scalar) -> SimpleNamespace:
            call_count[0] += 1
            if call_count[0] == 1:
                return h.stub_run()
            return h.stub_run(stdout="main.go", returncode=1)

        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_go_fallback_error_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "go.mod").write_text("module test")
        call_count = [0]

        def _fake_run(*_a: t.Scalar, **_kw: t.Scalar) -> SimpleNamespace:
            call_count[0] += 1
            if call_count[0] == 1:
                return h.stub_run(stderr="go vet failed", returncode=1)
            return h.stub_run()

        monkeypatch.setattr(checker, "_run", _fake_run)
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)


class TestWorkspaceCheckerRunCommand:
    """Test FlextInfraWorkspaceChecker._run method."""

    def test_run_command_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)

        def _fake_run(
            _self: FlextInfraUtilitiesSubprocess,
            _cmd: list[str],
            **_kw: t.Scalar,
        ) -> r[m.Infra.Core.CommandOutput]:
            return r[m.Infra.Core.CommandOutput].ok(
                m.Infra.Core.CommandOutput(stdout="output", stderr="", exit_code=0),
            )

        monkeypatch.setattr(FlextInfraUtilitiesSubprocess, "run_raw", _fake_run)
        result = checker._run(["echo", "test"], tmp_path)
        tm.that(result.stdout, eq="output")
        tm.that(result.exit_code, eq=0)

    def test_run_command_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)

        def _fake_run(
            _self: FlextInfraUtilitiesSubprocess,
            _cmd: list[str],
            **_kw: t.Scalar,
        ) -> r[m.Infra.Core.CommandOutput]:
            return r[m.Infra.Core.CommandOutput].fail("execution failed")

        monkeypatch.setattr(FlextInfraUtilitiesSubprocess, "run_raw", _fake_run)
        result = checker._run(["false"], tmp_path)
        tm.that(result.exit_code, eq=1)
        tm.that("execution failed" in result.stderr, eq=True)


class TestWorkspaceCheckerCollectMarkdownFiles:
    """Test FlextInfraWorkspaceChecker._collect_markdown_files method."""

    def test_collect_markdown_files_finds_files(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / "docs").mkdir()
        (proj_dir / "docs" / "guide.md").write_text("# Guide")
        files = checker._collect_markdown_files(proj_dir)
        tm.that(len(files), eq=2)

    def test_collect_markdown_files_excludes_dirs(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1", with_git=True)
        (proj_dir / "README.md").write_text("# Test")
        (proj_dir / ".git" / "README.md").write_text("# Git")
        files = checker._collect_markdown_files(proj_dir)
        tm.that(len(files), eq=1)
