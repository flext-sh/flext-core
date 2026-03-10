"""Tests for workspace checker runner — go vet and go fmt.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_core import t
from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm
from tests.infra import h


def _stub_run_seq(results: list[SimpleNamespace]) -> t.ContainerValue:
    idx = [0]

    def _run(_cmd: list[str], _cwd: Path, **_kw: t.ContainerValue) -> SimpleNamespace:
        i = idx[0]
        idx[0] += 1
        return results[i] if i < len(results) else results[-1]

    return _run


class TestRunGo:
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
        vet = h.stub_run(stdout="main.go:10:5: error message", returncode=1)
        fmt = h.stub_run()
        monkeypatch.setattr(checker, "_run", _stub_run_seq([vet, fmt]))
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
        vet = h.stub_run()
        fmt = h.stub_run(stdout="main.go", returncode=1)
        monkeypatch.setattr(checker, "_run", _stub_run_seq([vet, fmt]))
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=1)

    def test_run_go_skips_empty_lines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        proj_dir = h.mk_project(tmp_path, "p1")
        (proj_dir / "go.mod").write_text("module test\n")
        (proj_dir / "main.go").write_text("package main\n")
        vet = h.stub_run()
        fmt = h.stub_run(stdout="src/file.go\n\nsrc/other.go\n", returncode=1)
        monkeypatch.setattr(checker, "_run", _stub_run_seq([vet, fmt]))
        result = checker._run_go(proj_dir)
        tm.that(result.result.passed, eq=False)
        tm.that(len(result.issues), eq=2)
