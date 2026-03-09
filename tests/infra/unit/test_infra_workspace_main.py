"""Tests for workspace CLI entry point (__main__.py).

Uses real argparse.Namespace and monkeypatch instead of unittest.mock.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from flext_core import r
from flext_infra import m
from flext_infra.workspace import __main__ as workspace_main
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector, WorkspaceMode
from flext_infra.workspace.migrator import FlextInfraProjectMigrator
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService
from flext_infra.workspace.sync import FlextInfraSyncService
from flext_tests import tm

_MR = m.Infra.Workspace.MigrationResult
_CO = m.Infra.Core.CommandOutput


def _ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _cmd_out(exit_code: int = 0) -> _CO:
    return _CO(stdout="", stderr="", exit_code=exit_code, duration=0.0)


class TestRunDetect:
    def test_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _detect(self: FlextInfraWorkspaceDetector, root: Path) -> r[WorkspaceMode]:
            return r[WorkspaceMode].ok(WorkspaceMode.WORKSPACE)

        monkeypatch.setattr(FlextInfraWorkspaceDetector, "detect", _detect)
        tm.that(workspace_main._run_detect(_ns(project_root=tmp_path)), eq=0)

    def test_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _detect(self: FlextInfraWorkspaceDetector, root: Path) -> r[WorkspaceMode]:
            return r[WorkspaceMode].fail("Detection failed")

        monkeypatch.setattr(FlextInfraWorkspaceDetector, "detect", _detect)
        tm.that(workspace_main._run_detect(_ns(project_root=tmp_path)), eq=1)


class TestRunSync:
    def test_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _sync(self: FlextInfraSyncService, **kw: object) -> r[bool]:
            return r[bool].ok(True)

        monkeypatch.setattr(FlextInfraSyncService, "sync", _sync)
        args = _ns(project_root=tmp_path, canonical_root=None)
        tm.that(workspace_main._run_sync(args), eq=0)

    def test_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _sync(self: FlextInfraSyncService, **kw: object) -> r[bool]:
            return r[bool].fail("Sync failed")

        monkeypatch.setattr(FlextInfraSyncService, "sync", _sync)
        args = _ns(project_root=tmp_path, canonical_root=None)
        tm.that(workspace_main._run_sync(args), eq=1)


class TestRunOrchestrate:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        outputs = [_cmd_out(0), _cmd_out(0)]

        def _orch(self: FlextInfraOrchestratorService, **kw: object) -> r[list[_CO]]:
            return r[list[_CO]].ok(outputs)

        monkeypatch.setattr(FlextInfraOrchestratorService, "orchestrate", _orch)
        args = _ns(projects=["p-a", "p-b"], verb="check", fail_fast=False, make_arg=[])
        tm.that(workspace_main._run_orchestrate(args), eq=0)

    def test_no_projects(self) -> None:
        args = _ns(projects=[], verb="check", fail_fast=False, make_arg=[])
        tm.that(workspace_main._run_orchestrate(args), eq=1)

    def test_with_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        outputs = [_cmd_out(0), _cmd_out(1)]

        def _orch(self: FlextInfraOrchestratorService, **kw: object) -> r[list[_CO]]:
            return r[list[_CO]].ok(outputs)

        monkeypatch.setattr(FlextInfraOrchestratorService, "orchestrate", _orch)
        args = _ns(projects=["p-a", "p-b"], verb="check", fail_fast=False, make_arg=[])
        tm.that(workspace_main._run_orchestrate(args), eq=1)

    def test_orchestration_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _orch(self: FlextInfraOrchestratorService, **kw: object) -> r[list[_CO]]:
            return r[list[_CO]].fail("Orchestration failed")

        monkeypatch.setattr(FlextInfraOrchestratorService, "orchestrate", _orch)
        args = _ns(projects=["p-a"], verb="check", fail_fast=False, make_arg=[])
        tm.that(workspace_main._run_orchestrate(args), eq=1)


class TestRunMigrate:
    def test_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        migration = _MR(project="test", errors=[], changes=[])

        def _migrate(self: FlextInfraProjectMigrator, **kw: object) -> r[list[_MR]]:
            return r[list[_MR]].ok([migration])

        monkeypatch.setattr(FlextInfraProjectMigrator, "migrate", _migrate)
        tm.that(
            workspace_main._run_migrate(_ns(workspace_root=tmp_path, dry_run=False)),
            eq=0,
        )

    def test_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _migrate(self: FlextInfraProjectMigrator, **kw: object) -> r[list[_MR]]:
            return r[list[_MR]].fail("Migration failed")

        monkeypatch.setattr(FlextInfraProjectMigrator, "migrate", _migrate)
        tm.that(
            workspace_main._run_migrate(_ns(workspace_root=tmp_path, dry_run=False)),
            eq=1,
        )

    def test_with_project_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        migrations = [
            _MR(project="p1", errors=["Error 1"], changes=[]),
            _MR(project="p2", errors=[], changes=[]),
        ]

        def _migrate(self: FlextInfraProjectMigrator, **kw: object) -> r[list[_MR]]:
            return r[list[_MR]].ok(migrations)

        monkeypatch.setattr(FlextInfraProjectMigrator, "migrate", _migrate)
        tm.that(
            workspace_main._run_migrate(_ns(workspace_root=tmp_path, dry_run=False)),
            eq=1,
        )


class TestMainCli:
    def test_detect_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _run(_args: argparse.Namespace) -> int:
            return 0

        monkeypatch.setattr(workspace_main, "_run_detect", _run)
        tm.that(workspace_main.main(["detect", "--project-root", str(tmp_path)]), eq=0)

    def test_sync_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _run(_args: argparse.Namespace) -> int:
            return 0

        monkeypatch.setattr(workspace_main, "_run_sync", _run)
        tm.that(workspace_main.main(["sync", "--project-root", str(tmp_path)]), eq=0)

    def test_orchestrate_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _run(_args: argparse.Namespace) -> int:
            return 0

        monkeypatch.setattr(workspace_main, "_run_orchestrate", _run)
        tm.that(
            workspace_main.main(["orchestrate", "--verb", "check", "p-a", "p-b"]), eq=0
        )

    def test_migrate_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _run(_args: argparse.Namespace) -> int:
            return 0

        monkeypatch.setattr(workspace_main, "_run_migrate", _run)
        tm.that(
            workspace_main.main(["migrate", "--workspace-root", str(tmp_path)]), eq=0
        )

    def test_no_command(self) -> None:
        tm.that(workspace_main.main([]), eq=1)

    def test_orchestrate_with_fail_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[argparse.Namespace] = []

        def _run(args: argparse.Namespace) -> int:
            captured.append(args)
            return 0

        monkeypatch.setattr(workspace_main, "_run_orchestrate", _run)
        workspace_main.main(["orchestrate", "--verb", "check", "--fail-fast", "p-a"])
        tm.that(captured[0].fail_fast, eq=True)

    def test_orchestrate_with_make_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[argparse.Namespace] = []

        def _run(args: argparse.Namespace) -> int:
            captured.append(args)
            return 0

        monkeypatch.setattr(workspace_main, "_run_orchestrate", _run)
        argv = [
            "orchestrate",
            "--verb",
            "check",
            "--make-arg",
            "VERBOSE=1",
            "--make-arg",
            "PARALLEL=4",
            "p-a",
        ]
        workspace_main.main(argv)
        tm.that(captured[0].make_arg, has="VERBOSE=1")
        tm.that(captured[0].make_arg, has="PARALLEL=4")


__all__: list[str] = []
