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

_MR, _CO = m.Infra.Workspace.MigrationResult, m.Infra.Core.CommandOutput


def _ns(**kwargs: str | float | bool | list[str] | Path | None) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _cmd_out(code: int) -> _CO:
    return _CO(stdout="", stderr="", exit_code=code, duration=0.0)


def _orch_args(projects: list[str] | None = None) -> argparse.Namespace:
    return _ns(
        projects=["p-a", "p-b"] if projects is None else projects,
        verb="check",
        fail_fast=False,
        make_arg=[],
    )


class TestRunDetect:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (r[WorkspaceMode].ok(WorkspaceMode.WORKSPACE), 0),
            (r[WorkspaceMode].fail("Detection failed"), 1),
        ],
    )
    def test_detect(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: r[WorkspaceMode],
        expected: int,
    ) -> None:
        monkeypatch.setattr(
            FlextInfraWorkspaceDetector, "detect", lambda _s, _r: result
        )
        tm.that(workspace_main._run_detect(_ns(project_root=tmp_path)), eq=expected)


class TestRunSync:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (r[bool].ok(True), 0),
            (r[bool].fail("Sync failed"), 1),
        ],
    )
    def test_sync(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: r[bool],
        expected: int,
    ) -> None:
        monkeypatch.setattr(FlextInfraSyncService, "sync", lambda _s, **_kw: result)
        tm.that(
            workspace_main._run_sync(_ns(project_root=tmp_path, canonical_root=None)),
            eq=expected,
        )


class TestRunOrchestrate:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            FlextInfraOrchestratorService,
            "orchestrate",
            lambda _s, **_kw: r[list[_CO]].ok([_cmd_out(0), _cmd_out(0)]),
        )
        tm.that(workspace_main._run_orchestrate(_orch_args()), eq=0)

    def test_no_projects(self) -> None:
        tm.that(workspace_main._run_orchestrate(_orch_args([])), eq=1)

    def test_with_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            FlextInfraOrchestratorService,
            "orchestrate",
            lambda _s, **_kw: r[list[_CO]].ok([_cmd_out(0), _cmd_out(1)]),
        )
        tm.that(workspace_main._run_orchestrate(_orch_args()), eq=1)

    def test_orchestration_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            FlextInfraOrchestratorService,
            "orchestrate",
            lambda _s, **_kw: r[list[_CO]].fail("Orchestration failed"),
        )
        tm.that(workspace_main._run_orchestrate(_orch_args(["p-a"])), eq=1)


class TestRunMigrate:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (r[list[_MR]].ok([_MR(project="test", errors=[], changes=[])]), 0),
            (r[list[_MR]].fail("Migration failed"), 1),
        ],
    )
    def test_success_or_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: r[list[_MR]],
        expected: int,
    ) -> None:
        monkeypatch.setattr(
            FlextInfraProjectMigrator, "migrate", lambda _s, **_kw: result
        )
        tm.that(
            workspace_main._run_migrate(_ns(workspace_root=tmp_path, dry_run=False)),
            eq=expected,
        )

    def test_with_project_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mrs = [
            _MR(project="p1", errors=["Error 1"], changes=[]),
            _MR(project="p2", errors=[], changes=[]),
        ]
        monkeypatch.setattr(
            FlextInfraProjectMigrator, "migrate", lambda _s, **_kw: r[list[_MR]].ok(mrs)
        )
        tm.that(
            workspace_main._run_migrate(_ns(workspace_root=tmp_path, dry_run=False)),
            eq=1,
        )


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[argparse.Namespace]:
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        workspace_main, "_run_orchestrate", lambda args: (captured.append(args), 0)[1]
    )
    return captured


class TestMainCli:
    def test_detect(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(workspace_main, "_run_detect", lambda _a: 0)
        tm.that(workspace_main.main(["detect", "--project-root", str(tmp_path)]), eq=0)

    def test_sync(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(workspace_main, "_run_sync", lambda _a: 0)
        tm.that(workspace_main.main(["sync", "--project-root", str(tmp_path)]), eq=0)

    def test_orchestrate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(workspace_main, "_run_orchestrate", lambda _a: 0)
        tm.that(workspace_main.main(["orchestrate", "--verb", "check", "p-a"]), eq=0)

    def test_migrate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(workspace_main, "_run_migrate", lambda _a: 0)
        tm.that(
            workspace_main.main(["migrate", "--workspace-root", str(tmp_path)]), eq=0
        )

    def test_no_command(self) -> None:
        tm.that(workspace_main.main([]), eq=1)

    def test_fail_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch)
        workspace_main.main(["orchestrate", "--verb", "check", "--fail-fast", "p-a"])
        tm.that(captured[0].fail_fast, eq=True)

    def test_make_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch)
        workspace_main.main([
            "orchestrate",
            "--verb",
            "check",
            "--make-arg",
            "VERBOSE=1",
            "--make-arg",
            "PARALLEL=4",
            "p-a",
        ])
        tm.that(captured[0].make_arg, has=["VERBOSE=1", "PARALLEL=4"])
