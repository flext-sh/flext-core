from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra._utilities.cli import FlextInfraUtilitiesCli
from flext_infra.models import FlextInfraModels as m
from flext_infra.workspace import __main__ as workspace_main
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector, WorkspaceMode
from flext_infra.workspace.migrator import FlextInfraProjectMigrator
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService
from flext_infra.workspace.sync import FlextInfraSyncService
from flext_tests import tm


def _cli(workspace: Path) -> FlextInfraUtilitiesCli.CliArgs:
    return FlextInfraUtilitiesCli.CliArgs(workspace=workspace)


def _cmd_out(code: int) -> m.Infra.Core.CommandOutput:
    return m.Infra.Core.CommandOutput(
        stdout="", stderr="", exit_code=code, duration=0.0
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
        def _detect_stub(
            _self: FlextInfraWorkspaceDetector,
            project_root: Path,
        ) -> r[WorkspaceMode]:
            del _self, project_root
            return result

        monkeypatch.setattr(FlextInfraWorkspaceDetector, "detect", _detect_stub)
        tm.that(workspace_main._run_detect(_cli(tmp_path)), eq=expected)


class TestRunSync:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (
                r[m.Infra.Workspace.SyncResult].ok(
                    m.Infra.Workspace.SyncResult(
                        files_changed=1,
                        source=Path(),
                        target=Path(),
                    )
                ),
                0,
            ),
            (r[m.Infra.Workspace.SyncResult].fail("Sync failed"), 1),
        ],
    )
    def test_sync(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: r[m.Infra.Workspace.SyncResult],
        expected: int,
    ) -> None:
        def _sync_stub(
            _self: FlextInfraSyncService,
            _source: str | None = None,
            _target: str | None = None,
            *,
            workspace_root: Path | None = None,
            config: m.Infra.Basemk.BaseMkConfig | None = None,
            canonical_root: Path | None = None,
        ) -> r[m.Infra.Workspace.SyncResult]:
            del _self, _source, _target, workspace_root, config, canonical_root
            return result

        monkeypatch.setattr(FlextInfraSyncService, "sync", _sync_stub)
        tm.that(workspace_main._run_sync(_cli(tmp_path), None), eq=expected)


class TestRunOrchestrate:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _orchestrate_success(
            _self: FlextInfraOrchestratorService,
            projects: list[str],
            verb: str,
            fail_fast: bool = False,
            make_args: list[str] | None = None,
        ) -> r[list[m.Infra.Core.CommandOutput]]:
            del _self, projects, verb, fail_fast, make_args
            return r[list[m.Infra.Core.CommandOutput]].ok([_cmd_out(0), _cmd_out(0)])

        monkeypatch.setattr(
            FlextInfraOrchestratorService,
            "orchestrate",
            _orchestrate_success,
        )
        tm.that(
            workspace_main._run_orchestrate(
                ["p-a", "p-b"],
                "check",
                fail_fast=False,
                make_args=[],
            ),
            eq=0,
        )

    def test_no_projects(self) -> None:
        tm.that(
            workspace_main._run_orchestrate(
                [],
                "check",
                fail_fast=False,
                make_args=[],
            ),
            eq=1,
        )

    def test_with_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _orchestrate_partial(
            _self: FlextInfraOrchestratorService,
            projects: list[str],
            verb: str,
            fail_fast: bool = False,
            make_args: list[str] | None = None,
        ) -> r[list[m.Infra.Core.CommandOutput]]:
            del _self, projects, verb, fail_fast, make_args
            return r[list[m.Infra.Core.CommandOutput]].ok([_cmd_out(0), _cmd_out(1)])

        monkeypatch.setattr(
            FlextInfraOrchestratorService,
            "orchestrate",
            _orchestrate_partial,
        )
        tm.that(
            workspace_main._run_orchestrate(
                ["p-a", "p-b"],
                "check",
                fail_fast=False,
                make_args=[],
            ),
            eq=1,
        )

    def test_orchestration_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _orchestrate_failure(
            _self: FlextInfraOrchestratorService,
            projects: list[str],
            verb: str,
            fail_fast: bool = False,
            make_args: list[str] | None = None,
        ) -> r[list[m.Infra.Core.CommandOutput]]:
            del _self, projects, verb, fail_fast, make_args
            return r[list[m.Infra.Core.CommandOutput]].fail("Orchestration failed")

        monkeypatch.setattr(
            FlextInfraOrchestratorService,
            "orchestrate",
            _orchestrate_failure,
        )
        tm.that(
            workspace_main._run_orchestrate(
                ["p-a"],
                "check",
                fail_fast=False,
                make_args=[],
            ),
            eq=1,
        )


class TestRunMigrate:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (
                r[list[m.Infra.Workspace.MigrationResult]].ok([
                    m.Infra.Workspace.MigrationResult(
                        project="test", errors=[], changes=[]
                    )
                ]),
                0,
            ),
            (r[list[m.Infra.Workspace.MigrationResult]].fail("Migration failed"), 1),
        ],
    )
    def test_success_or_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: r[list[m.Infra.Workspace.MigrationResult]],
        expected: int,
    ) -> None:
        def _migrate_stub(
            _self: FlextInfraProjectMigrator,
            workspace_root: Path,
            dry_run: bool,
        ) -> r[list[m.Infra.Workspace.MigrationResult]]:
            del _self, workspace_root, dry_run
            return result

        monkeypatch.setattr(FlextInfraProjectMigrator, "migrate", _migrate_stub)
        tm.that(
            workspace_main._run_migrate(
                FlextInfraUtilitiesCli.CliArgs(workspace=tmp_path, apply=True)
            ),
            eq=expected,
        )

    def test_with_project_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mrs: list[m.Infra.Workspace.MigrationResult] = [
            m.Infra.Workspace.MigrationResult(
                project="p1", errors=["Error 1"], changes=[]
            ),
            m.Infra.Workspace.MigrationResult(project="p2", errors=[], changes=[]),
        ]

        def _migrate_with_errors(
            _self: FlextInfraProjectMigrator,
            workspace_root: Path,
            dry_run: bool,
        ) -> r[list[m.Infra.Workspace.MigrationResult]]:
            del _self, workspace_root, dry_run
            return r[list[m.Infra.Workspace.MigrationResult]].ok(mrs)

        monkeypatch.setattr(
            FlextInfraProjectMigrator,
            "migrate",
            _migrate_with_errors,
        )
        tm.that(
            workspace_main._run_migrate(
                FlextInfraUtilitiesCli.CliArgs(workspace=tmp_path, apply=True)
            ),
            eq=1,
        )


def _capture(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[str], str, bool, list[str]]]:
    captured: list[tuple[list[str], str, bool, list[str]]] = []

    def _capture_orchestrate(
        projects: list[str],
        verb: str,
        *,
        fail_fast: bool,
        make_args: list[str],
    ) -> int:
        captured.append((projects, verb, fail_fast, make_args))
        return 0

    monkeypatch.setattr(workspace_main, "_run_orchestrate", _capture_orchestrate)
    return captured


def _ok_main(*_args: object, **_kwargs: object) -> int:
    return 0


class TestMainCli:
    def test_detect(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(workspace_main, "_run_detect", _ok_main)
        tm.that(workspace_main.main(["detect", "--workspace", str(tmp_path)]), eq=0)

    def test_sync(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(workspace_main, "_run_sync", _ok_main)
        tm.that(workspace_main.main(["sync", "--workspace", str(tmp_path)]), eq=0)

    def test_orchestrate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(workspace_main, "_run_orchestrate", _ok_main)
        tm.that(workspace_main.main(["orchestrate", "--verb", "check", "p-a"]), eq=0)

    def test_migrate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(workspace_main, "_run_migrate", _ok_main)
        tm.that(workspace_main.main(["migrate", "--workspace", str(tmp_path)]), eq=0)

    def test_no_command(self) -> None:
        tm.that(workspace_main.main([]), eq=1)

    def test_fail_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture(monkeypatch)
        workspace_main.main(["orchestrate", "--verb", "check", "--fail-fast", "p-a"])
        tm.that(captured[0][2], eq=True)

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
        tm.that(captured[0][3], has=["VERBOSE=1", "PARALLEL=4"])
