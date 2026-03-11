from __future__ import annotations

import pytest

from flext_core import r
from flext_infra.models import FlextInfraModels as m
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService
from flext_tests import tm


def _cmd_out(exit_code: int = 0) -> m.Infra.Core.CommandOutput:
    return m.Infra.Core.CommandOutput(
        stdout="",
        stderr="",
        exit_code=exit_code,
        duration=0.0,
    )


@pytest.fixture
def orchestrator() -> FlextInfraOrchestratorService:
    return FlextInfraOrchestratorService()


class TestOrchestratorBasic:
    def test_executes_verb_across_projects(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        tm.ok(orchestrator.orchestrate(["project-a", "project-b"], "check"), len=2)

    def test_fail_fast(self, orchestrator: FlextInfraOrchestratorService) -> None:
        tm.ok(orchestrator.orchestrate(["p-a", "p-b", "p-c"], "test", fail_fast=True))

    def test_continues_without_fail_fast(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        tm.ok(orchestrator.orchestrate(["p-a", "p-b"], "test", fail_fast=False), len=2)

    def test_execute_returns_failure(self) -> None:
        tm.fail(FlextInfraOrchestratorService().execute())

    def test_empty_project_list(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        tm.ok(orchestrator.orchestrate([], "check"), len=0)


class TestOrchestratorFailures:
    def test_fail_fast_skips_remaining(
        self,
        orchestrator: FlextInfraOrchestratorService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _run_project_fail(
            self: FlextInfraOrchestratorService,
            project: str,
            verb: str,
            idx: int,
            make_args: list[str],
        ) -> r[m.Infra.Core.CommandOutput]:
            del self, project, verb, idx, make_args
            return r[m.Infra.Core.CommandOutput].fail("Failed")

        monkeypatch.setattr(
            FlextInfraOrchestratorService, "_run_project", _run_project_fail
        )
        tm.ok(
            orchestrator.orchestrate(["p-a", "p-b", "p-c"], "test", fail_fast=True),
            len=3,
        )

    def test_runner_exception(
        self,
        orchestrator: FlextInfraOrchestratorService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _run_project_raise(
            self: FlextInfraOrchestratorService,
            project: str,
            verb: str,
            idx: int,
            make_args: list[str],
        ) -> r[m.Infra.Core.CommandOutput]:
            del self, project, verb, idx, make_args
            msg = "Runner failed"
            raise OSError(msg)

        monkeypatch.setattr(
            FlextInfraOrchestratorService, "_run_project", _run_project_raise
        )
        tm.fail(orchestrator.orchestrate(["p-a"], "test"), has="Orchestration failed")

    def test_run_project_failure_with_fail_fast(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        orchestrator = FlextInfraOrchestratorService()
        call_count = [0]

        def _run_project(
            self: FlextInfraOrchestratorService,
            project: str,
            verb: str,
            idx: int,
            make_args: list[str],
        ) -> r[m.Infra.Core.CommandOutput]:
            del self, project, verb, idx, make_args
            call_count[0] += 1
            if call_count[0] == 1:
                return r[m.Infra.Core.CommandOutput].fail("project execution failed")
            return r[m.Infra.Core.CommandOutput].ok(_cmd_out(0))

        monkeypatch.setattr(FlextInfraOrchestratorService, "_run_project", _run_project)
        tm.ok(
            orchestrator.orchestrate(["p1", "p2", "p3"], "test", fail_fast=True), len=3
        )
