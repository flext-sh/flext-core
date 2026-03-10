"""Tests for FlextInfraOrchestratorService.

Uses real service instances with monkeypatch for runner control.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from flext_core import r, t
from flext_infra import m
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService
from flext_tests import tm

_CO = m.Infra.Core.CommandOutput


def _cmd_out(exit_code: int = 0) -> _CO:
    return _CO(stdout="", stderr="", exit_code=exit_code, duration=0.0)


def _stub_runner_ok(exit_code: int = 0) -> t.ContainerValue:
    """Create a runner stub that returns ok with given exit code."""

    class _Runner:
        @staticmethod
        def run_to_file(
            cmd: Sequence[str],
            output_file: Path,
            cwd: Path | None = None,
            timeout: int | None = None,
            env: t.ContainerValue = None,
        ) -> r[int]:
            return r[int].ok(exit_code)

    return _Runner()


def _stub_runner_fail(error: str) -> t.ContainerValue:
    """Create a runner stub that returns failure."""

    class _Runner:
        @staticmethod
        def run_to_file(
            cmd: Sequence[str],
            output_file: Path,
            cwd: Path | None = None,
            timeout: int | None = None,
            env: t.ContainerValue = None,
        ) -> r[int]:
            return r[int].fail(error)

    return _Runner()


def _stub_runner_raise(error: str) -> t.ContainerValue:
    """Create a runner stub that raises OSError."""

    class _Runner:
        @staticmethod
        def run_to_file(
            cmd: Sequence[str],
            output_file: Path,
            cwd: Path | None = None,
            timeout: int | None = None,
            env: t.ContainerValue = None,
        ) -> r[int]:
            raise OSError(error)

    return _Runner()


@pytest.fixture
def orchestrator() -> FlextInfraOrchestratorService:
    return FlextInfraOrchestratorService()


class TestOrchestratorBasic:
    def test_executes_verb_across_projects(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        result = orchestrator.orchestrate(["project-a", "project-b"], "check")
        tm.ok(result, len=2)

    def test_fail_fast(self, orchestrator: FlextInfraOrchestratorService) -> None:
        result = orchestrator.orchestrate(["p-a", "p-b", "p-c"], "test", fail_fast=True)
        tm.ok(result)

    def test_continues_without_fail_fast(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        result = orchestrator.orchestrate(["p-a", "p-b"], "test", fail_fast=False)
        tm.ok(result, len=2)

    def test_execute_returns_failure(self) -> None:
        tm.fail(FlextInfraOrchestratorService().execute())

    def test_empty_project_list(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        tm.ok(orchestrator.orchestrate([], "check"), len=0)

    def test_captures_per_project_output(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        tm.ok(orchestrator.orchestrate(["p-a"], "check"), len=1)


class TestOrchestratorWithRunner:
    def test_fail_fast_skips_remaining(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        orchestrator._runner = _stub_runner_fail("Failed")
        result = orchestrator.orchestrate(["p-a", "p-b", "p-c"], "test", fail_fast=True)
        tm.ok(result, len=3)

    def test_runner_exception(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        orchestrator._runner = _stub_runner_raise("Runner failed")
        result = orchestrator.orchestrate(["p-a"], "test")
        tm.fail(result, has="Orchestration failed")

    def test_fail_fast_exit_codes(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        orchestrator._runner = _stub_runner_fail("Failed")
        result = orchestrator.orchestrate(["p-a", "p-b", "p-c"], "test", fail_fast=True)
        value = tm.ok(result)
        tm.that(value[0].exit_code, eq=1)
        tm.that(value[1].exit_code, eq=0)
        tm.that(value[2].exit_code, eq=0)

    def test_execution_failure_all_projects(
        self, orchestrator: FlextInfraOrchestratorService
    ) -> None:
        orchestrator._runner = _stub_runner_fail("Execution failed")
        result = orchestrator.orchestrate(["p1", "p2"], "test", fail_fast=False)
        value = tm.ok(result, len=2)
        tm.that(value[0].exit_code, eq=1)
        tm.that(value[1].exit_code, eq=1)


class TestOrchestratorRunProject:
    def test_run_project_failure_with_fail_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraOrchestratorService()
        call_count = [0]

        def _run_project(
            self: t.ContainerValue,
            project: str,
            verb: str,
            idx: int,
            make_args: list[str],
        ) -> r[_CO]:
            call_count[0] += 1
            if call_count[0] == 1:
                return r[_CO].fail("project execution failed")
            return r[_CO].ok(_cmd_out(0))

        monkeypatch.setattr(FlextInfraOrchestratorService, "_run_project", _run_project)
        result = orchestrator.orchestrate(["p1", "p2", "p3"], "test", fail_fast=True)
        value = tm.ok(result, len=3)
        tm.that(value[0].exit_code, eq=1)
        tm.that(value[1].exit_code, eq=0)
        tm.that(value[2].exit_code, eq=0)
