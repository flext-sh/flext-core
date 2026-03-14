"""Tests for CLI entry points: workspace_check, fix_pyrefly_config, check __main__, run_cli.

Uses monkeypatch to inject controlled service behavior.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import flext_infra.check.__main__ as check_main_mod
import flext_infra.check.fix_pyrefly_config as fix_pyrefly_mod
import flext_infra.check.workspace_check as ws_mod
from flext_core import r, t
from flext_infra.check.services import (
    GateExecution,
    ProjectResult,
    run_cli,
)
from flext_tests import tm

from ...models import m


def _fake_checker_cls(
    parse_result: list[str],
    run_result: r[list[SimpleNamespace]] | r[list[ProjectResult]],
) -> type:
    class _Fake:
        def __init__(self, **_kw: t.Scalar) -> None:
            _ = _kw

        @staticmethod
        def parse_gate_csv(_gates: str) -> list[str]:
            return parse_result

        def run_projects(
            self,
            projects: list[str] | None = None,
            gates: list[str] | None = None,
            **kw: t.Scalar,
        ) -> r[list[SimpleNamespace]] | r[list[ProjectResult]]:
            _ = projects, gates, kw
            return run_result

    return _Fake


def _fake_fixer_cls(
    run_result: r[list[str]],
) -> type:
    class _Fake:
        def __init__(self, **_kw: t.Scalar) -> None:
            _ = _kw

        def run(
            self, _projects: list[str] | None = None, **kw: t.Scalar
        ) -> r[list[str]]:
            _ = _projects, kw
            return run_result

    return _Fake


class TestWorkspaceCheckCLI:
    def test_no_projects_error(self) -> None:
        tm.that(ws_mod.main([]), eq=1)

    def test_with_projects_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ok_result = r[list[SimpleNamespace]].ok([SimpleNamespace(passed=True)])
        monkeypatch.setattr(
            ws_mod,
            "FlextInfraWorkspaceChecker",
            _fake_checker_cls(["lint"], ok_result),
        )
        tm.that(ws_mod.main(["p1", "--gates", "lint"]), eq=0)

    def test_with_projects_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ok_result = r[list[SimpleNamespace]].ok([SimpleNamespace(passed=False)])
        monkeypatch.setattr(
            ws_mod,
            "FlextInfraWorkspaceChecker",
            _fake_checker_cls(["lint"], ok_result),
        )
        tm.that(ws_mod.main(["p1", "--gates", "lint"]), eq=1)

    def test_run_projects_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fail_result = r[list[SimpleNamespace]].fail("error")
        monkeypatch.setattr(
            ws_mod,
            "FlextInfraWorkspaceChecker",
            _fake_checker_cls(["lint"], fail_result),
        )
        tm.that(ws_mod.main(["p1", "--gates", "lint"]), eq=2)


class TestFixPyrelfyCLI:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            fix_pyrefly_mod,
            "FlextInfraConfigFixer",
            _fake_fixer_cls(r[list[str]].ok([])),
        )
        tm.that(fix_pyrefly_mod.main([]), eq=0)

    def test_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            fix_pyrefly_mod,
            "FlextInfraConfigFixer",
            _fake_fixer_cls(r[list[str]].fail("error")),
        )
        tm.that(fix_pyrefly_mod.main([]), eq=1)


def _const_cli_result(code: int) -> Callable[[list[str] | None], int]:
    def _runner(argv: list[str] | None = None) -> int:
        _ = argv
        return code

    return _runner


_fake_run_cli_zero = _const_cli_result(0)
_fake_run_cli_42 = _const_cli_result(42)


class _FakeRuntime:
    @staticmethod
    def ensure_structlog_configured() -> None:
        pass


class TestCheckMainEntryPoint:
    def test_calls_run_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(check_main_mod, "FlextRuntime", _FakeRuntime)
        monkeypatch.setattr(check_main_mod, "run_cli", _fake_run_cli_zero)
        tm.that(check_main_mod.main(), eq=0)

    def test_returns_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(check_main_mod, "FlextRuntime", _FakeRuntime)
        monkeypatch.setattr(check_main_mod, "run_cli", _fake_run_cli_42)
        tm.that(check_main_mod.main(), eq=42)


class TestRunCLIExtended:
    def test_fix_pyrefly_config_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ws_mod,
            "FlextInfraConfigFixer",
            _fake_fixer_cls(r[list[str]].ok([])),
        )
        tm.that(run_cli(["fix-pyrefly-config"]), eq=0)

    def test_fix_pyrefly_config_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ws_mod,
            "FlextInfraConfigFixer",
            _fake_fixer_cls(r[list[str]].fail("error")),
        )
        tm.that(run_cli(["fix-pyrefly-config"]), eq=1)

    def test_no_command_prints_help(self) -> None:
        tm.that(run_cli([]), eq=1)

    def test_with_relative_reports_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = m.Infra.Check.GateResult(
            gate="lint", project="p", passed=True, errors=[], duration=0.0
        )
        gate_exec = GateExecution(result=gate, issues=[], raw_output="")
        project = ProjectResult(project="p", gates={"lint": gate_exec})
        ok_result = r[list[ProjectResult]].ok([project])
        monkeypatch.setattr(
            ws_mod,
            "FlextInfraWorkspaceChecker",
            _fake_checker_cls(["lint"], ok_result),
        )
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
        exit_code = run_cli([
            "run",
            "--gates",
            "lint",
            "--project",
            "p",
            "--reports-dir",
            "reports/check",
        ])
        tm.that(exit_code, eq=0)
