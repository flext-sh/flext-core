"""Tests for CLI entry points: workspace_check, fix_pyrefly_config, check __main__, run_cli.

Uses monkeypatch to inject controlled service behavior instead of unittest.mock.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_core import r
from flext_infra import m
from flext_infra.check.__main__ import main as check_main
from flext_infra.check.fix_pyrefly_config import main as fix_pyrefly_main
from flext_infra.check.services import (
    FlextInfraConfigFixer,
    FlextInfraWorkspaceChecker,
    _GateExecution,
    _ProjectResult,
    run_cli,
)
from flext_infra.check.workspace_check import main as workspace_check_main
from flext_tests import tm

from ._stubs import Spy


def _fake_checker_cls(
    parse_result: list[str],
    run_result: r[list[SimpleNamespace]] | r[list[_ProjectResult]],
) -> type:
    """Build a fake FlextInfraWorkspaceChecker class."""

    class _Fake:
        def __init__(self, **_kw: object) -> None:
            self._run_spy = Spy()

        def parse_gate_csv(self, _gates: str) -> list[str]:
            return parse_result

        def run_projects(
            self,
            _projects: list[str],
            _gates: list[str],
            **_kw: object,
        ) -> r[list[SimpleNamespace]] | r[list[_ProjectResult]]:
            self._run_spy(*_projects, **_kw)
            return run_result

    return _Fake


def _fake_fixer_cls(
    run_result: r[list[str]],
) -> type:
    """Build a fake FlextInfraConfigFixer class."""

    class _Fake:
        def __init__(self, **_kw: object) -> None:
            self._run_spy = Spy()

        def run(self, _projects: list[str] | None = None, **kw: object) -> r[list[str]]:
            self._run_spy(**kw)
            return run_result

    return _Fake


class TestWorkspaceCheckCLI:
    """Test workspace_check CLI entry point."""

    def test_no_projects_error(self) -> None:
        tm.that(workspace_check_main([]), eq=1)

    def test_with_projects_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ok_result = r[list[SimpleNamespace]].ok([SimpleNamespace(passed=True)])
        monkeypatch.setattr(
            "flext_infra.check.workspace_check.FlextInfraWorkspaceChecker",
            _fake_checker_cls(["lint"], ok_result),
        )
        tm.that(workspace_check_main(["p1", "--gates", "lint"]), eq=0)

    def test_with_projects_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ok_result = r[list[SimpleNamespace]].ok([SimpleNamespace(passed=False)])
        monkeypatch.setattr(
            "flext_infra.check.workspace_check.FlextInfraWorkspaceChecker",
            _fake_checker_cls(["lint"], ok_result),
        )
        tm.that(workspace_check_main(["p1", "--gates", "lint"]), eq=1)

    def test_run_projects_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fail_result = r[list[SimpleNamespace]].fail("error")
        monkeypatch.setattr(
            "flext_infra.check.workspace_check.FlextInfraWorkspaceChecker",
            _fake_checker_cls(["lint"], fail_result),
        )
        tm.that(workspace_check_main(["p1", "--gates", "lint"]), eq=2)


class TestFixPyrelfyCLI:
    """Test fix_pyrefly_config CLI entry point."""

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "flext_infra.check.fix_pyrefly_config.FlextInfraConfigFixer",
            _fake_fixer_cls(r[list[str]].ok([])),
        )
        tm.that(fix_pyrefly_main([]), eq=0)

    def test_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "flext_infra.check.fix_pyrefly_config.FlextInfraConfigFixer",
            _fake_fixer_cls(r[list[str]].fail("error")),
        )
        tm.that(fix_pyrefly_main([]), eq=1)


class TestCheckMainEntryPoint:
    """Test check __main__ entry point."""

    def test_calls_run_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "flext_infra.check.__main__.FlextRuntime.ensure_structlog_configured",
            lambda: None,
        )
        monkeypatch.setattr("flext_infra.check.__main__.run_cli", lambda _args: 0)
        tm.that(check_main(), eq=0)

    def test_returns_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "flext_infra.check.__main__.FlextRuntime.ensure_structlog_configured",
            lambda: None,
        )
        monkeypatch.setattr("flext_infra.check.__main__.run_cli", lambda _args: 42)
        tm.that(check_main(), eq=42)


class TestRunCLIExtended:
    """Extended run_cli tests (fix-pyrefly-config, no command, reports dir)."""

    def test_fix_pyrefly_config_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "flext_infra.check.workspace_check.FlextInfraConfigFixer",
            _fake_fixer_cls(r[list[str]].ok([])),
        )
        tm.that(run_cli(["fix-pyrefly-config"]), eq=0)

    def test_fix_pyrefly_config_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "flext_infra.check.workspace_check.FlextInfraConfigFixer",
            _fake_fixer_cls(r[list[str]].fail("error")),
        )
        tm.that(run_cli(["fix-pyrefly-config"]), eq=1)

    def test_no_command_prints_help(self) -> None:
        tm.that(run_cli([]), eq=1)

    def test_with_relative_reports_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = m.Infra.Check.GateResult(gate="lint", project="p", passed=True)
        gate_exec = _GateExecution(result=gate, issues=[])
        project = _ProjectResult(project="p", gates={"lint": gate_exec})
        ok_result = r[list[_ProjectResult]].ok([project])
        monkeypatch.setattr(
            "flext_infra.check.workspace_check.FlextInfraWorkspaceChecker",
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
