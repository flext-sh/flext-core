"""Tests for FlextCheckCli to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from _pytest.monkeypatch import MonkeyPatch
from flext_core import FlextResult as r
from flext_infra.check.services import FlextInfraWorkspaceChecker, run_cli


def test_resolve_gates_maps_type_alias() -> None:
    result = FlextInfraWorkspaceChecker.resolve_gates(["lint", "type", "lint"])

    assert result.is_success
    assert result.value == ["lint", "pyrefly"]


def test_run_cli_run_returns_zero_for_pass(monkeypatch: MonkeyPatch) -> None:
    def _fake_run_projects(
        self: FlextInfraWorkspaceChecker,
        projects: list[str],
        gates: list[str],
        *,
        reports_dir: Path | None,
        fail_fast: bool,
    ) -> r[list[SimpleNamespace]]:
        del self, projects, gates, reports_dir, fail_fast
        return r[list[SimpleNamespace]].ok([SimpleNamespace(passed=True)])

    _ = monkeypatch.setattr(
        FlextInfraWorkspaceChecker, "run_projects", _fake_run_projects
    )

    exit_code = run_cli(["run", "--gates", "lint,type", "--project", "flext-core"])

    assert exit_code == 0


def test_run_cli_run_returns_one_for_fail(monkeypatch: MonkeyPatch) -> None:
    """Test that run_cli returns 1 when projects fail checks."""

    def _fake_run_projects(
        self: FlextInfraWorkspaceChecker,
        projects: list[str],
        gates: list[str],
        *,
        reports_dir: Path | None,
        fail_fast: bool,
    ) -> r[list[SimpleNamespace]]:
        del self, projects, gates, reports_dir, fail_fast
        return r[list[SimpleNamespace]].ok([SimpleNamespace(passed=False)])

    _ = monkeypatch.setattr(
        FlextInfraWorkspaceChecker, "run_projects", _fake_run_projects
    )

    exit_code = run_cli(["run", "--gates", "lint", "--project", "flext-core"])

    assert exit_code == 1


def test_run_cli_run_returns_two_for_error(monkeypatch: MonkeyPatch) -> None:
    """Test that run_cli returns 2 when run_projects fails."""

    def _fake_run_projects(
        self: FlextInfraWorkspaceChecker,
        projects: list[str],
        gates: list[str],
        *,
        reports_dir: Path | None,
        fail_fast: bool,
    ) -> r[list[SimpleNamespace]]:
        del self, projects, gates, reports_dir, fail_fast
        return r[list[SimpleNamespace]].fail("test error")

    _ = monkeypatch.setattr(
        FlextInfraWorkspaceChecker, "run_projects", _fake_run_projects
    )

    exit_code = run_cli(["run", "--gates", "lint", "--project", "flext-core"])

    assert exit_code == 2


def test_run_cli_with_multiple_projects(monkeypatch: MonkeyPatch) -> None:
    """Test that run_cli handles multiple projects."""
    captured_projects: list[str] = []

    def _fake_run_projects(
        self: FlextInfraWorkspaceChecker,
        projects: list[str],
        gates: list[str],
        *,
        reports_dir: Path | None,
        fail_fast: bool,
    ) -> r[list[SimpleNamespace]]:
        del self, gates, reports_dir, fail_fast
        captured_projects.extend(projects)
        return r[list[SimpleNamespace]].ok([SimpleNamespace(passed=True)])

    _ = monkeypatch.setattr(
        FlextInfraWorkspaceChecker, "run_projects", _fake_run_projects
    )

    exit_code = run_cli([
        "run",
        "--gates",
        "lint",
        "--project",
        "proj1",
        "--project",
        "proj2",
    ])

    assert exit_code == 0
    assert "proj1" in captured_projects
    assert "proj2" in captured_projects


def test_run_cli_with_fail_fast_flag(monkeypatch: MonkeyPatch) -> None:
    """Test that run_cli passes fail_fast flag to run_projects."""
    captured_fail_fast: list[bool] = []

    def _fake_run_projects(
        self: FlextInfraWorkspaceChecker,
        projects: list[str],
        gates: list[str],
        *,
        reports_dir: Path | None,
        fail_fast: bool,
    ) -> r[list[SimpleNamespace]]:
        del self, projects, gates, reports_dir
        captured_fail_fast.append(fail_fast)
        return r[list[SimpleNamespace]].ok([SimpleNamespace(passed=True)])

    _ = monkeypatch.setattr(
        FlextInfraWorkspaceChecker, "run_projects", _fake_run_projects
    )

    exit_code = run_cli([
        "run",
        "--gates",
        "lint",
        "--fail-fast",
        "--project",
        "flext-core",
    ])

    assert exit_code == 0
    assert captured_fail_fast[0] is True
