"""Tests for FlextCheckCli to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import SimpleNamespace

from _pytest.monkeypatch import MonkeyPatch
from flext_core.result import FlextResult as r
from flext_infra.check.services import WorkspaceChecker, run_cli


def test_resolve_gates_maps_type_alias() -> None:
    result = WorkspaceChecker.resolve_gates(["lint", "type", "lint"])

    assert result.is_success
    assert result.value == ["lint", "pyrefly"]


def test_run_cli_run_returns_zero_for_pass(monkeypatch: MonkeyPatch) -> None:
    def _fake_run_projects(
        self: WorkspaceChecker,
        projects: list[str],
        gates: list[str],
        *,
        reports_dir: object,
        fail_fast: bool,
    ) -> r[list[object]]:
        del self, projects, gates, reports_dir, fail_fast
        return r[list[object]].ok([SimpleNamespace(passed=True)])

    _ = monkeypatch.setattr(WorkspaceChecker, "run_projects", _fake_run_projects)

    exit_code = run_cli(["run", "--gates", "lint,type", "--project", "flext-core"])

    assert exit_code == 0
