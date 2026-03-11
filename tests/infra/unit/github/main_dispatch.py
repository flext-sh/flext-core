"""Tests for __main__.py _run_pr_workspace dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra.github import __main__ as github_main
from flext_tests import tm
from tests.infra.models import m
from tests.infra.unit.github._stubs import StubWorkspaceManager

run_pr_workspace = getattr(github_main, "_run_pr_workspace")


def _orch(*, fail: int = 0, total: int = 1) -> m.Infra.Github.PrOrchestrationResult:
    return m.Infra.Github.PrOrchestrationResult(
        total=total,
        success=max(total - fail, 0),
        fail=fail,
        results=(),
    )


class TestRunPrWorkspace:
    def _stub(
        self,
        *,
        orchestrate_returns: r[m.Infra.Github.PrOrchestrationResult] | None = None,
    ) -> StubWorkspaceManager:
        return StubWorkspaceManager(orchestrate_returns=orchestrate_returns)

    def test_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._stub(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].ok(
                _orch(fail=0),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        assert run_pr_workspace(["--workspace-root", str(tmp_path)]) == 0

    def test_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._stub(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].fail(
                "orchestration failed",
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        assert run_pr_workspace(["--workspace-root", str(tmp_path)]) == 1

    def test_with_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = self._stub(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].ok(
                _orch(fail=2, total=2),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        assert run_pr_workspace(["--workspace-root", str(tmp_path)]) == 1

    def test_with_projects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = self._stub(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].ok(
                _orch(fail=0, total=2),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        argv = ["--workspace-root", str(tmp_path), "--project", "p1", "--project", "p2"]
        assert run_pr_workspace(argv) == 0
        assert mgr.orchestrate_calls[0]["projects"] == ["p1", "p2"]

    def test_with_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = self._stub(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].ok(
                _orch(fail=0),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        run_pr_workspace([
            "--workspace-root",
            str(tmp_path),
            "--branch",
            "feature/test",
        ])
        assert mgr.orchestrate_calls[0]["branch"] == "feature/test"

    def test_with_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = self._stub(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].ok(
                _orch(fail=0),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        run_pr_workspace(["--workspace-root", str(tmp_path), "--checkpoint", "1"])
        assert mgr.orchestrate_calls[0]["checkpoint"] is True

    def test_with_fail_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = self._stub(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].ok(
                _orch(fail=0),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        run_pr_workspace(["--workspace-root", str(tmp_path), "--fail-fast", "1"])
        assert mgr.orchestrate_calls[0]["fail_fast"] is True

    def test_with_pr_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = self._stub(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].ok(
                _orch(fail=0),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        argv = [
            "--workspace-root",
            str(tmp_path),
            "--pr-action",
            "merge",
            "--pr-base",
            "main",
            "--pr-head",
            "feature/test",
        ]
        assert run_pr_workspace(argv) == 0
        pr_args = mgr.orchestrate_calls[0]["pr_args"]
        tm.that("action" in str(pr_args), eq=True)
        tm.that("base" in str(pr_args), eq=True)
        tm.that("head" in str(pr_args), eq=True)
