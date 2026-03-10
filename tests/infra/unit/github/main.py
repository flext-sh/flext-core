"""Tests for flext_infra.github.__main__ CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from flext_core import r
from flext_infra import m
from flext_infra.github import __main__ as github_main
from flext_infra.github.workflows import SyncOperation
from tests.infra.unit.github._stubs import (
    StubLinter,
    StubSyncer,
)

run_lint = getattr(github_main, "_run_lint")
run_pr = getattr(github_main, "_run_pr")
run_pr_workspace = getattr(github_main, "_run_pr_workspace")
run_workflows = getattr(github_main, "_run_workflows")
main = github_main.main


def _orchestration_result(
    *,
    fail: int = 0,
    total: int = 1,
) -> m.Infra.Github.PrOrchestrationResult:
    return m.Infra.Github.PrOrchestrationResult(
        total=total,
        success=max(total - fail, 0),
        fail=fail,
        results=(),
    )


class TestRunWorkflows:
    def test_success(self, tmp_path: Path, monkeypatch: object) -> None:
        syncer = StubSyncer(sync_returns=r[list[SyncOperation]].ok([]))
        monkeypatch.setattr(github_main, "FlextInfraWorkflowSyncer", lambda: syncer)
        assert run_workflows(["--workspace-root", str(tmp_path)]) == 0

    def test_failure(self, tmp_path: Path, monkeypatch: object) -> None:
        syncer = StubSyncer(sync_returns=r[list[SyncOperation]].fail("sync failed"))
        monkeypatch.setattr(github_main, "FlextInfraWorkflowSyncer", lambda: syncer)
        assert run_workflows(["--workspace-root", str(tmp_path)]) == 1

    def test_with_apply_flag(self, tmp_path: Path, monkeypatch: object) -> None:
        syncer = StubSyncer(sync_returns=r[list[SyncOperation]].ok([]))
        monkeypatch.setattr(github_main, "FlextInfraWorkflowSyncer", lambda: syncer)
        run_workflows(["--workspace-root", str(tmp_path), "--apply"])
        assert syncer.sync_workspace_calls[0]["apply"] is True

    def test_with_prune_flag(self, tmp_path: Path, monkeypatch: object) -> None:
        syncer = StubSyncer(sync_returns=r[list[SyncOperation]].ok([]))
        monkeypatch.setattr(github_main, "FlextInfraWorkflowSyncer", lambda: syncer)
        run_workflows(["--workspace-root", str(tmp_path), "--prune"])
        assert syncer.sync_workspace_calls[0]["prune"] is True

    def test_with_report(self, tmp_path: Path, monkeypatch: object) -> None:
        syncer = StubSyncer(sync_returns=r[list[SyncOperation]].ok([]))
        monkeypatch.setattr(github_main, "FlextInfraWorkflowSyncer", lambda: syncer)
        report = tmp_path / "report.json"
        run_workflows(["--workspace-root", str(tmp_path), "--report", str(report)])
        assert syncer.sync_workspace_calls[0]["report_path"] == report


class TestRunLint:
    def _lint_ok(self) -> StubLinter:
        return StubLinter(
            lint_returns=r[m.Infra.Github.WorkflowLintResult].ok(
                m.Infra.Github.WorkflowLintResult.model_validate({"status": "ok"}),
            )
        )

    def test_success(self, tmp_path: Path, monkeypatch: object) -> None:
        linter = self._lint_ok()
        monkeypatch.setattr(github_main, "FlextInfraWorkflowLinter", lambda: linter)
        assert run_lint(["--root", str(tmp_path)]) == 0

    def test_failure(self, tmp_path: Path, monkeypatch: object) -> None:
        linter = StubLinter(lint_returns=r[object].fail("lint failed"))
        monkeypatch.setattr(github_main, "FlextInfraWorkflowLinter", lambda: linter)
        assert run_lint(["--root", str(tmp_path)]) == 1

    def test_with_report(self, tmp_path: Path, monkeypatch: object) -> None:
        linter = self._lint_ok()
        monkeypatch.setattr(github_main, "FlextInfraWorkflowLinter", lambda: linter)
        report = tmp_path / "report.json"
        run_lint(["--root", str(tmp_path), "--report", str(report)])
        assert linter.lint_calls[0]["report_path"] == report

    def test_with_strict_flag(self, tmp_path: Path, monkeypatch: object) -> None:
        linter = self._lint_ok()
        monkeypatch.setattr(github_main, "FlextInfraWorkflowLinter", lambda: linter)
        run_lint(["--root", str(tmp_path), "--strict"])
        assert linter.lint_calls[0]["strict"] is True


class TestRunPr:
    def test_delegates_to_pr_main(self, monkeypatch: object) -> None:
        monkeypatch.setattr(github_main, "pr_main", lambda: 0)
        assert run_pr(["--repo-root", "/tmp", "--action", "status"]) == 0

    def test_sets_sys_argv(self, monkeypatch: object) -> None:
        original = sys.argv.copy()
        try:
            monkeypatch.setattr(github_main, "pr_main", lambda: 0)
            run_pr(["--repo-root", "/tmp", "--action", "status"])
            assert sys.argv[0] == "flext-infra github pr"
            assert "--repo-root" in sys.argv
        finally:
            sys.argv = original
