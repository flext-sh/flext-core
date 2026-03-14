"""Tests for __main__.py main() dispatch integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flext_core import r
from flext_infra.github import __main__ as github_main
from flext_infra.github.workflows import SyncOperation
from tests.infra.models import m
from tests.infra.unit.github._stubs import (
    StubLinter,
    StubSyncer,
    StubWorkspaceManager,
)

main = github_main.main


def _orch(*, fail: int = 0, total: int = 1) -> m.Infra.Github.PrOrchestrationResult:
    return m.Infra.Github.PrOrchestrationResult(
        total=total,
        success=max(total - fail, 0),
        fail=fail,
        results=(),
    )


class TestMain:
    def test_help_flag(self) -> None:
        original = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "-h"]
            assert main() == 0
        finally:
            sys.argv = original

    def test_no_args(self) -> None:
        original = sys.argv.copy()
        try:
            sys.argv = ["flext-infra"]
            assert main() == 1
        finally:
            sys.argv = original

    def test_workflows_subcommand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        syncer = StubSyncer(sync_returns=r[list[SyncOperation]].ok([]))
        monkeypatch.setattr(github_main, "FlextInfraWorkflowSyncer", lambda: syncer)
        original = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "workflows", "--workspace-root", str(tmp_path)]
            assert main() == 0
        finally:
            sys.argv = original

    def test_lint_subcommand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        linter = StubLinter(
            lint_returns=r[m.Infra.Github.WorkflowLintResult].ok(
                m.Infra.Github.WorkflowLintResult(status="ok"),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraWorkflowLinter", lambda: linter)
        original = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "lint", "--root", str(tmp_path)]
            assert main() == 0
        finally:
            sys.argv = original

    def test_pr_subcommand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(github_main, "pr_main", lambda: 0)
        original = sys.argv.copy()
        try:
            sys.argv = [
                "flext-infra",
                "pr",
                "--repo-root",
                str(tmp_path),
                "--action",
                "status",
            ]
            assert main() == 0
        finally:
            sys.argv = original

    def test_pr_workspace_subcommand(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mgr = StubWorkspaceManager(
            orchestrate_returns=r[m.Infra.Github.PrOrchestrationResult].ok(
                _orch(fail=0)
            ),
        )
        monkeypatch.setattr(github_main, "FlextInfraPrWorkspaceManager", lambda: mgr)
        original = sys.argv.copy()
        try:
            sys.argv = [
                "flext-infra",
                "pr-workspace",
                "--workspace-root",
                str(tmp_path),
            ]
            assert main() == 0
        finally:
            sys.argv = original

    def test_unknown_subcommand(self) -> None:
        original = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "unknown"]
            assert main() == 1
        finally:
            sys.argv = original

    def test_ensures_structlog_configured(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called: list[bool] = []
        monkeypatch.setattr(
            github_main,
            "FlextRuntime",
            type(
                "FakeRuntime",
                (),
                {
                    "ensure_structlog_configured": staticmethod(
                        lambda: called.append(True)
                    ),
                },
            ),
        )
        linter = StubLinter(
            lint_returns=r[m.Infra.Github.WorkflowLintResult].ok(
                m.Infra.Github.WorkflowLintResult(status="ok"),
            )
        )
        monkeypatch.setattr(github_main, "FlextInfraWorkflowLinter", lambda: linter)
        original = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "lint", "--root", str(tmp_path)]
            main()
            assert called
        finally:
            sys.argv = original

    def test_workflows_iterates_operations(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ops = [
            SyncOperation(project="p1", path="ci.yml", action="create", reason="new"),
            SyncOperation(
                project="p2",
                path="ci.yml",
                action="update",
                reason="changed",
            ),
        ]
        syncer = StubSyncer(sync_returns=r[list[SyncOperation]].ok(ops))
        monkeypatch.setattr(github_main, "FlextInfraWorkflowSyncer", lambda: syncer)
        original = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "workflows", "--workspace-root", str(tmp_path)]
            assert main() == 0
        finally:
            sys.argv = original
