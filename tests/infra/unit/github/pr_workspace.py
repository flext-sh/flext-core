"""Tests for FlextInfraPrWorkspaceManager — core methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra.github import pr_workspace as pw_mod
from flext_infra.github.pr_workspace import FlextInfraPrWorkspaceManager
from flext_tests import tm
from tests.infra.unit.github._stubs import StubProjectInfo, StubReporting, StubRunner


class TestFlextInfraPrWorkspaceManager:
    def test_has_changes_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test detecting uncommitted changes in repository."""
        monkeypatch.setattr(
            pw_mod.u.Infra, "git_has_changes", lambda _root: r[bool].ok(True)
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=StubProjectInfo(), reporting=StubReporting()
        )
        result = manager.has_changes(tmp_path)
        tm.ok(result, eq=True)

    def test_has_changes_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test when repository has no uncommitted changes."""
        monkeypatch.setattr(
            pw_mod.u.Infra, "git_has_changes", lambda _root: r[bool].ok(False)
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=StubProjectInfo(), reporting=StubReporting()
        )
        result = manager.has_changes(tmp_path)
        tm.ok(result, eq=False)

    def test_has_changes_command_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of git status command failure."""
        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_has_changes",
            lambda _root: r[bool].fail("not a git repository"),
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=StubProjectInfo(), reporting=StubReporting()
        )
        result = manager.has_changes(tmp_path)
        tm.fail(result)

    def test_checkout_branch_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test successful branch checkout."""
        calls: list[tuple[Path, str]] = []

        def _checkout(root: Path, branch: str) -> r[bool]:
            calls.append((root, branch))
            return r[bool].ok(True)

        monkeypatch.setattr(pw_mod.u.Infra, "git_checkout", _checkout)
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=StubProjectInfo(), reporting=StubReporting()
        )
        result = manager.checkout_branch(tmp_path, "feature/test")
        tm.ok(result)
        tm.that(len(calls), eq=1)
        tm.that(str(calls[0][0]), eq=str(tmp_path))
        tm.that(calls[0][1], eq="feature/test")

    def test_checkout_branch_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test checkout with empty branch is a no-op."""
        calls: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_checkout",
            lambda root, branch: (calls.append((root, branch)), r[bool].ok(True))[1],
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=StubProjectInfo(), reporting=StubReporting()
        )
        result = manager.checkout_branch(tmp_path, "")
        tm.ok(result, eq=True)
        tm.that(len(calls), eq=0)

    def test_checkout_branch_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test checkout failure propagation."""
        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_checkout",
            lambda _root, _branch: r[bool].fail("checkout failed"),
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=StubProjectInfo(), reporting=StubReporting()
        )
        result = manager.checkout_branch(tmp_path, "feature")
        tm.fail(result)

    def test_default_initialization(self) -> None:
        """Test manager initializes with default dependencies."""
        manager = FlextInfraPrWorkspaceManager()
        tm.that(getattr(manager, "_runner", None), none=False)
        tm.that(getattr(manager, "_selector", None), none=False)
        tm.that(getattr(manager, "_reporting", None), none=False)


class TestCheckpoint:
    def test_no_changes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test checkpoint exits when repository has no changes."""
        git_add_calls: list[Path] = []
        monkeypatch.setattr(
            pw_mod.u.Infra, "git_has_changes", lambda _root: r[bool].ok(False)
        )
        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_add",
            lambda root, *_paths: (git_add_calls.append(root), r[bool].ok(True))[1],
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=StubProjectInfo(), reporting=StubReporting()
        )
        result = manager.checkpoint(tmp_path, "feature")
        tm.ok(result, eq=True)
        tm.that(len(git_add_calls), eq=0)

    def test_checkpoint_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test checkpoint failure propagation."""
        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_has_changes",
            lambda _root: r[bool].fail("git error"),
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=StubProjectInfo(), reporting=StubReporting()
        )
        result = manager.checkpoint(tmp_path, "feature")
        tm.fail(result)


class TestRunPr:
    def test_root_repo(self, tmp_path: Path) -> None:
        """Test run_pr for root repository."""
        runner = StubRunner(run_to_file_returns=[r[int].ok(0)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=StubProjectInfo(), reporting=reporting
        )
        result = manager.run_pr(tmp_path, tmp_path, {"action": "status"})
        value = tm.ok(result)
        tm.that(value.status, eq="OK")

    def test_subproject(self, tmp_path: Path) -> None:
        """Test run_pr for subproject."""
        runner = StubRunner(run_to_file_returns=[r[int].ok(0)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        sub = tmp_path / "sub"
        sub.mkdir()
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=StubProjectInfo(), reporting=reporting
        )
        result = manager.run_pr(sub, tmp_path, {"action": "status"})
        tm.ok(result)

    def test_command_failure(self, tmp_path: Path) -> None:
        """Test run_pr command failure."""
        runner = StubRunner(run_to_file_returns=[r[int].fail("command error")])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=StubProjectInfo(), reporting=reporting
        )
        result = manager.run_pr(tmp_path, tmp_path, {"action": "status"})
        tm.fail(result)

    def test_nonzero_exit(self, tmp_path: Path) -> None:
        """Test run_pr with non-zero exit code."""
        runner = StubRunner(run_to_file_returns=[r[int].ok(1)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=StubProjectInfo(), reporting=reporting
        )
        result = manager.run_pr(tmp_path, tmp_path, {"action": "status"})
        value = tm.ok(result)
        tm.that(value.exit_code, eq=1)
