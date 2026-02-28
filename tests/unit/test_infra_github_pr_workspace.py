"""Tests for FlextInfraPrWorkspaceManager.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from flext_core import r
from flext_infra.github.pr_workspace import FlextInfraPrWorkspaceManager


class TestFlextInfraPrWorkspaceManager:
    """Test suite for FlextInfraPrWorkspaceManager."""

    def test_has_changes_true(self, tmp_path: Path) -> None:
        """Test detecting uncommitted changes in repository."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("M file.py\nA new.py")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.has_changes(tmp_path)
        assert result.is_success
        assert result.value is True

    def test_has_changes_false(self, tmp_path: Path) -> None:
        """Test when repository has no uncommitted changes."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.has_changes(tmp_path)
        assert result.is_success
        assert result.value is False

    def test_has_changes_command_failure(self, tmp_path: Path) -> None:
        """Test handling of git status command failure."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("not a git repository")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.has_changes(tmp_path)
        assert result.is_failure
        assert result.error

    def test_checkout_branch_success(self, tmp_path: Path) -> None:
        """Test successful branch checkout."""
        mock_runner = Mock()
        mock_runner.run.return_value = r[bool].ok(True)
        mock_git = Mock()
        mock_git.current_branch.return_value = r[str].ok("other")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "feature/test")
        assert result.is_success

    def test_checkout_branch_empty(self, tmp_path: Path) -> None:
        """Test checkout with empty branch returns True."""
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "")
        assert result.is_success
        assert result.value is True

    def test_checkout_branch_already_on(self, tmp_path: Path) -> None:
        """Test checkout when already on the branch."""
        mock_git = Mock()
        mock_git.current_branch.return_value = r[str].ok("feature")
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "feature")
        assert result.is_success
        assert result.value is True

    def test_checkout_branch_local_changes(self, tmp_path: Path) -> None:
        """Test checkout with local changes forces -B."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_git.current_branch.return_value = r[str].ok("other")
        mock_runner.run.return_value = r[bool].fail(
            "error: local changes would be overwritten"
        )
        mock_runner.run_checked.return_value = r[bool].ok(True)
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "feature")
        assert result.is_success

    def test_checkout_branch_fetch_from_origin(self, tmp_path: Path) -> None:
        """Test checkout fetches from origin when local branch missing."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_git.current_branch.return_value = r[str].ok("other")
        mock_runner.run.side_effect = [
            r[bool].fail("pathspec not found"),
            r[bool].ok(True),  # fetch succeeds
        ]
        mock_runner.run_checked.return_value = r[bool].ok(True)
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "feature")
        assert result.is_success

    def test_checkout_branch_fetch_fail_fallback(self, tmp_path: Path) -> None:
        """Test checkout when fetch also fails."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_git.current_branch.return_value = r[str].ok("other")
        mock_runner.run.side_effect = [
            r[bool].fail("pathspec not found"),
            r[bool].fail("fetch failed"),
        ]
        mock_runner.run_checked.return_value = r[bool].ok(True)
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "feature")
        assert result.is_success

    def test_default_initialization(self) -> None:
        """Test manager initializes with default dependencies."""
        manager = FlextInfraPrWorkspaceManager()
        assert manager._runner is not None
        assert manager._git is not None
        assert manager._selector is not None
        assert manager._reporting is not None


class TestCheckpoint:
    """Test checkpoint method."""

    def test_no_changes(self, tmp_path: Path) -> None:
        """Test checkpoint with no changes."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_success
        assert result.value is True

    def test_changes_full_flow(self, tmp_path: Path) -> None:
        """Test checkpoint with changes: add, commit, push."""
        mock_runner = Mock()
        mock_runner.capture.side_effect = [
            r[str].ok("M file.py"),  # has_changes
            r[str].ok("file.py"),  # staged diff
        ]
        mock_runner.run_checked.side_effect = [
            r[bool].ok(True),  # add
            r[bool].ok(True),  # commit
        ]
        mock_runner.run.return_value = r[bool].ok(True)  # push
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_success

    def test_changes_check_failure(self, tmp_path: Path) -> None:
        """Test checkpoint when changes check fails."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("git error")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_failure

    def test_add_failure(self, tmp_path: Path) -> None:
        """Test checkpoint when git add fails."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("M file.py")
        mock_runner.run_checked.return_value = r[bool].fail("add failed")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_failure

    def test_no_staged_after_add(self, tmp_path: Path) -> None:
        """Test checkpoint when nothing staged after add."""
        mock_runner = Mock()
        mock_runner.capture.side_effect = [
            r[str].ok("M file.py"),  # has_changes
            r[str].ok(""),  # staged diff empty
        ]
        mock_runner.run_checked.return_value = r[bool].ok(True)  # add
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_success
        assert result.value is True

    def test_commit_failure(self, tmp_path: Path) -> None:
        """Test checkpoint when commit fails."""
        mock_runner = Mock()
        mock_runner.capture.side_effect = [
            r[str].ok("M file.py"),
            r[str].ok("file.py"),
        ]
        mock_runner.run_checked.side_effect = [
            r[bool].ok(True),  # add
            r[bool].fail("commit failed"),  # commit
        ]
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_failure

    def test_push_fails_rebase_succeeds(self, tmp_path: Path) -> None:
        """Test push failure with successful rebase retry."""
        mock_runner = Mock()
        mock_runner.capture.side_effect = [
            r[str].ok("M file.py"),
            r[str].ok("file.py"),
        ]
        mock_runner.run_checked.side_effect = [
            r[bool].ok(True),  # add
            r[bool].ok(True),  # commit
            r[bool].ok(True),  # rebase
            r[bool].ok(True),  # push retry
        ]
        mock_runner.run.return_value = r[bool].fail("push rejected")  # first push
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_success

    def test_push_fails_rebase_fails(self, tmp_path: Path) -> None:
        """Test push failure with failed rebase."""
        mock_runner = Mock()
        mock_runner.capture.side_effect = [
            r[str].ok("M file.py"),
            r[str].ok("file.py"),
        ]
        mock_runner.run_checked.side_effect = [
            r[bool].ok(True),  # add
            r[bool].ok(True),  # commit
            r[bool].fail("rebase conflict"),  # rebase
        ]
        mock_runner.run.return_value = r[bool].fail("push rejected")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_failure


class TestRunPr:
    """Test run_pr method."""

    def test_root_repo(self, tmp_path: Path) -> None:
        """Test run_pr for root repository."""
        mock_runner = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=mock_reporting
        )
        result = manager.run_pr(tmp_path, tmp_path, {"action": "status"})
        assert result.is_success
        assert result.value["status"] == "OK"

    def test_subproject(self, tmp_path: Path) -> None:
        """Test run_pr for subproject."""
        mock_runner = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)
        sub = tmp_path / "sub"
        sub.mkdir()
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=mock_reporting
        )
        result = manager.run_pr(sub, tmp_path, {"action": "status"})
        assert result.is_success

    def test_command_failure(self, tmp_path: Path) -> None:
        """Test run_pr command failure."""
        mock_runner = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].fail("command error")
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=mock_reporting
        )
        result = manager.run_pr(tmp_path, tmp_path, {"action": "status"})
        assert result.is_failure

    def test_nonzero_exit(self, tmp_path: Path) -> None:
        """Test run_pr with non-zero exit code."""
        mock_runner = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(1)
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner, git=Mock(), selector=Mock(), reporting=mock_reporting
        )
        result = manager.run_pr(tmp_path, tmp_path, {"action": "status"})
        assert result.is_success
        assert result.value["exit_code"] == 1


class TestOrchestrate:
    """Test orchestrate method."""

    def test_all_success(self, tmp_path: Path) -> None:
        """Test orchestrate with all projects succeeding."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)
        mock_runner.capture.return_value = r[str].ok("")  # has_changes
        mock_runner.run.return_value = r[bool].ok(True)  # checkout

        proj = Mock()
        proj.path = tmp_path / "proj"
        proj.path.mkdir()
        mock_selector.resolve_projects.return_value = r[list].ok([proj])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=Mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, checkpoint=False, branch=""
        )
        assert result.is_success
        assert result.value["fail"] == 0

    def test_project_resolution_failure(self, tmp_path: Path) -> None:
        """Test orchestrate when project resolution fails."""
        mock_selector = Mock()
        mock_selector.resolve_projects.return_value = r[list].fail("no projects")
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=Mock(), selector=mock_selector, reporting=Mock()
        )
        result = manager.orchestrate(tmp_path)
        assert result.is_failure

    def test_fail_fast(self, tmp_path: Path) -> None:
        """Test orchestrate with fail_fast stopping on first failure."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(1)
        mock_runner.capture.return_value = r[str].ok("")
        mock_runner.run.return_value = r[bool].ok(True)

        proj1 = Mock()
        proj1.path = tmp_path / "p1"
        proj1.path.mkdir()
        proj2 = Mock()
        proj2.path = tmp_path / "p2"
        proj2.path.mkdir()
        mock_selector.resolve_projects.return_value = r[list].ok([proj1, proj2])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=Mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, fail_fast=True, checkpoint=False, branch=""
        )
        assert result.is_success
        assert result.value["fail"] >= 1

    def test_include_root(self, tmp_path: Path) -> None:
        """Test orchestrate includes root repository."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)
        mock_runner.capture.return_value = r[str].ok("")
        mock_runner.run.return_value = r[bool].ok(True)

        mock_selector.resolve_projects.return_value = r[list].ok([])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=Mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=True, checkpoint=False, branch=""
        )
        assert result.is_success
        assert result.value["total"] == 1

    def test_orchestrate_with_checkpoint(self, tmp_path: Path) -> None:
        """Test orchestrate with checkpoint enabled."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)
        mock_runner.capture.return_value = r[str].ok("M file.py")  # has_changes
        mock_runner.run.return_value = r[bool].ok(True)  # checkout

        proj = Mock()
        proj.path = tmp_path / "proj"
        proj.path.mkdir()
        mock_selector.resolve_projects.return_value = r[list].ok([proj])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=Mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, checkpoint=True, branch="test-branch"
        )
        assert result.is_success
        # checkpoint should be called, which calls has_changes, add, commit, push
        assert mock_runner.capture.called

    def test_orchestrate_failure_handling(self, tmp_path: Path) -> None:
        """Test orchestrate failure handling with fail_fast."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].fail("command error")
        mock_runner.capture.return_value = r[str].ok("")
        mock_runner.run.return_value = r[bool].ok(True)

        proj = Mock()
        proj.path = tmp_path / "proj"
        proj.path.mkdir()
        mock_selector.resolve_projects.return_value = r[list].ok([proj])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=Mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, fail_fast=True, checkpoint=False, branch=""
        )
        assert result.is_success
        assert result.value["fail"] == 1


class TestStaticMethods:
    """Test static utility methods."""

    def test_repo_display_name_root(self, tmp_path: Path) -> None:
        """Test display name for root repository."""
        result = FlextInfraPrWorkspaceManager._repo_display_name(tmp_path, tmp_path)
        assert result == tmp_path.name

    def test_repo_display_name_subproject(self, tmp_path: Path) -> None:
        """Test display name for subproject."""
        sub = tmp_path / "my-project"
        sub.mkdir()
        result = FlextInfraPrWorkspaceManager._repo_display_name(sub, tmp_path)
        assert result == "my-project"

    def test_build_root_command(self, tmp_path: Path) -> None:
        """Test root command building."""
        cmd = FlextInfraPrWorkspaceManager._build_root_command(
            tmp_path, {"action": "create", "head": "feature", "title": "Test"}
        )
        assert "python" in cmd
        assert "--action" in cmd
        assert "create" in cmd
        assert "--head" in cmd
        assert "feature" in cmd
        assert "--title" in cmd
        assert "Test" in cmd

    def test_build_subproject_command(self, tmp_path: Path) -> None:
        """Test subproject command building."""
        cmd = FlextInfraPrWorkspaceManager._build_subproject_command(
            tmp_path, {"action": "status", "head": "feat"}
        )
        assert "make" in cmd
        assert "-C" in cmd
        assert "pr" in cmd
        assert "PR_ACTION=status" in cmd
        assert "PR_HEAD=feat" in cmd

    def test_build_root_command_defaults(self, tmp_path: Path) -> None:
        """Test root command with default values."""
        cmd = FlextInfraPrWorkspaceManager._build_root_command(tmp_path, {})
        assert "--action" in cmd
        assert "status" in cmd

    def test_build_subproject_command_no_optional(self, tmp_path: Path) -> None:
        """Test subproject command without optional keys."""
        cmd = FlextInfraPrWorkspaceManager._build_subproject_command(tmp_path, {})
        assert "make" in cmd
        # head, number, title, body are optional — should not appear
        assert not [c for c in cmd if c.startswith("PR_HEAD=")]
