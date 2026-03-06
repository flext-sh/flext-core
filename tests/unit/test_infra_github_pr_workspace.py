"""Tests for FlextInfraPrWorkspaceManager.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from flext_core import r
from flext_infra.github.pr_workspace import FlextInfraPrWorkspaceManager


def _make_git_mock(**overrides: object) -> Mock:
    """Create a properly configured git service mock.

    Since pr_workspace delegates to high-level methods (smart_checkout,
    checkpoint, has_changes), those are the main ones we mock here.
    """
    git = Mock()
    git.smart_checkout.return_value = r[bool].ok(True)
    git.checkpoint.return_value = r[bool].ok(True)
    git.has_changes.return_value = r[bool].ok(False)
    for key, val in overrides.items():
        setattr(git, key, val)
    return git


class TestFlextInfraPrWorkspaceManager:
    """Test suite for FlextInfraPrWorkspaceManager."""

    def test_has_changes_true(self, tmp_path: Path) -> None:
        """Test detecting uncommitted changes in repository."""
        mock_git = _make_git_mock()
        mock_git.has_changes.return_value = r[bool].ok(True)
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.has_changes(tmp_path)
        assert result.is_success
        assert result.value is True

    def test_has_changes_false(self, tmp_path: Path) -> None:
        """Test when repository has no uncommitted changes."""
        mock_git = _make_git_mock()
        mock_git.has_changes.return_value = r[bool].ok(False)
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.has_changes(tmp_path)
        assert result.is_success
        assert result.value is False

    def test_has_changes_command_failure(self, tmp_path: Path) -> None:
        """Test handling of git status command failure."""
        mock_git = _make_git_mock()
        mock_git.has_changes.return_value = r[bool].fail("not a git repository")
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.has_changes(tmp_path)
        assert result.is_failure
        assert result.error

    def test_checkout_branch_success(self, tmp_path: Path) -> None:
        """Test successful branch checkout."""
        mock_git = _make_git_mock()
        mock_git.smart_checkout.return_value = r[bool].ok(True)
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "feature/test")
        assert result.is_success
        mock_git.smart_checkout.assert_called_once_with(tmp_path, "feature/test")

    def test_checkout_branch_empty(self, tmp_path: Path) -> None:
        """Test checkout with empty branch delegates to smart_checkout."""
        mock_git = _make_git_mock()
        mock_git.smart_checkout.return_value = r[bool].ok(True)
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "")
        assert result.is_success
        assert result.value is True

    def test_checkout_branch_already_on(self, tmp_path: Path) -> None:
        """Test checkout when already on the branch."""
        mock_git = _make_git_mock()
        mock_git.smart_checkout.return_value = r[bool].ok(True)
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "feature")
        assert result.is_success
        assert result.value is True

    def test_checkout_branch_failure(self, tmp_path: Path) -> None:
        """Test checkout failure propagation."""
        mock_git = _make_git_mock()
        mock_git.smart_checkout.return_value = r[bool].fail("checkout failed")
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkout_branch(tmp_path, "feature")
        assert result.is_failure

    def test_default_initialization(self) -> None:
        """Test manager initializes with default dependencies."""
        manager = FlextInfraPrWorkspaceManager()
        assert getattr(manager, "_runner", None) is not None
        assert getattr(manager, "_git", None) is not None
        assert getattr(manager, "_selector", None) is not None
        assert getattr(manager, "_reporting", None) is not None


class TestCheckpoint:
    """Test checkpoint method."""

    def test_no_changes(self, tmp_path: Path) -> None:
        """Test checkpoint delegates to git service."""
        mock_git = _make_git_mock()
        mock_git.checkpoint.return_value = r[bool].ok(True)
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
        )
        result = manager.checkpoint(tmp_path, "feature")
        assert result.is_success
        assert result.value is True
        mock_git.checkpoint.assert_called_once_with(tmp_path, "feature")

    def test_checkpoint_failure(self, tmp_path: Path) -> None:
        """Test checkpoint failure propagation."""
        mock_git = _make_git_mock()
        mock_git.checkpoint.return_value = r[bool].fail("git error")
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(), git=mock_git, selector=Mock(), reporting=Mock()
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
            runner=mock_runner,
            git=_make_git_mock(),
            selector=Mock(),
            reporting=mock_reporting,
        )
        result = manager.run_pr(tmp_path, tmp_path, {"action": "status"})
        assert result.is_success
        assert result.value.status == "OK"

    def test_subproject(self, tmp_path: Path) -> None:
        """Test run_pr for subproject."""
        mock_runner = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)
        sub = tmp_path / "sub"
        sub.mkdir()
        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=_make_git_mock(),
            selector=Mock(),
            reporting=mock_reporting,
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
            runner=mock_runner,
            git=_make_git_mock(),
            selector=Mock(),
            reporting=mock_reporting,
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
            runner=mock_runner,
            git=_make_git_mock(),
            selector=Mock(),
            reporting=mock_reporting,
        )
        result = manager.run_pr(tmp_path, tmp_path, {"action": "status"})
        assert result.is_success
        assert result.value.exit_code == 1


class TestOrchestrate:
    """Test orchestrate method."""

    def test_all_success(self, tmp_path: Path) -> None:
        """Test orchestrate with all projects succeeding."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)

        proj = Mock()
        proj.path = tmp_path / "proj"
        proj.path.mkdir()
        mock_selector.resolve_projects.return_value = r[list[Mock]].ok([proj])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=_make_git_mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, checkpoint=False, branch=""
        )
        assert result.is_success
        assert result.value.fail == 0

    def test_project_resolution_failure(self, tmp_path: Path) -> None:
        """Test orchestrate when project resolution fails."""
        mock_selector = Mock()
        mock_selector.resolve_projects.return_value = r[list[Mock]].fail("no projects")
        manager = FlextInfraPrWorkspaceManager(
            runner=Mock(),
            git=_make_git_mock(),
            selector=mock_selector,
            reporting=Mock(),
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

        proj1 = Mock()
        proj1.path = tmp_path / "p1"
        proj1.path.mkdir()
        proj2 = Mock()
        proj2.path = tmp_path / "p2"
        proj2.path.mkdir()
        mock_selector.resolve_projects.return_value = r[list[Mock]].ok([proj1, proj2])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=_make_git_mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, fail_fast=True, checkpoint=False, branch=""
        )
        assert result.is_success
        assert result.value.fail >= 1

    def test_include_root(self, tmp_path: Path) -> None:
        """Test orchestrate includes root repository."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)

        mock_selector.resolve_projects.return_value = r[list[Mock]].ok([])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=_make_git_mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=True, checkpoint=False, branch=""
        )
        assert result.is_success
        assert result.value.total == 1

    def test_orchestrate_with_checkpoint(self, tmp_path: Path) -> None:
        """Test orchestrate with checkpoint enabled calls git service."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].ok(0)

        mock_git = _make_git_mock()

        proj = Mock()
        proj.path = tmp_path / "proj"
        proj.path.mkdir()
        mock_selector.resolve_projects.return_value = r[list[Mock]].ok([proj])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=mock_git,
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, checkpoint=True, branch="test-branch"
        )
        assert result.is_success
        mock_git.checkpoint.assert_called()

    def test_orchestrate_failure_handling(self, tmp_path: Path) -> None:
        """Test orchestrate failure handling with fail_fast."""
        mock_runner = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"
        mock_runner.run_to_file.return_value = r[int].fail("command error")

        proj = Mock()
        proj.path = tmp_path / "proj"
        proj.path.mkdir()
        mock_selector.resolve_projects.return_value = r[list[Mock]].ok([proj])

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=_make_git_mock(),
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, fail_fast=True, checkpoint=False, branch=""
        )
        assert result.is_success
        assert result.value.fail == 1


class TestStaticMethods:
    """Test static utility methods."""

    def test_repo_display_name_root(self, tmp_path: Path) -> None:
        """Test display name for root repository."""
        display_name = getattr(FlextInfraPrWorkspaceManager, "_repo_display_name")
        result = display_name(tmp_path, tmp_path)
        assert result == tmp_path.name

    def test_repo_display_name_subproject(self, tmp_path: Path) -> None:
        """Test display name for subproject."""
        sub = tmp_path / "my-project"
        sub.mkdir()
        display_name = getattr(FlextInfraPrWorkspaceManager, "_repo_display_name")
        result = display_name(sub, tmp_path)
        assert result == "my-project"

    def test_build_root_command(self, tmp_path: Path) -> None:
        """Test root command building."""
        build_root_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_root_command"
        )
        cmd = build_root_command(
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
        build_subproject_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_subproject_command"
        )
        cmd = build_subproject_command(tmp_path, {"action": "status", "head": "feat"})
        assert "make" in cmd
        assert "-C" in cmd
        assert "pr" in cmd
        assert "PR_ACTION=status" in cmd
        assert "PR_HEAD=feat" in cmd

    def test_build_root_command_defaults(self, tmp_path: Path) -> None:
        """Test root command with default values."""
        build_root_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_root_command"
        )
        cmd = build_root_command(tmp_path, {})
        assert "--action" in cmd
        assert "status" in cmd

    def test_build_subproject_command_no_optional(self, tmp_path: Path) -> None:
        """Test subproject command without optional keys."""
        build_subproject_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_subproject_command"
        )
        cmd = build_subproject_command(tmp_path, {})
        assert "make" in cmd
        # head, number, title, body are optional — should not appear
        assert not [c for c in cmd if c.startswith("PR_HEAD=")]
