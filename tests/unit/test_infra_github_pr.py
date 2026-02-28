"""Tests for FlextInfraPrManager.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from flext_core import r
from flext_infra.github.pr import FlextInfraPrManager, _parse_args, _selector, main


class TestFlextInfraPrManager:
    """Test suite for FlextInfraPrManager."""

    def test_open_pr_for_head_found(self, tmp_path: Path) -> None:
        """Test finding an open PR for a given head branch."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_versioning = Mock()
        pr_data = {
            "number": 42,
            "title": "Feature: Add new endpoint",
            "state": "OPEN",
            "baseRefName": "main",
            "headRefName": "feature/new-endpoint",
            "url": "https://github.com/org/repo/pull/42",
            "isDraft": False,
        }
        mock_runner.capture.return_value = r[str].ok(json.dumps([pr_data]))

        manager = FlextInfraPrManager(
            runner=mock_runner, git=mock_git, versioning=mock_versioning
        )
        result = manager.open_pr_for_head(tmp_path, "feature/new-endpoint")

        assert result.is_success
        assert result.value["number"] == 42

    def test_open_pr_for_head_not_found(self, tmp_path: Path) -> None:
        """Test when no open PR exists for head branch."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_versioning = Mock()
        mock_runner.capture.return_value = r[str].ok("[]")

        manager = FlextInfraPrManager(
            runner=mock_runner, git=mock_git, versioning=mock_versioning
        )
        result = manager.open_pr_for_head(tmp_path, "feature/nonexistent")

        assert result.is_success
        assert result.value == {}

    def test_open_pr_for_head_json_error(self, tmp_path: Path) -> None:
        """Test handling of invalid JSON response."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_versioning = Mock()
        mock_runner.capture.return_value = r[str].ok("invalid json")

        manager = FlextInfraPrManager(
            runner=mock_runner, git=mock_git, versioning=mock_versioning
        )
        result = manager.open_pr_for_head(tmp_path, "feature/test")

        assert result.is_failure
        assert result.error

    def test_open_pr_for_head_command_failure(self, tmp_path: Path) -> None:
        """Test handling of gh command failure."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_versioning = Mock()
        mock_runner.capture.return_value = r[str].fail("gh command failed")

        manager = FlextInfraPrManager(
            runner=mock_runner, git=mock_git, versioning=mock_versioning
        )
        result = manager.open_pr_for_head(tmp_path, "feature/test")

        assert result.is_failure
        assert result.error

    def test_default_initialization(self) -> None:
        """Test manager initializes with default dependencies."""
        manager = FlextInfraPrManager()
        assert manager._runner is not None
        assert manager._git is not None
        assert manager._versioning is not None

    def test_open_pr_for_head_non_dict_first(self, tmp_path: Path) -> None:
        """Test open_pr_for_head with non-dict first element."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok(json.dumps(["not-a-dict"]))
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.open_pr_for_head(tmp_path, "head")
        assert result.is_success
        assert result.value == {}


class TestStatus:
    """Test status method."""

    def test_status_open_pr(self, tmp_path: Path) -> None:
        """Test status with open PR found."""
        mock_runner = Mock()
        pr_data = {
            "number": 10,
            "title": "Test PR",
            "state": "OPEN",
            "url": "https://github.com/o/r/pull/10",
            "isDraft": False,
        }
        mock_runner.capture.return_value = r[str].ok(json.dumps([pr_data]))
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.status(tmp_path, "main", "feature")
        assert result.is_success
        assert result.value["status"] == "open"
        assert result.value["pr_number"] == 10

    def test_status_no_pr(self, tmp_path: Path) -> None:
        """Test status with no open PR."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("[]")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.status(tmp_path, "main", "feature")
        assert result.is_success
        assert result.value["status"] == "no-open-pr"

    def test_status_failure(self, tmp_path: Path) -> None:
        """Test status when list fails."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("gh error")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.status(tmp_path, "main", "feature")
        assert result.is_failure


class TestCreate:
    """Test create method."""

    def test_create_new(self, tmp_path: Path) -> None:
        """Test creating a new PR."""
        mock_runner = Mock()
        mock_runner.capture.side_effect = [
            r[str].ok("[]"),
            r[str].ok("https://github.com/o/r/pull/99"),
        ]
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.create(tmp_path, "main", "feature", "title", "body")
        assert result.is_success
        assert result.value["status"] == "created"

    def test_create_already_open(self, tmp_path: Path) -> None:
        """Test creating when PR already exists."""
        mock_runner = Mock()
        pr_data = {"url": "https://github.com/o/r/pull/10"}
        mock_runner.capture.return_value = r[str].ok(json.dumps([pr_data]))
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.create(tmp_path, "main", "feature", "title", "body")
        assert result.is_success
        assert result.value["status"] == "already-open"

    def test_create_failure(self, tmp_path: Path) -> None:
        """Test creation failure."""
        mock_runner = Mock()
        mock_runner.capture.side_effect = [
            r[str].ok("[]"),
            r[str].fail("create failed"),
        ]
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.create(tmp_path, "main", "feature", "title", "body")
        assert result.is_failure

    def test_create_with_draft(self, tmp_path: Path) -> None:
        """Test creating a draft PR."""
        mock_runner = Mock()
        mock_runner.capture.side_effect = [
            r[str].ok("[]"),
            r[str].ok("https://github.com/o/r/pull/100"),
        ]
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.create(
            tmp_path, "main", "feature", "title", "body", draft=True
        )
        assert result.is_success
        # Verify --draft was in the command
        call_args = mock_runner.capture.call_args_list[1]
        assert "--draft" in call_args[0][0]

    def test_create_check_existing_failure(self, tmp_path: Path) -> None:
        """Test create when checking existing PR fails."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("gh error")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.create(tmp_path, "main", "feature", "title", "body")
        assert result.is_failure


class TestView:
    """Test view method."""

    def test_view_success(self, tmp_path: Path) -> None:
        """Test viewing a PR successfully."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("PR details")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.view(tmp_path, "42")
        assert result.is_success

    def test_view_failure(self, tmp_path: Path) -> None:
        """Test view failure."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("not found")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.view(tmp_path, "999")
        assert result.is_failure


class TestChecks:
    """Test checks method."""

    def test_checks_pass(self, tmp_path: Path) -> None:
        """Test checks passing."""
        mock_runner = Mock()
        mock_runner.run.return_value = r[bool].ok(True)
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.checks(tmp_path, "42")
        assert result.is_success
        assert result.value["status"] == "checks-passed"

    def test_checks_fail_non_strict(self, tmp_path: Path) -> None:
        """Test checks fail in non-strict mode."""
        mock_runner = Mock()
        mock_runner.run.return_value = r[bool].fail("checks failed")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.checks(tmp_path, "42")
        assert result.is_success
        assert result.value["status"] == "checks-nonblocking"

    def test_checks_fail_strict(self, tmp_path: Path) -> None:
        """Test checks fail in strict mode."""
        mock_runner = Mock()
        mock_runner.run.return_value = r[bool].fail("checks failed")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.checks(tmp_path, "42", strict=True)
        assert result.is_failure


class TestMerge:
    """Test merge method."""

    def test_merge_success(self, tmp_path: Path) -> None:
        """Test successful merge."""
        mock_runner = Mock()
        mock_runner.run.return_value = r[bool].ok(True)
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.merge(tmp_path, "42", "feature", release_on_merge=False)
        assert result.is_success
        assert result.value["status"] == "merged"

    def test_merge_failure(self, tmp_path: Path) -> None:
        """Test merge failure."""
        mock_runner = Mock()
        mock_runner.run.return_value = r[bool].fail("merge conflict")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.merge(tmp_path, "42", "feature")
        assert result.is_failure

    def test_merge_not_mergeable_retry(self, tmp_path: Path) -> None:
        """Test merge retry when not mergeable."""
        mock_runner = Mock()
        mock_runner.run.side_effect = [
            r[bool].fail("not mergeable"),
            r[bool].ok(True),  # update-branch success
            r[bool].ok(True),  # retry merge
        ]
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.merge(tmp_path, "42", "feature", release_on_merge=False)
        assert result.is_success

    def test_merge_selector_same_as_head_no_pr(self, tmp_path: Path) -> None:
        """Test merge when selector=head and no open PR."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("[]")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.merge(tmp_path, "feature", "feature")
        assert result.is_success
        assert result.value["status"] == "no-open-pr"

    def test_merge_with_release(self, tmp_path: Path) -> None:
        """Test merge triggers release workflow."""
        mock_runner = Mock()
        mock_versioning = Mock()
        mock_versioning.release_tag_from_branch.return_value = r[str].ok("v1.0.0")
        mock_runner.run.side_effect = [
            r[bool].ok(True),  # merge
            r[bool].ok(True),  # release view (exists)
        ]
        manager = FlextInfraPrManager(
            runner=mock_runner, git=Mock(), versioning=mock_versioning
        )
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: Release")
        result = manager.merge(tmp_path, "42", "release/1.0", release_on_merge=True)
        assert result.is_success

    def test_merge_auto_and_delete_branch(self, tmp_path: Path) -> None:
        """Test merge with auto and delete-branch flags."""
        mock_runner = Mock()
        mock_runner.run.return_value = r[bool].ok(True)
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.merge(
            tmp_path,
            "42",
            "feature",
            auto=True,
            delete_branch=True,
            release_on_merge=False,
        )
        assert result.is_success
        call_args = mock_runner.run.call_args_list[0][0][0]
        assert "--auto" in call_args
        assert "--delete-branch" in call_args

    def test_merge_rebase_method(self, tmp_path: Path) -> None:
        """Test merge with rebase method."""
        mock_runner = Mock()
        mock_runner.run.return_value = r[bool].ok(True)
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.merge(
            tmp_path, "42", "feature", method="rebase", release_on_merge=False
        )
        assert result.is_success
        call_args = mock_runner.run.call_args_list[0][0][0]
        assert "--rebase" in call_args


class TestClose:
    """Test close method."""

    def test_close_success(self, tmp_path: Path) -> None:
        """Test closing a PR successfully."""
        mock_runner = Mock()
        mock_runner.run_checked.return_value = r[bool].ok(True)
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.close(tmp_path, "42")
        assert result.is_success

    def test_close_failure(self, tmp_path: Path) -> None:
        """Test close failure."""
        mock_runner = Mock()
        mock_runner.run_checked.return_value = r[bool].fail("close failed")
        manager = FlextInfraPrManager(runner=mock_runner, git=Mock(), versioning=Mock())
        result = manager.close(tmp_path, "42")
        assert result.is_failure


class TestTriggerRelease:
    """Test _trigger_release_if_needed method."""

    def test_no_release_workflow(self, tmp_path: Path) -> None:
        """Test when no release.yml exists."""
        manager = FlextInfraPrManager(runner=Mock(), git=Mock(), versioning=Mock())
        result = manager._trigger_release_if_needed(tmp_path, "feature")
        assert result.is_success
        assert result.value["status"] == "no-release-workflow"

    def test_no_release_tag(self, tmp_path: Path) -> None:
        """Test when no release tag can be derived."""
        mock_versioning = Mock()
        mock_versioning.release_tag_from_branch.return_value = r[str].fail("no tag")
        manager = FlextInfraPrManager(
            runner=Mock(), git=Mock(), versioning=mock_versioning
        )
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: R")
        result = manager._trigger_release_if_needed(tmp_path, "feature")
        assert result.is_success
        assert result.value["status"] == "no-release-tag"

    def test_release_exists(self, tmp_path: Path) -> None:
        """Test when release already exists."""
        mock_runner = Mock()
        mock_versioning = Mock()
        mock_versioning.release_tag_from_branch.return_value = r[str].ok("v1.0.0")
        mock_runner.run.return_value = r[bool].ok(True)
        manager = FlextInfraPrManager(
            runner=mock_runner, git=Mock(), versioning=mock_versioning
        )
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: R")
        result = manager._trigger_release_if_needed(tmp_path, "release/1.0")
        assert result.is_success
        assert result.value["status"] == "release-exists"

    def test_release_dispatched(self, tmp_path: Path) -> None:
        """Test successful release dispatch."""
        mock_runner = Mock()
        mock_versioning = Mock()
        mock_versioning.release_tag_from_branch.return_value = r[str].ok("v1.0.0")
        mock_runner.run.side_effect = [
            r[bool].fail("not found"),  # view fails
            r[bool].ok(True),  # dispatch succeeds
        ]
        manager = FlextInfraPrManager(
            runner=mock_runner, git=Mock(), versioning=mock_versioning
        )
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: R")
        result = manager._trigger_release_if_needed(tmp_path, "release/1.0")
        assert result.is_success
        assert result.value["status"] == "release-dispatched"

    def test_release_dispatch_failed(self, tmp_path: Path) -> None:
        """Test failed release dispatch."""
        mock_runner = Mock()
        mock_versioning = Mock()
        mock_versioning.release_tag_from_branch.return_value = r[str].ok("v1.0.0")
        mock_runner.run.side_effect = [
            r[bool].fail("not found"),  # view fails
            r[bool].fail("dispatch failed"),  # dispatch fails
        ]
        manager = FlextInfraPrManager(
            runner=mock_runner, git=Mock(), versioning=mock_versioning
        )
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: R")
        result = manager._trigger_release_if_needed(tmp_path, "release/1.0")
        assert result.is_success
        assert result.value["status"] == "release-dispatch-failed"


class TestSelectorFunction:
    """Test _selector module function."""

    def test_with_pr_number(self) -> None:
        """Test selector returns pr_number when provided."""
        assert _selector("42", "feature") == "42"

    def test_with_head_only(self) -> None:
        """Test selector returns head when no pr_number."""
        assert _selector("", "feature") == "feature"


class TestMainFunction:
    """Test main() CLI entry point."""

    def test_main_status_success(self) -> None:
        """Test main with status action success."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="status",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="",
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.status.return_value = r[dict].ok({"status": "open"})
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 0

    def test_main_status_failure(self) -> None:
        """Test main with status action failure."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="status",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="",
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.status.return_value = r[dict].fail("error")
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 1

    def test_main_create_success(self) -> None:
        """Test main with create action."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="create",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="",
                title="",
                body="",
                draft=0,
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.create.return_value = r[dict].ok({"status": "created"})
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 0

    def test_main_view_success(self) -> None:
        """Test main with view action."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="view",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="42",
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.view.return_value = r[str].ok("PR view output")
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 0

    def test_main_checks_success(self) -> None:
        """Test main with checks action."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="checks",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="42",
                checks_strict=0,
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.checks.return_value = r[dict].ok({"status": "checks-passed"})
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 0

    def test_main_merge_success(self) -> None:
        """Test main with merge action."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="merge",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="42",
                merge_method="squash",
                auto=0,
                delete_branch=0,
                release_on_merge=1,
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.merge.return_value = r[dict].ok({"status": "merged"})
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 0

    def test_main_close_success(self) -> None:
        """Test main with close action."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="close",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="42",
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.close.return_value = r[bool].ok(True)
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 0

    def test_main_close_failure(self) -> None:
        """Test main with close action failure."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="close",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="42",
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.close.return_value = r[bool].fail("close failed")
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 1

    def test_main_view_failure(self) -> None:
        """Test main with view action failure."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="view",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="42",
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.view.return_value = r[str].fail("not found")
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 1

    def test_main_checks_failure(self) -> None:
        """Test main with checks action failure."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="checks",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="42",
                checks_strict=1,
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.checks.return_value = r[dict].fail("checks failed")
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 1

    def test_main_create_failure(self) -> None:
        """Test main with create action failure."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="create",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="",
                title="",
                body="",
                draft=0,
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.create.return_value = r[dict].fail("create failed")
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 1

    def test_main_merge_failure(self) -> None:
        """Test main with merge action failure."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="merge",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="42",
                merge_method="squash",
                auto=0,
                delete_branch=0,
                release_on_merge=0,
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_manager.merge.return_value = r[dict].fail("merge failed")
            mock_cls.return_value = mock_manager
            result = main()
        assert result == 1


class TestParseArgs:
    """Test _parse_args function."""

    def test_parse_args_defaults(self) -> None:
        """Test _parse_args with default values."""
        with patch("sys.argv", ["prog"]):
            args = _parse_args()
            assert args.action == "status"
            assert args.base == "main"
            assert args.head == ""
            assert args.number == ""
            assert args.title == ""
            assert args.body == ""
            assert args.draft == 0
            assert args.merge_method == "squash"
            assert args.auto == 0
            assert args.delete_branch == 0
            assert args.checks_strict == 0
            assert args.release_on_merge == 1

    def test_parse_args_custom_values(self) -> None:
        """Test _parse_args with custom values."""
        with patch(
            "sys.argv",
            [
                "prog",
                "--action",
                "create",
                "--base",
                "develop",
                "--head",
                "feature/test",
                "--number",
                "42",
                "--title",
                "Test PR",
                "--body",
                "Test body",
                "--draft",
                "1",
                "--merge-method",
                "rebase",
                "--auto",
                "1",
                "--delete-branch",
                "1",
                "--checks-strict",
                "1",
                "--release-on-merge",
                "0",
            ],
        ):
            args = _parse_args()
            assert args.action == "create"
            assert args.base == "develop"
            assert args.head == "feature/test"
            assert args.number == "42"
            assert args.title == "Test PR"
            assert args.body == "Test body"
            assert args.draft == 1
            assert args.merge_method == "rebase"
            assert args.auto == 1
            assert args.delete_branch == 1
            assert args.checks_strict == 1
            assert args.release_on_merge == 0

    def test_main_unknown_action(self) -> None:
        """Test main with unknown action raises RuntimeError."""
        with (
            patch("flext_infra.github.pr._parse_args") as mock_args,
            patch("flext_infra.github.pr.FlextInfraPrManager") as mock_cls,
            patch("flext_infra.github.pr.FlextInfraGitService") as mock_git_cls,
        ):
            mock_args.return_value = Mock(
                action="invalid_action",
                repo_root=Path("/tmp/test"),
                base="main",
                head="feature",
                number="",
            )
            mock_git = Mock()
            mock_git.current_branch.return_value = r[str].ok("feature")
            mock_git_cls.return_value = mock_git
            mock_manager = Mock()
            mock_cls.return_value = mock_manager
            with pytest.raises(RuntimeError, match="unknown action"):
                main()


class TestGithubInit:
    """Test github module __init__.py lazy imports."""

    def test_lazy_import_pr_manager(self) -> None:
        """Test lazy import of FlextInfraPrManager."""
        import flext_infra.github as github_module  # noqa: PLC0415

        manager = github_module.FlextInfraPrManager()
        assert isinstance(manager, FlextInfraPrManager)

    def test_getattr_invalid_attribute(self) -> None:
        """Test __getattr__ raises AttributeError for invalid attribute."""
        import flext_infra.github as github_module  # noqa: PLC0415

        with pytest.raises(AttributeError, match=r"module.*has no attribute"):
            _ = github_module.NonexistentAttribute

    def test_dir_returns_all_exports(self) -> None:
        """Test __dir__ returns all exported attributes."""
        import flext_infra.github as github_module  # noqa: PLC0415

        exports = dir(github_module)
        assert "FlextInfraPrManager" in exports
        assert "FlextInfraPrWorkspaceManager" in exports
        assert "FlextInfraWorkflowLinter" in exports
        assert "FlextInfraWorkflowSyncer" in exports
        assert "SyncOperation" in exports
