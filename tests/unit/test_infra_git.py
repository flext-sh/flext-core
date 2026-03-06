"""Tests for FlextInfraGitService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from flext_core import r
from flext_infra import FlextInfraGitService


class TestFlextInfraGitService:
    """Test suite for FlextInfraGitService."""

    def test_current_branch_success(self, tmp_path: Path) -> None:
        """Test successful branch name retrieval."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("main")
        service = FlextInfraGitService(runner=mock_runner)

        result = service.current_branch(tmp_path)

        assert result.is_success
        assert result.value == "main"
        mock_runner.capture.assert_called_once()

    def test_current_branch_failure(self, tmp_path: Path) -> None:
        """Test branch retrieval failure."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("not a git repo")
        service = FlextInfraGitService(runner=mock_runner)

        result = service.current_branch(tmp_path)

        assert result.is_failure
        assert isinstance(result.error, str)
        assert isinstance(result.error, str)
        assert "not a git repo" in result.error

    def test_tag_exists_true(self, tmp_path: Path) -> None:
        """Test tag existence check returns true."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("v1.0.0")
        service = FlextInfraGitService(runner=mock_runner)

        result = service.tag_exists(tmp_path, "v1.0.0")

        assert result.is_success
        assert result.value is True

    def test_tag_exists_false(self, tmp_path: Path) -> None:
        """Test tag existence check returns false."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("")
        service = FlextInfraGitService(runner=mock_runner)

        result = service.tag_exists(tmp_path, "v1.0.0")

        assert result.is_success
        assert result.value is False

    def test_tag_exists_failure(self, tmp_path: Path) -> None:
        """Test tag check failure."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("command failed")
        service = FlextInfraGitService(runner=mock_runner)

        result = service.tag_exists(tmp_path, "v1.0.0")

        assert result.is_failure

    def test_run_arbitrary_command(self, tmp_path: Path) -> None:
        """Test running arbitrary git command."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("output")
        service = FlextInfraGitService(runner=mock_runner)

        result = service.run(["log", "--oneline"], cwd=tmp_path)

        assert result.is_success
        assert result.value == "output"

    def test_run_command_failure(self, tmp_path: Path) -> None:
        """Test arbitrary command failure."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("error")
        service = FlextInfraGitService(runner=mock_runner)

        result = service.run(["invalid"], cwd=tmp_path)

        assert result.is_failure

    def test_default_runner_initialization(self) -> None:
        """Test service initializes with default runner."""
        service = FlextInfraGitService()
        assert service._runner is not None


class TestSmartCheckout:
    """Tests for FlextInfraGitService.smart_checkout."""

    def test_empty_branch_is_noop(self, tmp_path: Path) -> None:
        """Empty branch name returns ok immediately."""
        service = FlextInfraGitService(runner=Mock())
        result = service.smart_checkout(tmp_path, "")
        assert result.is_success

    def test_already_on_branch(self, tmp_path: Path) -> None:
        """No checkout when already on the target branch."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("feature")
        service = FlextInfraGitService(runner=runner)

        result = service.smart_checkout(tmp_path, "feature")

        assert result.is_success
        runner.run_checked.assert_not_called()

    def test_simple_checkout_succeeds(self, tmp_path: Path) -> None:
        """Simple checkout works when branch exists."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("main")
        runner.run_checked.return_value = r[bool].ok(True)
        service = FlextInfraGitService(runner=runner)

        result = service.smart_checkout(tmp_path, "feature")

        assert result.is_success

    def test_force_create_on_local_changes(self, tmp_path: Path) -> None:
        """Fallback to -B when local changes would be overwritten."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("main")
        runner.run_checked.side_effect = [
            r[bool].fail("error: local changes would be overwritten"),
            r[bool].ok(True),
        ]
        service = FlextInfraGitService(runner=runner)

        result = service.smart_checkout(tmp_path, "feature")

        assert result.is_success

    def test_fetch_and_track(self, tmp_path: Path) -> None:
        """Fallback to fetch+track when branch absent locally."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("main")
        runner.run_checked.side_effect = [
            r[bool].fail("error: pathspec 'feature' did not match"),
            r[bool].ok(True),  # fetch succeeds
            r[bool].ok(True),  # checkout -B with track
        ]
        service = FlextInfraGitService(runner=runner)

        result = service.smart_checkout(tmp_path, "feature")

        assert result.is_success


class TestCheckpoint:
    """Tests for FlextInfraGitService.checkpoint."""

    def test_no_changes_is_noop(self, tmp_path: Path) -> None:
        """No commit when there are no changes."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("")
        service = FlextInfraGitService(runner=runner)

        result = service.checkpoint(tmp_path, "main")

        assert result.is_success
        runner.run_checked.assert_not_called()

    def test_full_flow(self, tmp_path: Path) -> None:
        """Full add+commit+push flow works."""
        runner = Mock()
        runner.capture.side_effect = [
            r[str].ok("M file.py"),  # has_changes -> porcelain
            r[str].ok("file.py"),  # diff_names
        ]
        runner.run_checked.side_effect = [
            r[bool].ok(True),  # add
            r[bool].ok(True),  # commit
            r[bool].ok(True),  # push
        ]
        service = FlextInfraGitService(runner=runner)

        result = service.checkpoint(tmp_path, "main")

        assert result.is_success

    def test_push_failure_triggers_rebase(self, tmp_path: Path) -> None:
        """Push failure triggers rebase + retry."""
        runner = Mock()
        runner.capture.side_effect = [
            r[str].ok("M file.py"),  # has_changes
            r[str].ok("file.py"),  # diff_names
        ]
        runner.run_checked.side_effect = [
            r[bool].ok(True),  # add
            r[bool].ok(True),  # commit
            r[bool].fail("rejected"),  # push fails
            r[bool].ok(True),  # pull --rebase
            r[bool].ok(True),  # push retry
        ]
        service = FlextInfraGitService(runner=runner)

        result = service.checkpoint(tmp_path, "main")

        assert result.is_success

    def test_changes_check_failure(self, tmp_path: Path) -> None:
        """Failure checking changes propagates error."""
        runner = Mock()
        runner.capture.return_value = r[str].fail("git status failed")
        service = FlextInfraGitService(runner=runner)

        result = service.checkpoint(tmp_path, "main")

        assert result.is_failure


class TestCreateTagIfMissing:
    """Tests for FlextInfraGitService.create_tag_if_missing."""

    def test_tag_exists_returns_ok(self, tmp_path: Path) -> None:
        """No-op when tag already exists."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("v1.0.0")
        service = FlextInfraGitService(runner=runner)

        result = service.create_tag_if_missing(tmp_path, "v1.0.0")

        assert result.is_success

    def test_creates_tag_when_missing(self, tmp_path: Path) -> None:
        """Creates tag when it doesn't exist."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("")
        runner.run_checked.return_value = r[bool].ok(True)
        service = FlextInfraGitService(runner=runner)

        result = service.create_tag_if_missing(tmp_path, "v2.0.0")

        assert result.is_success
        runner.run_checked.assert_called_once()


class TestPreviousTag:
    """Tests for FlextInfraGitService.previous_tag."""

    def test_finds_previous(self, tmp_path: Path) -> None:
        """Returns the tag after current in sorted list."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("v2.0.0\nv1.0.0\nv0.1.0\n")
        service = FlextInfraGitService(runner=runner)

        result = service.previous_tag(tmp_path, "v2.0.0")

        assert result.is_success
        assert result.value == "v1.0.0"

    def test_no_previous(self, tmp_path: Path) -> None:
        """Returns empty string when tag is the only one."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("v1.0.0\n")
        service = FlextInfraGitService(runner=runner)

        result = service.previous_tag(tmp_path, "v1.0.0")

        assert result.is_success
        assert result.value == ""

    def test_tag_not_in_list(self, tmp_path: Path) -> None:
        """Returns first non-matching tag if target not in list."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("v2.0.0\nv1.0.0\n")
        service = FlextInfraGitService(runner=runner)

        result = service.previous_tag(tmp_path, "v3.0.0")

        assert result.is_success
        assert result.value == "v2.0.0"


class TestPushRelease:
    """Tests for FlextInfraGitService.push_release."""

    def test_push_release_success(self, tmp_path: Path) -> None:
        """Push HEAD and tag both succeed."""
        runner = Mock()
        runner.run_checked.return_value = r[bool].ok(True)
        service = FlextInfraGitService(runner=runner)

        result = service.push_release(tmp_path, "v1.0.0")

        assert result.is_success
        assert runner.run_checked.call_count == 2


class TestCollectChanges:
    """Tests for FlextInfraGitService.collect_changes."""

    def test_with_previous(self, tmp_path: Path) -> None:
        """Builds correct revision range with previous tag."""
        runner = Mock()
        runner.capture.side_effect = [
            r[str].ok("v2.0.0"),  # tag_exists
            r[str].ok("- abc123 fix (dev)"),  # log
        ]
        service = FlextInfraGitService(runner=runner)

        result = service.collect_changes(tmp_path, "v2.0.0", "v1.0.0")

        assert result.is_success
        assert "abc123" in result.value

    def test_without_previous(self, tmp_path: Path) -> None:
        """Uses tag only when no previous."""
        runner = Mock()
        runner.capture.side_effect = [
            r[str].ok("v1.0.0"),  # tag_exists
            r[str].ok("- abc123 init (dev)"),  # log
        ]
        service = FlextInfraGitService(runner=runner)

        result = service.collect_changes(tmp_path, "v1.0.0")

        assert result.is_success
