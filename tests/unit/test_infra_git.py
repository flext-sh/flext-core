"""Tests for FlextInfraGitService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from flext_core import r
from flext_infra import FlextInfraUtilitiesGit


class TestFlextInfraGitService:
    """Test suite for FlextInfraGitService."""

    def test_current_branch_success(self, tmp_path: Path) -> None:
        """Test successful branch name retrieval."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("main")
        service = FlextInfraUtilitiesGit(runner=mock_runner)
        result = service.current_branch(tmp_path)
        assert result.is_success
        assert result.value == "main"
        mock_runner.capture.assert_called_once()

    def test_current_branch_failure(self, tmp_path: Path) -> None:
        """Test branch retrieval failure."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("not a git repo")
        service = FlextInfraUtilitiesGit(runner=mock_runner)
        result = service.current_branch(tmp_path)
        assert result.is_failure
        assert isinstance(result.error, str)
        assert isinstance(result.error, str)
        assert "not a git repo" in result.error

    def test_tag_exists_true(self, tmp_path: Path) -> None:
        """Test tag existence check returns true."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("v1.0.0")
        service = FlextInfraUtilitiesGit(runner=mock_runner)
        result = service.tag_exists(tmp_path, "v1.0.0")
        assert result.is_success
        assert result.value is True

    def test_tag_exists_false(self, tmp_path: Path) -> None:
        """Test tag existence check returns false."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("")
        service = FlextInfraUtilitiesGit(runner=mock_runner)
        result = service.tag_exists(tmp_path, "v1.0.0")
        assert result.is_success
        assert result.value is False

    def test_tag_exists_failure(self, tmp_path: Path) -> None:
        """Test tag check failure."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("command failed")
        service = FlextInfraUtilitiesGit(runner=mock_runner)
        result = service.tag_exists(tmp_path, "v1.0.0")
        assert result.is_failure

    def test_run_arbitrary_command(self, tmp_path: Path) -> None:
        """Test running arbitrary git command."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].ok("output")
        service = FlextInfraUtilitiesGit(runner=mock_runner)
        result = service.run(["log", "--oneline"], cwd=tmp_path)
        assert result.is_success
        assert result.value == "output"

    def test_run_command_failure(self, tmp_path: Path) -> None:
        """Test arbitrary command failure."""
        mock_runner = Mock()
        mock_runner.capture.return_value = r[str].fail("error")
        service = FlextInfraUtilitiesGit(runner=mock_runner)
        result = service.run(["invalid"], cwd=tmp_path)
        assert result.is_failure


class TestRemovedCompatibilityMethods:
    """Removed compatibility methods are not callable anymore."""

    def test_removed_methods_raise_attribute_error(self) -> None:
        """Removed fallback helper methods are absent from git service."""
        service = FlextInfraUtilitiesGit(runner=Mock())
        with pytest.raises(AttributeError):
            _ = getattr(service, "smart_checkout")
        with pytest.raises(AttributeError):
            _ = getattr(service, "checkpoint")
        with pytest.raises(AttributeError):
            _ = getattr(service, "create_tag_if_missing")
        with pytest.raises(AttributeError):
            _ = getattr(service, "collect_changes")


class TestPreviousTag:
    """Tests for FlextInfraGitService.previous_tag."""

    def test_finds_previous(self, tmp_path: Path) -> None:
        """Returns the tag after current in sorted list."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("v2.0.0\nv1.0.0\nv0.1.0\n")
        service = FlextInfraUtilitiesGit(runner=runner)
        result = service.previous_tag(tmp_path, "v2.0.0")
        assert result.is_success
        assert result.value == "v1.0.0"

    def test_no_previous(self, tmp_path: Path) -> None:
        """Returns empty string when tag is the only one."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("v1.0.0\n")
        service = FlextInfraUtilitiesGit(runner=runner)
        result = service.previous_tag(tmp_path, "v1.0.0")
        assert result.is_success
        assert result.value == ""

    def test_tag_not_in_list(self, tmp_path: Path) -> None:
        """Returns first non-matching tag if target not in list."""
        runner = Mock()
        runner.capture.return_value = r[str].ok("v2.0.0\nv1.0.0\n")
        service = FlextInfraUtilitiesGit(runner=runner)
        result = service.previous_tag(tmp_path, "v3.0.0")
        assert result.is_success
        assert result.value == "v2.0.0"


class TestPushRelease:
    """Tests for FlextInfraGitService.push_release."""

    def test_push_release_success(self, tmp_path: Path) -> None:
        """Push HEAD and tag both succeed."""
        runner = Mock()
        runner.run_checked.return_value = r[bool].ok(True)
        service = FlextInfraUtilitiesGit(runner=runner)
        result = service.push_release(tmp_path, "v1.0.0")
        assert result.is_success
        assert runner.run_checked.call_count == 2
