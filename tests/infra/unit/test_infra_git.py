"""Tests for FlextInfraGitService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from flext_core import r
from flext_infra import FlextInfraUtilitiesGit


class TestFlextInfraGitService:
    """Test suite for FlextInfraGitService."""

    def test_current_branch_success(self, tmp_path: Path) -> None:
        """Test successful branch name retrieval."""
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].ok("main"),
        ) as mock_capture:
            result = FlextInfraUtilitiesGit.git_current_branch(tmp_path)
        assert result.is_success
        assert result.value == "main"
        mock_capture.assert_called_once()

    def test_current_branch_failure(self, tmp_path: Path) -> None:
        """Test branch retrieval failure."""
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].fail("not a git repo"),
        ):
            result = FlextInfraUtilitiesGit.git_current_branch(tmp_path)
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "not a git repo" in result.error

    def test_tag_exists_true(self, tmp_path: Path) -> None:
        """Test tag existence check returns true."""
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].ok("v1.0.0"),
        ):
            result = FlextInfraUtilitiesGit.git_tag_exists(tmp_path, "v1.0.0")
        assert result.is_success
        assert result.value is True

    def test_tag_exists_false(self, tmp_path: Path) -> None:
        """Test tag existence check returns false."""
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].ok(""),
        ):
            result = FlextInfraUtilitiesGit.git_tag_exists(tmp_path, "v1.0.0")
        assert result.is_success
        assert result.value is False

    def test_tag_exists_failure(self, tmp_path: Path) -> None:
        """Test tag check failure."""
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].fail("command failed"),
        ):
            result = FlextInfraUtilitiesGit.git_tag_exists(tmp_path, "v1.0.0")
        assert result.is_failure

    def test_run_arbitrary_command(self, tmp_path: Path) -> None:
        """Test running arbitrary git command."""
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].ok("output"),
        ):
            result = FlextInfraUtilitiesGit.git_run(["log", "--oneline"], cwd=tmp_path)
        assert result.is_success
        assert result.value == "output"

    def test_run_command_failure(self, tmp_path: Path) -> None:
        """Test arbitrary command failure."""
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].fail("error"),
        ):
            result = FlextInfraUtilitiesGit.git_run(["invalid"], cwd=tmp_path)
        assert result.is_failure


class TestRemovedCompatibilityMethods:
    """Removed compatibility methods are not callable anymore."""

    def test_removed_methods_raise_attribute_error(self) -> None:
        """Removed fallback helper methods are absent from git service."""
        with pytest.raises(AttributeError):
            _ = getattr(FlextInfraUtilitiesGit, "smart_checkout")
        with pytest.raises(AttributeError):
            _ = getattr(FlextInfraUtilitiesGit, "checkpoint")
        with pytest.raises(AttributeError):
            _ = getattr(FlextInfraUtilitiesGit, "create_tag_if_missing")
        with pytest.raises(AttributeError):
            _ = getattr(FlextInfraUtilitiesGit, "collect_changes")


class TestPreviousTag:
    def test_finds_previous(self, tmp_path: Path) -> None:
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].ok("v2.0.0\nv1.0.0\nv0.1.0\n"),
        ):
            result = FlextInfraUtilitiesGit.git_list_tags(tmp_path)
        assert result.is_success
        assert result.value == "v2.0.0\nv1.0.0\nv0.1.0\n"

    def test_no_previous(self, tmp_path: Path) -> None:
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.capture",
            return_value=r[str].fail("tag listing failed"),
        ):
            result = FlextInfraUtilitiesGit.git_list_tags(tmp_path)
        assert result.is_failure

    def test_tag_not_in_list(self, tmp_path: Path) -> None:
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.run_checked",
            return_value=r[bool].ok(True),
        ) as mock_run_checked:
            result = FlextInfraUtilitiesGit.git_create_tag(tmp_path, "v3.0.0")
        assert result.is_success
        cmd = mock_run_checked.call_args.args[0]
        assert "release: v3.0.0" in cmd


class TestPushRelease:
    def test_push_release_success(self, tmp_path: Path) -> None:
        with patch(
            "flext_infra._utilities.git.FlextInfraUtilitiesSubprocess.run_checked",
            return_value=r[bool].ok(True),
        ) as mock_run_checked:
            result = FlextInfraUtilitiesGit.git_push(
                tmp_path,
                remote="origin",
                branch="main",
                set_upstream=True,
            )
        assert result.is_success
        cmd = mock_run_checked.call_args.args[0]
        assert cmd == ["git", "push", "-u", "origin", "main"]
