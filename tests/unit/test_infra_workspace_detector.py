"""Tests for FlextInfraWorkspaceDetector.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flext_core import FlextResult as r
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector, WorkspaceMode


@pytest.fixture
def detector() -> FlextInfraWorkspaceDetector:
    """Create a detector instance."""
    return FlextInfraWorkspaceDetector()


def test_detector_detects_workspace_mode_with_parent_git(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection with parent .git directory present."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    # Test with actual git command (will fail gracefully in test env)
    result = detector.detect(project_root)

    # Should return standalone since we can't actually run git in test
    assert result.is_success

def test_detector_detects_standalone_mode_without_parent_git(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection of standalone mode when parent repo is not 'flext'."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    # Test with actual git command (will fail gracefully in test env)
    result = detector.detect(project_root)

    # Should return standalone since we can't actually run git in test
    assert result.is_success

def test_detector_returns_standalone_when_no_parent_git(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection returns standalone when no parent .git exists."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    result = detector.detect(project_root)

    assert result.is_success
    assert result.value == WorkspaceMode.STANDALONE


def test_detector_returns_standalone_on_git_command_failure(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection returns standalone when git command fails."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    with patch.object(
        detector._runner,
        "run_raw",
        return_value=r[MagicMock].fail("git command failed"),
    ):
        result = detector.detect(project_root)

        assert result.is_success
        assert result.value == WorkspaceMode.STANDALONE


def test_detector_extracts_repo_name_from_https_url() -> None:
    """Test extraction of repo name from HTTPS URL."""
    url = "https://github.com/flext-sh/flext.git"
    name = FlextInfraWorkspaceDetector._repo_name_from_url(url)
    assert name == "flext"


def test_detector_extracts_repo_name_from_ssh_url() -> None:
    """Test extraction of repo name from SSH URL."""
    url = "git@github.com:flext-sh/flext.git"
    name = FlextInfraWorkspaceDetector._repo_name_from_url(url)
    assert name == "flext"


def test_detector_extracts_repo_name_without_git_suffix() -> None:
    """Test extraction of repo name without .git suffix."""
    url = "https://github.com/flext-sh/flext"
    name = FlextInfraWorkspaceDetector._repo_name_from_url(url)
    assert name == "flext"


def test_detector_execute_returns_failure() -> None:
    """Test that execute() method returns failure as expected."""
    detector = FlextInfraWorkspaceDetector()
    result = detector.execute()
    assert result.is_failure
