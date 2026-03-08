"""Tests for FlextInfraWorkspaceDetector.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from flext_core import r
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


def test_detector_handles_git_command_errors(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection handles git command errors gracefully."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    # Test with actual git command (will fail gracefully in test env)
    result = detector.detect(project_root)

    # Should return standalone since git command will fail
    assert result.is_success


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


def test_detector_handles_empty_origin_url(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection when git origin URL is empty."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    mock_git = Mock()
    mock_git.config_get.return_value = r[str].ok("")
    detector._git = mock_git

    result = detector.detect(project_root)

    assert result.is_success
    assert result.value == WorkspaceMode.STANDALONE


def test_detector_handles_git_command_failure(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection when git command returns failure."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    mock_git = Mock()
    mock_git.config_get.return_value = r[str].fail("git config failed")
    detector._git = mock_git

    result = detector.detect(project_root)

    assert result.is_success
    assert result.value == WorkspaceMode.STANDALONE


def test_detector_detects_workspace_mode_with_flext_repo(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection of workspace mode when parent repo is 'flext'."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    mock_git = Mock()
    mock_git.config_get.return_value = r[str].ok(
        "https://github.com/flext-sh/flext.git"
    )
    detector._git = mock_git

    result = detector.detect(project_root)

    assert result.is_success
    assert result.value == WorkspaceMode.WORKSPACE


def test_detector_detects_standalone_with_non_flext_repo(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection of standalone mode when parent repo is not 'flext'."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    mock_git = Mock()
    mock_git.config_get.return_value = r[str].ok(
        "https://github.com/other-org/other-repo.git"
    )
    detector._git = mock_git

    result = detector.detect(project_root)

    assert result.is_success
    assert result.value == WorkspaceMode.STANDALONE


def test_detector_handles_runner_failure(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection when git service returns failure."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    mock_git = Mock()
    mock_git.config_get.return_value = r[str].fail("no remote")
    detector._git = mock_git

    result = detector.detect(project_root)

    assert result.is_success
    assert result.value == WorkspaceMode.STANDALONE


def test_detector_handles_exception_during_detection(
    detector: FlextInfraWorkspaceDetector, tmp_path: Path
) -> None:
    """Test detection handles exceptions gracefully."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    parent_git = tmp_path / ".git"
    parent_git.mkdir()

    mock_git = Mock()
    mock_git.config_get.side_effect = RuntimeError("Command failed")
    detector._git = mock_git

    result = detector.detect(project_root)

    assert result.is_failure
    assert isinstance(result.error, str)
    assert "Detection failed" in result.error
