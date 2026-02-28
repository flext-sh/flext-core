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
        mock_git = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_runner.capture.return_value = r[str].ok("M file.py\nA new.py")

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=mock_git,
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.has_changes(tmp_path)

        assert result.is_success
        assert result.value is True

    def test_has_changes_false(self, tmp_path: Path) -> None:
        """Test when repository has no uncommitted changes."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_runner.capture.return_value = r[str].ok("")

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=mock_git,
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.has_changes(tmp_path)

        assert result.is_success
        assert result.value is False

    def test_has_changes_command_failure(self, tmp_path: Path) -> None:
        """Test handling of git status command failure."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_runner.capture.return_value = r[str].fail("not a git repository")

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=mock_git,
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.has_changes(tmp_path)

        assert result.is_failure
        assert result.error

    def test_checkout_branch_success(self, tmp_path: Path) -> None:
        """Test successful branch checkout."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_runner.run.return_value = r[Mock].ok(Mock())

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=mock_git,
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.checkout_branch(tmp_path, "feature/test")

        assert result.is_success
        assert result.value is True

    def test_checkout_branch_failure(self, tmp_path: Path) -> None:
        """Test branch checkout failure."""
        mock_runner = Mock()
        mock_git = Mock()
        mock_selector = Mock()
        mock_reporting = Mock()
        mock_runner.run.return_value = r[Mock].fail("checkout failed")

        manager = FlextInfraPrWorkspaceManager(
            runner=mock_runner,
            git=mock_git,
            selector=mock_selector,
            reporting=mock_reporting,
        )
        result = manager.checkout_branch(tmp_path, "invalid-branch")

        assert result.is_failure

    def test_default_initialization(self) -> None:
        """Test manager initializes with default dependencies."""
        manager = FlextInfraPrWorkspaceManager()
        assert manager._runner is not None
        assert manager._git is not None
        assert manager._selector is not None
        assert manager._reporting is not None
