"""Tests for FlextInfraPrManager.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from flext_core import r
from flext_infra.github.pr import FlextInfraPrManager


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
