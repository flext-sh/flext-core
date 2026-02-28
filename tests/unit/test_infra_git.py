"""Tests for FlextInfraGitService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

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
