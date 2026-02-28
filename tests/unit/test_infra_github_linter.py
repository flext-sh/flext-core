"""Tests for FlextInfraWorkflowLinter.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from flext_core import r
from flext_infra.github.linter import FlextInfraWorkflowLinter


class TestFlextInfraWorkflowLinter:
    """Test suite for FlextInfraWorkflowLinter."""

    def test_lint_success_with_actionlint_installed(self, tmp_path: Path) -> None:
        """Test successful linting when actionlint is available."""
        mock_runner = Mock()
        mock_json = Mock()
        mock_output = Mock()
        mock_output.exit_code = 0
        mock_output.stdout = "All workflows valid"
        mock_output.stderr = ""
        mock_runner.run.return_value = r[Mock].ok(mock_output)

        with patch("shutil.which", return_value="/usr/bin/actionlint"):
            linter = FlextInfraWorkflowLinter(runner=mock_runner, json_io=mock_json)
            result = linter.lint(tmp_path)

        assert result.is_success
        assert result.value["status"] == "ok"
        assert result.value["exit_code"] == 0

    def test_lint_skipped_when_actionlint_not_installed(self, tmp_path: Path) -> None:
        """Test linting skipped when actionlint is not available."""
        mock_runner = Mock()
        mock_json = Mock()

        with patch("shutil.which", return_value=None):
            linter = FlextInfraWorkflowLinter(runner=mock_runner, json_io=mock_json)
            result = linter.lint(tmp_path)

        assert result.is_success
        assert result.value["status"] == "skipped"
        assert "actionlint not installed" in result.value["reason"]

    def test_lint_with_report_path(self, tmp_path: Path) -> None:
        """Test linting with JSON report output."""
        mock_runner = Mock()
        mock_json = Mock()
        mock_output = Mock()
        mock_output.exit_code = 0
        mock_output.stdout = "Valid"
        mock_output.stderr = ""
        mock_runner.run.return_value = r[Mock].ok(mock_output)

        report_path = tmp_path / "report.json"
        with patch("shutil.which", return_value="/usr/bin/actionlint"):
            linter = FlextInfraWorkflowLinter(runner=mock_runner, json_io=mock_json)
            result = linter.lint(tmp_path, report_path=report_path)

        assert result.is_success
        mock_json.write.assert_called_once()

    def test_lint_strict_mode_fails_on_issues(self, tmp_path: Path) -> None:
        """Test strict mode returns failure when actionlint finds issues."""
        mock_runner = Mock()
        mock_json = Mock()
        mock_runner.run.return_value = r[Mock].fail("workflow has errors")

        with patch("shutil.which", return_value="/usr/bin/actionlint"):
            linter = FlextInfraWorkflowLinter(runner=mock_runner, json_io=mock_json)
            result = linter.lint(tmp_path, strict=True)

        assert result.is_failure
        assert result.error

    def test_lint_default_runner_initialization(self) -> None:
        """Test linter initializes with default runner and json service."""
        linter = FlextInfraWorkflowLinter()
        assert linter._runner is not None
        assert linter._json is not None

    def test_lint_skipped_with_report(self, tmp_path: Path) -> None:
        """Test linting skipped with report output."""
        mock_runner = Mock()
        mock_json = Mock()
        report_path = tmp_path / "report.json"

        with patch("shutil.which", return_value=None):
            linter = FlextInfraWorkflowLinter(runner=mock_runner, json_io=mock_json)
            result = linter.lint(tmp_path, report_path=report_path)

        assert result.is_success
        mock_json.write.assert_called_once()
