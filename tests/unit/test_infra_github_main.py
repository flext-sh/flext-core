"""Tests for flext_infra.github.__main__ CLI entry point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

from flext_core import r
from flext_infra.github.__main__ import (
    _run_lint,
    _run_pr,
    _run_pr_workspace,
    _run_workflows,
    main,
)


class TestRunWorkflows:
    """Test suite for _run_workflows handler."""

    def test_run_workflows_success(self, tmp_path: Path) -> None:
        """Test successful workflow sync."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowSyncer"
        ) as mock_syncer_class:
            mock_syncer = Mock()
            mock_syncer_class.return_value = mock_syncer
            mock_syncer.sync_workspace.return_value = r[list].ok([])

            argv = ["--workspace-root", str(tmp_path)]
            result = _run_workflows(argv)

            assert result == 0

    def test_run_workflows_failure(self, tmp_path: Path) -> None:
        """Test workflow sync failure."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowSyncer"
        ) as mock_syncer_class:
            mock_syncer = Mock()
            mock_syncer_class.return_value = mock_syncer
            mock_syncer.sync_workspace.return_value = r[list].fail("sync failed")

            argv = ["--workspace-root", str(tmp_path)]
            result = _run_workflows(argv)

            assert result == 1

    def test_run_workflows_with_apply_flag(self, tmp_path: Path) -> None:
        """Test workflow sync with apply flag."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowSyncer"
        ) as mock_syncer_class:
            mock_syncer = Mock()
            mock_syncer_class.return_value = mock_syncer
            mock_syncer.sync_workspace.return_value = r[list].ok([])

            argv = ["--workspace-root", str(tmp_path), "--apply"]
            result = _run_workflows(argv)

            assert result == 0
            mock_syncer.sync_workspace.assert_called_once()
            call_kwargs = mock_syncer.sync_workspace.call_args[1]
            assert call_kwargs["apply"] is True

    def test_run_workflows_with_prune_flag(self, tmp_path: Path) -> None:
        """Test workflow sync with prune flag."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowSyncer"
        ) as mock_syncer_class:
            mock_syncer = Mock()
            mock_syncer_class.return_value = mock_syncer
            mock_syncer.sync_workspace.return_value = r[list].ok([])

            argv = ["--workspace-root", str(tmp_path), "--prune"]
            result = _run_workflows(argv)

            assert result == 0
            call_kwargs = mock_syncer.sync_workspace.call_args[1]
            assert call_kwargs["prune"] is True

    def test_run_workflows_with_report(self, tmp_path: Path) -> None:
        """Test workflow sync with report output."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowSyncer"
        ) as mock_syncer_class:
            mock_syncer = Mock()
            mock_syncer_class.return_value = mock_syncer
            mock_syncer.sync_workspace.return_value = r[list].ok([])

            report_path = tmp_path / "report.json"
            argv = ["--workspace-root", str(tmp_path), "--report", str(report_path)]
            result = _run_workflows(argv)

            assert result == 0
            call_kwargs = mock_syncer.sync_workspace.call_args[1]
            assert call_kwargs["report_path"] == report_path


class TestRunLint:
    """Test suite for _run_lint handler."""

    def test_run_lint_success(self, tmp_path: Path) -> None:
        """Test successful linting."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowLinter"
        ) as mock_linter_class:
            mock_linter = Mock()
            mock_linter_class.return_value = mock_linter
            mock_linter.lint.return_value = r[dict].ok({"status": "ok"})

            argv = ["--root", str(tmp_path)]
            result = _run_lint(argv)

            assert result == 0

    def test_run_lint_failure(self, tmp_path: Path) -> None:
        """Test linting failure."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowLinter"
        ) as mock_linter_class:
            mock_linter = Mock()
            mock_linter_class.return_value = mock_linter
            mock_linter.lint.return_value = r[dict].fail("lint failed")

            argv = ["--root", str(tmp_path)]
            result = _run_lint(argv)

            assert result == 1

    def test_run_lint_with_report(self, tmp_path: Path) -> None:
        """Test linting with report output."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowLinter"
        ) as mock_linter_class:
            mock_linter = Mock()
            mock_linter_class.return_value = mock_linter
            mock_linter.lint.return_value = r[dict].ok({"status": "ok"})

            report_path = tmp_path / "report.json"
            argv = ["--root", str(tmp_path), "--report", str(report_path)]
            result = _run_lint(argv)

            assert result == 0
            call_kwargs = mock_linter.lint.call_args[1]
            assert call_kwargs["report_path"] == report_path

    def test_run_lint_with_strict_flag(self, tmp_path: Path) -> None:
        """Test linting with strict flag."""
        with patch(
            "flext_infra.github.__main__.FlextInfraWorkflowLinter"
        ) as mock_linter_class:
            mock_linter = Mock()
            mock_linter_class.return_value = mock_linter
            mock_linter.lint.return_value = r[dict].ok({"status": "ok"})

            argv = ["--root", str(tmp_path), "--strict"]
            result = _run_lint(argv)

            assert result == 0
            call_kwargs = mock_linter.lint.call_args[1]
            assert call_kwargs["strict"] is True


class TestRunPr:
    """Test suite for _run_pr handler."""

    def test_run_pr_delegates_to_pr_main(self) -> None:
        """Test that _run_pr delegates to pr_main."""
        with patch("flext_infra.github.__main__.pr_main") as mock_pr_main:
            mock_pr_main.return_value = 0

            argv = ["--repo-root", "/tmp", "--action", "status"]
            result = _run_pr(argv)

            assert result == 0
            mock_pr_main.assert_called_once()

    def test_run_pr_sets_sys_argv(self) -> None:
        """Test that _run_pr sets sys.argv correctly."""
        original_argv = sys.argv.copy()
        try:
            with patch("flext_infra.github.__main__.pr_main") as mock_pr_main:
                mock_pr_main.return_value = 0

                argv = ["--repo-root", "/tmp", "--action", "status"]
                _run_pr(argv)

                # sys.argv should be set with the command and remaining args
                assert sys.argv[0] == "flext-infra github pr"
                assert "--repo-root" in sys.argv
        finally:
            sys.argv = original_argv


class TestRunPrWorkspace:
    """Test suite for _run_pr_workspace handler."""

    def test_run_pr_workspace_success(self, tmp_path: Path) -> None:
        """Test successful PR workspace orchestration."""
        with patch(
            "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.orchestrate.return_value = r[dict].ok({"fail": 0})

            argv = ["--workspace-root", str(tmp_path)]
            result = _run_pr_workspace(argv)

            assert result == 0

    def test_run_pr_workspace_failure(self, tmp_path: Path) -> None:
        """Test PR workspace orchestration failure."""
        with patch(
            "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.orchestrate.return_value = r[dict].fail("orchestration failed")

            argv = ["--workspace-root", str(tmp_path)]
            result = _run_pr_workspace(argv)

            assert result == 1

    def test_run_pr_workspace_with_failures(self, tmp_path: Path) -> None:
        """Test PR workspace with failures in results."""
        with patch(
            "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.orchestrate.return_value = r[dict].ok({"fail": 2})

            argv = ["--workspace-root", str(tmp_path)]
            result = _run_pr_workspace(argv)

            assert result == 1

    def test_run_pr_workspace_with_projects(self, tmp_path: Path) -> None:
        """Test PR workspace with specific projects."""
        with patch(
            "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.orchestrate.return_value = r[dict].ok({"fail": 0})

            argv = [
                "--workspace-root",
                str(tmp_path),
                "--project",
                "proj1",
                "--project",
                "proj2",
            ]
            result = _run_pr_workspace(argv)

            assert result == 0
            call_kwargs = mock_manager.orchestrate.call_args[1]
            assert call_kwargs["projects"] == ["proj1", "proj2"]

    def test_run_pr_workspace_with_branch(self, tmp_path: Path) -> None:
        """Test PR workspace with branch checkout."""
        with patch(
            "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.orchestrate.return_value = r[dict].ok({"fail": 0})

            argv = ["--workspace-root", str(tmp_path), "--branch", "feature/test"]
            result = _run_pr_workspace(argv)

            assert result == 0
            call_kwargs = mock_manager.orchestrate.call_args[1]
            assert call_kwargs["branch"] == "feature/test"

    def test_run_pr_workspace_with_checkpoint(self, tmp_path: Path) -> None:
        """Test PR workspace with checkpoint enabled."""
        with patch(
            "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.orchestrate.return_value = r[dict].ok({"fail": 0})

            argv = ["--workspace-root", str(tmp_path), "--checkpoint", "1"]
            result = _run_pr_workspace(argv)

            assert result == 0
            call_kwargs = mock_manager.orchestrate.call_args[1]
            assert call_kwargs["checkpoint"] is True

    def test_run_pr_workspace_with_fail_fast(self, tmp_path: Path) -> None:
        """Test PR workspace with fail-fast enabled."""
        with patch(
            "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.orchestrate.return_value = r[dict].ok({"fail": 0})

            argv = ["--workspace-root", str(tmp_path), "--fail-fast", "1"]
            result = _run_pr_workspace(argv)

            assert result == 0
            call_kwargs = mock_manager.orchestrate.call_args[1]
            assert call_kwargs["fail_fast"] is True

    def test_run_pr_workspace_with_pr_args(self, tmp_path: Path) -> None:
        """Test PR workspace with PR operation arguments."""
        with patch(
            "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.orchestrate.return_value = r[dict].ok({"fail": 0})

            argv = [
                "--workspace-root",
                str(tmp_path),
                "--pr-action",
                "merge",
                "--pr-base",
                "main",
                "--pr-head",
                "feature/test",
            ]
            result = _run_pr_workspace(argv)

            assert result == 0
            call_kwargs = mock_manager.orchestrate.call_args[1]
            pr_args = call_kwargs["pr_args"]
            assert pr_args["action"] == "merge"
            assert pr_args["base"] == "main"
            assert pr_args["head"] == "feature/test"


class TestMain:
    """Test suite for main CLI entry point."""

    def test_main_help_flag(self) -> None:
        """Test main with help flag."""
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "-h"]
            result = main()
            assert result == 0
        finally:
            sys.argv = original_argv

    def test_main_no_args(self) -> None:
        """Test main with no arguments."""
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["flext-infra"]
            result = main()
            assert result == 1
        finally:
            sys.argv = original_argv

    def test_main_workflows_subcommand(self, tmp_path: Path) -> None:
        """Test main dispatching to workflows subcommand."""
        original_argv = sys.argv.copy()
        try:
            with patch(
                "flext_infra.github.__main__.FlextInfraWorkflowSyncer"
            ) as mock_syncer_class:
                mock_syncer = Mock()
                mock_syncer_class.return_value = mock_syncer
                mock_syncer.sync_workspace.return_value = r[list].ok([])

                sys.argv = [
                    "flext-infra",
                    "workflows",
                    "--workspace-root",
                    str(tmp_path),
                ]
                result = main()
                assert result == 0
        finally:
            sys.argv = original_argv

    def test_main_lint_subcommand(self, tmp_path: Path) -> None:
        """Test main dispatching to lint subcommand."""
        original_argv = sys.argv.copy()
        try:
            with patch(
                "flext_infra.github.__main__.FlextInfraWorkflowLinter"
            ) as mock_linter_class:
                mock_linter = Mock()
                mock_linter_class.return_value = mock_linter
                mock_linter.lint.return_value = r[dict].ok({"status": "ok"})

                sys.argv = ["flext-infra", "lint", "--root", str(tmp_path)]
                result = main()
                assert result == 0
        finally:
            sys.argv = original_argv

    def test_main_pr_subcommand(self, tmp_path: Path) -> None:
        """Test main dispatching to pr subcommand."""
        original_argv = sys.argv.copy()
        try:
            with patch("flext_infra.github.__main__.pr_main") as mock_pr_main:
                mock_pr_main.return_value = 0

                sys.argv = [
                    "flext-infra",
                    "pr",
                    "--repo-root",
                    str(tmp_path),
                    "--action",
                    "status",
                ]
                result = main()
                assert result == 0
        finally:
            sys.argv = original_argv

    def test_main_pr_workspace_subcommand(self, tmp_path: Path) -> None:
        """Test main dispatching to pr-workspace subcommand."""
        original_argv = sys.argv.copy()
        try:
            with patch(
                "flext_infra.github.__main__.FlextInfraPrWorkspaceManager"
            ) as mock_manager_class:
                mock_manager = Mock()
                mock_manager_class.return_value = mock_manager
                mock_manager.orchestrate.return_value = r[dict].ok({"fail": 0})

                sys.argv = [
                    "flext-infra",
                    "pr-workspace",
                    "--workspace-root",
                    str(tmp_path),
                ]
                result = main()
                assert result == 0
        finally:
            sys.argv = original_argv

    def test_main_unknown_subcommand(self) -> None:
        """Test main with unknown subcommand."""
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["flext-infra", "unknown"]
            result = main()
            assert result == 1
        finally:
            sys.argv = original_argv

    def test_main_ensures_structlog_configured(self, tmp_path: Path) -> None:
        """Test that main ensures structlog is configured."""
        original_argv = sys.argv.copy()
        try:
            with patch("flext_infra.github.__main__.FlextRuntime") as mock_runtime:
                with patch(
                    "flext_infra.github.__main__.FlextInfraWorkflowLinter"
                ) as mock_linter_class:
                    mock_linter = Mock()
                    mock_linter_class.return_value = mock_linter
                    mock_linter.lint.return_value = r[dict].ok({"status": "ok"})

                    sys.argv = [
                        "flext-infra",
                        "lint",
                        "--root",
                        str(tmp_path),
                    ]
                    main()

                    mock_runtime.ensure_structlog_configured.assert_called_once()
        finally:
            sys.argv = original_argv
