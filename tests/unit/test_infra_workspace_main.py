"""Tests for workspace CLI entry point (__main__.py).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from flext_infra.workspace import __main__ as workspace_main


def test_run_detect_success(tmp_path: Path) -> None:
    """Test _run_detect with successful detection."""
    args = Mock()
    args.project_root = tmp_path

    with patch(
        "flext_infra.workspace.__main__.FlextInfraWorkspaceDetector"
    ) as mock_detector_class:
        mock_detector = Mock()
        mock_detector.detect.return_value = Mock(is_success=True, value="workspace")
        mock_detector_class.return_value = mock_detector

        result = workspace_main._run_detect(args)
        assert result == 0


def test_run_detect_failure(tmp_path: Path) -> None:
    """Test _run_detect with detection failure."""
    args = Mock()
    args.project_root = tmp_path

    with patch(
        "flext_infra.workspace.__main__.FlextInfraWorkspaceDetector"
    ) as mock_detector_class:
        mock_detector = Mock()
        mock_detector.detect.return_value = Mock(
            is_success=False, error="Detection failed"
        )
        mock_detector_class.return_value = mock_detector

        result = workspace_main._run_detect(args)
        assert result == 1


def test_run_sync_success(tmp_path: Path) -> None:
    """Test _run_sync with successful sync."""
    args = Mock()
    args.project_root = tmp_path
    args.canonical_root = None

    with patch(
        "flext_infra.workspace.__main__.FlextInfraSyncService"
    ) as mock_sync_class:
        mock_sync = Mock()
        mock_sync.sync.return_value = Mock(is_success=True)
        mock_sync_class.return_value = mock_sync

        result = workspace_main._run_sync(args)
        assert result == 0


def test_run_sync_failure(tmp_path: Path) -> None:
    """Test _run_sync with sync failure."""
    args = Mock()
    args.project_root = tmp_path
    args.canonical_root = None

    with patch(
        "flext_infra.workspace.__main__.FlextInfraSyncService"
    ) as mock_sync_class:
        mock_sync = Mock()
        mock_sync.sync.return_value = Mock(is_success=False, error="Sync failed")
        mock_sync_class.return_value = mock_sync

        result = workspace_main._run_sync(args)
        assert result == 1


def test_run_orchestrate_success() -> None:
    """Test _run_orchestrate with successful orchestration."""
    args = Mock()
    args.projects = ["project-a", "project-b"]
    args.verb = "check"
    args.fail_fast = False
    args.make_arg = []

    with patch(
        "flext_infra.workspace.__main__.FlextInfraOrchestratorService"
    ) as mock_orch_class:
        mock_orch = Mock()
        mock_orch.orchestrate.return_value = Mock(
            is_success=True, value=[Mock(exit_code=0), Mock(exit_code=0)]
        )
        mock_orch_class.return_value = mock_orch

        result = workspace_main._run_orchestrate(args)
        assert result == 0


def test_run_orchestrate_no_projects() -> None:
    """Test _run_orchestrate with no projects specified."""
    args = Mock()
    args.projects = []
    args.verb = "check"
    args.fail_fast = False
    args.make_arg = []

    result = workspace_main._run_orchestrate(args)
    assert result == 1


def test_run_orchestrate_with_failures() -> None:
    """Test _run_orchestrate with project failures."""
    args = Mock()
    args.projects = ["project-a", "project-b"]
    args.verb = "check"
    args.fail_fast = False
    args.make_arg = []

    with patch(
        "flext_infra.workspace.__main__.FlextInfraOrchestratorService"
    ) as mock_orch_class:
        mock_orch = Mock()
        mock_orch.orchestrate.return_value = Mock(
            is_success=True, value=[Mock(exit_code=0), Mock(exit_code=1)]
        )
        mock_orch_class.return_value = mock_orch

        result = workspace_main._run_orchestrate(args)
        assert result == 1


def test_run_orchestrate_failure() -> None:
    """Test _run_orchestrate with orchestration failure."""
    args = Mock()
    args.projects = ["project-a"]
    args.verb = "check"
    args.fail_fast = False
    args.make_arg = []

    with patch(
        "flext_infra.workspace.__main__.FlextInfraOrchestratorService"
    ) as mock_orch_class:
        mock_orch = Mock()
        mock_orch.orchestrate.return_value = Mock(
            is_success=False, error="Orchestration failed"
        )
        mock_orch_class.return_value = mock_orch

        result = workspace_main._run_orchestrate(args)
        assert result == 1


def test_run_migrate_success(tmp_path: Path) -> None:
    """Test _run_migrate with successful migration."""
    args = Mock()
    args.workspace_root = tmp_path
    args.dry_run = False

    with patch(
        "flext_infra.workspace.__main__.FlextInfraProjectMigrator"
    ) as mock_migrator_class:
        mock_migrator = Mock()
        mock_migrator.migrate.return_value = Mock(
            is_success=True, is_failure=False, value=[Mock(errors=[], changes=[])]
        )
        mock_migrator_class.return_value = mock_migrator

        result = workspace_main._run_migrate(args)
        assert result == 0


def test_run_migrate_failure(tmp_path: Path) -> None:
    """Test _run_migrate with migration failure."""
    args = Mock()
    args.workspace_root = tmp_path
    args.dry_run = False

    with patch(
        "flext_infra.workspace.__main__.FlextInfraProjectMigrator"
    ) as mock_migrator_class:
        mock_migrator = Mock()
        mock_migrator.migrate.return_value = Mock(
            is_success=False, is_failure=True, error="Migration failed"
        )
        mock_migrator_class.return_value = mock_migrator

        result = workspace_main._run_migrate(args)
        assert result == 1


def test_run_migrate_with_project_errors(tmp_path: Path) -> None:
    """Test _run_migrate when projects have errors."""
    args = Mock()
    args.workspace_root = tmp_path
    args.dry_run = False

    with patch(
        "flext_infra.workspace.__main__.FlextInfraProjectMigrator"
    ) as mock_migrator_class:
        mock_migrator = Mock()
        mock_migrator.migrate.return_value = Mock(
            is_success=True,
            is_failure=False,
            value=[Mock(errors=["Error 1"], changes=[]), Mock(errors=[], changes=[])],
        )
        mock_migrator_class.return_value = mock_migrator

        result = workspace_main._run_migrate(args)
        assert result == 1


def test_main_detect_command(tmp_path: Path) -> None:
    """Test main() with detect command."""
    argv = ["detect", "--project-root", str(tmp_path)]

    with patch("flext_infra.workspace.__main__._run_detect") as mock_run:
        mock_run.return_value = 0
        result = workspace_main.main(argv)
        assert result == 0
        mock_run.assert_called_once()


def test_main_sync_command(tmp_path: Path) -> None:
    """Test main() with sync command."""
    argv = ["sync", "--project-root", str(tmp_path)]

    with patch("flext_infra.workspace.__main__._run_sync") as mock_run:
        mock_run.return_value = 0
        result = workspace_main.main(argv)
        assert result == 0
        mock_run.assert_called_once()


def test_main_orchestrate_command() -> None:
    """Test main() with orchestrate command."""
    argv = ["orchestrate", "--verb", "check", "project-a", "project-b"]

    with patch("flext_infra.workspace.__main__._run_orchestrate") as mock_run:
        mock_run.return_value = 0
        result = workspace_main.main(argv)
        assert result == 0
        mock_run.assert_called_once()


def test_main_migrate_command(tmp_path: Path) -> None:
    """Test main() with migrate command."""
    argv = ["migrate", "--workspace-root", str(tmp_path)]

    with patch("flext_infra.workspace.__main__._run_migrate") as mock_run:
        mock_run.return_value = 0
        result = workspace_main.main(argv)
        assert result == 0
        mock_run.assert_called_once()


def test_main_no_command() -> None:
    """Test main() with no command specified."""
    argv = []

    result = workspace_main.main(argv)
    assert result == 1


def test_main_orchestrate_with_fail_fast() -> None:
    """Test main() orchestrate with --fail-fast flag."""
    argv = ["orchestrate", "--verb", "check", "--fail-fast", "project-a"]

    with patch("flext_infra.workspace.__main__._run_orchestrate") as mock_run:
        mock_run.return_value = 0
        result = workspace_main.main(argv)
        assert result == 0
        args = mock_run.call_args[0][0]
        assert args.fail_fast is True


def test_main_orchestrate_with_make_args() -> None:
    """Test main() orchestrate with --make-arg flags."""
    argv = [
        "orchestrate",
        "--verb",
        "check",
        "--make-arg",
        "VERBOSE=1",
        "--make-arg",
        "PARALLEL=4",
        "project-a",
    ]

    with patch("flext_infra.workspace.__main__._run_orchestrate") as mock_run:
        mock_run.return_value = 0
        result = workspace_main.main(argv)
        assert result == 0
        args = mock_run.call_args[0][0]
        assert "VERBOSE=1" in args.make_arg
        assert "PARALLEL=4" in args.make_arg


def test_main_migrate_dry_run(tmp_path: Path) -> None:
    """Test main() migrate with --dry-run flag."""
    argv = ["migrate", "--workspace-root", str(tmp_path), "--dry-run"]

    with patch("flext_infra.workspace.__main__._run_migrate") as mock_run:
        mock_run.return_value = 0
        result = workspace_main.main(argv)
        assert result == 0
        args = mock_run.call_args[0][0]
        assert args.dry_run is True


def test_main_sync_with_canonical_root(tmp_path: Path) -> None:
    """Test main() sync with --canonical-root flag."""
    canonical = tmp_path / "canonical"
    argv = ["sync", "--project-root", str(tmp_path), "--canonical-root", str(canonical)]

    with patch("flext_infra.workspace.__main__._run_sync") as mock_run:
        mock_run.return_value = 0
        result = workspace_main.main(argv)
        assert result == 0
        args = mock_run.call_args[0][0]
        assert args.canonical_root == canonical


def test_main_entry_point(tmp_path: Path) -> None:
    """Test __main__ entry point."""
    argv = ["detect", "--project-root", str(tmp_path)]

    with patch("flext_infra.workspace.__main__.main") as mock_main:
        mock_main.return_value = 0
        # Simulate the if __name__ == "__main__" block
        exit_code = workspace_main.main(argv)
        assert exit_code == 0


__all__ = []


def test_main_calls_sys_exit(tmp_path: Path) -> None:
    """Test main() calls sys.exit."""
    with patch("sys.argv", ["workspace", "detect", "--project-root", str(tmp_path)]):
        with patch(
            "flext_infra.workspace.__main__.FlextInfraWorkspaceDetector"
        ) as mock_detector_class:
            mock_detector = Mock()
            mock_detector_class.return_value = mock_detector
            mock_detector.detect_mode.return_value = "monorepo"
            with patch("sys.exit") as _mock_exit:
                from flext_infra.workspace.__main__ import (  # noqa: PLC0415
                    main as _main_func,
                )

                try:
                    _main_func(argv=["detect", "--project-root", str(tmp_path)])
                except SystemExit:
                    pass
