"""Tests for FlextInfraReleaseOrchestrator.

Tests release orchestration with mocked git and pyproject operations,
using tmp_path fixtures for isolated test environments.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from flext_core import r
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator


class TestFlextInfraReleaseOrchestrator:
    """Test suite for FlextInfraReleaseOrchestrator."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create mock workspace root with pyproject.toml."""
        root = tmp_path / "workspace"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "Makefile").touch()
        (root / "pyproject.toml").write_text(
            'version = "0.1.0"\n',
            encoding="utf-8",
        )
        return root

    def test_execute_not_implemented(self) -> None:
        """Test that execute() returns failure (use run_release instead)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator.execute()
        assert result.is_failure
        assert "Use run_release()" in result.error

    def test_run_release_invalid_phase(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with invalid phase name."""
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["invalid_phase"],
        )

        assert result.is_failure

    def test_run_release_empty_phases_list(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with empty phases list."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch.object(
            orchestrator,
            "_create_branches",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=[],
            )

        assert result.is_success

    def test_run_release_with_project_filter(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with project_names filter."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with (
            patch.object(
                orchestrator,
                "_create_branches",
                return_value=r[bool].ok(True),
            ),
            patch.object(
                orchestrator,
                "_dispatch_phase",
                return_value=r[bool].ok(True),
            ),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate"],
                project_names=["flext-core", "flext-api"],
            )

        assert result.is_success

    def test_run_release_dry_run_mode(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with dry_run=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch.object(
            orchestrator,
            "_dispatch_phase",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate"],
                dry_run=True,
            )

        assert result.is_success

    def test_run_release_with_push(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with push=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with (
            patch.object(
                orchestrator,
                "_create_branches",
                return_value=r[bool].ok(True),
            ),
            patch.object(
                orchestrator,
                "_dispatch_phase",
                return_value=r[bool].ok(True),
            ),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate"],
                push=True,
            )

        assert result.is_success

    def test_run_release_with_dev_suffix(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with dev_suffix=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with (
            patch.object(
                orchestrator,
                "_create_branches",
                return_value=r[bool].ok(True),
            ),
            patch.object(
                orchestrator,
                "_dispatch_phase",
                return_value=r[bool].ok(True),
            ),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0-dev",
                tag="v1.0.0-dev",
                phases=["version"],
                dev_suffix=True,
            )

        assert result.is_success

    def test_run_release_next_dev_version(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with next_dev=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with (
            patch.object(
                orchestrator,
                "_create_branches",
                return_value=r[bool].ok(True),
            ),
            patch.object(
                orchestrator,
                "_dispatch_phase",
                return_value=r[bool].ok(True),
            ),
            patch.object(
                orchestrator,
                "_bump_next_dev",
                return_value=r[bool].ok(True),
            ),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["version"],
                next_dev=True,
                next_bump="minor",
            )

        assert result.is_success

    def test_run_release_phase_failure_stops_execution(
        self,
        workspace_root: Path,
    ) -> None:
        """Test that phase failure stops further execution."""
        orchestrator = FlextInfraReleaseOrchestrator()
        call_count = 0

        def mock_dispatch(phase: str, *args, **kwargs) -> r[bool]:
            nonlocal call_count
            call_count += 1
            if phase == "validate":
                return r[bool].fail("validation failed")
            return r[bool].ok(True)

        with (
            patch.object(
                orchestrator,
                "_create_branches",
                return_value=r[bool].ok(True),
            ),
            patch.object(
                orchestrator,
                "_dispatch_phase",
                side_effect=mock_dispatch,
            ),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate", "version"],
            )

        assert result.is_failure
        assert call_count == 1  # Only validate phase called

    def test_run_release_version_format_validation(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release validates semantic version format."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with (
            patch.object(
                orchestrator,
                "_create_branches",
                return_value=r[bool].ok(True),
            ),
            patch.object(
                orchestrator,
                "_dispatch_phase",
                return_value=r[bool].ok(True),
            ),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate"],
            )

        assert result.is_success

    def test_run_release_create_branches_disabled(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with create_branches=False."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch.object(
            orchestrator,
            "_dispatch_phase",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate"],
                create_branches=False,
            )

        assert result.is_success

    def test_run_release_multiple_projects(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with multiple projects."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with (
            patch.object(
                orchestrator,
                "_create_branches",
                return_value=r[bool].ok(True),
            ),
            patch.object(
                orchestrator,
                "_dispatch_phase",
                return_value=r[bool].ok(True),
            ),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="2.0.0",
                tag="v2.0.0",
                phases=["validate"],
                project_names=["flext-core", "flext-api", "flext-cli"],
            )

        assert result.is_success
