"""Tests for FlextInfraPythonVersionEnforcer.

Tests Python version enforcement across workspace projects with mocked
pyproject.toml operations and tmp_path fixtures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from flext_core import FlextResult as r
from flext_infra.discovery import FlextInfraDiscoveryService
from flext_infra.maintenance.python_version import FlextInfraPythonVersionEnforcer


class TestFlextInfraPythonVersionEnforcer:
    """Test suite for FlextInfraPythonVersionEnforcer."""

    @pytest.fixture
    def enforcer(self) -> FlextInfraPythonVersionEnforcer:
        """Create enforcer instance."""
        return FlextInfraPythonVersionEnforcer()

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create mock workspace root with required markers."""
        root = tmp_path / "workspace"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "Makefile").touch()
        (root / "pyproject.toml").write_text(
            'requires-python = ">=3.13"\n',
            encoding="utf-8",
        )
        return root

    def test_execute_check_only_success(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        workspace_root: Path,
    ) -> None:
        """Test execute with check_only=True returns success."""
        with (
            patch.object(
                enforcer,
                "_discover_projects",
                return_value=[workspace_root],
            ),
            patch("sys.version_info") as mock_version,
        ):
            mock_version.minor = 13
            result = enforcer.execute(check_only=True, verbose=False)

        assert result.is_success
        assert result.value == 0

    def test_execute_enforce_mode(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        workspace_root: Path,
    ) -> None:
        """Test execute in enforce mode (not check_only)."""
        with (
            patch.object(
                enforcer,
                "_discover_projects",
                return_value=[workspace_root],
            ),
            patch("sys.version_info") as mock_version,
        ):
            mock_version.minor = 13
            result = enforcer.execute(check_only=False, verbose=False)

        assert result.is_success
        assert result.value == 0

    def test_read_required_minor_from_pyproject(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        workspace_root: Path,
    ) -> None:
        """Test reading required Python minor version from pyproject.toml."""
        minor = enforcer._read_required_minor(workspace_root)
        assert minor == 13

    def test_read_required_minor_fallback_to_13(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test fallback to minor version 13 when pyproject.toml missing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        minor = enforcer._read_required_minor(empty_dir)
        assert minor == 13

    def test_read_required_minor_malformed_pyproject(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test fallback when pyproject.toml is malformed."""
        root = tmp_path / "malformed"
        root.mkdir()
        (root / "pyproject.toml").write_text(
            "# No requires-python field\n",
            encoding="utf-8",
        )
        minor = enforcer._read_required_minor(root)
        assert minor == 13

    def test_workspace_root_from_file_success(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        workspace_root: Path,
    ) -> None:
        """Test resolving workspace root from file path."""
        test_file = workspace_root / "src" / "module.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        resolved = enforcer._workspace_root_from_file(test_file)
        assert resolved == workspace_root

    def test_workspace_root_from_file_not_found(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test RuntimeError when workspace root cannot be found."""
        orphan_file = tmp_path / "orphan.py"
        orphan_file.touch()

        with pytest.raises(RuntimeError, match="workspace root not found"):
            enforcer._workspace_root_from_file(orphan_file)

    def test_discover_projects_empty(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        workspace_root: Path,
    ) -> None:
        """Test discovering projects when none exist."""
        with (
            patch.object(
                enforcer,
                "_discover_projects",
                return_value=[],
            ),
            patch("sys.version_info") as mock_version,
        ):
            mock_version.minor = 13
            result = enforcer.execute(check_only=True)

        assert result.is_success

    def test_ensure_python_version_file_mismatch(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test validation fails when local minor version mismatches."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            'requires-python = ">=3.12"\n',
            encoding="utf-8",
        )

        enforcer.check_only = True
        result = enforcer._ensure_python_version_file(project, required_minor=13)
        assert result is False

    def test_ensure_python_version_file_match(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test validation succeeds when versions match."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            'requires-python = ">=3.13"\n',
            encoding="utf-8",
        )

        enforcer.check_only = True
        enforcer.verbose = False
        with patch("sys.version_info") as mock_version:
            mock_version.minor = 13
            result = enforcer._ensure_python_version_file(project, required_minor=13)

        assert result is True

    def test_execute_with_verbose_logging(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        workspace_root: Path,
    ) -> None:
        """Test execute with verbose=True logs detailed output."""
        with (
            patch.object(
                enforcer,
                "_discover_projects",
                return_value=[workspace_root],
            ),
            patch("sys.version_info") as mock_version,
        ):
            mock_version.minor = 13
            result = enforcer.execute(check_only=True, verbose=True)

        assert result.is_success
        assert result.is_success
        assert enforcer.verbose is True

    def test_ensure_python_version_file_runtime_mismatch(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test validation fails when runtime minor version mismatches."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            'requires-python = ">=3.13"\n',
            encoding="utf-8",
        )

        enforcer.check_only = False
        with patch("sys.version_info") as mock_version:
            mock_version.minor = 12  # Mismatch
            result = enforcer._ensure_python_version_file(project, required_minor=13)

        assert result is False

    def test_ensure_python_version_file_verbose_logging(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test verbose logging when validation succeeds."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            'requires-python = ">=3.13"\n',
            encoding="utf-8",
        )

        enforcer.check_only = True
        enforcer.verbose = True
        with patch("sys.version_info") as mock_version:
            mock_version.minor = 13
            result = enforcer._ensure_python_version_file(project, required_minor=13)

        assert result is True

    def test_execute_failure_on_workspace_root_mismatch(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test execute fails when workspace root has version mismatch."""
        root = tmp_path / "workspace"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "Makefile").touch()
        (root / "pyproject.toml").write_text(
            'requires-python = ">=3.12"\n',
            encoding="utf-8",
        )

        with patch.object(enforcer, "_workspace_root_from_file", return_value=root):
            result = enforcer.execute(check_only=True, verbose=False)

        assert result.is_failure

    def test_execute_failure_on_project_mismatch(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        workspace_root: Path,
    ) -> None:
        """Test execute fails when a project has version mismatch."""
        project = workspace_root / "project-a"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            'requires-python = ">=3.12"\n',
            encoding="utf-8",
        )

        with (
            patch.object(
                enforcer,
                "_discover_projects",
                return_value=[project],
            ),
            patch("sys.version_info") as mock_version,
        ):
            mock_version.minor = 13
            result = enforcer.execute(check_only=True, verbose=False)

        assert result.is_failure

    def test_discover_projects_failure_returns_empty_list(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        workspace_root: Path,
    ) -> None:
        """Test _discover_projects returns empty list when discovery fails (line 167).

        When discovery.discover_projects returns a failure result,
        _discover_projects should return an empty list.
        """
        with patch.object(
            FlextInfraDiscoveryService,
            "discover_projects",
            return_value=r[list].fail("discovery error"),
        ):
            result = enforcer._discover_projects(workspace_root)

            assert result == []

    def test_ensure_python_version_file_enforce_mode_logs_error(
        self,
        enforcer: FlextInfraPythonVersionEnforcer,
        tmp_path: Path,
    ) -> None:
        """Test _ensure_python_version_file logs error in enforce mode (lines 196-202).

        When check_only=False and version mismatch is found,
        logger.error should be called with mismatch and manual update messages.
        """
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            'requires-python = ">=3.12"\n',
            encoding="utf-8",
        )

        enforcer.check_only = False
        with patch("flext_infra.maintenance.python_version.logger") as mock_logger:
            with patch("sys.version_info") as mock_version:
                mock_version.minor = 13
                result = enforcer._ensure_python_version_file(
                    project, required_minor=13
                )

                # Should return False due to mismatch
                assert result is False
                # logger.error should be called twice for enforce mode
                assert mock_logger.error.call_count >= 2
