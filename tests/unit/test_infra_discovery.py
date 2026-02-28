"""Tests for FlextInfraDiscoveryService.

Tests cover project discovery, pyproject file discovery, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_core import r
from flext_infra import FlextInfraDiscoveryService, m


class TestFlextInfraDiscoveryService:
    """Test suite for FlextInfraDiscoveryService."""

    @pytest.fixture
    def service(self) -> FlextInfraDiscoveryService:
        """Create a discovery service instance."""
        return FlextInfraDiscoveryService()

    @pytest.fixture
    def workspace_with_projects(self, tmp_path: Path) -> Path:
        """Create a temporary workspace with test projects."""
        # Create project1 (python/submodule)
        proj1 = tmp_path / "project1"
        proj1.mkdir()
        (proj1 / ".git").mkdir()
        (proj1 / "Makefile").touch()
        (proj1 / "pyproject.toml").touch()
        (proj1 / "src").mkdir()
        (proj1 / "tests").mkdir()

        # Create project2 (go/external)
        proj2 = tmp_path / "project2"
        proj2.mkdir()
        (proj2 / ".git").mkdir()
        (proj2 / "Makefile").touch()
        (proj2 / "go.mod").touch()

        # Create invalid project (no Makefile)
        invalid = tmp_path / "invalid"
        invalid.mkdir()
        (invalid / ".git").mkdir()
        (invalid / "pyproject.toml").touch()

        # Create hidden directory (should be skipped)
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / ".git").mkdir()
        (hidden / "Makefile").touch()

        return tmp_path

    def test_discover_projects_happy_path(
        self,
        service: FlextInfraDiscoveryService,
        workspace_with_projects: Path,
    ) -> None:
        """Test discovering projects in a valid workspace."""
        result = service.discover_projects(workspace_with_projects)

        assert result.is_success
        projects = result.value
        assert len(projects) == 2
        assert projects[0].name == "project1"
        assert projects[1].name == "project2"
        assert projects[0].has_tests is True
        assert projects[0].has_src is True
        assert projects[1].has_src is False

    def test_discover_projects_empty_workspace(
        self,
        service: FlextInfraDiscoveryService,
        tmp_path: Path,
    ) -> None:
        """Test discovering projects in an empty workspace."""
        result = service.discover_projects(tmp_path)

        assert result.is_success
        assert result.value == []

    def test_discover_projects_nonexistent_path(
        self,
        service: FlextInfraDiscoveryService,
    ) -> None:
        """Test discovering projects with nonexistent path."""
        nonexistent = Path("/nonexistent/path/to/workspace")
        result = service.discover_projects(nonexistent)

        assert result.is_failure
        assert "discovery failed" in result.error

    def test_find_all_pyproject_files_happy_path(
        self,
        service: FlextInfraDiscoveryService,
        tmp_path: Path,
    ) -> None:
        """Test finding all pyproject.toml files."""
        # Create nested structure
        (tmp_path / "project1").mkdir()
        (tmp_path / "project1" / "pyproject.toml").touch()
        (tmp_path / "project2").mkdir()
        (tmp_path / "project2" / "pyproject.toml").touch()
        (tmp_path / "project2" / "subdir").mkdir()
        (tmp_path / "project2" / "subdir" / "pyproject.toml").touch()

        result = service.find_all_pyproject_files(tmp_path)

        assert result.is_success
        files = result.value
        assert len(files) == 3
        assert all(f.name == "pyproject.toml" for f in files)

    def test_find_all_pyproject_files_with_skip_dirs(
        self,
        service: FlextInfraDiscoveryService,
        tmp_path: Path,
    ) -> None:
        """Test finding pyproject files with directory exclusion."""
        (tmp_path / "project1").mkdir()
        (tmp_path / "project1" / "pyproject.toml").touch()
        (tmp_path / "skip_me").mkdir()
        (tmp_path / "skip_me" / "pyproject.toml").touch()

        result = service.find_all_pyproject_files(
            tmp_path,
            skip_dirs=frozenset({"skip_me"}),
        )

        assert result.is_success
        files = result.value
        assert len(files) == 1
        assert "skip_me" not in str(files[0])

    def test_find_all_pyproject_files_with_project_paths(
        self,
        service: FlextInfraDiscoveryService,
        tmp_path: Path,
    ) -> None:
        """Test finding pyproject files for specific projects."""
        proj1 = tmp_path / "project1"
        proj2 = tmp_path / "project2"
        proj1.mkdir()
        proj2.mkdir()
        (proj1 / "pyproject.toml").touch()
        (proj2 / "pyproject.toml").touch()

        result = service.find_all_pyproject_files(
            tmp_path,
            project_paths=[proj1],
        )

        assert result.is_success
        files = result.value
        assert len(files) == 1
        assert files[0].parent == proj1

    def test_discover_projects_result_type(
        self,
        service: FlextInfraDiscoveryService,
        workspace_with_projects: Path,
    ) -> None:
        """Test that result is properly typed FlextResult."""
        result = service.discover_projects(workspace_with_projects)

        assert isinstance(result, type(r[list[m.ProjectInfo]].ok([])))
        assert result.is_success
        assert isinstance(result.value, list)
        assert all(isinstance(p, m.ProjectInfo) for p in result.value)

    def test_discover_projects_empty_workspace_v2(
        self, service: FlextInfraDiscoveryService, tmp_path: Path
    ) -> None:
        """Test discover_projects returns empty list for empty workspace."""
        result = service.discover_projects(tmp_path)
        assert result.is_success
        assert result.value == []
