"""Tests for FlextInfraDiscoveryService.

Tests cover project discovery, pyproject file discovery, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraUtilitiesDiscovery, m
from flext_tests import tm


class TestFlextInfraDiscoveryService:
    @pytest.fixture
    def service(self) -> FlextInfraUtilitiesDiscovery:
        return FlextInfraUtilitiesDiscovery()

    @pytest.fixture
    def workspace_with_projects(self, tmp_path: Path) -> Path:
        proj1 = tmp_path / "project1"
        proj1.mkdir()
        (proj1 / ".git").mkdir()
        (proj1 / "Makefile").touch()
        (proj1 / "pyproject.toml").touch()
        (proj1 / "src").mkdir()
        (proj1 / "tests").mkdir()
        proj2 = tmp_path / "project2"
        proj2.mkdir()
        (proj2 / ".git").mkdir()
        (proj2 / "Makefile").touch()
        (proj2 / "go.mod").touch()
        invalid = tmp_path / "invalid"
        invalid.mkdir()
        (invalid / ".git").mkdir()
        (invalid / "pyproject.toml").touch()
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / ".git").mkdir()
        (hidden / "Makefile").touch()
        return tmp_path

    def test_discover_projects_happy_path(
        self,
        service: FlextInfraUtilitiesDiscovery,
        workspace_with_projects: Path,
    ) -> None:
        result = service.discover_projects(workspace_with_projects)
        tm.ok(result)
        projects = result.value
        assert len(projects) == 2
        assert projects[0].name == "project1"
        assert projects[1].name == "project2"
        assert projects[0].has_tests is True
        assert projects[0].has_src is True
        assert projects[1].has_src is False

    def test_discover_projects_empty_workspace(
        self,
        service: FlextInfraUtilitiesDiscovery,
        tmp_path: Path,
    ) -> None:
        result = service.discover_projects(tmp_path)
        tm.ok(result)
        assert result.value == []

    def test_discover_projects_nonexistent_path(
        self,
        service: FlextInfraUtilitiesDiscovery,
    ) -> None:
        nonexistent = Path("/nonexistent/path/to/workspace")
        result = service.discover_projects(nonexistent)
        tm.fail(result)
        assert isinstance(result.error, str)
        assert isinstance(result.error, str)
        assert "discovery failed" in result.error

    def test_find_all_pyproject_files_happy_path(
        self,
        service: FlextInfraUtilitiesDiscovery,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "project1").mkdir()
        (tmp_path / "project1" / "pyproject.toml").touch()
        (tmp_path / "project2").mkdir()
        (tmp_path / "project2" / "pyproject.toml").touch()
        (tmp_path / "project2" / "subdir").mkdir()
        (tmp_path / "project2" / "subdir" / "pyproject.toml").touch()
        result = service.find_all_pyproject_files(tmp_path)
        tm.ok(result)
        files = result.value
        assert len(files) == 3
        assert all(f.name == "pyproject.toml" for f in files)

    def test_find_all_pyproject_files_with_skip_dirs(
        self,
        service: FlextInfraUtilitiesDiscovery,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "project1").mkdir()
        (tmp_path / "project1" / "pyproject.toml").touch()
        (tmp_path / "skip_me").mkdir()
        (tmp_path / "skip_me" / "pyproject.toml").touch()
        result = service.find_all_pyproject_files(
            tmp_path,
            skip_dirs=frozenset({"skip_me"}),
        )
        tm.ok(result)
        files = result.value
        assert len(files) == 1
        assert "skip_me" not in str(files[0])

    def test_find_all_pyproject_files_with_project_paths(
        self,
        service: FlextInfraUtilitiesDiscovery,
        tmp_path: Path,
    ) -> None:
        proj1 = tmp_path / "project1"
        proj2 = tmp_path / "project2"
        proj1.mkdir()
        proj2.mkdir()
        (proj1 / "pyproject.toml").touch()
        (proj2 / "pyproject.toml").touch()
        result = service.find_all_pyproject_files(tmp_path, project_paths=[proj1])
        tm.ok(result)
        files = result.value
        assert len(files) == 1
        assert files[0].parent == proj1

    def test_discover_projects_result_type(
        self,
        service: FlextInfraUtilitiesDiscovery,
        workspace_with_projects: Path,
    ) -> None:
        result = service.discover_projects(workspace_with_projects)
        assert isinstance(result, type(r[list[m.Infra.Workspace.ProjectInfo]].ok([])))
        tm.ok(result)
        projects: list[m.Infra.Workspace.ProjectInfo] = result.value
        for item in projects:
            assert isinstance(item, m.Infra.Workspace.ProjectInfo)

    def test_discover_projects_empty_workspace_v2(
        self,
        service: FlextInfraUtilitiesDiscovery,
        tmp_path: Path,
    ) -> None:
        result = service.discover_projects(tmp_path)
        tm.ok(result)
        assert result.value == []


__all__: list[str] = []
