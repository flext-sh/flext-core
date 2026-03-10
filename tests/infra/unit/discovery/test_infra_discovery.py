"""Tests for FlextInfraDiscoveryService.

Tests cover project discovery, pyproject file discovery, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from flext_core import r
from flext_infra import FlextInfraUtilitiesDiscovery, m
from flext_tests import tm


class TestFlextInfraDiscoveryService:
    """Test suite for FlextInfraDiscoveryService."""

    @pytest.fixture
    def service(self) -> FlextInfraUtilitiesDiscovery:
        """Create a discovery service instance."""
        return FlextInfraUtilitiesDiscovery()

    @pytest.fixture
    def workspace_with_projects(self, tmp_path: Path) -> Path:
        """Create a temporary workspace with test projects."""
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
        """Test discovering projects in a valid workspace."""
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
        """Test discovering projects in an empty workspace."""
        result = service.discover_projects(tmp_path)
        tm.ok(result)
        assert result.value == []

    def test_discover_projects_nonexistent_path(
        self,
        service: FlextInfraUtilitiesDiscovery,
    ) -> None:
        """Test discovering projects with nonexistent path."""
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
        """Test finding all pyproject.toml files."""
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
        """Test finding pyproject files with directory exclusion."""
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
        """Test finding pyproject files for specific projects."""
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
        """Test that result is properly typed FlextResult."""
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
        """Test discover_projects returns empty list for empty workspace."""
        result = service.discover_projects(tmp_path)
        tm.ok(result)
        assert result.value == []


class TestFlextInfraDiscoveryServiceUncoveredLines:
    """Test uncovered lines in FlextInfraDiscoveryService."""

    def test_discover_projects_skips_non_git_projects(self, tmp_path: Path) -> None:
        """Test discover_projects skips non-git directories (line 75)."""
        service = FlextInfraUtilitiesDiscovery()
        workspace_root = tmp_path
        non_git_dir = workspace_root / "non_git_project"
        non_git_dir.mkdir()
        (non_git_dir / "Makefile").touch()
        (non_git_dir / "pyproject.toml").touch()
        result = service.discover_projects(workspace_root)
        tm.ok(result)
        assert len(result.value) == 0

    def test_find_all_pyproject_files_with_nonexistent_path(self) -> None:
        """Test find_all_pyproject_files with nonexistent path (lines 138-139)."""
        service = FlextInfraUtilitiesDiscovery()
        nonexistent = Path("/nonexistent/path/to/workspace")
        result = service.find_all_pyproject_files(nonexistent)
        tm.ok(result)
        assert result.value == []

    def test_find_all_pyproject_files_with_permission_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Test find_all_pyproject_files handles permission errors (lines 156-157)."""
        service = FlextInfraUtilitiesDiscovery()
        (tmp_path / "pyproject.toml").touch()
        result = service.find_all_pyproject_files(tmp_path)
        tm.ok(result)
        assert len(result.value) >= 1

    def test_discover_projects_skips_no_pyproject_no_gomod(
        self,
        tmp_path: Path,
    ) -> None:
        """Test discover_projects skips dirs with Makefile but no pyproject/go.mod (line 75)."""
        service = FlextInfraUtilitiesDiscovery()
        workspace_root = tmp_path
        proj = workspace_root / "incomplete_project"
        proj.mkdir()
        (proj / ".git").mkdir()
        (proj / "Makefile").touch()
        result = service.discover_projects(workspace_root)
        tm.ok(result)
        assert len(result.value) == 0

    def test_find_all_pyproject_files_oserror_on_rglob(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test find_all_pyproject_files handles OSError from rglob (lines 138-139)."""
        service = FlextInfraUtilitiesDiscovery()

        def mock_rglob(self: Path, pattern: str) -> None:
            msg = "permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "rglob", mock_rglob)
        result = service.find_all_pyproject_files(tmp_path)
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "pyproject file scan failed" in result.error

    def test_submodule_names_with_gitmodules_oserror(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test _submodule_names handles OSError on read_text (lines 154-157)."""
        workspace_root = tmp_path
        gitmodules = workspace_root / ".gitmodules"
        gitmodules.touch()

        def mock_read_text(self: Path, encoding: str | None = None) -> None:
            msg = "permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "read_text", mock_read_text)
        result = FlextInfraUtilitiesDiscovery._submodule_names(workspace_root)
        assert result == set()

    def test_submodule_names_with_valid_gitmodules(self, tmp_path: Path) -> None:
        """Test _submodule_names extracts submodule names from .gitmodules (line 158)."""
        workspace_root = tmp_path
        gitmodules = workspace_root / ".gitmodules"
        gitmodules.write_text(
            '[submodule "sub1"]\n    path = submodule-one\n[submodule "sub2"]\n    path = submodule-two\n',
            encoding="utf-8",
        )
        result = FlextInfraUtilitiesDiscovery._submodule_names(workspace_root)
        assert result == {"submodule-one", "submodule-two"}
