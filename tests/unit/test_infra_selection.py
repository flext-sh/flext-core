"""Tests for FlextInfraProjectSelector.

Tests cover project resolution, filtering, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraDiscoveryService, FlextInfraProjectSelector, m


class TestFlextInfraProjectSelector:
    """Test suite for FlextInfraProjectSelector."""

    @pytest.fixture
    def workspace_with_projects(self, tmp_path: Path) -> Path:
        """Create a temporary workspace with test projects."""
        for name in ["alpha", "beta", "gamma"]:
            proj = tmp_path / name
            proj.mkdir()
            (proj / ".git").mkdir()
            (proj / "Makefile").touch()
            (proj / "pyproject.toml").touch()
            (proj / "src").mkdir()

        return tmp_path

    @pytest.fixture
    def discovery_service(self) -> FlextInfraDiscoveryService:
        """Create a discovery service."""
        return FlextInfraDiscoveryService()

    @pytest.fixture
    def selector(
        self,
        discovery_service: FlextInfraDiscoveryService,
    ) -> FlextInfraProjectSelector:
        """Create a project selector with discovery service."""
        return FlextInfraProjectSelector(discovery=discovery_service)

    def test_resolve_projects_all_projects(
        self,
        selector: FlextInfraProjectSelector,
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving all projects when names list is empty."""
        result = selector.resolve_projects(workspace_with_projects, [])

        assert result.is_success
        projects = result.value
        assert len(projects) == 3
        assert [p.name for p in projects] == ["alpha", "beta", "gamma"]

    def test_resolve_projects_specific_names(
        self,
        selector: FlextInfraProjectSelector,
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving specific projects by name."""
        result = selector.resolve_projects(
            workspace_with_projects,
            ["beta", "alpha"],
        )

        assert result.is_success
        projects = result.value
        assert len(projects) == 2
        assert [p.name for p in projects] == ["alpha", "beta"]

    def test_resolve_projects_single_project(
        self,
        selector: FlextInfraProjectSelector,
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving a single project."""
        result = selector.resolve_projects(
            workspace_with_projects,
            ["gamma"],
        )

        assert result.is_success
        projects = result.value
        assert len(projects) == 1
        assert projects[0].name == "gamma"

    def test_resolve_projects_unknown_project(
        self,
        selector: FlextInfraProjectSelector,
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving with unknown project name."""
        result = selector.resolve_projects(
            workspace_with_projects,
            ["unknown"],
        )

        assert result.is_failure
        assert result.error and "unknown projects" in result.error
        assert result.error and "unknown" in result.error

    def test_resolve_projects_mixed_known_unknown(
        self,
        selector: FlextInfraProjectSelector,
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving with mix of known and unknown projects."""
        result = selector.resolve_projects(
            workspace_with_projects,
            ["alpha", "unknown", "beta"],
        )

        assert result.is_failure
        assert result.error and "unknown projects" in result.error
        assert result.error and "unknown" in result.error

    def test_resolve_projects_discovery_failure(
        self,
        workspace_with_projects: Path,
    ) -> None:
        """Test handling discovery service failure."""
        selector = FlextInfraProjectSelector(discovery=None)
        nonexistent = Path("/nonexistent/workspace")

        result = selector.resolve_projects(nonexistent, ["alpha"])

        assert result.is_failure
        assert result.error and ("discovery failed" in result.error or "failed" in result.error)

    def test_resolve_projects_sorted_output(
        self,
        selector: FlextInfraProjectSelector,
        workspace_with_projects: Path,
    ) -> None:
        """Test that resolved projects are sorted by name."""
        result = selector.resolve_projects(
            workspace_with_projects,
            ["gamma", "alpha", "beta"],
        )

        assert result.is_success
        projects = result.value
        assert [p.name for p in projects] == ["alpha", "beta", "gamma"]

    def test_resolve_projects_result_type(
        self,
        selector: FlextInfraProjectSelector,
        workspace_with_projects: Path,
    ) -> None:
        """Test that result is properly typed FlextResult."""
        result = selector.resolve_projects(workspace_with_projects, [])

        assert isinstance(result, type(r[list[m.ProjectInfo]].ok([])))
        assert result.is_success
        assert isinstance(result.value, list)
        assert all(isinstance(p, m.ProjectInfo) for p in result.value)

    def test_selector_with_default_discovery(
        self,
        workspace_with_projects: Path,
    ) -> None:
        """Test selector creates default discovery service if not provided."""
        selector = FlextInfraProjectSelector()
        result = selector.resolve_projects(workspace_with_projects, [])

        assert result.is_success
        assert len(result.value) == 3
