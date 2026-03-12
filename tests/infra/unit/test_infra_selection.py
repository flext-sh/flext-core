"""Tests for FlextInfraUtilitiesSelection.

Tests cover project resolution, filtering, and error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesSelection
from flext_tests import tm


class TestFlextInfraUtilitiesSelection:
    """Test suite for FlextInfraUtilitiesSelection."""

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
    def selector(self) -> type[FlextInfraUtilitiesSelection]:
        """Provide project selector utilities class."""
        return FlextInfraUtilitiesSelection

    def test_resolve_projects_all_projects(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving all projects when names list is empty."""
        result = selector.resolve_projects(workspace_with_projects, [])
        projects = tm.ok(result)
        tm.that(projects, length=3)
        tm.that([p.name for p in projects], eq=["alpha", "beta", "gamma"])

    def test_resolve_projects_specific_names(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving specific projects by name."""
        result = selector.resolve_projects(workspace_with_projects, ["beta", "alpha"])
        projects = tm.ok(result)
        tm.that(projects, length=2)
        tm.that([p.name for p in projects], eq=["alpha", "beta"])

    def test_resolve_projects_single_project(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving a single project."""
        result = selector.resolve_projects(workspace_with_projects, ["gamma"])
        projects = tm.ok(result)
        tm.that(projects, length=1)
        tm.that(projects[0].name, eq="gamma")

    def test_resolve_projects_unknown_project(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving with unknown project name."""
        result = selector.resolve_projects(workspace_with_projects, ["unknown"])
        tm.fail(result, has="unknown projects")

    def test_resolve_projects_mixed_known_unknown(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        workspace_with_projects: Path,
    ) -> None:
        """Test resolving with mix of known and unknown projects."""
        result = selector.resolve_projects(
            workspace_with_projects, ["alpha", "unknown", "beta"]
        )
        tm.fail(result, has="unknown projects")

    def test_resolve_projects_discovery_failure(
        self,
        selector: type[FlextInfraUtilitiesSelection],
    ) -> None:
        """Test handling discovery failure with non-existent path."""
        result = selector.resolve_projects(Path("/nonexistent/path"), ["alpha"])
        tm.fail(result)

    def test_resolve_projects_sorted_output(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        workspace_with_projects: Path,
    ) -> None:
        """Test that resolved projects are sorted by name."""
        result = selector.resolve_projects(
            workspace_with_projects, ["gamma", "alpha", "beta"]
        )
        projects = tm.ok(result)
        tm.that([p.name for p in projects], eq=["alpha", "beta", "gamma"])

    def test_resolve_projects_result_type(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        workspace_with_projects: Path,
    ) -> None:
        """Test that result contains properly typed ProjectInfo items."""
        result = selector.resolve_projects(workspace_with_projects, [])
        assert result.is_success
        projects = result.value
        tm.that(len(projects), eq=3)
        tm.that([p.name for p in projects], eq=["alpha", "beta", "gamma"])

    def test_selector_with_default_discovery(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        workspace_with_projects: Path,
    ) -> None:
        """Test selector uses default discovery service implicitly."""
        result = selector.resolve_projects(workspace_with_projects, [])
        projects = tm.ok(result)
        tm.that(projects, length=3)

    def test_selector_resolve_projects_empty_list(
        self,
        selector: type[FlextInfraUtilitiesSelection],
        tmp_path: Path,
    ) -> None:
        """Test resolve_projects returns empty list when no projects match."""
        result = selector.resolve_projects(tmp_path, [])
        projects = tm.ok(result)
        tm.that(projects, eq=[])
