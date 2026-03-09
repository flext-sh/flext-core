"""Tests for FlextInfraPrWorkspaceManager — orchestrate and static methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra.github import pr_workspace as pw_mod
from flext_infra.github.pr_workspace import FlextInfraPrWorkspaceManager
from flext_tests import tm
from tests.infra.unit.github._stubs import StubProjectInfo, StubReporting, StubRunner


class TestOrchestrate:
    def test_all_success(self, tmp_path: Path) -> None:
        """Test orchestrate with all projects succeeding."""
        runner = StubRunner(run_to_file_returns=[r[int].ok(0)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        proj = StubProjectInfo(name="proj", path=tmp_path / "proj")
        proj.path.mkdir()
        selector = _StubSelectorWithProjects(projects=[proj])
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=selector, reporting=reporting
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, checkpoint=False, branch=""
        )
        value = tm.ok(result)
        tm.that(value.fail, eq=0)

    def test_project_resolution_failure(self, tmp_path: Path) -> None:
        """Test orchestrate when project resolution fails."""
        selector = _StubSelectorFailing(error="no projects")
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=selector, reporting=StubReporting()
        )
        result = manager.orchestrate(tmp_path)
        tm.fail(result)

    def test_fail_fast(self, tmp_path: Path) -> None:
        """Test orchestrate with fail_fast stopping on first failure."""
        runner = StubRunner(run_to_file_returns=[r[int].ok(1)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        p1 = StubProjectInfo(name="p1", path=tmp_path / "p1")
        p1.path.mkdir()
        p2 = StubProjectInfo(name="p2", path=tmp_path / "p2")
        p2.path.mkdir()
        selector = _StubSelectorWithProjects(projects=[p1, p2])
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=selector, reporting=reporting
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, fail_fast=True, checkpoint=False, branch=""
        )
        value = tm.ok(result)
        tm.that(value.fail >= 1, eq=True)

    def test_include_root(self, tmp_path: Path) -> None:
        """Test orchestrate includes root repository."""
        runner = StubRunner(run_to_file_returns=[r[int].ok(0)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        selector = _StubSelectorWithProjects(projects=[])
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=selector, reporting=reporting
        )
        result = manager.orchestrate(
            tmp_path, include_root=True, checkpoint=False, branch=""
        )
        value = tm.ok(result)
        tm.that(value.total, eq=1)

    def test_orchestrate_with_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test orchestrate with checkpoint enabled runs checkpoint flow."""
        runner = StubRunner(run_to_file_returns=[r[int].ok(0)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        proj = StubProjectInfo(name="proj", path=tmp_path / "proj")
        proj.path.mkdir()
        selector = _StubSelectorWithProjects(projects=[proj])
        has_changes_calls: list[Path] = []
        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_has_changes",
            lambda root: (has_changes_calls.append(root), r[bool].ok(False))[1],
        )
        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_checkout",
            lambda _root, _branch: r[bool].ok(True),
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=selector, reporting=reporting
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, checkpoint=True, branch="test-branch"
        )
        tm.ok(result)
        tm.that(len(has_changes_calls) >= 1, eq=True)

    def test_orchestrate_failure_handling(self, tmp_path: Path) -> None:
        """Test orchestrate failure handling with fail_fast."""
        runner = StubRunner(run_to_file_returns=[r[int].fail("command error")])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        proj = StubProjectInfo(name="proj", path=tmp_path / "proj")
        proj.path.mkdir()
        selector = _StubSelectorWithProjects(projects=[proj])
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=selector, reporting=reporting
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, fail_fast=True, checkpoint=False, branch=""
        )
        value = tm.ok(result)
        tm.that(value.fail, eq=1)


class TestStaticMethods:
    """Test static utility methods."""

    def test_repo_display_name_root(self, tmp_path: Path) -> None:
        """Test display name for root repository."""
        display_name = getattr(FlextInfraPrWorkspaceManager, "_repo_display_name")
        result = display_name(tmp_path, tmp_path)
        tm.that(result, eq=tmp_path.name)

    def test_repo_display_name_subproject(self, tmp_path: Path) -> None:
        """Test display name for subproject."""
        sub = tmp_path / "my-project"
        sub.mkdir()
        display_name = getattr(FlextInfraPrWorkspaceManager, "_repo_display_name")
        result = display_name(sub, tmp_path)
        tm.that(result, eq="my-project")

    def test_build_root_command(self, tmp_path: Path) -> None:
        """Test root command building."""
        build_root_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_root_command"
        )
        cmd = build_root_command(
            tmp_path, {"action": "create", "head": "feature", "title": "Test"}
        )
        tm.that("python" in cmd, eq=True)
        tm.that("--action" in cmd, eq=True)
        tm.that("create" in cmd, eq=True)

    def test_build_subproject_command(self, tmp_path: Path) -> None:
        """Test subproject command building."""
        build_subproject_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_subproject_command"
        )
        cmd = build_subproject_command(tmp_path, {"action": "status", "head": "feat"})
        tm.that("make" in cmd, eq=True)
        tm.that("-C" in cmd, eq=True)
        tm.that("PR_ACTION=status" in cmd, eq=True)

    def test_build_root_command_defaults(self, tmp_path: Path) -> None:
        """Test root command with default values."""
        build_root_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_root_command"
        )
        cmd = build_root_command(tmp_path, {})
        tm.that("--action" in cmd, eq=True)
        tm.that("status" in cmd, eq=True)

    def test_build_subproject_command_no_optional(self, tmp_path: Path) -> None:
        """Test subproject command without optional keys."""
        build_subproject_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_subproject_command"
        )
        cmd = build_subproject_command(tmp_path, {})
        tm.that("make" in cmd, eq=True)
        tm.that(not [c for c in cmd if c.startswith("PR_HEAD=")], eq=True)


# ---------------------------------------------------------------------------
# Internal stubs for orchestrate tests (selector with project list)
# ---------------------------------------------------------------------------


class _StubSelectorWithProjects:
    """Selector stub that returns a fixed project list."""

    def __init__(self, projects: list[StubProjectInfo]) -> None:
        self._projects = projects

    def resolve_projects(self, *_args: object, **_kwargs: object) -> object:
        return r[list[StubProjectInfo]].ok(self._projects)


class _StubSelectorFailing:
    """Selector stub that always fails."""

    def __init__(self, error: str = "no projects") -> None:
        self._error = error

    def resolve_projects(self, *_args: object, **_kwargs: object) -> object:
        return r[list[StubProjectInfo]].fail(self._error)
