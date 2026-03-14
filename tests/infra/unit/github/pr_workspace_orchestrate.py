"""Tests for FlextInfraPrWorkspaceManager — orchestrate and static methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import m
from flext_infra.github import pr_workspace as pw_mod
from flext_infra.github.pr_workspace import FlextInfraPrWorkspaceManager
from flext_tests import tm
from tests.infra.unit.github._stubs import (
    StubProjectInfo,
    StubReporting,
    StubRunner,
    StubSelector,
)


def _as_project(info: StubProjectInfo) -> m.Infra.Workspace.ProjectInfo:
    return m.Infra.Workspace.ProjectInfo.model_validate(info.model_dump())


class TestOrchestrate:
    def test_all_success(self, tmp_path: Path) -> None:
        runner = StubRunner(run_to_file_returns=[r[int].ok(0)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        proj = StubProjectInfo(name="proj", path=tmp_path / "proj", stack="python")
        proj.path.mkdir()
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].ok([
                _as_project(proj)
            ])
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=selector, reporting=reporting
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, checkpoint=False, branch=""
        )
        value = tm.ok(result)
        tm.that(value.fail, eq=0)

    def test_project_resolution_failure(self, tmp_path: Path) -> None:
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].fail("no projects")
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=StubRunner(), selector=selector, reporting=StubReporting()
        )
        result = manager.orchestrate(tmp_path)
        tm.fail(result)

    def test_fail_fast(self, tmp_path: Path) -> None:
        runner = StubRunner(run_to_file_returns=[r[int].ok(1)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        p1 = StubProjectInfo(name="p1", path=tmp_path / "p1", stack="python")
        p1.path.mkdir()
        p2 = StubProjectInfo(name="p2", path=tmp_path / "p2", stack="python")
        p2.path.mkdir()
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].ok([
                _as_project(p1),
                _as_project(p2),
            ])
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=selector, reporting=reporting
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, fail_fast=True, checkpoint=False, branch=""
        )
        value = tm.ok(result)
        tm.that(value.fail >= 1, eq=True)

    def test_include_root(self, tmp_path: Path) -> None:
        runner = StubRunner(run_to_file_returns=[r[int].ok(0)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].ok([])
        )
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
        runner = StubRunner(run_to_file_returns=[r[int].ok(0)])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        proj = StubProjectInfo(name="proj", path=tmp_path / "proj", stack="python")
        proj.path.mkdir()
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].ok([
                _as_project(proj)
            ])
        )
        has_changes_calls: list[Path] = []

        def _has_changes(root: Path) -> r[bool]:
            has_changes_calls.append(root)
            return r[bool].ok(False)

        def _git_checkout(root: Path, branch: str) -> r[bool]:
            _ = root, branch
            return r[bool].ok(True)

        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_has_changes",
            _has_changes,
        )
        monkeypatch.setattr(
            pw_mod.u.Infra,
            "git_checkout",
            _git_checkout,
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
        runner = StubRunner(run_to_file_returns=[r[int].fail("command error")])
        reporting = StubReporting(report_dir=tmp_path / "reports")
        proj = StubProjectInfo(name="proj", path=tmp_path / "proj", stack="python")
        proj.path.mkdir()
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].ok([
                _as_project(proj)
            ])
        )
        manager = FlextInfraPrWorkspaceManager(
            runner=runner, selector=selector, reporting=reporting
        )
        result = manager.orchestrate(
            tmp_path, include_root=False, fail_fast=True, checkpoint=False, branch=""
        )
        value = tm.ok(result)
        tm.that(value.fail, eq=1)


class TestStaticMethods:
    def test_repo_display_name_root(self, tmp_path: Path) -> None:
        display_name = getattr(FlextInfraPrWorkspaceManager, "_repo_display_name")
        result = display_name(tmp_path, tmp_path)
        tm.that(result, eq=tmp_path.name)

    def test_repo_display_name_subproject(self, tmp_path: Path) -> None:
        sub = tmp_path / "my-project"
        sub.mkdir()
        display_name = getattr(FlextInfraPrWorkspaceManager, "_repo_display_name")
        result = display_name(sub, tmp_path)
        tm.that(result, eq="my-project")

    def test_build_root_command(self, tmp_path: Path) -> None:
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
        build_subproject_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_subproject_command"
        )
        cmd = build_subproject_command(tmp_path, {"action": "status", "head": "feat"})
        tm.that("make" in cmd, eq=True)
        tm.that("-C" in cmd, eq=True)
        tm.that("PR_ACTION=status" in cmd, eq=True)

    def test_build_root_command_defaults(self, tmp_path: Path) -> None:
        build_root_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_root_command"
        )
        cmd = build_root_command(tmp_path, {})
        tm.that("--action" in cmd, eq=True)
        tm.that("status" in cmd, eq=True)

    def test_build_subproject_command_no_optional(self, tmp_path: Path) -> None:
        build_subproject_command = getattr(
            FlextInfraPrWorkspaceManager, "_build_subproject_command"
        )
        cmd = build_subproject_command(tmp_path, {})
        tm.that("make" in cmd, eq=True)
        tm.that(not [c for c in cmd if c.startswith("PR_HEAD=")], eq=True)
