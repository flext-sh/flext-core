"""Tests for FlextInfraWorkflowSyncer workspace sync and report writing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra import m
from flext_infra.github.workflows import FlextInfraWorkflowSyncer, SyncOperation
from flext_tests import tm
from tests.infra.unit.github._stubs import (
    StubJsonIo,
    StubSelector,
    StubTemplates,
)


def _as_project(info: m.Infra.Workspace.ProjectInfo) -> m.Infra.Workspace.ProjectInfo:
    return info


def _syncer(
    selector: StubSelector | None = None,
    json_io: StubJsonIo | None = None,
    templates: StubTemplates | None = None,
) -> FlextInfraWorkflowSyncer:
    return FlextInfraWorkflowSyncer(
        selector=selector or StubSelector(),
        json_io=json_io or StubJsonIo(),
        templates=templates or StubTemplates(),
    )


class TestSyncWorkspace:
    """Test sync_workspace method."""

    def test_full_flow(self, tmp_path: Path) -> None:
        """Test full workspace sync flow."""
        wf = tmp_path / ".github" / "workflows"
        wf.mkdir(parents=True)
        (wf / "ci.yml").write_text("name: CI\n")
        proj = m.Infra.Workspace.ProjectInfo(
            name="my-proj",
            path=tmp_path / "my-proj",
            stack="python",
        )
        proj.path.mkdir()
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].ok([
                _as_project(proj)
            ]),
        )
        json_io = StubJsonIo()
        tm.ok(
            _syncer(selector=selector, json_io=json_io).sync_workspace(
                tmp_path,
                apply=True,
            )
        )

    def test_source_resolution_failure(self, tmp_path: Path) -> None:
        """Test workspace sync with source resolution failure."""
        tm.fail(_syncer().sync_workspace(tmp_path))

    def test_project_discovery_failure(self, tmp_path: Path) -> None:
        """Test workspace sync with project discovery failure."""
        wf = tmp_path / ".github" / "workflows"
        wf.mkdir(parents=True)
        (wf / "ci.yml").write_text("name: CI")
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].fail("no projects"),
        )
        tm.fail(_syncer(selector=selector).sync_workspace(tmp_path))

    def test_with_report_path(self, tmp_path: Path) -> None:
        """Test workspace sync writes report."""
        wf = tmp_path / ".github" / "workflows"
        wf.mkdir(parents=True)
        (wf / "ci.yml").write_text("name: CI\n")
        selector = StubSelector(
            resolve_returns=r[list[m.Infra.Workspace.ProjectInfo]].ok([]),
        )
        json_io = StubJsonIo()
        report = tmp_path / "report.json"
        tm.ok(
            _syncer(selector=selector, json_io=json_io).sync_workspace(
                tmp_path,
                report_path=report,
            )
        )
        tm.that(len(json_io.write_json_calls), eq=1)

    def test_template_render_failure(self, tmp_path: Path) -> None:
        """Test workspace sync with template render failure."""
        wf = tmp_path / ".github" / "workflows"
        wf.mkdir(parents=True)
        ci = wf / "ci.yml"
        ci.write_text("name: CI")
        syncer = _syncer()

        def _render_template(path: Path) -> r[str]:
            _ = path
            return r[str].fail("render error")

        def _resolve_source_workflow(
            root: Path,
            source: Path | None = None,
        ) -> r[Path]:
            _ = root, source
            return r[Path].ok(ci)

        object.__setattr__(
            syncer,
            "render_template",
            _render_template,
        )
        object.__setattr__(
            syncer,
            "resolve_source_workflow",
            _resolve_source_workflow,
        )
        tm.fail(syncer.sync_workspace(tmp_path))


class TestWriteReport:
    """Test _write_report method."""

    def test_write_report(self, tmp_path: Path) -> None:
        """Test report writing with operations."""
        json_io = StubJsonIo()
        syncer = _syncer(json_io=json_io)
        ops = [
            SyncOperation(project="p1", path="ci.yml", action="create", reason="new"),
            SyncOperation(project="p2", path="ci.yml", action="noop", reason="synced"),
        ]
        report_path = tmp_path / "report.json"
        syncer._write_report(report_path, apply=True, operations=ops)
        tm.that(len(json_io.write_json_calls), eq=1)
        payload = json_io.write_json_calls[0][1]
        tm.that("mode" in str(payload), eq=True)
        tm.that("summary" in str(payload), eq=True)

    def test_write_report_dry_run(self, tmp_path: Path) -> None:
        """Test report writing in dry-run mode."""
        json_io = StubJsonIo()
        syncer = _syncer(json_io=json_io)
        report_path = tmp_path / "report.json"
        syncer._write_report(report_path, apply=False, operations=[])
        payload = json_io.write_json_calls[0][1]
        tm.that("mode" in str(payload), eq=True)
