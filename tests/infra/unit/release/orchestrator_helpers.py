"""Tests for FlextInfraReleaseOrchestrator helper methods.

Tests _version_files, _build_targets, _run_make, _generate_notes,
_update_changelog, _bump_next_dev, and _dispatch_phase.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from flext_core import r
from flext_infra import FlextInfraModels
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
from flext_tests import tm
from tests.infra.unit.release._stubs import (
    FakeReporting,
    FakeSelection,
    FakeSubprocess,
    FakeUtilsNamespace,
    FakeVersioning,
)

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

_m = FlextInfraModels
_U_PATH = "flext_infra.release.orchestrator.u"


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Create workspace root with pyproject.toml."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "Makefile").touch()
    (root / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
    return root


class TestVersionFiles:
    """Tests for _version_files."""

    def test_includes_workspace_root(self, workspace_root: Path) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        files = orchestrator._version_files(workspace_root, [])
        tm.that(any(f.name == "pyproject.toml" for f in files), eq=True)

    def test_discovery(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        proj_dir = workspace_root / "proj1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").touch()
        fake_sel = FakeSelection()
        mock_project = SimpleNamespace(name="proj1", path=proj_dir)
        fake_sel._resolve_result = r[list[SimpleNamespace]].ok([mock_project])
        monkeypatch.setattr(
            "flext_infra.release.orchestrator.FlextInfraUtilitiesSelection",
            lambda *a, **kw: fake_sel,
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator._version_files(workspace_root, ["proj1"])
        tm.that(len(result), length_gt=0)


class TestBuildTargets:
    """Tests for _build_targets."""

    def test_includes_root(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_sel = FakeSelection()
        monkeypatch.setattr(
            "flext_infra.release.orchestrator.FlextInfraUtilitiesSelection",
            lambda *a, **kw: fake_sel,
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        targets = orchestrator._build_targets(workspace_root, [])
        tm.that(targets[0], eq=("root", workspace_root))

    def test_deduplication(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_sel = FakeSelection()
        mock_project = SimpleNamespace(name="proj1", path=workspace_root / "proj1")
        fake_sel._resolve_result = r[list[SimpleNamespace]].ok([mock_project])
        monkeypatch.setattr(
            "flext_infra.release.orchestrator.FlextInfraUtilitiesSelection",
            lambda *a, **kw: fake_sel,
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator._build_targets(workspace_root, ["proj1"])
        names = [name for name, _ in result]
        tm.that(len(names), eq=len(set(names)))


class TestRunMake:
    """Tests for _run_make."""

    def test_success(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        fake_sp = FakeSubprocess()
        output_model = _m.Infra.Core.CommandOutput(
            exit_code=0, stdout="build ok", stderr=""
        )
        fake_sp._run_raw_result = r[_m.Infra.Core.CommandOutput].ok(output_model)
        monkeypatch.setattr(
            "flext_infra.release.orchestrator.FlextInfraUtilitiesSubprocess",
            lambda *a, **kw: fake_sp,
        )
        result = FlextInfraReleaseOrchestrator._run_make(workspace_root, "build")
        tm.ok(result)
        code, _output = result.value
        tm.that(code, eq=0)

    def test_failure(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        fake_sp = FakeSubprocess()
        fake_sp._run_raw_result = r[_m.Infra.Core.CommandOutput].fail("command failed")
        monkeypatch.setattr(
            "flext_infra.release.orchestrator.FlextInfraUtilitiesSubprocess",
            lambda *a, **kw: fake_sp,
        )
        result = FlextInfraReleaseOrchestrator._run_make(workspace_root, "build")
        tm.fail(result)


class TestGenerateNotes:
    """Tests for _generate_notes."""

    def test_writes_file(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        FakeUtilsNamespace.Infra.reset()
        monkeypatch.setattr(_U_PATH, FakeUtilsNamespace)
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_previous_tag",
            lambda *a, **kw: r[str].ok(""),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_collect_changes",
            lambda *a, **kw: r[str].ok(""),
        )
        fake_sel = FakeSelection()
        monkeypatch.setattr(
            "flext_infra.release.orchestrator.FlextInfraUtilitiesSelection",
            lambda *a, **kw: fake_sel,
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "notes.md"
        result = orchestrator._generate_notes(
            workspace_root, "1.0.0", "v1.0.0", [], notes_path
        )
        tm.ok(result)
        tm.that(notes_path.exists(), eq=True)


class TestUpdateChangelog:
    """Tests for _update_changelog."""

    def test_creates_files(self, workspace_root: Path) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "notes.md"
        notes_path.write_text("# Release v1.0.0\n")
        result = orchestrator._update_changelog(
            workspace_root, "1.0.0", "v1.0.0", notes_path
        )
        tm.ok(result)
        changelog = workspace_root / "docs" / "CHANGELOG.md"
        tm.that(changelog.exists(), eq=True)

    def test_appends_to_existing(self, workspace_root: Path) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        changelog = workspace_root / "docs" / "CHANGELOG.md"
        changelog.parent.mkdir(parents=True)
        changelog.write_text("# Changelog\n\n## 0.9.0 - 2025-01-01\n")
        notes_path = workspace_root / "notes.md"
        notes_path.write_text("# Release v1.0.0\n")
        result = orchestrator._update_changelog(
            workspace_root, "1.0.0", "v1.0.0", notes_path
        )
        tm.ok(result)
        tm.that(changelog.read_text(), contains="1.0.0")


class TestBumpNextDev:
    """Tests for _bump_next_dev."""

    def test_bumps_version(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "flext_infra.release.orchestrator.FlextInfraUtilitiesVersioning",
            lambda *a, **kw: FakeVersioning(),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "phase_version",
            lambda *a, **kw: r[bool].ok(True),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator._bump_next_dev(workspace_root, "1.0.0", [], "minor"))

    def test_bump_failure(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        fake_vs = FakeVersioning()
        fake_vs._bump_result = r[str].fail("invalid bump")
        monkeypatch.setattr(
            "flext_infra.release.orchestrator.FlextInfraUtilitiesVersioning",
            lambda *a, **kw: fake_vs,
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator._bump_next_dev(workspace_root, "1.0.0", [], "invalid"))


class TestDispatchPhase:
    """Tests for _dispatch_phase."""

    def test_unknown_phase(self, workspace_root: Path) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator._dispatch_phase(
            "unknown",
            workspace_root,
            "1.0.0",
            "v1.0.0",
            [],
            dry_run=False,
            push=False,
            dev_suffix=False,
        )
        tm.fail(result)
        tm.that(isinstance(result.error, str), eq=True)
        tm.that(result.error, contains="unknown phase")

    def test_routes_validate(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "phase_validate",
            lambda *a, **kw: r[bool].ok(True),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(
            orchestrator._dispatch_phase(
                "validate",
                workspace_root,
                "1.0.0",
                "v1.0.0",
                [],
                dry_run=False,
                push=False,
                dev_suffix=False,
            )
        )
