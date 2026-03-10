"""Tests for FlextInfraReleaseOrchestrator phase methods.

Tests phase_validate, phase_version, and phase_build using monkeypatch
and tmp_path fixtures for isolated test environments.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core import r
from flext_infra import FlextInfraModels
from flext_infra.release import orchestrator as _orch_mod
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
from flext_tests import tm
from tests.infra.unit.release._stubs import (
    FakeReporting,
    FakeSubprocess,
    FakeVersioning,
)

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

_m = FlextInfraModels


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Create workspace root with pyproject.toml."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "Makefile").touch()
    (root / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
    return root


class TestPhaseValidate:
    """Tests for phase_validate."""

    def test_dry_run(self, workspace_root: Path) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator.phase_validate(workspace_root, dry_run=True))

    def test_executes_make(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_sp = FakeSubprocess()
        monkeypatch.setattr(
            _orch_mod, "FlextInfraUtilitiesSubprocess", lambda *a, **kw: fake_sp
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator.phase_validate(workspace_root, dry_run=False))
        tm.that(fake_sp._run_checked_called, eq=True)


class TestPhaseVersion:
    """Tests for phase_version."""

    def test_updates_files(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesVersioning",
            lambda *a, **kw: FakeVersioning(),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator.phase_version(workspace_root, "1.0.0", [], dry_run=False))

    def test_invalid_semver(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_vs = FakeVersioning()
        fake_vs._parse_result = r[str].fail("invalid version")
        monkeypatch.setattr(
            _orch_mod, "FlextInfraUtilitiesVersioning", lambda *a, **kw: fake_vs
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator.phase_version(workspace_root, "invalid", []))

    def test_with_dev_suffix(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesVersioning",
            lambda *a, **kw: FakeVersioning(),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator.phase_version(workspace_root, "1.0.0", [], dev_suffix=True))

    def test_dry_run(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesVersioning",
            lambda *a, **kw: FakeVersioning(),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator.phase_version(workspace_root, "1.0.0", [], dry_run=True))

    def test_skips_missing_files(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            orchestrator,
            "_version_files",
            lambda *a, **kw: [workspace_root / "nonexistent.toml"],
        )
        tm.ok(orchestrator.phase_version(workspace_root, "1.0.0", []))


class TestPhaseBuild:
    """Tests for phase_build."""

    def test_creates_report_dir(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_rep = FakeReporting()
        fake_rep._report_dir = workspace_root / "reports"
        monkeypatch.setattr(
            _orch_mod, "FlextInfraUtilitiesReporting", lambda *a, **kw: fake_rep
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_run_make",
            lambda *a, **kw: r[tuple[int, str]].ok((0, "ok")),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator.phase_build(workspace_root, "1.0.0", []))

    def test_report_dir_creation_fails(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_rep = FakeReporting()
        fake_rep._report_dir = workspace_root / "reports"
        monkeypatch.setattr(
            _orch_mod, "FlextInfraUtilitiesReporting", lambda *a, **kw: fake_rep
        )
        monkeypatch.setattr(
            "pathlib.Path.mkdir",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("permission denied")),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator.phase_build(workspace_root, "1.0.0", []))

    def test_with_make_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_build_targets",
            lambda *a, **kw: [("root", workspace_root)],
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_run_make",
            lambda *a, **kw: r[tuple[int, str]].fail("make failed"),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator.phase_build(workspace_root, "1.0.0", []))
