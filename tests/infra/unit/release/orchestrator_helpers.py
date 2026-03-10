"""Tests for FlextInfraReleaseOrchestrator helper methods."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from flext_core import r
from flext_infra.release import orchestrator as _orch_mod
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
from flext_tests import tm
from tests.infra.models import m as _m
from tests.infra.unit.release._stubs import (
    FakeSelection,
    FakeSubprocess,
    FakeUtilsNamespace,
    FakeVersioning,
)

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

_CLS = FlextInfraReleaseOrchestrator


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "Makefile").touch()
    (root / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
    return root


def _patch_sel(mp: MonkeyPatch, sel: FakeSelection) -> None:
    mp.setattr(_orch_mod, "FlextInfraUtilitiesSelection", lambda *a, **kw: sel)


def _patch_sp(mp: MonkeyPatch, sp: FakeSubprocess) -> None:
    mp.setattr(_orch_mod, "FlextInfraUtilitiesSubprocess", lambda *a, **kw: sp)


class TestVersionFiles:
    def test_includes_workspace_root(self, workspace_root: Path) -> None:
        files = _CLS()._version_files(workspace_root, [])
        tm.that(any(f.name == "pyproject.toml" for f in files), eq=True)

    def test_discovery(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        proj_dir = workspace_root / "proj1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").touch()
        fake_sel = FakeSelection()
        fake_sel._resolve_result = r[list[SimpleNamespace]].ok([
            SimpleNamespace(name="proj1", path=proj_dir)
        ])
        _patch_sel(monkeypatch, fake_sel)
        tm.that(len(_CLS()._version_files(workspace_root, ["proj1"])), gt=0)


class TestBuildTargets:
    """Tests for _build_targets."""

    def test_includes_root(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        _patch_sel(monkeypatch, FakeSelection())
        targets = _CLS()._build_targets(workspace_root, [])
        name, path = targets[0]
        tm.that(name, eq="root")
        tm.that(str(path), eq=str(workspace_root))

    def test_deduplication(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_sel = FakeSelection()
        fake_sel._resolve_result = r[list[SimpleNamespace]].ok([
            SimpleNamespace(name="proj1", path=workspace_root / "proj1")
        ])
        _patch_sel(monkeypatch, fake_sel)
        names = [n for n, _ in _CLS()._build_targets(workspace_root, ["proj1"])]
        tm.that(len(names), eq=len(set(names)))


class TestRunMake:
    """Tests for _run_make."""

    def test_success(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        fake_sp = FakeSubprocess()
        output = _m.Infra.Core.CommandOutput(exit_code=0, stdout="ok", stderr="")
        fake_sp._run_raw_result = r[_m.Infra.Core.CommandOutput].ok(output)
        _patch_sp(monkeypatch, fake_sp)
        result = _CLS._run_make(workspace_root, "build")
        tm.ok(result)
        code, _out = result.value
        tm.that(code, eq=0)

    def test_failure(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        fake_sp = FakeSubprocess()
        fake_sp._run_raw_result = r[_m.Infra.Core.CommandOutput].fail("failed")
        _patch_sp(monkeypatch, fake_sp)
        tm.fail(_CLS._run_make(workspace_root, "build"))


class TestGenerateNotes:
    """Tests for _generate_notes."""

    def test_writes_file(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        FakeUtilsNamespace.Infra.reset()
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        monkeypatch.setattr(_CLS, "_previous_tag", lambda *a, **kw: r[str].ok(""))
        monkeypatch.setattr(_CLS, "_collect_changes", lambda *a, **kw: r[str].ok(""))
        _patch_sel(monkeypatch, FakeSelection())
        notes_path = workspace_root / "notes.md"
        result = _CLS()._generate_notes(
            workspace_root, "1.0.0", "v1.0.0", [], notes_path
        )
        tm.ok(result)
        tm.that(notes_path.exists(), eq=True)


class TestUpdateChangelog:
    """Tests for _update_changelog."""

    def test_creates_files(self, workspace_root: Path) -> None:
        notes = workspace_root / "notes.md"
        notes.write_text("# Release v1.0.0\n")
        result = _CLS()._update_changelog(workspace_root, "1.0.0", "v1.0.0", notes)
        tm.ok(result)
        tm.that((workspace_root / "docs" / "CHANGELOG.md").exists(), eq=True)

    def test_appends_to_existing(self, workspace_root: Path) -> None:
        changelog = workspace_root / "docs" / "CHANGELOG.md"
        changelog.parent.mkdir(parents=True)
        changelog.write_text("# Changelog\n\n## 0.9.0 - 2025-01-01\n")
        notes = workspace_root / "notes.md"
        notes.write_text("# Release v1.0.0\n")
        tm.ok(_CLS()._update_changelog(workspace_root, "1.0.0", "v1.0.0", notes))
        tm.that(changelog.read_text(), contains="1.0.0")


class TestBumpNextDev:
    """Tests for _bump_next_dev."""

    def test_bumps_version(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesVersioning",
            lambda *a, **kw: FakeVersioning(),
        )
        monkeypatch.setattr(_CLS, "phase_version", lambda *a, **kw: r[bool].ok(True))
        tm.ok(_CLS()._bump_next_dev(workspace_root, "1.0.0", [], "minor"))

    def test_bump_failure(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        fake_vs = FakeVersioning()
        fake_vs._bump_result = r[str].fail("invalid bump")
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesVersioning",
            lambda *a, **kw: fake_vs,
        )
        tm.fail(_CLS()._bump_next_dev(workspace_root, "1.0.0", [], "invalid"))


class TestDispatchPhase:
    """Tests for _dispatch_phase."""

    @staticmethod
    def _dispatch(
        orch: FlextInfraReleaseOrchestrator, phase: str, root: Path
    ) -> r[bool]:
        return orch._dispatch_phase(
            phase,
            root,
            "1.0.0",
            "v1.0.0",
            [],
            dry_run=False,
            push=False,
            dev_suffix=False,
        )

    def test_unknown_phase(self, workspace_root: Path) -> None:
        result = self._dispatch(_CLS(), "unknown", workspace_root)
        tm.fail(result)
        tm.that(result.error, contains="unknown phase")

    def test_routes_validate(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_CLS, "phase_validate", lambda *a, **kw: r[bool].ok(True))
        tm.ok(self._dispatch(_CLS(), "validate", workspace_root))
