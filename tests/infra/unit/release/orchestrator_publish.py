"""Tests for FlextInfraReleaseOrchestrator phase_publish."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core import r, t
from flext_infra.release import orchestrator as _orch_mod
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
from flext_tests import tm
from tests.infra.unit.release._stubs import FakeReporting

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


def _stub_publish(mp: MonkeyPatch, root: Path) -> None:
    """Stub reporting + notes for publish tests."""
    fake_rep = FakeReporting()
    fake_rep._report_dir = root / "reports"

    def _reporting_factory(*a: t.Scalar, **kw: t.Scalar) -> FakeReporting:
        del a, kw
        return fake_rep

    def _generate_notes(*a: t.Scalar, **kw: t.Scalar) -> r[bool]:
        del a, kw
        return r[bool].ok(True)

    mp.setattr(_orch_mod, "FlextInfraUtilitiesReporting", _reporting_factory)
    mp.setattr(_CLS, "_generate_notes", _generate_notes)


def _stub_full_publish(mp: MonkeyPatch, root: Path) -> None:
    """Stub reporting + notes + changelog + tag for full publish."""
    _stub_publish(mp, root)

    def _update_changelog(*a: t.Scalar, **kw: t.Scalar) -> r[bool]:
        del a, kw
        return r[bool].ok(True)

    def _create_tag(*a: t.Scalar, **kw: t.Scalar) -> r[bool]:
        del a, kw
        return r[bool].ok(True)

    mp.setattr(_CLS, "_update_changelog", _update_changelog)
    mp.setattr(_CLS, "_create_tag", _create_tag)


class TestPhasePublish:
    def test_generates_notes(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        _stub_publish(monkeypatch, workspace_root)
        tm.ok(_CLS().phase_publish(workspace_root, "1.0.0", "v1.0.0", [], dry_run=True))

    def test_dry_run_skips_changelog(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        changelog_called = False

        def fake_changelog(*args: str, **kwargs: str) -> r[bool]:
            nonlocal changelog_called
            changelog_called = True
            return r[bool].ok(True)

        _stub_publish(monkeypatch, workspace_root)
        monkeypatch.setattr(_CLS, "_update_changelog", fake_changelog)
        tm.ok(_CLS().phase_publish(workspace_root, "1.0.0", "v1.0.0", [], dry_run=True))
        tm.that(changelog_called, eq=False)

    def test_updates_changelog(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        _stub_full_publish(monkeypatch, workspace_root)
        tm.ok(
            _CLS().phase_publish(workspace_root, "1.0.0", "v1.0.0", [], dry_run=False)
        )

    def test_with_push(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        push_called = False

        def fake_push(*args: str, **kwargs: str) -> r[bool]:
            nonlocal push_called
            push_called = True
            return r[bool].ok(True)

        _stub_full_publish(monkeypatch, workspace_root)
        monkeypatch.setattr(_CLS, "_push_release", fake_push)
        tm.ok(
            _CLS().phase_publish(
                workspace_root, "1.0.0", "v1.0.0", [], dry_run=False, push=True
            )
        )
        tm.that(push_called, eq=True)

    def test_notes_generation_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        def _generate_notes(*a: t.Scalar, **kw: t.Scalar) -> r[bool]:
            del a, kw
            return r[bool].fail("notes failed")

        monkeypatch.setattr(
            _CLS,
            "_generate_notes",
            _generate_notes,
        )
        tm.fail(
            _CLS().phase_publish(workspace_root, "1.0.0", "v1.0.0", [], dry_run=False)
        )

    def test_changelog_update_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        _stub_publish(monkeypatch, workspace_root)

        def _update_changelog(*a: t.Scalar, **kw: t.Scalar) -> r[bool]:
            del a, kw
            return r[bool].fail("changelog failed")

        monkeypatch.setattr(
            _CLS,
            "_update_changelog",
            _update_changelog,
        )
        tm.fail(
            _CLS().phase_publish(workspace_root, "1.0.0", "v1.0.0", [], dry_run=False)
        )

    def test_tag_creation_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        _stub_publish(monkeypatch, workspace_root)

        def _update_changelog(*a: t.Scalar, **kw: t.Scalar) -> r[bool]:
            del a, kw
            return r[bool].ok(True)

        def _create_tag(*a: t.Scalar, **kw: t.Scalar) -> r[bool]:
            del a, kw
            return r[bool].fail("tag failed")

        monkeypatch.setattr(_CLS, "_update_changelog", _update_changelog)
        monkeypatch.setattr(
            _CLS,
            "_create_tag",
            _create_tag,
        )
        tm.fail(
            _CLS().phase_publish(workspace_root, "1.0.0", "v1.0.0", [], dry_run=False)
        )
