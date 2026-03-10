"""Tests for FlextInfraReleaseOrchestrator phase_publish.

Tests release publish phase using monkeypatch and tmp_path fixtures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core import r
from flext_infra.release import orchestrator as _orch_mod
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
from flext_tests import tm
from tests.infra.unit.release._stubs import FakeReporting

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Create workspace root with pyproject.toml."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "Makefile").touch()
    (root / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
    return root


class TestPhasePublish:
    """Tests for phase_publish."""

    def test_generates_notes(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_rep = FakeReporting()
        fake_rep._report_dir = workspace_root / "reports"
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesReporting",
            lambda *a, **kw: fake_rep,
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_generate_notes",
            lambda *a, **kw: r[bool].ok(True),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(
            orchestrator.phase_publish(
                workspace_root, "1.0.0", "v1.0.0", [], dry_run=True
            )
        )

    def test_dry_run_skips_changelog(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        changelog_called = False

        def fake_changelog(*args: str, **kwargs: str) -> r[bool]:
            nonlocal changelog_called
            changelog_called = True
            return r[bool].ok(True)

        fake_rep = FakeReporting()
        fake_rep._report_dir = workspace_root / "reports"
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesReporting",
            lambda *a, **kw: fake_rep,
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_generate_notes",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator, "_update_changelog", fake_changelog
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(
            orchestrator.phase_publish(
                workspace_root, "1.0.0", "v1.0.0", [], dry_run=True
            )
        )
        tm.that(changelog_called, eq=False)

    def test_updates_changelog(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        fake_rep = FakeReporting()
        fake_rep._report_dir = workspace_root / "reports"
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesReporting",
            lambda *a, **kw: fake_rep,
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_generate_notes",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_update_changelog",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_tag",
            lambda *a, **kw: r[bool].ok(True),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(
            orchestrator.phase_publish(
                workspace_root, "1.0.0", "v1.0.0", [], dry_run=False
            )
        )

    def test_with_push(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        push_called = False

        def fake_push(*args: str, **kwargs: str) -> r[bool]:
            nonlocal push_called
            push_called = True
            return r[bool].ok(True)

        fake_rep = FakeReporting()
        fake_rep._report_dir = workspace_root / "reports"
        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesReporting",
            lambda *a, **kw: fake_rep,
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_generate_notes",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_update_changelog",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_tag",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(FlextInfraReleaseOrchestrator, "_push_release", fake_push)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(
            orchestrator.phase_publish(
                workspace_root, "1.0.0", "v1.0.0", [], dry_run=False, push=True
            )
        )
        tm.that(push_called, eq=True)

    def test_notes_generation_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_generate_notes",
            lambda *a, **kw: r[bool].fail("notes generation failed"),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(
            orchestrator.phase_publish(
                workspace_root, "1.0.0", "v1.0.0", [], dry_run=False
            )
        )

    def test_changelog_update_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        notes_path = workspace_root / "RELEASE_NOTES.md"
        notes_path.write_text("# Release v1.0.0\n")
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_generate_notes",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_update_changelog",
            lambda *a, **kw: r[bool].fail("changelog update failed"),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(
            orchestrator.phase_publish(
                workspace_root, "1.0.0", "v1.0.0", [], dry_run=False
            )
        )

    def test_tag_creation_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_generate_notes",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_update_changelog",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_tag",
            lambda *a, **kw: r[bool].fail("tag creation failed"),
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(
            orchestrator.phase_publish(
                workspace_root, "1.0.0", "v1.0.0", [], dry_run=False
            )
        )
