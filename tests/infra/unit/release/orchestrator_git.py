"""Tests for FlextInfraReleaseOrchestrator git operations.

Tests _create_branches, _create_tag, _push_release, _previous_tag,
and _collect_changes using monkeypatch and tmp_path fixtures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from flext_core import r, t
from flext_infra.release import orchestrator as _orch_mod
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
from flext_tests import tm
from tests.infra.unit.release._stubs import FakeSelection, FakeUtilsNamespace

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


class TestCreateBranches:
    """Tests for _create_branches."""

    def test_workspace_only(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        FakeUtilsNamespace.Infra.reset()
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator._create_branches(workspace_root, "1.0.0", []))

    def test_failure(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_checkout_result = r[bool].fail("git failed")
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator._create_branches(workspace_root, "1.0.0", []))

    def test_project_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_checkout_side_effects = [
            r[bool].ok(True),
            r[bool].fail("project branch failed"),
        ]
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        fake_sel = FakeSelection()
        mock_project = SimpleNamespace(name="proj1", path=workspace_root / "proj1")
        fake_sel._resolve_result = r[list[SimpleNamespace]].ok([mock_project])

        def _selection_factory(*a: t.Scalar, **kw: t.Scalar) -> FakeSelection:
            del a, kw
            return fake_sel

        monkeypatch.setattr(
            _orch_mod,
            "FlextInfraUtilitiesSelection",
            _selection_factory,
        )
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator._create_branches(workspace_root, "1.0.0", ["proj1"]))


class TestCreateTag:
    """Tests for _create_tag."""

    def test_creates_new_tag(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        FakeUtilsNamespace.Infra.reset()
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator._create_tag(workspace_root, "v1.0.0"))

    def test_skips_existing(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_tag_exists_result = r[bool].ok(True)
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator._create_tag(workspace_root, "v1.0.0"))

    def test_check_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_tag_exists_result = r[bool].fail(
            "tag check failed"
        )
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator._create_tag(workspace_root, "v1.0.0"))


class TestPushRelease:
    """Tests for _push_release."""

    def test_pushes_branch_and_tag(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        FakeUtilsNamespace.Infra.reset()
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator._push_release(workspace_root, "v1.0.0"))

    def test_branch_failure(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_run_checked_result = r[bool].fail("push failed")
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator._push_release(workspace_root, "v1.0.0"))


class TestPreviousTag:
    """Tests for _previous_tag."""

    def test_finds_tag(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_run_result = r[str].ok("v0.9.0")
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator._previous_tag(workspace_root, "v1.0.0")
        tm.ok(result, eq="v0.9.0")

    def test_no_previous(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_run_result = r[str].ok("")
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator._previous_tag(workspace_root, "v1.0.0")
        tm.ok(result, eq="")

    def test_git_failure(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_run_result = r[str].fail("git error")
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator._previous_tag(workspace_root, "v1.0.0"))


class TestCollectChanges:
    """Tests for _collect_changes."""

    def test_with_tag(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_run_result = r[str].ok(
            "- abc1234 fix: bug (author)"
        )
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator._collect_changes(workspace_root, "v0.9.0", "v1.0.0"))

    def test_git_failure(self, workspace_root: Path, monkeypatch: MonkeyPatch) -> None:
        FakeUtilsNamespace.Infra.reset()
        FakeUtilsNamespace.Infra._git_run_result = r[str].fail("git error")
        monkeypatch.setattr(_orch_mod, "u", FakeUtilsNamespace)
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.fail(orchestrator._collect_changes(workspace_root, "", "HEAD"))
