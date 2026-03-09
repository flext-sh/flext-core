"""Tests for FlextInfraReleaseOrchestrator run_release and execute.

Tests release orchestration top-level methods using monkeypatch
and tmp_path fixtures for isolated test environments.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core import r
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
from flext_tests import tm

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


class TestReleaseOrchestratorExecute:
    """Tests for execute() and run_release() top-level."""

    def test_execute_returns_ok(self) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        tm.ok(orchestrator.execute(), eq=True)

    def test_run_release_invalid_phase(self, workspace_root: Path) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["invalid_phase"],
        )
        tm.fail(result)

    def test_run_release_empty_phases(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_branches",
            lambda *a, **kw: r[bool].ok(True),
        )
        result = orchestrator.run_release(
            root=workspace_root, version="1.0.0", tag="v1.0.0", phases=[]
        )
        tm.ok(result)

    def test_run_release_with_project_filter(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_branches",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_dispatch_phase",
            lambda *a, **kw: r[bool].ok(True),
        )
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["validate"],
            project_names=["flext-core", "flext-api"],
        )
        tm.ok(result)

    def test_run_release_dry_run(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_dispatch_phase",
            lambda *a, **kw: r[bool].ok(True),
        )
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["validate"],
            dry_run=True,
        )
        tm.ok(result)

    def test_run_release_with_push(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_branches",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_dispatch_phase",
            lambda *a, **kw: r[bool].ok(True),
        )
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["validate"],
            push=True,
        )
        tm.ok(result)

    def test_run_release_with_dev_suffix(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_branches",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_dispatch_phase",
            lambda *a, **kw: r[bool].ok(True),
        )
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0-dev",
            tag="v1.0.0-dev",
            phases=["version"],
            dev_suffix=True,
        )
        tm.ok(result)

    def test_run_release_next_dev(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_branches",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_dispatch_phase",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_bump_next_dev",
            lambda *a, **kw: r[bool].ok(True),
        )
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["version"],
            next_dev=True,
            next_bump="minor",
        )
        tm.ok(result)

    def test_run_release_phase_failure_stops(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        call_count = 0

        def fake_dispatch(phase: str, *args: str, **kwargs: str) -> r[bool]:
            nonlocal call_count
            call_count += 1
            if phase == "validate":
                return r[bool].fail("validation failed")
            return r[bool].ok(True)

        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_create_branches",
            lambda *a, **kw: r[bool].ok(True),
        )
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator, "_dispatch_phase", fake_dispatch
        )
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["validate", "version"],
        )
        tm.fail(result)
        tm.that(call_count, eq=1)

    def test_run_release_create_branches_disabled(
        self, workspace_root: Path, monkeypatch: MonkeyPatch
    ) -> None:
        orchestrator = FlextInfraReleaseOrchestrator()
        monkeypatch.setattr(
            FlextInfraReleaseOrchestrator,
            "_dispatch_phase",
            lambda *a, **kw: r[bool].ok(True),
        )
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["validate"],
            create_branches=False,
        )
        tm.ok(result)
