"""Tests for FlextInfraWorkspaceDetector.

Uses real detector instances with monkeypatch for git_run control.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector, WorkspaceMode
from flext_tests import tm


@pytest.fixture
def detector() -> FlextInfraWorkspaceDetector:
    """Create a detector instance."""
    return FlextInfraWorkspaceDetector()


def _setup_project_with_git(tmp_path: Path) -> Path:
    """Create project dir with parent .git."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    (tmp_path / ".git").mkdir()
    return project_root


class TestDetectorBasicDetection:
    """Tests for basic workspace detection scenarios."""

    def test_detects_with_parent_git(
        self, detector: FlextInfraWorkspaceDetector, tmp_path: Path
    ) -> None:
        project_root = _setup_project_with_git(tmp_path)
        result = detector.detect(project_root)
        tm.ok(result)

    def test_standalone_without_parent_git(
        self, detector: FlextInfraWorkspaceDetector, tmp_path: Path
    ) -> None:
        project_root = tmp_path / "project"
        project_root.mkdir()
        result = detector.detect(project_root)
        tm.ok(result, eq=WorkspaceMode.STANDALONE)

    def test_handles_git_command_errors(
        self, detector: FlextInfraWorkspaceDetector, tmp_path: Path
    ) -> None:
        project_root = _setup_project_with_git(tmp_path)
        result = detector.detect(project_root)
        tm.ok(result)

    def test_execute_returns_failure(self) -> None:
        result = FlextInfraWorkspaceDetector().execute()
        tm.fail(result)


class TestDetectorRepoNameExtraction:
    """Tests for URL-based repo name extraction."""

    def test_https_url(self) -> None:
        name = FlextInfraWorkspaceDetector._repo_name_from_url(
            "https://github.com/flext-sh/flext.git"
        )
        tm.that(name, eq="flext")

    def test_ssh_url(self) -> None:
        name = FlextInfraWorkspaceDetector._repo_name_from_url(
            "git@github.com:flext-sh/flext.git"
        )
        tm.that(name, eq="flext")

    def test_without_git_suffix(self) -> None:
        name = FlextInfraWorkspaceDetector._repo_name_from_url(
            "https://github.com/flext-sh/flext"
        )
        tm.that(name, eq="flext")


class TestDetectorGitRunScenarios:
    """Tests for detection with controlled git_run responses."""

    def test_empty_origin_url(
        self,
        detector: FlextInfraWorkspaceDetector,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_root = _setup_project_with_git(tmp_path)
        monkeypatch.setattr(
            "flext_infra.workspace.detector.u.Infra.git_run",
            lambda *_a, **_kw: r[str].ok(""),
        )
        result = detector.detect(project_root)
        tm.ok(result, eq=WorkspaceMode.STANDALONE)

    def test_git_command_failure(
        self,
        detector: FlextInfraWorkspaceDetector,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_root = _setup_project_with_git(tmp_path)
        monkeypatch.setattr(
            "flext_infra.workspace.detector.u.Infra.git_run",
            lambda *_a, **_kw: r[str].fail("git config failed"),
        )
        result = detector.detect(project_root)
        tm.ok(result, eq=WorkspaceMode.STANDALONE)

    def test_flext_repo_detected(
        self,
        detector: FlextInfraWorkspaceDetector,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_root = _setup_project_with_git(tmp_path)
        monkeypatch.setattr(
            "flext_infra.workspace.detector.u.Infra.git_run",
            lambda *_a, **_kw: r[str].ok("https://github.com/flext-sh/flext.git"),
        )
        result = detector.detect(project_root)
        tm.ok(result, eq=WorkspaceMode.WORKSPACE)

    def test_non_flext_repo(
        self,
        detector: FlextInfraWorkspaceDetector,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_root = _setup_project_with_git(tmp_path)
        monkeypatch.setattr(
            "flext_infra.workspace.detector.u.Infra.git_run",
            lambda *_a, **_kw: r[str].ok("https://github.com/other-org/other-repo.git"),
        )
        result = detector.detect(project_root)
        tm.ok(result, eq=WorkspaceMode.STANDALONE)

    def test_runner_failure(
        self,
        detector: FlextInfraWorkspaceDetector,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_root = _setup_project_with_git(tmp_path)
        monkeypatch.setattr(
            "flext_infra.workspace.detector.u.Infra.git_run",
            lambda *_a, **_kw: r[str].fail("no remote"),
        )
        result = detector.detect(project_root)
        tm.ok(result, eq=WorkspaceMode.STANDALONE)

    def test_exception_during_detection(
        self,
        detector: FlextInfraWorkspaceDetector,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_root = _setup_project_with_git(tmp_path)

        def _raise(*_a: object, **_kw: object) -> r[str]:
            msg = "Command failed"
            raise RuntimeError(msg)

        monkeypatch.setattr("flext_infra.workspace.detector.u.Infra.git_run", _raise)
        result = detector.detect(project_root)
        tm.fail(result, has="Detection failed")
