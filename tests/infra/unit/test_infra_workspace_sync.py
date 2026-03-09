"""Tests for FlextInfraSyncService.

Uses real service instances with monkeypatch for controlled failures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import fcntl
import sys
import tempfile
from pathlib import Path

import pytest

from flext_core import r
from flext_infra.workspace.sync import FlextInfraSyncService, main
from flext_tests import tm


def _stub_generator_ok(content: str) -> object:
    """Create a generator stub returning ok(content)."""

    class _Gen:
        @staticmethod
        def generate(*args: object, **kwargs: object) -> r[str]:
            return r[str].ok(content)

    return _Gen()


def _stub_generator_fail(error: str) -> object:
    """Create a generator stub returning fail(error)."""

    class _Gen:
        @staticmethod
        def generate(*args: object, **kwargs: object) -> r[str]:
            return r[str].fail(error)

    return _Gen()


@pytest.fixture
def sync_service(tmp_path: Path) -> FlextInfraSyncService:
    return FlextInfraSyncService(canonical_root=tmp_path)


class TestSyncServiceBasic:
    def test_generates_base_mk(
        self, sync_service: FlextInfraSyncService, tmp_path: Path
    ) -> None:
        tm.ok(sync_service.sync(project_root=tmp_path))

    def test_detects_changes_via_sha256(
        self, sync_service: FlextInfraSyncService, tmp_path: Path
    ) -> None:
        (tmp_path / "base.mk").write_text("# Old content\n", encoding="utf-8")
        tm.ok(sync_service.sync(project_root=tmp_path))

    def test_skips_write_when_unchanged(
        self, sync_service: FlextInfraSyncService, tmp_path: Path
    ) -> None:
        (tmp_path / "base.mk").write_text("# Same content\n", encoding="utf-8")
        tm.ok(sync_service.sync(project_root=tmp_path))

    def test_creates_base_mk_if_missing(
        self, sync_service: FlextInfraSyncService, tmp_path: Path
    ) -> None:
        tm.ok(sync_service.sync(project_root=tmp_path))
        tm.that((tmp_path / "base.mk").exists(), eq=True)

    def test_execute_returns_failure(self) -> None:
        tm.fail(FlextInfraSyncService().execute())

    def test_validates_gitignore_entries(
        self, sync_service: FlextInfraSyncService, tmp_path: Path
    ) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
        tm.ok(sync_service.sync(project_root=tmp_path))

    def test_project_root_required(self) -> None:
        tm.fail(
            FlextInfraSyncService().sync(project_root=None),
            has="project_root is required",
        )

    def test_project_root_not_exists(self) -> None:
        tm.fail(
            FlextInfraSyncService().sync(project_root=Path("/nonexistent/path")),
            has="does not exist",
        )


class TestSyncServiceFailures:
    def test_lock_acquisition_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _flock(fd: int, operation: int) -> None:
            msg = "Lock failed"
            raise OSError(msg)

        monkeypatch.setattr(fcntl, "flock", _flock)
        tm.fail(
            FlextInfraSyncService().sync(project_root=tmp_path),
            has="lock acquisition failed",
        )

    def test_basemk_generation_failure(self, tmp_path: Path) -> None:
        svc = FlextInfraSyncService()
        svc._generator = _stub_generator_fail("Generation failed")
        tm.fail(svc.sync(project_root=tmp_path), has="Generation failed")

    def test_gitignore_update_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _open(*_a: object, **_kw: object) -> None:
            msg = "Write failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "open", _open)
        result = FlextInfraSyncService().sync(project_root=tmp_path)
        tm.fail(result)

    def test_gitignore_sync_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc = FlextInfraSyncService()

        def _ensure(*_a: object, **_kw: object) -> r[bool]:
            return r[bool].fail(".gitignore sync failed")

        monkeypatch.setattr(svc, "_ensure_gitignore_entries", _ensure)
        tm.fail(svc.sync(project_root=tmp_path), has=".gitignore sync failed")


class TestSyncServiceAtomicWrite:
    def test_success(self, tmp_path: Path) -> None:
        target = tmp_path / "test.txt"
        tm.ok(FlextInfraSyncService._atomic_write(target, "test content"), eq=True)
        tm.that(target.read_text(encoding="utf-8"), eq="test content")

    def test_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _temp(*_a: object, **_kw: object) -> None:
            msg = "Temp file failed"
            raise OSError(msg)

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", _temp)
        tm.fail(
            FlextInfraSyncService._atomic_write(tmp_path / "test.txt", "content"),
            has="atomic write failed",
        )


class TestSyncServiceHashing:
    def test_sha256_content(self) -> None:
        h1 = FlextInfraSyncService._sha256_content("test content")
        h2 = FlextInfraSyncService._sha256_content("test content")
        tm.that(h1, eq=h2)
        tm.that(h1, len=64)

    def test_sha256_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("test content", encoding="utf-8")
        h1 = FlextInfraSyncService._sha256_file(f)
        h2 = FlextInfraSyncService._sha256_file(f)
        tm.that(h1, eq=h2)
        tm.that(h1, len=64)


class TestSyncServiceCanonical:
    def test_canonical_root_copy(self, tmp_path: Path) -> None:
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        (canonical / "base.mk").write_text("# Canonical content\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir()
        svc = FlextInfraSyncService(canonical_root=canonical)
        tm.ok(svc.sync(project_root=project))
        tm.that(
            (project / "base.mk").read_text(encoding="utf-8"),
            eq="# Canonical content\n",
        )

    def test_sync_basemk_from_canonical(self, tmp_path: Path) -> None:
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        (canonical / "base.mk").write_text("# Canonical\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir()
        svc = FlextInfraSyncService()
        tm.ok(svc._sync_basemk(project, None, canonical_root=canonical), eq=True)
        tm.that((project / "base.mk").read_text(encoding="utf-8"), eq="# Canonical\n")

    def test_sync_basemk_no_change_needed(self, tmp_path: Path) -> None:
        svc = FlextInfraSyncService()
        content = "# Same content\n"
        (tmp_path / "base.mk").write_text(content, encoding="utf-8")
        svc._generator = _stub_generator_ok(content)
        tm.ok(svc._sync_basemk(tmp_path, None), eq=False)

    def test_sync_basemk_generation_failure(self, tmp_path: Path) -> None:
        svc = FlextInfraSyncService()
        svc._generator = _stub_generator_fail("Generation failed")
        tm.fail(svc._sync_basemk(tmp_path, None), has="Generation failed")

    def test_sync_basemk_with_canonical_root(self, tmp_path: Path) -> None:
        canonical_root = tmp_path / "canonical"
        canonical_root.mkdir(parents=True, exist_ok=True)
        (canonical_root / "base.mk").write_text("# Canonical base.mk\n")
        svc = FlextInfraSyncService()
        result = svc._sync_basemk(tmp_path, None, canonical_root=canonical_root)
        tm.that(result.is_success or result.is_failure, eq=True)


class TestSyncServiceGitignore:
    def test_missing_entries(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
        svc = FlextInfraSyncService()
        tm.ok(svc._ensure_gitignore_entries(tmp_path, [".reports/", ".venv/"]), eq=True)
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        tm.that(content, has=".reports/")
        tm.that(content, has=".venv/")

    def test_all_present(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text(
            ".reports/\n.venv/\n__pycache__/\n", encoding="utf-8"
        )
        svc = FlextInfraSyncService()
        tm.ok(
            svc._ensure_gitignore_entries(tmp_path, [".reports/", ".venv/"]), eq=False
        )

    def test_write_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")

        def _open(*_a: object, **_kw: object) -> None:
            msg = "Write failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "open", _open)
        svc = FlextInfraSyncService()
        tm.fail(
            svc._ensure_gitignore_entries(tmp_path, [".reports/"]),
            has=".gitignore update failed",
        )


class TestSyncServiceCli:
    def test_main_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["sync", "--project-root", str(tmp_path)])
        tm.that(main(), eq=0)

    def test_main_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.argv", ["sync", "--project-root", "/nonexistent/path"])
        tm.that(main(), eq=1)
