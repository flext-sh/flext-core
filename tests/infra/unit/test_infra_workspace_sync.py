"""Tests for FlextInfraSyncService."""

from __future__ import annotations

import fcntl
import sys
import tempfile
from pathlib import Path

import pytest

from flext_core import r, t
from flext_infra.workspace.sync import FlextInfraSyncService, main
from flext_tests import tm

_S = FlextInfraSyncService


def _stub_gen(content: str, *, fail: bool = False) -> t.ContainerValue:
    class _Gen:
        @staticmethod
        def generate(*args: t.ContainerValue, **kwargs: t.ContainerValue) -> r[str]:
            return r[str].fail(content) if fail else r[str].ok(content)

    return _Gen()


@pytest.fixture
def svc(tmp_path: Path) -> FlextInfraSyncService:
    return _S(canonical_root=tmp_path)


class TestSyncBasic:
    def test_generates_base_mk(self, svc: _S, tmp_path: Path) -> None:
        tm.ok(svc.sync(project_root=tmp_path))

    def test_creates_base_mk_if_missing(self, svc: _S, tmp_path: Path) -> None:
        tm.ok(svc.sync(project_root=tmp_path))
        tm.that((tmp_path / "base.mk").exists(), eq=True)

    def test_detects_changes(self, svc: _S, tmp_path: Path) -> None:
        (tmp_path / "base.mk").write_text("# Old content\n", encoding="utf-8")
        tm.ok(svc.sync(project_root=tmp_path))

    def test_execute_returns_failure(self) -> None:
        tm.fail(_S().execute())

    def test_validates_gitignore(self, svc: _S, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
        tm.ok(svc.sync(project_root=tmp_path))

    def test_project_root_required(self) -> None:
        tm.fail(_S().sync(project_root=None), has="project_root is required")

    def test_project_root_not_exists(self) -> None:
        tm.fail(_S().sync(project_root=Path("/nonexistent/path")), has="does not exist")

    def test_cli_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["sync", "--project-root", str(tmp_path)])
        tm.that(main(), eq=0)

    def test_cli_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["sync", "--project-root", "/nonexistent/path"])
        tm.that(main(), eq=1)


class TestSyncFailures:
    def test_lock_acquisition(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _flock(fd: int, operation: int) -> None:
            msg = "Lock failed"
            raise OSError(msg)

        monkeypatch.setattr(fcntl, "flock", _flock)
        tm.fail(_S().sync(project_root=tmp_path), has="lock acquisition failed")

    def test_basemk_generation(self, tmp_path: Path) -> None:
        s = _S()
        s._generator = _stub_gen("Generation failed", fail=True)
        tm.fail(s.sync(project_root=tmp_path), has="Generation failed")

    def test_gitignore_update(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _open(*_a: t.ContainerValue, **_kw: t.ContainerValue) -> None:
            msg = "Write failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "open", _open)
        tm.fail(_S().sync(project_root=tmp_path))

    def test_gitignore_sync(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        s = _S()

        def _ensure(*_a: t.ContainerValue, **_kw: t.ContainerValue) -> r[bool]:
            return r[bool].fail(".gitignore sync failed")

        monkeypatch.setattr(s, "_ensure_gitignore_entries", _ensure)
        tm.fail(s.sync(project_root=tmp_path), has=".gitignore sync failed")


class TestSyncInternals:
    def test_atomic_write_ok(self, tmp_path: Path) -> None:
        target = tmp_path / "test.txt"
        tm.ok(_S._atomic_write(target, "test content"), eq=True)
        tm.that(target.read_text(encoding="utf-8"), eq="test content")

    def test_atomic_write_fail(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _temp(*_a: t.ContainerValue, **_kw: t.ContainerValue) -> None:
            msg = "Temp file failed"
            raise OSError(msg)

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", _temp)
        tm.fail(_S._atomic_write(tmp_path / "t.txt", "c"), has="atomic write failed")

    def test_sha256_content(self) -> None:
        h1 = _S._sha256_content("test content")
        tm.that(h1, eq=_S._sha256_content("test content"))
        tm.that(h1, len=64)

    def test_sha256_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("test content", encoding="utf-8")
        h1 = _S._sha256_file(f)
        tm.that(h1, eq=_S._sha256_file(f))
        tm.that(h1, len=64)

    def test_canonical_root_copy(self, tmp_path: Path) -> None:
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        (canonical / "base.mk").write_text("# Canonical\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir()
        tm.ok(_S(canonical_root=canonical).sync(project_root=project))
        tm.that((project / "base.mk").read_text(encoding="utf-8"), eq="# Canonical\n")

    def test_sync_basemk_no_change(self, tmp_path: Path) -> None:
        s = _S()
        content = "# Same content\n"
        (tmp_path / "base.mk").write_text(content, encoding="utf-8")
        s._generator = _stub_gen(content)
        tm.ok(s._sync_basemk(tmp_path, None), eq=False)

    def test_sync_basemk_gen_failure(self, tmp_path: Path) -> None:
        s = _S()
        s._generator = _stub_gen("Generation failed", fail=True)
        tm.fail(s._sync_basemk(tmp_path, None), has="Generation failed")


class TestSyncGitignore:
    def test_missing_entries(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
        tm.ok(
            _S()._ensure_gitignore_entries(tmp_path, [".reports/", ".venv/"]), eq=True
        )
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        tm.that(content, has=".reports/")
        tm.that(content, has=".venv/")

    def test_all_present(self, tmp_path: Path) -> None:
        gi = ".reports/\n.venv/\n__pycache__/\n"
        (tmp_path / ".gitignore").write_text(gi, encoding="utf-8")
        tm.ok(
            _S()._ensure_gitignore_entries(tmp_path, [".reports/", ".venv/"]), eq=False
        )

    def test_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")

        def _open(*_a: t.ContainerValue, **_kw: t.ContainerValue) -> None:
            msg = "Write failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "open", _open)
        tm.fail(
            _S()._ensure_gitignore_entries(tmp_path, [".reports/"]),
            has=".gitignore update failed",
        )


__all__: list[str] = []
