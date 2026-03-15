from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraInternalDependencySyncService
from flext_infra.deps import internal_sync
from flext_tests import tm


class TestEnsureSymlink:
    def test_create_new_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        tm.ok(FlextInfraInternalDependencySyncService.ensure_symlink(target, source))
        tm.that(target.is_symlink(), eq=True)

    def test_existing_correct_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        target.symlink_to(source.resolve(), target_is_directory=True)
        tm.ok(FlextInfraInternalDependencySyncService.ensure_symlink(target, source))

    def test_replace_existing_dir(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        target.mkdir()
        (target / "file.txt").write_text("old")
        tm.ok(FlextInfraInternalDependencySyncService.ensure_symlink(target, source))
        tm.that(target.is_symlink(), eq=True)

    def test_replace_existing_wrong_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        target = tmp_path / "target"
        target.symlink_to(other.resolve(), target_is_directory=True)
        tm.ok(FlextInfraInternalDependencySyncService.ensure_symlink(target, source))
        tm.that(str(target.resolve()), eq=str(source.resolve()))


class TestEnsureSymlinkEdgeCases:
    def test_ensure_symlink_replaces_file(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        target.write_text("content")
        tm.ok(FlextInfraInternalDependencySyncService.ensure_symlink(target, source))
        tm.that(target.is_symlink(), eq=True)

    def test_ensure_symlink_permission_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"

        def _raise_symlink_to(
            self: Path,
            target_path: Path,
            target_is_directory: bool = False,
        ) -> None:
            msg = "Permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "symlink_to", _raise_symlink_to)
        error = tm.fail(
            FlextInfraInternalDependencySyncService.ensure_symlink(target, source)
        )
        tm.that(error, contains="failed to ensure symlink")


class TestEnsureCheckout:
    def test_clone_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _git_run_checked(_cmd: list[str]) -> r[bool]:
            return r[bool].ok(True)

        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_run_checked",
            _git_run_checked,
        )
        result = FlextInfraInternalDependencySyncService().ensure_checkout(
            tmp_path / "dep",
            "https://github.com/flext-sh/flext.git",
            "main",
        )
        tm.ok(result)

    def test_clone_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _git_run_checked(_cmd: list[str]) -> r[bool]:
            return r[bool].fail("fatal: repo not found")

        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_run_checked",
            _git_run_checked,
        )
        result = FlextInfraInternalDependencySyncService().ensure_checkout(
            tmp_path / "dep",
            "https://github.com/flext-sh/flext.git",
            "main",
        )
        tm.fail(result)

    def test_fetch_and_checkout_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()

        def _git_fetch(_a: Path, _b: str) -> r[bool]:
            return r[bool].ok(True)

        def _git_checkout(_a: Path, _b: str) -> r[bool]:
            return r[bool].ok(True)

        def _git_pull(_a: Path, remote: str, branch: str) -> r[bool]:
            _ = remote
            _ = branch
            return r[bool].ok(True)

        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_fetch",
            _git_fetch,
        )
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_checkout",
            _git_checkout,
        )
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_pull",
            _git_pull,
        )
        tm.ok(
            FlextInfraInternalDependencySyncService().ensure_checkout(
                dep_path,
                "https://github.com/flext-sh/flext.git",
                "main",
            ),
        )

    def test_invalid_repo_and_ref(self, tmp_path: Path) -> None:
        service = FlextInfraInternalDependencySyncService()
        tm.fail(service.ensure_checkout(tmp_path / "dep-a", "not-a-url", "main"))
        tm.fail(
            service.ensure_checkout(
                tmp_path / "dep-b",
                "https://github.com/flext-sh/flext.git",
                "invalid@ref!",
            )
        )

    def test_fetch_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()

        def _git_fetch(_a: Path, _b: str) -> r[bool]:
            return r[bool].fail("fetch failed")

        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_fetch",
            _git_fetch,
        )
        tm.fail(
            FlextInfraInternalDependencySyncService().ensure_checkout(
                dep_path,
                "https://github.com/flext-sh/flext.git",
                "main",
            ),
        )

    def test_checkout_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()

        def _git_fetch(_a: Path, _b: str) -> r[bool]:
            return r[bool].ok(True)

        def _git_checkout(_a: Path, _b: str) -> r[bool]:
            return r[bool].fail("checkout error")

        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_fetch",
            _git_fetch,
        )
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_checkout",
            _git_checkout,
        )
        tm.fail(
            FlextInfraInternalDependencySyncService().ensure_checkout(
                dep_path,
                "https://github.com/flext-sh/flext.git",
                "main",
            ),
        )

    def test_clone_replaces_existing_symlink_and_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _git_run_checked(_cmd: list[str]) -> r[bool]:
            return r[bool].ok(True)

        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_run_checked",
            _git_run_checked,
        )
        other = tmp_path / "other"
        other.mkdir()
        dep_symlink = tmp_path / "dep-symlink"
        dep_symlink.symlink_to(other)
        dep_dir = tmp_path / "dep-dir"
        dep_dir.mkdir()
        (dep_dir / "somefile").write_text("old")
        service = FlextInfraInternalDependencySyncService()
        tm.ok(
            service.ensure_checkout(
                dep_symlink, "https://github.com/flext-sh/flext.git", "main"
            )
        )
        tm.ok(
            service.ensure_checkout(
                dep_dir, "https://github.com/flext-sh/flext.git", "main"
            )
        )
