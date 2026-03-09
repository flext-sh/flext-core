from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra.deps import internal_sync
from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import h


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
        tm.that(target.resolve(), eq=source.resolve())


class TestEnsureSymlinkEdgeCases:
    def test_ensure_symlink_replaces_file(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        target.write_text("content")
        tm.ok(FlextInfraInternalDependencySyncService.ensure_symlink(target, source))
        tm.that(target.is_symlink(), eq=True)

    def test_ensure_symlink_permission_error(self, tmp_path: Path, monkeypatch) -> None:
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"

        def _raise_symlink_to(
            self, target_path: Path, target_is_directory: bool = False
        ) -> None:
            raise OSError("Permission denied")

        monkeypatch.setattr(Path, "symlink_to", _raise_symlink_to)
        error = tm.fail(
            FlextInfraInternalDependencySyncService.ensure_symlink(target, source)
        )
        tm.that(error, contains="failed to ensure symlink")


class TestEnsureCheckout:
    def test_clone_success(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_run_checked", lambda _cmd: r[bool].ok(True)
        )
        result = FlextInfraInternalDependencySyncService().ensure_checkout(
            tmp_path / "dep",
            "https://github.com/flext-sh/flext.git",
            "main",
        )
        tm.ok(result)

    def test_clone_failure(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_run_checked",
            lambda _cmd: r[bool].fail("fatal: repo not found"),
        )
        result = FlextInfraInternalDependencySyncService().ensure_checkout(
            tmp_path / "dep",
            "https://github.com/flext-sh/flext.git",
            "main",
        )
        tm.fail(result)

    def test_fetch_and_checkout_existing(self, tmp_path: Path, monkeypatch) -> None:
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_fetch", lambda _a, _b: r[bool].ok(True)
        )
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_checkout", lambda _a, _b: r[bool].ok(True)
        )
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_pull",
            lambda _a, remote, branch: r[bool].ok(True),
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

    def test_fetch_failure(self, tmp_path: Path, monkeypatch) -> None:
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_fetch",
            lambda _a, _b: r[bool].fail("fetch failed"),
        )
        tm.fail(
            FlextInfraInternalDependencySyncService().ensure_checkout(
                dep_path,
                "https://github.com/flext-sh/flext.git",
                "main",
            ),
        )

    def test_checkout_failure(self, tmp_path: Path, monkeypatch) -> None:
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_fetch", lambda _a, _b: r[bool].ok(True)
        )
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_checkout",
            lambda _a, _b: r[bool].fail("checkout error"),
        )
        tm.fail(
            FlextInfraInternalDependencySyncService().ensure_checkout(
                dep_path,
                "https://github.com/flext-sh/flext.git",
                "main",
            ),
        )

    def test_clone_replaces_existing_symlink_and_dir(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_run_checked", lambda _cmd: r[bool].ok(True)
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


class TestEnsureCheckoutEdgeCases:
    def test_ensure_checkout_cleanup_failure(self, tmp_path: Path, monkeypatch) -> None:
        dep_path = tmp_path / "dep"
        dep_path.mkdir()
        (dep_path / "file.txt").write_text("content")

        def _raise_rmtree(_path: Path) -> None:
            raise OSError("Permission denied")

        monkeypatch.setattr(internal_sync.shutil, "rmtree", _raise_rmtree)
        error = tm.fail(
            FlextInfraInternalDependencySyncService().ensure_checkout(
                dep_path,
                "https://github.com/test/repo.git",
                "main",
            ),
        )
        tm.that(error, contains="cleanup failed")
        tm.that(h is not None, eq=True)
