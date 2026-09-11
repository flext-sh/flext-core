"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from git import GitCommandError

from flext_cli import u
from flext_core import r
from flext_infra._utilities._git.worktree_discovery import (
    FlextInfraUtilitiesGitWorktreeDiscoveryMixin,
)
from flext_infra._utilities._git.worktree_io import git_stdin
from flext_infra.constants import c
from flext_infra.typings import t

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitWorktreeMaterializationMixin(
    FlextInfraUtilitiesGitWorktreeDiscoveryMixin
):
    """Own worktree materialization operations."""

    @classmethod
    def git_add_detached_worktree(
        cls, source_root: Path, worktree_root: Path
    ) -> p.Result[str]:
        """Create a detached worktree at the source repository HEAD."""
        ensure_parent = u.Cli.ensure_dir(worktree_root.parent)
        if ensure_parent.failure:
            return r[str].fail(
                ensure_parent.error or "failed to create worktree parent"
            )
        if worktree_root.exists():
            try:
                worktree_root.rmdir()
            except OSError as exc:
                return r[str].fail(f"worktree target is not empty: {exc}")
        head_result = cls._git_head_oid(source_root)
        if head_result.failure:
            return head_result
        # An isolated transaction is a generator-validation boundary, not a user
        # checkout. Host post-checkout hooks may depend on a toolchain which the
        # generated project is about to declare, so they cannot be its prerequisite.
        # Transaction validators still exercise the generated artifact explicitly.
        try:
            repo = cls._repo(source_root)
            repo.git.execute([
                c.Infra.GIT,
                "-c",
                "core.hooksPath=/dev/null",
                "worktree",
                "add",
                "--detach",
                str(worktree_root),
                head_result.value,
            ])
        except GitCommandError as exc:
            return r[str].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[str].fail(f"failed to add detached worktree: {exc}")
        return head_result

    @staticmethod
    def _git_path_is_excluded(path: Path, excluded: t.SequenceOf[Path]) -> bool:
        """Return whether a relative path belongs to an excluded subtree."""
        return any(path == prefix or prefix in path.parents for prefix in excluded)

    @classmethod
    def _git_copy_untracked(
        cls, source_root: Path, worktree_root: Path, excluded: t.SequenceOf[Path]
    ) -> p.Result[bool]:
        """Copy non-ignored untracked files into an isolated worktree."""
        try:
            repo = cls._repo(source_root)
            untracked = repo.git.ls_files("--others", "--exclude-standard", "-z")
        except GitCommandError as exc:
            return r[bool].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[bool].fail(f"failed to list untracked files: {exc}")
        for raw_path in untracked.split("\0"):
            if not raw_path:
                continue
            relative_path = Path(raw_path)
            if cls._git_path_is_excluded(relative_path, excluded):
                continue
            source_path = source_root / relative_path
            if source_path.is_dir():
                continue
            destination_path = worktree_root / relative_path
            ensure_parent = u.Cli.ensure_dir(destination_path.parent)
            if ensure_parent.failure:
                return r[bool].fail(
                    ensure_parent.error or f"failed to create {destination_path.parent}"
                )
            if source_path.is_symlink():
                try:
                    destination_path.symlink_to(source_path.readlink())
                except OSError as exc:
                    return r[bool].fail(
                        f"failed to copy symlink {relative_path}: {exc}"
                    )
                continue
            copy_result = u.Cli.files_copy(source_path, destination_path)
            if copy_result.failure:
                return r[bool].fail(
                    copy_result.error or f"failed to copy untracked {relative_path}"
                )
        return r[bool].ok(True)

    @classmethod
    def git_copy_worktree_state(
        cls,
        source_root: Path,
        worktree_root: Path,
        *,
        excluded: t.SequenceOf[Path] = (),
    ) -> p.Result[bool]:
        """Reproduce tracked, staged, unstaged, and untracked source state."""
        pathspecs = tuple(f":(exclude){path.as_posix()}" for path in excluded)
        try:
            repo = cls._repo(source_root)
            patch_bytes = repo.git.diff(
                "--binary", c.Infra.GIT_HEAD, "--", ".", *pathspecs
            ).encode(c.Cli.ENCODING_DEFAULT)
        except GitCommandError as exc:
            return r[bool].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[bool].fail(f"failed to capture dirty patch: {exc}")
        if patch_bytes:
            # git apply rejects a patch whose final line lacks the terminating
            # newline ("corrupt patch"); `git diff --binary` can emit exactly
            # that, so restore the single trailing newline the format requires.
            if not patch_bytes.endswith(b"\n"):
                patch_bytes += b"\n"
            try:
                worktree_repo = cls._repo(worktree_root)
                with git_stdin(patch_bytes) as istream:
                    worktree_repo.git.apply("--binary", "-", istream=istream)
            except GitCommandError as exc:
                return r[bool].fail(str(exc))
            except (OSError, ValueError) as exc:
                return r[bool].fail(f"dirty patch did not apply: {exc}")
        return cls._git_copy_untracked(source_root, worktree_root, tuple(excluded))


__all__: list[str] = ["FlextInfraUtilitiesGitWorktreeMaterializationMixin"]
