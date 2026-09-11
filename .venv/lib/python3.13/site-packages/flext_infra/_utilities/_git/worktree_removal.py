"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from git import GitCommandError, Repo

from flext_core import r
from flext_infra._utilities._git.worktree_patch import (
    FlextInfraUtilitiesGitWorktreePatchMixin,
)

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitWorktreeRemovalMixin(
    FlextInfraUtilitiesGitWorktreePatchMixin
):
    """Own worktree removal operations."""

    @staticmethod
    def _worktree_entry(listed: str, worktree_root: Path) -> str:
        return next(
            (
                block
                for block in listed.split("\n\n")
                if f"worktree {worktree_root.resolve()}" in block.splitlines()
            ),
            "",
        )

    @staticmethod
    def _nested_submodule_changes(repo: Repo) -> tuple[str, ...]:
        nested = repo.git.submodule(
            "foreach",
            "--recursive",
            "git status --porcelain --untracked-files=all",
            with_exceptions=False,
        )
        return tuple(
            line
            for line in nested.splitlines()
            if line and not line.startswith("Entering '")
        )

    @classmethod
    def _preflight_clean_worktree(
        cls, source_root: Path, worktree_root: Path
    ) -> p.Result[Repo]:
        try:
            repo = cls._repo(source_root)
            entry = cls._worktree_entry(
                repo.git.worktree("list", "--porcelain"), worktree_root
            )
            worktree_repo = cls._repo(worktree_root)
            dirty = cls._nested_submodule_changes(worktree_repo)
            porcelain = worktree_repo.git.status("--porcelain", "--untracked-files=all")
        except GitCommandError as exc:
            return r[Repo].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[Repo].fail(f"failed to inspect clean worktree: {exc}")
        if "\nlocked" in f"\n{entry}":
            return r[Repo].fail(f"locked worktree: {worktree_root}")
        if dirty:
            return r[Repo].fail(
                f"dirty nested submodule in {worktree_root}: {'; '.join(dirty)}"
            )
        if porcelain.strip():
            return r[Repo].fail(f"dirty worktree: {worktree_root}")
        return r[Repo].ok(repo)

    @classmethod
    def git_remove_worktree(
        cls, source_root: Path, worktree_root: Path
    ) -> p.Result[bool]:
        """Remove one explicitly selected temporary worktree and prune metadata."""
        try:
            repo = cls._repo(source_root)
            repo.git.worktree("remove", "--force", str(worktree_root))
            repo.git.worktree("prune")
        except GitCommandError as exc:
            return r[bool].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[bool].fail(f"failed to remove worktree: {exc}")
        return r[bool].ok(True)

    @classmethod
    def git_remove_clean_worktree(
        cls, source_root: Path, worktree_root: Path
    ) -> p.Result[bool]:
        """Remove an explicitly selected clean worktree and prune metadata."""
        preflight = cls._preflight_clean_worktree(source_root, worktree_root)
        if preflight.failure:
            return r[bool].fail(preflight.error or "clean worktree preflight failed")
        try:
            preflight.value.git.worktree("remove", "--force", str(worktree_root))
            preflight.value.git.worktree("prune")
        except GitCommandError as exc:
            return r[bool].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[bool].fail(f"failed to remove clean worktree: {exc}")
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraUtilitiesGitWorktreeRemovalMixin"]
