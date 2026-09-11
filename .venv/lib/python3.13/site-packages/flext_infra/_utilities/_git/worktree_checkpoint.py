"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from git import GitCommandError

from flext_core import r
from flext_infra._utilities._git.worktree_materialization import (
    FlextInfraUtilitiesGitWorktreeMaterializationMixin,
)
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitWorktreeCheckpointMixin(
    FlextInfraUtilitiesGitWorktreeMaterializationMixin
):
    """Own worktree checkpoint operations."""

    @classmethod
    def git_checkpoint_worktree(
        cls, worktree_root: Path, *, message: str, excluded: t.SequenceOf[Path] = ()
    ) -> p.Result[str]:
        """Commit the complete isolated state as a synthetic checkpoint."""
        # `make setup` fast-forwards every declared submodule to its branch tip by
        # contract, so staging gitlinks made the checkpoint differ from HEAD before
        # the verb even ran: every later verb then reported pending changes for
        # pointers it never touched, and `gen` aborted before applying anything.
        submodules_result = cls.git_declared_submodule_paths(worktree_root)
        if submodules_result.failure:
            return r[str].fail(
                submodules_result.error or "failed to resolve declared submodules"
            )
        gitlink_exclusions = tuple(
            f":(exclude){path.as_posix()}" for path in submodules_result.value
        )
        try:
            commit_sha = cls._git_create_checkpoint_commit(
                worktree_root, gitlink_exclusions, excluded, message
            )
        except GitCommandError as exc:
            return r[str].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[str].fail(f"failed to create checkpoint: {exc}")
        return r[str].ok(commit_sha)

    @classmethod
    def _git_create_checkpoint_commit(
        cls,
        worktree_root: Path,
        gitlink_exclusions: tuple[str, ...],
        excluded: t.SequenceOf[Path],
        message: str,
    ) -> str:
        """Stage all state and create a synthetic checkpoint commit-tree."""
        repo = cls._repo(worktree_root)
        if excluded:
            tracked_output = repo.git.ls_files(
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                ".",
                *(f":(exclude){path.as_posix()}" for path in excluded),
            )
            tracked_paths = tuple(path for path in tracked_output.split("\0") if path)
            if tracked_paths:
                repo.git.add("-A", "-f", "--", *tracked_paths, *gitlink_exclusions)
        else:
            # `-f` matches the tracked-paths branch and the operation delta:
            # the checkpoint must capture ignored-but-tracked paths.
            repo.git.add("-A", "-f", "--", *gitlink_exclusions)
        tree = repo.git.write_tree().strip()
        parent_result = cls._git_head_oid(worktree_root)
        if parent_result.failure:
            raise OSError(parent_result.error or "failed to resolve checkpoint parent")
        parent = parent_result.value
        identity_output = repo.git.show("-s", "--format=%an%x00%ae", parent).rstrip(
            "\n"
        )
        identity = identity_output.split("\0")
        match identity:
            case [author_name, author_email] if (
                author_name.strip() and author_email.strip()
            ):
                pass
            case _:
                detail = "checkpoint parent has invalid author identity"
                raise OSError(detail)
        commit_sha = str(
            repo.git.execute([
                c.Infra.GIT,
                "-c",
                f"user.name={author_name}",
                "-c",
                f"user.email={author_email}",
                "commit-tree",
                tree,
                "-p",
                parent,
                "-m",
                message,
            ])
        ).strip()
        repo.git.update_ref(c.Infra.GIT_HEAD, commit_sha)
        return commit_sha

    @staticmethod
    def _transaction_exclusion_pathspecs() -> tuple[str, ...]:
        """Pathspecs that exclude tool-cache directories from operation deltas."""
        return tuple(
            f":(exclude){name}"
            for name in sorted(c.Infra.WORKTREE_TRANSACTION_EXCLUDED_DIRS)
        )

    @classmethod
    def git_repository_delta(
        cls,
        repository: m.Infra.RepositoryWorktree,
        *,
        source_gitlinks: t.MappingKV[str, str] | None = None,
    ) -> p.Result[m.Infra.RepositoryDelta]:
        """Stage and capture the operation-only patch after a checkpoint."""
        head_result = cls._git_head_oid(repository.worktree_root)
        if head_result.failure or head_result.value != repository.checkpoint_sha:
            return r[m.Infra.RepositoryDelta].fail(
                head_result.error
                if head_result.failure
                else "isolated command moved repository HEAD"
            )
        exclusions = cls._transaction_exclusion_pathspecs()
        try:
            repo = cls._repo(repository.worktree_root)
            repo.git.add("-A", "-f", *exclusions)
            for path, source_head in (source_gitlinks or {}).items():
                repo.git.update_index(
                    "--add",
                    "--cacheinfo",
                    c.Infra.GIT_CACHEINFO_GITLINK,
                    source_head,
                    path,
                )
            # Gitlinks are owned by `make setup`, which fast-forwards each
            # declared submodule to its branch tip. Including them here made
            # every verb that runs after setup report "pending changes" for
            # pointers it never touched, so `gen` aborted before applying.
            names_output = repo.git.diff(
                "--cached",
                "--name-only",
                "-z",
                "--ignore-submodules=all",
                repository.checkpoint_sha,
                "--",
                *exclusions,
            )
            patch_bytes = repo.git.diff(
                "--cached",
                "--binary",
                "--ignore-submodules=all",
                repository.checkpoint_sha,
                "--",
                *exclusions,
            ).encode(c.Cli.ENCODING_DEFAULT)
        except GitCommandError as exc:
            return r[m.Infra.RepositoryDelta].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.RepositoryDelta].fail(
                f"failed to capture operation patch: {exc}"
            )
        # git apply rejects a patch whose final line has no terminating newline
        # ("corrupt patch"). `git diff --binary` can emit exactly that when the
        # last hunk ends on a context line, so restore the single trailing
        # newline the patch format requires before the delta is applied.
        if patch_bytes and not patch_bytes.endswith(b"\n"):
            patch_bytes += b"\n"
        return r[m.Infra.RepositoryDelta].ok(
            m.Infra.RepositoryDelta(
                relative_path=repository.relative_path,
                source_root=repository.source_root,
                worktree_root=repository.worktree_root,
                checkpoint_sha=repository.checkpoint_sha,
                changed_files=tuple(name for name in names_output.split("\0") if name),
                patch=patch_bytes,
            )
        )


__all__: list[str] = ["FlextInfraUtilitiesGitWorktreeCheckpointMixin"]
