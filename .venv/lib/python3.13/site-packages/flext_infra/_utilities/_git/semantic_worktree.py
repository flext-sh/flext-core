"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from git import GitCommandError, Repo

from flext_core import r
from flext_infra._utilities._git.semantic_index import (
    FlextInfraUtilitiesGitSemanticIndexMixin,
)
from flext_infra.models import m

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitSemanticWorktreeMixin(
    FlextInfraUtilitiesGitSemanticIndexMixin
):
    """Own semantic worktree operations."""

    @classmethod
    def git_add_lane_worktree(
        cls, request: m.Infra.GitWorktreeAddRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Add a development lane worktree for an existing or new branch."""
        try:
            repo = cls._repo(request.repo_root)
            text = cls._git_add_worktree_args(repo, request)
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitTextReport].fail(
                f"failed to add worktree for {request.branch}: {exc}"
            )
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=text))

    @staticmethod
    def _git_add_worktree_args(
        repo: Repo, request: m.Infra.GitWorktreeAddRequest
    ) -> str:
        """Select and execute the correct ``git worktree add`` variant."""
        if request.local_branch_exists:
            return str(repo.git.worktree("add", str(request.lane), request.branch))
        if request.track_remote:
            return str(
                repo.git.worktree(
                    "add",
                    "--track",
                    "-b",
                    request.branch,
                    str(request.lane),
                    f"origin/{request.branch}",
                )
            )
        return str(
            repo.git.worktree(
                "add", "-b", request.branch, str(request.lane), request.base
            )
        )

    @classmethod
    def git_attach_branch_at_head(
        cls, request: m.Infra.GitBranchRequest
    ) -> p.Result[m.Infra.GitBoolReport]:
        """Point ``branch`` at HEAD and attach HEAD to it without moving the tree.

        A detached checkout carries real work, so it is attached by rewriting the
        ref and the symbolic HEAD rather than by ``checkout``, which would touch
        the working tree. Upstream tracking is best effort: a branch that has no
        counterpart on origin yet is still a valid attachment.
        """
        try:
            repo = cls._repo(request.repo_root)
            repo.git.branch("--quiet", "-f", request.branch, "HEAD")
            repo.git.symbolic_ref("HEAD", f"refs/heads/{request.branch}")
        except (GitCommandError, OSError, ValueError) as exc:
            return r[m.Infra.GitBoolReport].fail(
                f"failed to attach {request.branch} at HEAD: {exc}"
            )
        try:
            repo.git.branch(
                "--quiet",
                "--set-upstream-to",
                f"origin/{request.branch}",
                request.branch,
            )
        except GitCommandError:
            # No counterpart on origin yet; the attachment itself still stands.
            return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=True))
        return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=True))


__all__: list[str] = ["FlextInfraUtilitiesGitSemanticWorktreeMixin"]
