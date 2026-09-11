"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from git import BadName, GitCommandError

from flext_core import r
from flext_infra._utilities._git.worktree import FlextInfraUtilitiesGitWorktreeMixin
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitSemanticRefsMixin(FlextInfraUtilitiesGitWorktreeMixin):
    """Own semantic refs operations."""

    @classmethod
    def git_list_worktrees(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """List registered worktrees in porcelain form."""
        try:
            repo = cls._repo(request.repo_root)
            text = repo.git.worktree("list", "--porcelain")
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitTextReport].fail(f"failed to list Git worktrees: {exc}")
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=text))

    @classmethod
    def git_check_branch_format(
        cls, request: m.Infra.GitBranchRequest
    ) -> p.Result[m.Infra.GitBoolReport]:
        """Validate a branch name with ``git check-ref-format --branch``."""
        try:
            repo = cls._repo(request.repo_root)
            repo.git.check_ref_format("--branch", request.branch)
        except GitCommandError:
            return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=False))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitBoolReport].fail(
                f"failed to validate branch name: {exc}"
            )
        return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=True))

    @classmethod
    def git_ref_exists(
        cls, request: m.Infra.GitRefRequest
    ) -> p.Result[m.Infra.GitBoolReport]:
        """Return whether an exact Git ref exists (exit 0/1 only)."""
        try:
            repo = cls._repo(request.repo_root)
            repo.git.show_ref("--verify", "--quiet", request.reference)
        except GitCommandError:
            # show-ref exits 1 when the ref does not exist — not an error.
            return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=False))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitBoolReport].fail(f"failed to inspect Git ref: {exc}")
        return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=True))

    @classmethod
    def git_superproject_working_tree(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Capture ``rev-parse --show-superproject-working-tree`` stdout."""
        try:
            repo = cls._repo(request.repo_root)
            text = repo.git.rev_parse("--show-superproject-working-tree")
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitTextReport].fail(
                f"failed to resolve superproject working tree: {exc}"
            )
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=text))

    @classmethod
    def git_show_toplevel(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitRootReport]:
        """Resolve ``rev-parse --show-toplevel`` as a repository root report."""
        try:
            repo = cls._repo(request.repo_root)
            root = (
                Path(repo.working_tree_dir).resolve() if repo.working_tree_dir else None
            )
            if root is None:
                return r[m.Infra.GitRootReport].fail(
                    "failed to resolve Git top level: working tree is None"
                )
        except GitCommandError as exc:
            return r[m.Infra.GitRootReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitRootReport].fail(
                f"failed to resolve Git top level: {exc}"
            )
        return r[m.Infra.GitRootReport].ok(m.Infra.GitRootReport(repository_root=root))

    @classmethod
    def git_current_branch(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Resolve the current non-detached branch name."""
        try:
            repo = cls._repo(request.repo_root)
            branch = repo.active_branch.name
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (TypeError, OSError, ValueError) as exc:
            # active_branch raises TypeError on detached HEAD.
            return r[m.Infra.GitTextReport].fail(
                f"head branch is required from a detached HEAD: {exc}"
            )
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=branch))

    @classmethod
    def git_symbolic_ref_short(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Resolve ``symbolic-ref --quiet --short HEAD``."""
        try:
            repo = cls._repo(request.repo_root)
            text = repo.git.symbolic_ref("--quiet", "--short", c.Infra.GIT_HEAD)
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitTextReport].fail(
                f"failed to resolve symbolic-ref HEAD: {exc}"
            )
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=text.strip()))

    @classmethod
    def git_resolve_commit(
        cls, request: m.Infra.GitCommitishRequest
    ) -> p.Result[m.Infra.GitOidReport]:
        """Resolve a commit-ish to an oid via ``rev-parse --verify``."""
        try:
            repo = cls._repo(request.repo_root)
            oid = repo.commit(request.commitish).hexsha
        except (BadName, GitCommandError) as exc:
            return r[m.Infra.GitOidReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitOidReport].fail(f"cannot resolve commitish: {exc}")
        return r[m.Infra.GitOidReport].ok(m.Infra.GitOidReport(oid=oid))

    @classmethod
    def git_abbrev_ref_head(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Resolve ``rev-parse --abbrev-ref HEAD``."""
        try:
            repo = cls._repo(request.repo_root)
            branch = repo.active_branch.name
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (TypeError, OSError, ValueError):
            # Detached HEAD — fall back to the proxy for the ``HEAD`` text.
            try:
                detached_repo = cls._repo(request.repo_root)
                text = detached_repo.git.rev_parse("--abbrev-ref", c.Infra.GIT_HEAD)
            except GitCommandError as exc:
                return r[m.Infra.GitTextReport].fail(str(exc))
            return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=text.strip()))
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=branch))

    @classmethod
    def git_is_ancestor(
        cls, request: m.Infra.GitCommitishRequest
    ) -> p.Result[m.Infra.GitBoolReport]:
        """Return whether ``commitish`` is an ancestor of HEAD."""
        try:
            repo = cls._repo(request.repo_root)
            ancestor = repo.commit(request.commitish)
            head = repo.commit(c.Infra.GIT_HEAD)
            result = repo.is_ancestor(ancestor, head)
        except GitCommandError as exc:
            return r[m.Infra.GitBoolReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitBoolReport].fail(f"failed to inspect ancestry: {exc}")
        return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=result))

    @classmethod
    def git_rev_parse(
        cls, request: m.Infra.GitCommitishRequest
    ) -> p.Result[m.Infra.GitOidReport]:
        """Resolve an arbitrary rev-parse argument to stripped text oid."""
        try:
            repo = cls._repo(request.repo_root)
            oid = repo.git.rev_parse(request.commitish).strip()
        except GitCommandError as exc:
            return r[m.Infra.GitOidReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitOidReport].fail(
                f"rev-parse failed for {request.commitish}: {exc}"
            )
        return r[m.Infra.GitOidReport].ok(m.Infra.GitOidReport(oid=oid))

    @classmethod
    def git_rev_parse_parent(
        cls, request: m.Infra.GitCommitishRequest
    ) -> p.Result[m.Infra.GitOidReport]:
        """Resolve ``commitish^`` via rev-parse."""
        try:
            repo = cls._repo(request.repo_root)
            oid = repo.commit(f"{request.commitish}^").hexsha
        except GitCommandError as exc:
            return r[m.Infra.GitOidReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitOidReport].fail(
                f"failed to resolve parent of {request.commitish}: {exc}"
            )
        return r[m.Infra.GitOidReport].ok(m.Infra.GitOidReport(oid=oid))


__all__: list[str] = ["FlextInfraUtilitiesGitSemanticRefsMixin"]
