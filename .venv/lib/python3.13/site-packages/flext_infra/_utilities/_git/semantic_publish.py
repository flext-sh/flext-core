"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from git import GitCommandError

from flext_core import r
from flext_infra._utilities._git.semantic_refs import (
    FlextInfraUtilitiesGitSemanticRefsMixin,
)
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitSemanticPublishMixin(
    FlextInfraUtilitiesGitSemanticRefsMixin
):
    """Own semantic publish operations."""

    @classmethod
    def git_merge_no_edit(
        cls, request: m.Infra.GitCommitishRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Merge ``commitish`` into HEAD with an explicit merge commit."""
        try:
            repo = cls._repo(request.repo_root)
            text = repo.git.merge("--no-ff", "--no-edit", request.commitish)
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitTextReport].fail(
                f"merge failed for {request.commitish}: {exc}"
            )
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=text))

    @classmethod
    def git_delete_ref(
        cls, request: m.Infra.GitDeleteRefRequest
    ) -> p.Result[m.Infra.GitBoolReport]:
        """CAS-delete a ref when it still points at ``expected_oid``."""
        try:
            repo = cls._repo(request.repo_root)
            repo.git.update_ref("-d", request.reference, request.expected_oid)
        except GitCommandError as exc:
            return r[m.Infra.GitBoolReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitBoolReport].fail(
                f"failed to delete ref {request.reference}: {exc}"
            )
        return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=True))

    @classmethod
    def git_fetch_origin(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitBoolReport]:
        """Fetch from origin."""
        try:
            repo = cls._repo(request.repo_root)
            repo.remotes[c.Infra.GIT_DEFAULT_REMOTE].fetch()
        except GitCommandError as exc:
            return r[m.Infra.GitBoolReport].fail(str(exc))
        except (OSError, ValueError, AssertionError) as exc:
            return r[m.Infra.GitBoolReport].fail(f"failed to fetch origin: {exc}")
        return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=True))

    @classmethod
    def git_push_upstream(
        cls, request: m.Infra.GitPushRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Push HEAD to ``remote`` as ``refs/heads/<branch>`` with ``-u``."""
        try:
            repo = cls._repo(request.repo_root)
            text = repo.git.push(
                "-u", request.remote, f"HEAD:refs/heads/{request.branch}"
            )
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitTextReport].fail(
                f"failed to push {request.branch}: {exc}"
            )
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=text))

    @classmethod
    def git_remote_url(
        cls, request: m.Infra.GitRemoteUrlRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Resolve ``remote get-url <remote>`` as a text report."""
        try:
            repo = cls._repo(request.repo_root)
            url = repo.remotes[request.remote].url
        except GitCommandError as exc:
            return r[m.Infra.GitTextReport].fail(str(exc))
        except (OSError, ValueError, IndexError, AssertionError) as exc:
            return r[m.Infra.GitTextReport].fail(f"failed to resolve remote URL: {exc}")
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=url))


__all__: list[str] = ["FlextInfraUtilitiesGitSemanticPublishMixin"]
