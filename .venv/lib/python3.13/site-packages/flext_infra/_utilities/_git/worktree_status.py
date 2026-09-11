"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from git import GitCommandError, Repo

from flext_core import r
from flext_infra._utilities._git.repo import FlextInfraUtilitiesGitRepo
from flext_infra.models import m

_PORCELAIN_PATH_OFFSET = 3

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitWorktreeStatusMixin(FlextInfraUtilitiesGitRepo):
    """Own worktree status operations."""

    @classmethod
    def _lifecycle_porcelain(cls, repo: Repo, repo_path: Path, porcelain: str) -> str:
        listed = repo.git.worktree("list", "--porcelain")
        registered = {
            Path(line.removeprefix("worktree ")).expanduser().resolve()
            for line in listed.splitlines()
            if line.startswith("worktree ")
        }
        administrative = {
            path.relative_to(repo_path).as_posix().rstrip("/")
            for path in registered
            if path != repo_path and path.is_relative_to(repo_path)
        }
        retained: list[str] = []
        for line in porcelain.splitlines():
            candidate = (
                line[_PORCELAIN_PATH_OFFSET:].rstrip("/")
                if len(line) > _PORCELAIN_PATH_OFFSET
                else ""
            )
            if line.startswith("?? ") and candidate in administrative:
                continue
            retained.append(line)
        return "\n".join(retained)

    @classmethod
    def git_status(
        cls, request: m.Infra.GitStatusRequest
    ) -> p.Result[m.Infra.GitStatusReport]:
        """Capture porcelain status for one repository."""
        repo_path = request.repo_root.expanduser().resolve()
        try:
            repo = cls._repo(repo_path)
            porcelain = repo.git.status("--porcelain", "--untracked-files=all")
            lifecycle = cls._lifecycle_porcelain(repo, repo_path, porcelain)
        except GitCommandError as exc:
            return r[m.Infra.GitStatusReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitStatusReport].fail(f"git status failed: {exc}")
        return r[m.Infra.GitStatusReport].ok(
            m.Infra.GitStatusReport(
                repo_root=repo_path, porcelain=porcelain, dirty=bool(lifecycle.strip())
            )
        )

    @classmethod
    def git_repository_head(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitOidReport]:
        """Capture the current repository HEAD as a typed oid report."""
        oid = cls._git_head_oid(request.repo_root)
        if oid.failure:
            return r[m.Infra.GitOidReport].fail(oid.error or "failed to resolve HEAD")
        return r[m.Infra.GitOidReport].ok(m.Infra.GitOidReport(oid=oid.value))

    @classmethod
    def _git_head_oid(cls, repo_root: Path) -> p.Result[str]:
        """Private Path-based HEAD oid resolver for facet-internal callers."""
        opened = cls._open_repo(repo_root)
        if opened.failure:
            return r[str].fail(opened.error or "failed to open git repository")
        try:
            return r[str].ok(opened.value.head.commit.hexsha)
        except (ValueError, TypeError, OSError) as exc:
            return r[str].fail(f"failed to resolve HEAD: {exc}")


__all__: list[str] = ["FlextInfraUtilitiesGitWorktreeStatusMixin"]
