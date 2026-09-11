"""GitPython repository helpers for the private git facet.

Only ``FlextInfraUtilitiesGitRepo`` lives here. Semantic operations use
GitPython's object-oriented API (``Repo``, ``IndexFile``, ``Remote``,
``BaseIndexEntry``) or the ``repo.git.<cmd>(args)`` proxy directly;
``Git(path).execute(tuple)`` with manual cast/protocol is eliminated.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from git import (
    Git,
    GitCommandNotFound,
    InvalidGitRepositoryError,
    NoSuchPathError,
    Repo,
)

from flext_core import r
from flext_infra.constants import c

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitRepo:
    """Git repository opener with GitPython native OO API."""

    @classmethod
    def refresh_binary(cls) -> p.Result[bool]:
        """Point GitPython at the absolute path of the canonical git binary."""
        # Git.refresh resolves relative names against cwd; always pass an absolute path.
        resolved = shutil.which(c.Infra.GIT)
        if resolved is None:
            return r[bool].fail(f"git executable not found on PATH: {c.Infra.GIT}")
        try:
            Git.refresh(resolved)
        except (FileNotFoundError, OSError) as exc:
            return r[bool].fail(f"git binary refresh failed: {exc}")
        return r[bool].ok(True)

    @classmethod
    def _open_repo(cls, repo_root: Path) -> p.Result[Repo]:
        """Open the non-bare worktree repository containing ``repo_root``.

        The path may name the checkout root, a directory inside it, or a file
        inside it; the repository is the one that contains it. Callers depend
        on that -- ``git_is_work_tree`` and ``git_semantic_identity`` already
        open with the same contract, and one of them documents it as "the same
        nested-path contract as git_open_repo" -- but this opener had lost it,
        so every nested path failed with "cannot open git repository".
        """
        resolved = repo_root.expanduser().resolve()
        try:
            refreshed = cls.refresh_binary()
            if refreshed.failure:
                return r[Repo].fail(refreshed.error or "git binary unavailable")
            repo = Repo(resolved, search_parent_directories=True)
        except (
            GitCommandNotFound,
            ImportError,
            InvalidGitRepositoryError,
            NoSuchPathError,
            OSError,
            ValueError,
        ) as exc:
            return r[Repo].fail(f"cannot open git repository at {resolved}: {exc}")
        if repo.bare or repo.working_tree_dir is None:
            return r[Repo].fail(f"bare or worktree-less repository at {resolved}")
        return r[Repo].ok(repo)

    @classmethod
    def _repo(cls, repo_root: Path) -> Repo:
        """Open a repo and unwrap, raising on failure.

        This is the canonical helper for semantic methods that prefer
        try/except → ``r[...].fail()`` over ``Result`` chaining.
        """
        opened = cls._open_repo(repo_root)
        if opened.failure:
            raise OSError(opened.error or "failed to open git repository")
        return opened.value


__all__: list[str] = ["FlextInfraUtilitiesGitRepo"]
