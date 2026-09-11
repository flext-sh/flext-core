"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from git import GitCommandError

from flext_core import r
from flext_infra._utilities._git.worktree_status import (
    FlextInfraUtilitiesGitWorktreeStatusMixin,
)
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitWorktreeRootsMixin(
    FlextInfraUtilitiesGitWorktreeStatusMixin
):
    """Own worktree roots operations."""

    @classmethod
    def git_repository_root(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitRootReport]:
        """Resolve the superproject root or the repository's own top level."""
        root = cls._git_repository_root_path(request.repo_root)
        if root.failure:
            return r[m.Infra.GitRootReport].fail(
                root.error or "failed to resolve repository root"
            )
        return r[m.Infra.GitRootReport].ok(
            m.Infra.GitRootReport(repository_root=root.value)
        )

    @classmethod
    def git_primary_worktree_root(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitPrimaryRootReport]:
        """Resolve the primary worktree from Git's canonical storage topology."""
        primary = cls._git_primary_worktree_root_path(request.repo_root)
        if primary.failure:
            return r[m.Infra.GitPrimaryRootReport].fail(
                primary.error or "failed to resolve primary worktree"
            )
        return r[m.Infra.GitPrimaryRootReport].ok(
            m.Infra.GitPrimaryRootReport(primary_root=primary.value)
        )

    @classmethod
    def _git_repository_root_path(cls, repository_path: Path) -> p.Result[Path]:
        """Private Path-based workspace/superproject resolver."""
        try:
            repo = cls._repo(repository_path)
            superproject = repo.git.rev_parse(
                "--show-superproject-working-tree"
            ).strip()
        except GitCommandError:
            # Not inside any superproject — check if we're in a worktree at all.
            try:
                fallback_repo = cls._repo(repository_path)
                inside = fallback_repo.git.rev_parse("--is-inside-work-tree").strip()
            except GitCommandError:
                return r[Path].ok(repository_path.expanduser().resolve())
            if inside != "true":
                return r[Path].ok(repository_path.expanduser().resolve())
            return r[Path].fail("failed to resolve Git superproject")
        except (OSError, ValueError) as exc:
            return r[Path].fail(f"failed to resolve repository root: {exc}")
        if superproject:
            return r[Path].ok(Path(superproject).resolve())
        try:
            top_level = repo.git.rev_parse("--show-toplevel").strip()
        except GitCommandError as exc:
            return r[Path].fail(str(exc))
        return r[Path].ok(Path(top_level).resolve())

    @classmethod
    def _git_primary_worktree_root_path(cls, repository_path: Path) -> p.Result[Path]:
        """Private Path-based primary worktree resolver."""
        try:
            repo = cls._repo(repository_path)
            common_dir_text = repo.git.rev_parse(
                "--path-format=absolute", "--git-common-dir"
            ).strip()
            common_dir = Path(common_dir_text).resolve()
            configured_output = repo.git.config(
                "--path", "--get", "core.worktree", with_exceptions=False
            ).strip()
        except GitCommandError as exc:
            return r[Path].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[Path].fail(f"failed to resolve primary worktree: {exc}")

        if configured_output:
            configured = Path(configured_output)
            primary_root = (
                configured if configured.is_absolute() else common_dir / configured
            ).resolve()
        elif common_dir.name == c.Infra.GIT_DIR:
            primary_root = common_dir.parent
        else:
            try:
                git_dir_text = repo.git.rev_parse(
                    "--path-format=absolute", "--git-dir"
                ).strip()
                git_dir = Path(git_dir_text).resolve()
            except GitCommandError as exc:
                return r[Path].fail(str(exc))
            if git_dir != common_dir:
                listed = repo.git.worktree("list", "--porcelain")
                registered = tuple(
                    Path(line.removeprefix("worktree ").strip()).resolve()
                    for line in listed.splitlines()
                    if line.startswith("worktree ")
                )
                if not registered:
                    return r[Path].fail(
                        f"Git worktree registry is empty for {repository_path}"
                    )
                primary_root = registered[0]
                # Verify primary_root is a real worktree top-level by opening
                # a separate repo context against it.
                primary_repo_result = cls._open_repo(primary_root)
                primary_top: Path | None = None
                if primary_repo_result.success:
                    try:
                        primary_top = Path(
                            primary_repo_result.value.git.rev_parse(
                                "--show-toplevel"
                            ).strip()
                        ).resolve()
                    except GitCommandError:
                        primary_top = None
                if primary_top != primary_root:
                    caller_top = repo.git.rev_parse("--show-toplevel").strip()
                    caller_root = Path(caller_top).resolve()
                    if caller_root not in registered:
                        return r[Path].fail(
                            "current worktree is absent from Git's canonical registry: "
                            f"{caller_root}"
                        )
                    primary_root = caller_root
            else:
                caller_top = repo.git.rev_parse("--show-toplevel").strip()
                primary_root = Path(caller_top).resolve()

        # Verify the resolved primary_root is a valid worktree top-level.
        primary_repo_result = cls._open_repo(primary_root)
        if primary_repo_result.failure:
            return r[Path].fail(
                f"invalid primary worktree: {primary_root}: {primary_repo_result.error}"
            )
        try:
            resolved_top = Path(
                primary_repo_result.value.git.rev_parse("--show-toplevel").strip()
            ).resolve()
        except GitCommandError as exc:
            return r[Path].fail(f"invalid primary worktree: {primary_root}: {exc}")
        if resolved_top != primary_root:
            return r[Path].fail(
                f"Git primary worktree mismatch: {primary_root} != {resolved_top}"
            )
        return r[Path].ok(primary_root)


__all__: list[str] = ["FlextInfraUtilitiesGitWorktreeRootsMixin"]
