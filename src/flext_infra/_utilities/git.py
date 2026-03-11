"""Git operations utilities for repository interaction.

Wraps Git commands with r error handling as static methods.
All subprocess delegation goes through FlextInfraUtilitiesSubprocess.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra.constants import FlextInfraConstants as c


class FlextInfraUtilitiesGit:
    """Static Git operations utilities.

    All methods delegate to ``FlextInfraUtilitiesSubprocess`` and are
    exposed directly via ``u.Infra.git_*()`` through MRO.
    """

    @staticmethod
    def git_run(cmd: list[str], cwd: Path | None = None) -> r[str]:
        """Run an arbitrary git command and capture output.

        Args:
            cmd: Git command arguments (without 'git' prefix).
            cwd: Working directory.

        Returns:
            r[str] with command output.

        """
        return FlextInfraUtilitiesSubprocess.capture(
            [c.Infra.Cli.GIT, *cmd],
            cwd=cwd,
        )

    @staticmethod
    def git_run_checked(cmd: list[str], cwd: Path | None = None) -> r[bool]:
        """Run an arbitrary git command and return success/failure.

        Args:
            cmd: Git command arguments (without 'git' prefix).
            cwd: Working directory.

        Returns:
            r[bool] with True on success.

        """
        return FlextInfraUtilitiesSubprocess.run_checked(
            [c.Infra.Cli.GIT, *cmd],
            cwd=cwd,
        )

    @staticmethod
    def git_is_repo(path: Path) -> bool:
        """Check whether *path* sits inside a Git work-tree."""
        result = FlextInfraUtilitiesSubprocess.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.REV_PARSE, "--is-inside-work-tree"],
            cwd=path,
        )
        return result.is_success

    @staticmethod
    def git_current_branch(repo_root: Path) -> r[str]:
        """Return the name of the current active branch."""
        return FlextInfraUtilitiesSubprocess.capture(
            [c.Infra.Cli.GIT, "rev-parse", "--abbrev-ref", c.Infra.Git.HEAD],
            cwd=repo_root,
        )

    @staticmethod
    def git_has_changes(repo_root: Path) -> r[bool]:
        """Check if the repository has uncommitted changes."""
        result = FlextInfraUtilitiesSubprocess.capture(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.STATUS, "--porcelain"],
            cwd=repo_root,
        )
        return result.fold(
            on_failure=lambda e: r[bool].fail(e or "git status failed"),
            on_success=lambda v: r[bool].ok(bool(v.strip())),
        )

    @staticmethod
    def git_diff_names(repo_root: Path, *, cached: bool = False) -> r[str]:
        """Return names of changed files."""
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.DIFF, "--name-only"]
        if cached:
            cmd.insert(3, "--cached")
        return FlextInfraUtilitiesSubprocess.capture(cmd, cwd=repo_root)

    @staticmethod
    def git_checkout(
        repo_root: Path,
        branch: str,
        *,
        create: bool = False,
        track: str | None = None,
    ) -> r[bool]:
        """Checkout a branch."""
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.CHECKOUT]
        if create:
            cmd.append("-B")
        cmd.append(branch)
        if track:
            cmd.append(track)
        return FlextInfraUtilitiesSubprocess.run_checked(cmd, cwd=repo_root)

    @staticmethod
    def git_fetch(
        repo_root: Path,
        remote: str = "",
        branch: str = "",
    ) -> r[bool]:
        """Fetch from a remote."""
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.FETCH]
        if remote:
            cmd.append(remote)
        if branch:
            cmd.append(branch)
        return FlextInfraUtilitiesSubprocess.run_checked(cmd, cwd=repo_root)

    @staticmethod
    def git_add(repo_root: Path, *paths: str) -> r[bool]:
        """Stage files for commit."""
        targets = list(paths) if paths else ["-A"]
        return FlextInfraUtilitiesSubprocess.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.ADD, *targets],
            cwd=repo_root,
        )

    @staticmethod
    def git_commit(repo_root: Path, message: str) -> r[bool]:
        """Create a commit with the given message."""
        return FlextInfraUtilitiesSubprocess.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.COMMIT, "-m", message],
            cwd=repo_root,
        )

    @staticmethod
    def git_push(
        repo_root: Path,
        remote: str = "",
        branch: str = "",
        *,
        set_upstream: bool = False,
    ) -> r[bool]:
        """Push commits to a remote."""
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.PUSH]
        if set_upstream:
            cmd.append("-u")
        if remote:
            cmd.append(remote)
        if branch:
            cmd.append(branch)
        return FlextInfraUtilitiesSubprocess.run_checked(cmd, cwd=repo_root)

    @staticmethod
    def git_pull(
        repo_root: Path,
        *,
        rebase: bool = False,
        remote: str = "",
        branch: str = "",
    ) -> r[bool]:
        """Pull from a remote."""
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.PULL]
        if rebase:
            cmd.append("--rebase")
        if remote:
            cmd.append(remote)
        if branch:
            cmd.append(branch)
        return FlextInfraUtilitiesSubprocess.run_checked(cmd, cwd=repo_root)

    @staticmethod
    def git_tag_exists(repo_root: Path, tag: str) -> r[bool]:
        """Check if a specific tag exists in the repository."""
        result = FlextInfraUtilitiesSubprocess.capture(
            [c.Infra.Cli.GIT, c.Infra.ReportKeys.TAG, "-l", tag],
            cwd=repo_root,
        )
        if result.is_success:
            return r[bool].ok(result.value.strip() == tag)
        return r[bool].fail(result.error or "tag check failed")

    @staticmethod
    def git_create_tag(
        repo_root: Path,
        tag: str,
        message: str = "",
    ) -> r[bool]:
        """Create an annotated Git tag."""
        msg = message or f"release: {tag}"
        return FlextInfraUtilitiesSubprocess.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.TAG, "-a", tag, "-m", msg],
            cwd=repo_root,
        )

    @staticmethod
    def git_list_tags(
        repo_root: Path,
        *,
        sort: str = "-v:refname",
    ) -> r[str]:
        """List tags with optional sorting."""
        return FlextInfraUtilitiesSubprocess.capture(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.TAG, f"--sort={sort}"],
            cwd=repo_root,
        )


__all__ = ["FlextInfraUtilitiesGit"]
