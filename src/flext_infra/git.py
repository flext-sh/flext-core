"""Git operations service for repository interaction.

Wraps Git commands with r error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from flext_core import r, s
from flext_infra import FlextInfraCommandRunner, c, p


class FlextInfraGitService(s[str]):
    """Infrastructure service for Git operations.

    Delegates to ``FlextInfraCommandRunner`` for subprocess execution.
    Provides high-level convenience methods so callers never need to
    build raw ``[c.Infra.Cli.GIT, ...]`` command arrays.
    """

    def __init__(self, runner: p.Infra.SafetyRunner | None = None) -> None:
        """Initialize the Git service."""
        super().__init__()
        selected_runner = runner or FlextInfraCommandRunner()
        self._runner: p.Infra.SafetyRunner = selected_runner

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    @override
    def execute(self) -> r[str]:
        """Execute the service (required by s base class)."""
        return r[str].ok("")

    def run(
        self,
        cmd: list[str],
        cwd: Path | None = None,
    ) -> r[str]:
        """Run an arbitrary git command and capture output.

        Args:
            cmd: Git command arguments (without 'git' prefix).
            cwd: Working directory.

        Returns:
            r[str] with command output.

        """
        return self._runner.capture([c.Infra.Cli.GIT, *cmd], cwd=cwd)

    def run_checked(
        self,
        cmd: list[str],
        cwd: Path | None = None,
    ) -> r[bool]:
        """Run an arbitrary git command and return success/failure.

        Args:
            cmd: Git command arguments (without 'git' prefix).
            cwd: Working directory.

        Returns:
            r[bool] with True on success.

        """
        return self._runner.run_checked([c.Infra.Cli.GIT, *cmd], cwd=cwd)

    # ------------------------------------------------------------------
    # Repository queries
    # ------------------------------------------------------------------

    def is_repo(self, path: Path) -> bool:
        """Check whether *path* sits inside a Git work-tree."""
        result = self._runner.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.REV_PARSE, "--is-inside-work-tree"],
            cwd=path,
        )
        return result.is_success

    def current_branch(self, repo_root: Path) -> r[str]:
        """Return the name of the current active branch.

        Args:
            repo_root: The root directory of the Git repository.

        Returns:
            r[str] with the branch name.

        """
        return self._runner.capture(
            [c.Infra.Cli.GIT, "rev-parse", "--abbrev-ref", c.Infra.Git.HEAD],
            cwd=repo_root,
        )

    def has_changes(self, repo_root: Path) -> r[bool]:
        """Check if the repository has uncommitted changes.

        Args:
            repo_root: Repository root directory.

        Returns:
            r[bool] with True if changes exist.

        """
        result = self._runner.capture(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.STATUS, "--porcelain"],
            cwd=repo_root,
        )
        if result.is_failure:
            return r[bool].fail(result.error or "git status failed")
        return r[bool].ok(bool(result.value.strip()))

    def diff_names(self, repo_root: Path, *, cached: bool = False) -> r[str]:
        """Return names of changed files.

        Args:
            repo_root: Repository root directory.
            cached: If True, show staged changes only.

        Returns:
            r[str] with newline-separated file names.

        """
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.DIFF, "--name-only"]
        if cached:
            cmd.insert(3, "--cached")
        return self._runner.capture(cmd, cwd=repo_root)

    # ------------------------------------------------------------------
    # Branch operations
    # ------------------------------------------------------------------

    def checkout(
        self,
        repo_root: Path,
        branch: str,
        *,
        create: bool = False,
        track: str | None = None,
    ) -> r[bool]:
        """Checkout a branch.

        Args:
            repo_root: Repository root directory.
            branch: Branch name.
            create: If True, use ``-B`` to force-create.
            track: Optional remote tracking ref (e.g. ``origin/branch``).

        Returns:
            r[bool] with True on success.

        """
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.CHECKOUT]
        if create:
            cmd.append("-B")
        cmd.append(branch)
        if track:
            cmd.append(track)
        return self._runner.run_checked(cmd, cwd=repo_root)

    def fetch(
        self,
        repo_root: Path,
        remote: str = "",
        branch: str = "",
    ) -> r[bool]:
        """Fetch from a remote.

        Args:
            repo_root: Repository root directory.
            remote: Remote name (e.g. ``origin``).
            branch: Optional branch to fetch.

        Returns:
            r[bool] with True on success.

        """
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.FETCH]
        if remote:
            cmd.append(remote)
        if branch:
            cmd.append(branch)
        return self._runner.run_checked(cmd, cwd=repo_root)

    # ------------------------------------------------------------------
    # Index & commit operations
    # ------------------------------------------------------------------

    def add(self, repo_root: Path, *paths: str) -> r[bool]:
        """Stage files for commit.

        Args:
            repo_root: Repository root directory.
            paths: Paths to add; defaults to ``-A`` (all).

        Returns:
            r[bool] with True on success.

        """
        targets = list(paths) if paths else ["-A"]
        return self._runner.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.ADD, *targets],
            cwd=repo_root,
        )

    def commit(self, repo_root: Path, message: str) -> r[bool]:
        """Create a commit with the given message.

        Args:
            repo_root: Repository root directory.
            message: Commit message.

        Returns:
            r[bool] with True on success.

        """
        return self._runner.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.COMMIT, "-m", message],
            cwd=repo_root,
        )

    # ------------------------------------------------------------------
    # Remote operations
    # ------------------------------------------------------------------

    def push(
        self,
        repo_root: Path,
        remote: str = "",
        branch: str = "",
        *,
        set_upstream: bool = False,
    ) -> r[bool]:
        """Push commits to a remote.

        Args:
            repo_root: Repository root directory.
            remote: Remote name (default: omit for implicit push).
            branch: Branch to push (default: current).
            set_upstream: If True, add ``-u`` flag.

        Returns:
            r[bool] with True on success.

        """
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.PUSH]
        if set_upstream:
            cmd.append("-u")
        if remote:
            cmd.append(remote)
        if branch:
            cmd.append(branch)
        return self._runner.run_checked(cmd, cwd=repo_root)

    def pull(
        self,
        repo_root: Path,
        *,
        rebase: bool = False,
        remote: str = "",
        branch: str = "",
    ) -> r[bool]:
        """Pull from a remote.

        Args:
            repo_root: Repository root directory.
            rebase: If True, use ``--rebase``.
            remote: Remote name.
            branch: Branch name.

        Returns:
            r[bool] with True on success.

        """
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.PULL]
        if rebase:
            cmd.append("--rebase")
        if remote:
            cmd.append(remote)
        if branch:
            cmd.append(branch)
        return self._runner.run_checked(cmd, cwd=repo_root)

    # ------------------------------------------------------------------
    # Tag operations
    # ------------------------------------------------------------------

    def tag_exists(self, repo_root: Path, tag: str) -> r[bool]:
        """Check if a specific tag exists in the repository.

        Args:
            repo_root: The root directory of the Git repository.
            tag: The tag name to check.

        Returns:
            r[bool] with True if the tag exists.

        """
        result = self._runner.capture(
            [c.Infra.Cli.GIT, c.Infra.ReportKeys.TAG, "-l", tag],
            cwd=repo_root,
        )
        if result.is_success:
            return r[bool].ok(result.value.strip() == tag)
        return r[bool].fail(result.error or "tag check failed")

    def create_tag(
        self,
        repo_root: Path,
        tag: str,
        message: str = "",
    ) -> r[bool]:
        """Create an annotated Git tag.

        Args:
            repo_root: Repository root directory.
            tag: Tag name.
            message: Tag message (defaults to ``release: <tag>``).

        Returns:
            r[bool] with True on success.

        """
        msg = message or f"release: {tag}"
        return self._runner.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.TAG, "-a", tag, "-m", msg],
            cwd=repo_root,
        )

    def list_tags(self, repo_root: Path, *, sort: str = "-v:refname") -> r[str]:
        """List tags with optional sorting.

        Args:
            repo_root: Repository root directory.
            sort: Sort key (default: version descending).

        Returns:
            r[str] with newline-separated tag names.

        """
        return self._runner.capture(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.TAG, f"--sort={sort}"],
            cwd=repo_root,
        )

    def push_tag(
        self,
        repo_root: Path,
        tag: str,
        remote: str = "",
    ) -> r[bool]:
        """Push a single tag to a remote.

        Args:
            repo_root: Repository root directory.
            tag: Tag name.
            remote: Remote name (default: ``origin``).

        Returns:
            r[bool] with True on success.

        """
        target = remote or c.Infra.Git.ORIGIN
        return self._runner.run_checked(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.PUSH, target, tag],
            cwd=repo_root,
        )

    # ------------------------------------------------------------------
    # Stash operations
    # ------------------------------------------------------------------

    def stash_push(
        self,
        repo_root: Path,
        message: str = "",
        *,
        include_untracked: bool = False,
    ) -> r[bool]:
        """Push changes to the stash.

        Args:
            repo_root: Repository root directory.
            message: Stash message.
            include_untracked: If True, include untracked files.

        Returns:
            r[bool] with True on success.

        """
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.STASH, "push"]
        if include_untracked:
            cmd.append("--include-untracked")
        if message:
            cmd.extend(["-m", message])
        return self._runner.run_checked(cmd, cwd=repo_root)

    def stash_pop(self, repo_root: Path, ref: str = "") -> r[bool]:
        """Pop the most recent stash entry.

        Args:
            repo_root: Repository root directory.
            ref: Optional stash reference (e.g. ``stash@{0}``).

        Returns:
            r[bool] with True on success.

        """
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.STASH, "pop"]
        if ref:
            cmd.append(ref)
        return self._runner.run_checked(cmd, cwd=repo_root)

    def stash_list(
        self,
        repo_root: Path,
        *,
        limit: int = 1,
        fmt: str = "%gd",
    ) -> r[str]:
        """List stash entries.

        Args:
            repo_root: Repository root directory.
            limit: Max entries to return.
            fmt: Format string for output.

        Returns:
            r[str] with formatted stash list.

        """
        return self._runner.capture(
            [
                c.Infra.Cli.GIT,
                c.Infra.Cli.GitCmd.STASH,
                "list",
                "-n",
                str(limit),
                f"--format={fmt}",
            ],
            cwd=repo_root,
        )

    # ------------------------------------------------------------------
    # Log operations
    # ------------------------------------------------------------------

    def log(
        self,
        repo_root: Path,
        rev_range: str = "",
        *,
        fmt: str = "- %h %s (%an)",
    ) -> r[str]:
        """Return formatted git log.

        Args:
            repo_root: Repository root directory.
            rev_range: Revision range (e.g. ``v1.0..HEAD``).
            fmt: Pretty format string.

        Returns:
            r[str] with formatted log output.

        """
        cmd = [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.LOG, f"--pretty=format:{fmt}"]
        if rev_range:
            cmd.append(rev_range)
        return self._runner.capture(cmd, cwd=repo_root)

    # ------------------------------------------------------------------
    # Config operations
    # ------------------------------------------------------------------

    def config_get(
        self,
        path: Path,
        key: str,
    ) -> r[str]:
        """Get a git config value.

        Args:
            path: Working directory (can use ``-C`` semantics).
            key: Config key (e.g. ``remote.origin.url``).

        Returns:
            r[str] with the config value.

        """
        return self._runner.capture(
            [c.Infra.Cli.GIT, c.Infra.Cli.GitCmd.CONFIG, "--get", key],
            cwd=path,
        )

    # ------------------------------------------------------------------
    # Composite helpers
    # ------------------------------------------------------------------

    def previous_tag(self, repo_root: Path, tag: str) -> r[str]:
        """Find the tag immediately preceding *tag*.

        Returns:
            r[str] with the previous tag name, or empty string if none.

        """
        tags_result = self.list_tags(repo_root)
        if tags_result.is_failure:
            return r[str].fail(tags_result.error or "failed to list tags")
        tags = [line.strip() for line in tags_result.value.splitlines() if line.strip()]
        if tag in tags:
            idx = tags.index(tag)
            if idx + 1 < len(tags):
                return r[str].ok(tags[idx + 1])
        for candidate in tags:
            if candidate != tag:
                return r[str].ok(candidate)
        return r[str].ok("")

    def push_release(self, repo_root: Path, tag: str) -> r[bool]:
        """Push HEAD and a tag to origin — common release workflow.

        Returns:
            r[bool] with True on success.

        """
        head_result = self.push(
            repo_root,
            remote=c.Infra.Git.ORIGIN,
            branch=c.Infra.Git.HEAD,
        )
        if head_result.is_failure:
            return head_result
        return self.push_tag(repo_root, tag, remote=c.Infra.Git.ORIGIN)


__all__ = ["FlextInfraGitService"]
