"""Workspace-wide PR automation service.

Wraps multi-repository PR operations with r error handling,
replacing scripts/github/pr_workspace.py with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from flext_core import r, t

from flext_infra.constants import c
from flext_infra.git import GitService
from flext_infra.reporting import ReportingService
from flext_infra.selection import ProjectSelector
from flext_infra.subprocess import CommandRunner


class PrWorkspaceManager:
    """Infrastructure service for workspace-wide PR automation.

    Orchestrates PR operations (status, create, merge, etc.) across all
    workspace repositories with checkpoint and branch management.
    """

    def __init__(
        self,
        runner: CommandRunner | None = None,
        git: GitService | None = None,
        selector: ProjectSelector | None = None,
        reporting: ReportingService | None = None,
    ) -> None:
        """Initialize the workspace PR manager."""
        self._runner = runner or CommandRunner()
        self._git = git or GitService(self._runner)
        self._selector = selector or ProjectSelector()
        self._reporting = reporting or ReportingService()

    def has_changes(self, repo_root: Path) -> r[bool]:
        """Check if the repository has uncommitted changes.

        Args:
            repo_root: Repository root directory.

        Returns:
            r[bool] with True if changes exist.

        """
        result = self._runner.capture(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
        )
        if result.is_failure:
            return r[bool].fail(result.error or "git status failed")
        return r[bool].ok(bool(result.value.strip()))

    def checkout_branch(
        self,
        repo_root: Path,
        branch: str,
    ) -> r[bool]:
        """Checkout or create a branch in the repository.

        Handles local changes, fetch from origin, and force-create scenarios.

        Args:
            repo_root: Repository root directory.
            branch: Branch name to checkout.

        Returns:
            r[bool] with True on success.

        """
        if not branch:
            return r[bool].ok(True)

        current_result = self._git.current_branch(repo_root)
        if current_result.is_success and current_result.value == branch:
            return r[bool].ok(True)

        checkout_result = self._runner.run(
            ["git", "checkout", branch],
            cwd=repo_root,
        )
        if checkout_result.is_success:
            return r[bool].ok(True)

        detail = (checkout_result.error or "").lower()
        if "local changes" in detail or "would be overwritten" in detail:
            return self._runner.run_checked(
                ["git", "checkout", "-B", branch],
                cwd=repo_root,
            )

        fetch_result = self._runner.run(
            ["git", "fetch", "origin", branch],
            cwd=repo_root,
        )
        if fetch_result.is_success:
            return self._runner.run_checked(
                ["git", "checkout", "-B", branch, f"origin/{branch}"],
                cwd=repo_root,
            )
        return self._runner.run_checked(
            ["git", "checkout", "-B", branch],
            cwd=repo_root,
        )

    def checkpoint(
        self,
        repo_root: Path,
        branch: str,
    ) -> r[bool]:
        """Commit and push pending changes.

        Args:
            repo_root: Repository root directory.
            branch: Branch name for push target.

        Returns:
            r[bool] with True on success.

        """
        changes_result = self.has_changes(repo_root)
        if changes_result.is_failure:
            return r[bool].fail(changes_result.error or "changes check failed")
        if not changes_result.value:
            return r[bool].ok(True)

        add_result = self._runner.run_checked(
            ["git", "add", "-A"],
            cwd=repo_root,
        )
        if add_result.is_failure:
            return r[bool].fail(add_result.error or "git add failed")

        staged_result = self._runner.capture(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo_root,
        )
        if staged_result.is_success and not staged_result.value.strip():
            return r[bool].ok(True)

        commit_result = self._runner.run_checked(
            ["git", "commit", "-m", "chore: checkpoint pending changes"],
            cwd=repo_root,
        )
        if commit_result.is_failure:
            return r[bool].fail(commit_result.error or "git commit failed")

        push_cmd = (
            ["git", "push", "-u", "origin", branch] if branch else ["git", "push"]
        )
        push_result = self._runner.run(push_cmd, cwd=repo_root)
        if push_result.is_success:
            return r[bool].ok(True)

        rebase_cmd = (
            ["git", "pull", "--rebase", "origin", branch]
            if branch
            else ["git", "pull", "--rebase"]
        )
        rebase_result = self._runner.run_checked(rebase_cmd, cwd=repo_root)
        if rebase_result.is_failure:
            return r[bool].fail(rebase_result.error or "git pull --rebase failed")

        return self._runner.run_checked(push_cmd, cwd=repo_root)

    def run_pr(
        self,
        repo_root: Path,
        workspace_root: Path,
        pr_args: Mapping[str, str],
    ) -> r[Mapping[str, t.ScalarValue]]:
        """Execute a PR operation on a single repository.

        Args:
            repo_root: Repository root directory.
            workspace_root: Workspace root directory.
            pr_args: PR argument dictionary.

        Returns:
            r with execution result info.

        """
        display = self._repo_display_name(repo_root, workspace_root)
        report_dir_result = self._reporting.ensure_report_dir(
            workspace_root,
            "workspace",
            "pr",
        )

        log_path: Path | None = None
        if report_dir_result.is_success:
            log_path = report_dir_result.value / f"{display}.log"

        if repo_root == workspace_root:
            command = self._build_root_command(repo_root, pr_args)
        else:
            command = self._build_subproject_command(repo_root, pr_args)

        started = time.monotonic()
        if log_path is not None:
            to_file_result = self._runner.run_to_file(command, log_path)
            if to_file_result.is_failure:
                return r[Mapping[str, t.ScalarValue]].fail(
                    to_file_result.error or "command execution error",
                )
            exit_code: int = to_file_result.value
        else:
            raw_result = self._runner.run_raw(command)
            if raw_result.is_failure:
                return r[Mapping[str, t.ScalarValue]].fail(
                    raw_result.error or "command execution error",
                )
            exit_code = raw_result.value.exit_code

        elapsed = int(time.monotonic() - started)
        status = c.Status.OK if exit_code == 0 else c.Status.FAIL
        return r[Mapping[str, t.ScalarValue]].ok({
            "display": display,
            "status": status,
            "elapsed": elapsed,
            "exit_code": exit_code,
            "log_path": str(log_path) if log_path else None,
        })

    def orchestrate(
        self,
        workspace_root: Path,
        *,
        projects: list[str] | None = None,
        include_root: bool = True,
        branch: str = "",
        checkpoint: bool = True,
        fail_fast: bool = False,
        pr_args: Mapping[str, str] | None = None,
    ) -> r[dict[str, int | list[Mapping[str, t.ScalarValue]]]]:
        """Run PR operations across workspace repositories.

        Args:
            workspace_root: Workspace root directory.
            projects: Optional list of project names to include.
            include_root: If True, include workspace root repo.
            branch: Optional branch to checkout.
            checkpoint: If True, commit/push pending changes.
            fail_fast: If True, stop on first failure.
            pr_args: PR operation arguments.

        Returns:
            r with orchestration summary.

        """
        projects_result = self._selector.resolve_projects(
            workspace_root,
            projects or [],
        )
        if projects_result.is_failure:
            return r[dict[str, int | list[Mapping[str, t.ScalarValue]]]].fail(
                projects_result.error or "project resolution failed",
            )

        repos = [p.path for p in projects_result.value]
        if include_root:
            repos.append(workspace_root)

        effective_args = pr_args or {"action": "status", "base": "main"}
        failures = 0
        results: list[Mapping[str, t.ScalarValue]] = []

        for repo_root in repos:
            self.checkout_branch(repo_root, branch)
            if checkpoint:
                self.checkpoint(repo_root, branch)

            run_result = self.run_pr(repo_root, workspace_root, effective_args)
            if run_result.is_success:
                results.append(run_result.value)
                if run_result.value.get("exit_code", 0) != 0:
                    failures += 1
                    if fail_fast:
                        break
            else:
                failures += 1
                if fail_fast:
                    break

        total = len(repos)
        return r[dict[str, int | list[Mapping[str, t.ScalarValue]]]].ok({
            "total": total,
            "success": total - failures,
            "fail": failures,
            "results": results,
        })

    @staticmethod
    def _repo_display_name(repo_root: Path, workspace_root: Path) -> str:
        return workspace_root.name if repo_root == workspace_root else repo_root.name

    @staticmethod
    def _build_root_command(
        repo_root: Path,
        pr_args: Mapping[str, str],
    ) -> list[str]:
        command = [
            "python",
            "-m",
            "flext_infra.github.pr",
            "--repo-root",
            str(repo_root),
            "--action",
            pr_args.get("action", "status"),
            "--base",
            pr_args.get("base", "main"),
            "--draft",
            pr_args.get("draft", "0"),
            "--merge-method",
            pr_args.get("merge_method", "squash"),
            "--auto",
            pr_args.get("auto", "0"),
            "--delete-branch",
            pr_args.get("delete_branch", "0"),
            "--checks-strict",
            pr_args.get("checks_strict", "0"),
            "--release-on-merge",
            pr_args.get("release_on_merge", "1"),
        ]
        for key in ("head", "number", "title", "body"):
            value = pr_args.get(key, "")
            if value:
                command.extend([f"--{key}", value])
        return command

    @staticmethod
    def _build_subproject_command(
        repo_root: Path,
        pr_args: Mapping[str, str],
    ) -> list[str]:
        command = [
            "make",
            "-C",
            str(repo_root),
            "pr",
            f"PR_ACTION={pr_args.get('action', 'status')}",
            f"PR_BASE={pr_args.get('base', 'main')}",
            f"PR_DRAFT={pr_args.get('draft', '0')}",
            f"PR_MERGE_METHOD={pr_args.get('merge_method', 'squash')}",
            f"PR_AUTO={pr_args.get('auto', '0')}",
            f"PR_DELETE_BRANCH={pr_args.get('delete_branch', '0')}",
            f"PR_CHECKS_STRICT={pr_args.get('checks_strict', '0')}",
            f"PR_RELEASE_ON_MERGE={pr_args.get('release_on_merge', '1')}",
        ]
        for key, flag in (
            ("head", "PR_HEAD"),
            ("number", "PR_NUMBER"),
            ("title", "PR_TITLE"),
            ("body", "PR_BODY"),
        ):
            value = pr_args.get(key, "")
            if value:
                command.append(f"{flag}={value}")
        return command


__all__ = ["PrWorkspaceManager"]
