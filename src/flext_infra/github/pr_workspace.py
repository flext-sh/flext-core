"""Workspace-wide PR automation service.

Wraps multi-repository PR operations with r error handling,
replacing scripts/github/pr_workspace.py with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from pathlib import Path

from flext_core import r
from flext_infra import (
    FlextInfraUtilitiesReporting,
    FlextInfraUtilitiesSelection,
    c,
    m,
    p,
    u,
)


class FlextInfraPrWorkspaceManager:
    """Infrastructure service for workspace-wide PR automation.

    Orchestrates PR operations (status, create, merge, etc.) across all
    workspace repositories with checkpoint and branch management.
    """

    def __init__(
        self,
        runner: p.Infra.CommandRunner | None = None,
        selector: FlextInfraUtilitiesSelection | None = None,
        reporting: FlextInfraUtilitiesReporting | None = None,
    ) -> None:
        """Initialize the workspace PR manager."""
        self._runner = runner
        self._selector = selector
        self._reporting = reporting

    @staticmethod
    def _build_root_command(repo_root: Path, pr_args: Mapping[str, str]) -> list[str]:
        command = [
            c.Infra.Toml.PYTHON,
            "-m",
            "flext_infra.github.pr",
            "--repo-root",
            str(repo_root),
            "--action",
            pr_args.get(c.Infra.ReportKeys.ACTION, c.Infra.ReportKeys.STATUS),
            "--base",
            pr_args.get("base", c.Infra.Git.MAIN),
            "--draft",
            pr_args.get("draft", "0"),
            "--merge-method",
            pr_args.get("merge_method", c.Infra.Cli.GhCmd.SQUASH),
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
            c.Infra.Cli.MAKE,
            "-C",
            str(repo_root),
            c.Infra.Cli.GhCmd.PR,
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

    @staticmethod
    def _repo_display_name(repo_root: Path, workspace_root: Path) -> str:
        return workspace_root.name if repo_root == workspace_root else repo_root.name

    def checkout_branch(self, repo_root: Path, branch: str) -> r[bool]:
        """Checkout a branch with canonical git contract only."""
        if not branch:
            return r[bool].ok(True)
        return u.Infra.git_checkout(repo_root, branch)

    def checkpoint(self, repo_root: Path, branch: str) -> r[bool]:
        """Commit and push pending changes without fallback strategies."""
        changes_result = u.Infra.git_has_changes(repo_root)
        if changes_result.is_failure:
            return r[bool].fail(changes_result.error or "changes check failed")
        if not changes_result.value:
            return r[bool].ok(True)
        add_result = u.Infra.git_add(repo_root)
        if add_result.is_failure:
            return r[bool].fail(add_result.error or "git add failed")
        staged_result = u.Infra.git_diff_names(repo_root, cached=True)
        if staged_result.is_success and (not staged_result.value.strip()):
            return r[bool].ok(True)
        commit_result = u.Infra.git_commit(
            repo_root, "chore: checkpoint pending changes"
        )
        if commit_result.is_failure:
            return r[bool].fail(commit_result.error or "git commit failed")
        return u.Infra.git_push(
            repo_root,
            remote=c.Infra.Git.ORIGIN if branch else "",
            branch=branch,
            set_upstream=bool(branch),
        )

    def has_changes(self, repo_root: Path) -> r[bool]:
        """Check if the repository has uncommitted changes."""
        return u.Infra.git_has_changes(repo_root)

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
    ) -> r[m.Infra.Github.PrOrchestrationResult]:
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
        if self._selector is not None:
            projects_result: r[list[m.Infra.Workspace.ProjectInfo]] = (
                self._selector.resolve_projects(workspace_root, projects or [])
            )
        else:
            projects_result = u.Infra.resolve_projects(workspace_root, projects or [])
        if projects_result.is_failure:
            return r[m.Infra.Github.PrOrchestrationResult].fail(
                projects_result.error or "project resolution failed",
            )
        repos = [p.path for p in projects_result.value]
        if include_root:
            repos.append(workspace_root)
        effective_args = pr_args or {
            c.Infra.ReportKeys.ACTION: c.Infra.ReportKeys.STATUS,
            "base": c.Infra.Git.MAIN,
        }
        failures = 0
        results: list[m.Infra.Github.PrExecutionResult] = []
        for repo_root in repos:
            self.checkout_branch(repo_root, branch)
            if checkpoint:
                self.checkpoint(repo_root, branch)
            run_result: r[m.Infra.Github.PrExecutionResult] = self.run_pr(
                repo_root,
                workspace_root,
                effective_args,
            )
            if run_result.is_success:
                pr_data = run_result.value
                results.append(pr_data)
                if pr_data.exit_code != 0:
                    failures += 1
                    if fail_fast:
                        break
            else:
                failures += 1
                if fail_fast:
                    break
        total = len(repos)
        orchestration_results: tuple[m.Infra.Github.PrExecutionResult, ...] = tuple(
            results,
        )
        return r[m.Infra.Github.PrOrchestrationResult].ok(
            m.Infra.Github.PrOrchestrationResult(
                total=total,
                success=total - failures,
                fail=failures,
                results=orchestration_results,
            ),
        )

    def run_pr(
        self,
        repo_root: Path,
        workspace_root: Path,
        pr_args: Mapping[str, str],
    ) -> r[m.Infra.Github.PrExecutionResult]:
        """Execute a PR operation on a single repository.

        Args:
            repo_root: Repository root directory.
            workspace_root: Workspace root directory.
            pr_args: PR argument dictionary.

        Returns:
            r with execution result info.

        """
        display = self._repo_display_name(repo_root, workspace_root)
        if self._reporting is not None:
            report_dir = self._reporting.get_report_dir(
                workspace_root,
                c.Infra.ReportKeys.WORKSPACE,
                c.Infra.Cli.GhCmd.PR,
            )
        else:
            report_dir = u.Infra.get_report_dir(
                workspace_root,
                c.Infra.ReportKeys.WORKSPACE,
                c.Infra.Cli.GhCmd.PR,
            )
        with contextlib.suppress(OSError):
            report_dir.mkdir(parents=True, exist_ok=True)
        log_path = report_dir / f"{display}.log"
        if repo_root == workspace_root:
            command = self._build_root_command(repo_root, pr_args)
        else:
            command = self._build_subproject_command(repo_root, pr_args)
        started = time.monotonic()
        if self._runner is not None:
            to_file_result: r[int] = self._runner.run_to_file(command, log_path)
        else:
            to_file_result = u.Infra.run_to_file(command, log_path)
        if to_file_result.is_failure:
            return r[m.Infra.Github.PrExecutionResult].fail(
                to_file_result.error or "command execution error",
            )
        exit_code = to_file_result.value
        elapsed = int(time.monotonic() - started)
        status = c.Infra.Status.OK if exit_code == 0 else c.Infra.Status.FAIL
        log_str = str(log_path)
        return r[m.Infra.Github.PrExecutionResult].ok(
            m.Infra.Github.PrExecutionResult(
                display=display,
                status=status,
                elapsed=elapsed,
                exit_code=exit_code,
                log_path=log_str,
            ),
        )


__all__ = ["FlextInfraPrWorkspaceManager", "u"]
