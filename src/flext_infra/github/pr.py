"""Pull request lifecycle management service.

Wraps GitHub CLI (gh) PR operations with FlextResult error handling.

Usage:
    python -m flext_infra github pr --repo-root <path> --action status

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from flext_core.result import FlextResult, r
from flext_core.typings import t

from flext_infra.git import GitService
from flext_infra.subprocess import CommandRunner
from flext_infra.versioning import VersioningService


class PrManager:
    """Infrastructure service for pull request lifecycle management.

    Provides FlextResult-wrapped PR operations (status, create, merge,
    view, checks, close) via the ``gh`` CLI.
    """

    def __init__(
        self,
        runner: CommandRunner | None = None,
        git: GitService | None = None,
        versioning: VersioningService | None = None,
    ) -> None:
        """Initialize the PR manager."""
        self._runner = runner or CommandRunner()
        self._git = git or GitService(self._runner)
        self._versioning = versioning or VersioningService()

    def open_pr_for_head(
        self,
        repo_root: Path,
        head: str,
    ) -> FlextResult[Mapping[str, t.ScalarValue]]:
        """Find an open PR for the given head branch.

        Args:
            repo_root: Repository root directory.
            head: Head branch name.

        Returns:
            FlextResult with PR dict. Empty dict means no open PR found.

        """
        result = self._runner.capture(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--head",
                head,
                "--json",
                "number,title,state,baseRefName,headRefName,url,isDraft",
                "--limit",
                "1",
            ],
            cwd=repo_root,
        )
        if result.is_failure:
            return r[Mapping[str, t.ScalarValue]].fail(
                result.error or "failed to list PRs",
            )
        try:
            payload = json.loads(result.value)
        except json.JSONDecodeError as exc:
            return r[Mapping[str, t.ScalarValue]].fail(f"invalid JSON: {exc}")

        if not payload:
            return r[Mapping[str, t.ScalarValue]].ok({})
        first = payload[0]
        if type(first) is not dict:
            return r[Mapping[str, t.ScalarValue]].ok({})
        return r[Mapping[str, t.ScalarValue]].ok(first)

    def status(
        self,
        repo_root: Path,
        base: str,
        head: str,
    ) -> FlextResult[Mapping[str, t.ScalarValue]]:
        """Get PR status for the given head branch.

        Args:
            repo_root: Repository root directory.
            base: Base branch name.
            head: Head branch name.

        Returns:
            FlextResult with status info dict.

        """
        pr_result = self.open_pr_for_head(repo_root, head)
        if pr_result.is_failure:
            return r[Mapping[str, t.ScalarValue]].fail(pr_result.error or "status check failed")

        info: MutableMapping[str, t.ScalarValue] = {
            "repo": str(repo_root),
            "base": base,
            "head": head,
        }
        pr = pr_result.value
        if not pr:
            info["status"] = "no-open-pr"
        else:
            info["status"] = "open"
            info["pr_number"] = pr.get("number")
            info["pr_title"] = pr.get("title")
            info["pr_url"] = pr.get("url")
            info["pr_state"] = pr.get("state")
            info["pr_draft"] = pr.get("isDraft")
        return r[Mapping[str, t.ScalarValue]].ok(info)

    def create(
        self,
        repo_root: Path,
        base: str,
        head: str,
        title: str,
        body: str,
        *,
        draft: bool = False,
    ) -> FlextResult[Mapping[str, t.ScalarValue]]:
        """Create a new PR or report existing one.

        Args:
            repo_root: Repository root directory.
            base: Base branch name.
            head: Head branch name.
            title: PR title.
            body: PR body.
            draft: Whether to create as draft.

        Returns:
            FlextResult with creation status info.

        """
        existing_result = self.open_pr_for_head(repo_root, head)
        if existing_result.is_failure:
            return r[Mapping[str, t.ScalarValue]].fail(
                existing_result.error or "failed to check existing PRs",
            )

        existing = existing_result.value
        if existing:
            return r[Mapping[str, t.ScalarValue]].ok({
                "status": "already-open",
                "pr_url": existing.get("url"),
            })

        command = [
            "gh",
            "pr",
            "create",
            "--base",
            base,
            "--head",
            head,
            "--title",
            title,
            "--body",
            body,
        ]
        if draft:
            command.append("--draft")

        result = self._runner.capture(command, cwd=repo_root)
        if result.is_failure:
            return r[Mapping[str, t.ScalarValue]].fail(result.error or "PR creation failed")
        return r[Mapping[str, t.ScalarValue]].ok({
            "status": "created",
            "pr_url": result.value,
        })

    def view(self, repo_root: Path, selector: str) -> FlextResult[str]:
        """View a PR by selector (number or branch name).

        Args:
            repo_root: Repository root directory.
            selector: PR number or head branch.

        Returns:
            FlextResult with command output.

        """
        return self._runner.capture(
            ["gh", "pr", "view", selector],
            cwd=repo_root,
        )

    def checks(
        self,
        repo_root: Path,
        selector: str,
        *,
        strict: bool = False,
    ) -> FlextResult[Mapping[str, t.ScalarValue]]:
        """Run PR checks.

        Args:
            repo_root: Repository root directory.
            selector: PR number or head branch.
            strict: If True, treat check failures as errors.

        Returns:
            FlextResult with check status info.

        """
        result = self._runner.run(
            ["gh", "pr", "checks", selector],
            cwd=repo_root,
        )
        if result.is_success:
            return r[Mapping[str, t.ScalarValue]].ok({"status": "checks-passed"})
        if not strict:
            return r[Mapping[str, t.ScalarValue]].ok({"status": "checks-nonblocking"})
        return r[Mapping[str, t.ScalarValue]].fail(result.error or "checks failed")

    def merge(
        self,
        repo_root: Path,
        selector: str,
        head: str,
        *,
        method: str = "squash",
        auto: bool = False,
        delete_branch: bool = False,
        release_on_merge: bool = True,
    ) -> FlextResult[Mapping[str, t.ScalarValue]]:
        """Merge a PR with retry on rebase.

        Args:
            repo_root: Repository root directory.
            selector: PR number or head branch.
            head: Head branch name.
            method: Merge method (merge, rebase, squash).
            auto: Enable auto-merge.
            delete_branch: Delete branch after merge.
            release_on_merge: Trigger release workflow on merge.

        Returns:
            FlextResult with merge status info.

        """
        if selector == head:
            pr_result = self.open_pr_for_head(repo_root, head)
            if pr_result.is_success and not pr_result.value:
                return r[Mapping[str, t.ScalarValue]].ok({"status": "no-open-pr"})

        merge_flag = {
            "merge": "--merge",
            "rebase": "--rebase",
            "squash": "--squash",
        }.get(method, "--squash")

        command = ["gh", "pr", "merge", selector, merge_flag]
        if auto:
            command.append("--auto")
        if delete_branch:
            command.append("--delete-branch")

        result = self._runner.run(command, cwd=repo_root)
        if result.is_failure:
            stderr = result.error or ""
            if "not mergeable" in stderr:
                update_result = self._runner.run(
                    ["gh", "pr", "update-branch", selector, "--rebase"],
                    cwd=repo_root,
                )
                if update_result.is_success:
                    result = self._runner.run(command, cwd=repo_root)

        if result.is_failure:
            return r[Mapping[str, t.ScalarValue]].fail(result.error or "merge failed")

        info: MutableMapping[str, t.ScalarValue] = {"status": "merged"}
        if release_on_merge:
            release_result = self._trigger_release_if_needed(repo_root, head)
            if release_result.is_success:
                info["release"] = release_result.value
        return r[Mapping[str, t.ScalarValue]].ok(info)

    def close(self, repo_root: Path, selector: str) -> FlextResult[bool]:
        """Close a PR.

        Args:
            repo_root: Repository root directory.
            selector: PR number or head branch.

        Returns:
            FlextResult[bool] with True on success.

        """
        return self._runner.run_checked(
            ["gh", "pr", "close", selector],
            cwd=repo_root,
        )

    def _trigger_release_if_needed(
        self,
        repo_root: Path,
        head: str,
    ) -> FlextResult[Mapping[str, str]]:
        """Trigger release workflow if repo supports it.

        Args:
            repo_root: Repository root directory.
            head: Head branch name.

        Returns:
            FlextResult with release info or skip reason.

        """
        release_yml = repo_root / ".github" / "workflows" / "release.yml"
        if not release_yml.exists():
            return r[Mapping[str, str]].ok({"status": "no-release-workflow"})

        tag_result = self._versioning.release_tag_from_branch(head)
        if tag_result.is_failure:
            return r[Mapping[str, str]].ok({"status": "no-release-tag"})

        tag = tag_result.value
        view_result = self._runner.run(
            ["gh", "release", "view", tag],
            cwd=repo_root,
        )
        if view_result.is_success:
            return r[Mapping[str, str]].ok({"status": "release-exists", "tag": tag})

        dispatch_result = self._runner.run(
            ["gh", "workflow", "run", "release.yml", "-f", f"tag={tag}"],
            cwd=repo_root,
        )
        if dispatch_result.is_success:
            return r[Mapping[str, str]].ok({"status": "release-dispatched", "tag": tag})
        return r[Mapping[str, str]].ok({"status": "release-dispatch-failed", "tag": tag})


def _selector(pr_number: str, head: str) -> str:
    return pr_number or head


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PR lifecycle management")
    _ = parser.add_argument("--repo-root", type=Path, default=Path())
    _ = parser.add_argument(
        "--action",
        default="status",
        choices=["status", "create", "view", "checks", "merge", "close"],
    )
    _ = parser.add_argument("--base", default="main")
    _ = parser.add_argument("--head", default="")
    _ = parser.add_argument("--number", default="")
    _ = parser.add_argument("--title", default="")
    _ = parser.add_argument("--body", default="")
    _ = parser.add_argument("--draft", type=int, default=0)
    _ = parser.add_argument("--merge-method", default="squash")
    _ = parser.add_argument("--auto", type=int, default=0)
    _ = parser.add_argument("--delete-branch", type=int, default=0)
    _ = parser.add_argument("--checks-strict", type=int, default=0)
    _ = parser.add_argument("--release-on-merge", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    """Dispatch requested PR action and return its exit code."""
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    manager = PrManager()

    git = GitService()
    head_result = git.current_branch(repo_root)
    head = args.head or (head_result.value if head_result.is_success else "HEAD")
    base = args.base
    selector = _selector(args.number, head)

    if args.action == "status":
        result = manager.status(repo_root, base, head)
        if result.is_success:
            for key, value in result.value.items():
                _ = sys.stdout.write(f"{key}={value}\n")
            return 0
        _ = sys.stderr.write(f"Error: {result.error}\n")
        return 1

    if args.action == "create":
        title = args.title or f"chore: sync {head}"
        body = args.body or "Automated PR managed by flext_infra.github.pr"
        result = manager.create(
            repo_root,
            base,
            head,
            title,
            body,
            draft=args.draft == 1,
        )
        if result.is_success:
            for key, value in result.value.items():
                _ = sys.stdout.write(f"{key}={value}\n")
            return 0
        _ = sys.stderr.write(f"Error: {result.error}\n")
        return 1

    if args.action == "view":
        result_view = manager.view(repo_root, selector)
        if result_view.is_success:
            _ = sys.stdout.write(f"{result_view.value}\n")
            return 0
        _ = sys.stderr.write(f"Error: {result_view.error}\n")
        return 1

    if args.action == "checks":
        result = manager.checks(
            repo_root,
            selector,
            strict=args.checks_strict == 1,
        )
        if result.is_success:
            for key, value in result.value.items():
                _ = sys.stdout.write(f"{key}={value}\n")
            return 0
        _ = sys.stderr.write(f"Error: {result.error}\n")
        return 1

    if args.action == "merge":
        result = manager.merge(
            repo_root,
            selector,
            head,
            method=args.merge_method,
            auto=args.auto == 1,
            delete_branch=args.delete_branch == 1,
            release_on_merge=args.release_on_merge == 1,
        )
        if result.is_success:
            for key, value in result.value.items():
                _ = sys.stdout.write(f"{key}={value}\n")
            return 0
        _ = sys.stderr.write(f"Error: {result.error}\n")
        return 1

    if args.action == "close":
        result_close = manager.close(repo_root, selector)
        if result_close.is_success:
            _ = sys.stdout.write("status=closed\n")
            return 0
        _ = sys.stderr.write(f"Error: {result_close.error}\n")
        return 1

    msg = f"unknown action: {args.action}"
    raise RuntimeError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
