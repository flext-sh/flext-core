"""Pull request lifecycle management service.

Wraps GitHub CLI (gh) PR operations with r error handling.

Usage:
    python -m flext_infra github pr --repo-root <path> --action status

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from flext_core import r
from flext_infra import (
    FlextInfraUtilitiesVersioning,
    c,
    m,
    output,
    p,
    t,
    u,
)


class FlextInfraPrManager:
    """Infrastructure service for pull request lifecycle management.

    Provides r-wrapped PR operations (status, create, merge,
    view, checks, close) via the ``gh`` CLI.
    """

    def __init__(
        self,
        runner: p.Infra.CommandRunner | None = None,
        versioning: FlextInfraUtilitiesVersioning | None = None,
    ) -> None:
        """Initialize the PR manager."""
        self._runner: p.Infra.CommandRunner | None = runner
        self._versioning: FlextInfraUtilitiesVersioning | None = versioning

    def _run(
        self, command: list[str], repo_root: Path
    ) -> r[m.Infra.Core.CommandOutput]:
        if self._runner is not None:
            return self._runner.run(command, cwd=repo_root)
        return u.Infra.run(command, cwd=repo_root)

    def _run_checked(self, command: list[str], repo_root: Path) -> r[bool]:
        if self._runner is not None:
            return self._runner.run_checked(command, cwd=repo_root)
        return u.Infra.run_checked(command, cwd=repo_root)

    def _capture(self, command: list[str], repo_root: Path) -> r[str]:
        if self._runner is not None:
            return self._runner.capture(command, cwd=repo_root)
        return u.Infra.capture(command, cwd=repo_root)

    def _release_tag_from_branch(self, head: str) -> r[str]:
        if self._versioning is not None:
            return self._versioning.release_tag_from_branch(head)
        return u.Infra.release_tag_from_branch(head)

    def checks(
        self,
        repo_root: Path,
        selector: str,
        *,
        strict: bool = False,
    ) -> r[Mapping[str, t.Scalar]]:
        """Run PR checks.

        Args:
            repo_root: Repository root directory.
            selector: PR number or head branch.
            strict: If True, treat check failures as errors.

        Returns:
            r with check status info.

        """
        result = self._run(
            [c.Infra.Cli.GH, c.Infra.Cli.GhCmd.PR, c.Infra.Verbs.CHECKS, selector],
            repo_root,
        )
        if result.is_success:
            return r[Mapping[str, t.Scalar]].ok({
                c.Infra.ReportKeys.STATUS: "checks-passed",
            })
        if not strict:
            return r[Mapping[str, t.Scalar]].ok({
                c.Infra.ReportKeys.STATUS: "checks-nonblocking",
            })
        return r[Mapping[str, t.Scalar]].fail(result.error or "checks failed")

    def close(self, repo_root: Path, selector: str) -> r[bool]:
        """Close a PR.

        Args:
            repo_root: Repository root directory.
            selector: PR number or head branch.

        Returns:
            r[bool] with True on success.

        """
        return self._run_checked(
            [c.Infra.Cli.GH, c.Infra.Cli.GhCmd.PR, c.Infra.Verbs.CLOSE, selector],
            repo_root,
        )

    def create(
        self,
        repo_root: Path,
        base: str,
        head: str,
        title: str,
        body: str,
        *,
        draft: bool = False,
    ) -> r[Mapping[str, t.Scalar]]:
        """Create a new PR or report existing one.

        Args:
            repo_root: Repository root directory.
            base: Base branch name.
            head: Head branch name.
            title: PR title.
            body: PR body.
            draft: Whether to create as draft.

        Returns:
            r with creation status info.

        """
        existing_result: r[Mapping[str, t.Scalar]] = self.open_pr_for_head(
            repo_root,
            head,
        )
        if existing_result.is_failure:
            return r[Mapping[str, t.Scalar]].fail(
                existing_result.error or "failed to check existing PRs",
            )
        existing = existing_result.unwrap()
        if existing:
            return r[Mapping[str, t.Scalar]].ok({
                c.Infra.ReportKeys.STATUS: "already-open",
                "pr_url": existing.get(c.Infra.ReportKeys.URL, ""),
            })
        command = [
            c.Infra.Cli.GH,
            c.Infra.Cli.GhCmd.PR,
            c.Infra.Cli.GhCmd.CREATE,
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
        result: r[str] = self._capture(command, repo_root)
        return result.fold(
            on_failure=lambda e: r[Mapping[str, t.Scalar]].fail(
                e or "PR creation failed"
            ),
            on_success=lambda v: r[Mapping[str, t.Scalar]].ok({
                c.Infra.ReportKeys.STATUS: "created",
                "pr_url": v,
            }),
        )

    def merge(
        self,
        repo_root: Path,
        selector: str,
        head: str,
        *,
        method: str = c.Infra.Cli.GhCmd.SQUASH,
        auto: bool = False,
        delete_branch: bool = False,
        release_on_merge: bool = True,
    ) -> r[t.Infra.ContainerDict]:
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
            r with merge status info.

        """
        if selector == head:
            pr_result = self.open_pr_for_head(repo_root, head)
            if pr_result.is_success and (not pr_result.unwrap()):
                return r[t.Infra.ContainerDict].ok({
                    c.Infra.ReportKeys.STATUS: "no-open-pr",
                })
        merge_flag = {
            c.Infra.Cli.GhCmd.MERGE: "--merge",
            c.Infra.Verbs.REBASE: "--rebase",
            c.Infra.Cli.GhCmd.SQUASH: "--squash",
        }.get(method, "--squash")
        command = [
            c.Infra.Cli.GH,
            c.Infra.Cli.GhCmd.PR,
            c.Infra.Cli.GhCmd.MERGE,
            selector,
            merge_flag,
        ]
        if auto:
            command.append("--auto")
        if delete_branch:
            command.append("--delete-branch")
        result = self._run(command, repo_root)
        if result.is_failure:
            stderr = result.error or ""
            if "not mergeable" in stderr:
                update_result = self._run(
                    [
                        c.Infra.Cli.GH,
                        c.Infra.Cli.GhCmd.PR,
                        c.Infra.Cli.GhCmd.UPDATE_BRANCH,
                        selector,
                        "--rebase",
                    ],
                    repo_root,
                )
                if update_result.is_success:
                    result = self._run(command, repo_root)
        if result.is_failure:
            return r[t.Infra.ContainerDict].fail(
                result.error or "merge failed",
            )
        info: t.Infra.ContainerDict = {
            c.Infra.ReportKeys.STATUS: "merged",
        }
        if release_on_merge:
            release_result = self._trigger_release_if_needed(repo_root, head)
            if release_result.is_success:
                release_data = release_result.value
                for key, value in release_data.items():
                    info[f"release_{key}"] = value
        return r[t.Infra.ContainerDict].ok(info)

    def open_pr_for_head(self, repo_root: Path, head: str) -> r[Mapping[str, t.Scalar]]:
        """Find an open PR for the given head branch.

        Args:
            repo_root: Repository root directory.
            head: Head branch name.

        Returns:
            r with PR dict. Empty dict means no open PR found.

        """
        result = self._capture(
            [
                c.Infra.Cli.GH,
                c.Infra.Cli.GhCmd.PR,
                c.Infra.Cli.GhCmd.LIST,
                "--state",
                c.Infra.Verbs.OPEN,
                "--head",
                head,
                "--json",
                "number,title,state,baseRefName,headRefName,url,isDraft",
                "--limit",
                "1",
            ],
            repo_root,
        )
        if result.is_failure:
            return r[Mapping[str, t.Scalar]].fail(result.error or "failed to list PRs")
        payload_result = u.Infra.parse(str(result.unwrap()))
        if payload_result.is_failure:
            return r[Mapping[str, t.Scalar]].fail(
                payload_result.error or "invalid JSON",
            )
        payload = payload_result.value
        if not isinstance(payload, list) or len(payload) == 0:
            return r[Mapping[str, t.Scalar]].ok({})
        first = payload[0]
        if not isinstance(first, dict):
            return r[Mapping[str, t.Scalar]].ok({})
        try:
            typed_first = TypeAdapter(dict[str, t.Scalar]).validate_python(
                first,
                strict=False,
            )
        except ValidationError:
            return r[Mapping[str, t.Scalar]].ok({})
        return r[Mapping[str, t.Scalar]].ok(typed_first)

    def status(
        self,
        repo_root: Path,
        base: str,
        head: str,
    ) -> r[Mapping[str, t.Scalar]]:
        """Get PR status for the given head branch.

        Args:
            repo_root: Repository root directory.
            base: Base branch name.
            head: Head branch name.

        Returns:
            r with status info dict.

        """
        pr_result: r[Mapping[str, t.Scalar]] = self.open_pr_for_head(repo_root, head)
        if pr_result.is_failure:
            return r[Mapping[str, t.Scalar]].fail(
                pr_result.error or "status check failed",
            )
        info: MutableMapping[str, t.Scalar] = {
            "repo": str(repo_root),
            "base": base,
            "head": head,
        }
        pr = pr_result.unwrap()
        if not pr:
            info[c.Infra.ReportKeys.STATUS] = "no-open-pr"
        else:
            info[c.Infra.ReportKeys.STATUS] = c.Infra.Verbs.OPEN
            info["pr_number"] = pr.get("number", "")
            info["pr_title"] = pr.get("title", "")
            info["pr_url"] = pr.get(c.Infra.ReportKeys.URL, "")
            info["pr_state"] = pr.get("state", "")
            info["pr_draft"] = pr.get("isDraft", False)
        return r[Mapping[str, t.Scalar]].ok(info)

    def view(self, repo_root: Path, selector: str) -> r[str]:
        """View a PR by selector (number or branch name).

        Args:
            repo_root: Repository root directory.
            selector: PR number or head branch.

        Returns:
            r with command output.

        """
        return self._capture(
            [c.Infra.Cli.GH, c.Infra.Cli.GhCmd.PR, c.Infra.Cli.GhCmd.VIEW, selector],
            repo_root,
        )

    def _trigger_release_if_needed(
        self,
        repo_root: Path,
        head: str,
    ) -> r[Mapping[str, str]]:
        """Trigger release workflow if repo supports it.

        Args:
            repo_root: Repository root directory.
            head: Head branch name.

        Returns:
            r with release info or skip reason.

        """
        release_yml = repo_root / ".github" / "workflows" / "release.yml"
        if not release_yml.exists():
            return r[Mapping[str, str]].ok({
                c.Infra.ReportKeys.STATUS: "no-release-workflow",
            })
        tag_result: r[str] = self._release_tag_from_branch(head)
        if tag_result.is_failure:
            return r[Mapping[str, str]].ok({
                c.Infra.ReportKeys.STATUS: "no-release-tag",
            })
        tag = tag_result.unwrap()
        view_result = self._run(
            [c.Infra.Cli.GH, c.Infra.ReportKeys.RELEASE, c.Infra.Cli.GhCmd.VIEW, tag],
            repo_root,
        )
        if view_result.is_success:
            return r[Mapping[str, str]].ok({
                c.Infra.ReportKeys.STATUS: "release-exists",
                c.Infra.ReportKeys.TAG: tag,
            })
        dispatch_result = self._run(
            [
                c.Infra.Cli.GH,
                c.Infra.Cli.GhCmd.WORKFLOW,
                c.Infra.Verbs.RUN,
                "release.yml",
                "-f",
                f"tag={tag}",
            ],
            repo_root,
        )
        if dispatch_result.is_success:
            return r[Mapping[str, str]].ok({
                c.Infra.ReportKeys.STATUS: "release-dispatched",
                c.Infra.ReportKeys.TAG: tag,
            })
        return r[Mapping[str, str]].ok({
            c.Infra.ReportKeys.STATUS: "release-dispatch-failed",
            c.Infra.ReportKeys.TAG: tag,
        })

    @staticmethod
    def selector(pr_number: str, head: str) -> str:
        """Resolve PR selector from number or head branch."""
        return pr_number or head

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Build and parse CLI arguments for PR actions."""
        parser = argparse.ArgumentParser(description="PR lifecycle management")
        _ = parser.add_argument("--repo-root", type=Path, default=Path())
        _ = parser.add_argument(
            "--action",
            default=c.Infra.ReportKeys.STATUS,
            choices=[
                c.Infra.ReportKeys.STATUS,
                c.Infra.Cli.GhCmd.CREATE,
                c.Infra.Cli.GhCmd.VIEW,
                c.Infra.Verbs.CHECKS,
                c.Infra.Cli.GhCmd.MERGE,
                c.Infra.Verbs.CLOSE,
            ],
        )
        _ = parser.add_argument("--base", default=c.Infra.Git.MAIN)
        _ = parser.add_argument("--head", default="")
        _ = parser.add_argument("--number", default="")
        _ = parser.add_argument("--title", default="")
        _ = parser.add_argument("--body", default="")
        _ = parser.add_argument("--draft", type=int, default=0)
        _ = parser.add_argument("--merge-method", default=c.Infra.Cli.GhCmd.SQUASH)
        _ = parser.add_argument("--auto", type=int, default=0)
        _ = parser.add_argument("--delete-branch", type=int, default=0)
        _ = parser.add_argument("--checks-strict", type=int, default=0)
        _ = parser.add_argument("--release-on-merge", type=int, default=1)
        return parser.parse_args()


def main() -> int:
    """Dispatch requested PR action and return its exit code."""
    args = FlextInfraPrManager.parse_args()
    repo_root = args.repo_root.resolve()
    manager = FlextInfraPrManager()
    head_result = u.Infra.git_current_branch(repo_root)
    head = args.head or head_result.unwrap_or(c.Infra.Git.HEAD)
    base = args.base
    selector = FlextInfraPrManager.selector(args.number, head)
    if args.action == c.Infra.ReportKeys.STATUS:
        result = manager.status(repo_root, base, head)
        if result.is_success:
            return 0
        output.error(result.error or "status failed")
        return 1
    if args.action == c.Infra.Cli.GhCmd.CREATE:
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
            return 0
        output.error(result.error or "create failed")
        return 1
    if args.action == c.Infra.Cli.GhCmd.VIEW:
        result_view = manager.view(repo_root, selector)
        if result_view.is_success:
            return 0
        output.error(result_view.error or "view failed")
        return 1
    if args.action == c.Infra.Verbs.CHECKS:
        result = manager.checks(repo_root, selector, strict=args.checks_strict == 1)
        if result.is_success:
            return 0
        output.error(result.error or "checks failed")
        return 1
    if args.action == c.Infra.Cli.GhCmd.MERGE:
        merge_result = manager.merge(
            repo_root,
            selector,
            head,
            method=args.merge_method,
            auto=args.auto == 1,
            delete_branch=args.delete_branch == 1,
            release_on_merge=args.release_on_merge == 1,
        )
        if merge_result.is_success:
            return 0
        output.error(merge_result.error or "merge failed")
        return 1
    if args.action == c.Infra.Verbs.CLOSE:
        result_close = manager.close(repo_root, selector)
        if result_close.is_success:
            return 0
        output.error(result_close.error or "close failed")
        return 1
    msg = f"unknown action: {args.action}"
    raise RuntimeError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
