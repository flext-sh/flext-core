#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.selection import resolve_projects
from libs.subprocess import run_capture, run_checked


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--workspace-root", type=Path, default=Path())
    _ = parser.add_argument("--project", action="append", default=[])
    _ = parser.add_argument("--include-root", type=int, default=1)
    _ = parser.add_argument("--branch", default="")
    _ = parser.add_argument("--fail-fast", type=int, default=0)
    _ = parser.add_argument("--checkpoint", type=int, default=1)
    _ = parser.add_argument("--pr-action", default="status")
    _ = parser.add_argument("--pr-base", default="main")
    _ = parser.add_argument("--pr-head", default="")
    _ = parser.add_argument("--pr-number", default="")
    _ = parser.add_argument("--pr-title", default="")
    _ = parser.add_argument("--pr-body", default="")
    _ = parser.add_argument("--pr-draft", default="0")
    _ = parser.add_argument("--pr-merge-method", default="squash")
    _ = parser.add_argument("--pr-auto", default="0")
    _ = parser.add_argument("--pr-delete-branch", default="0")
    _ = parser.add_argument("--pr-checks-strict", default="0")
    _ = parser.add_argument("--pr-release-on-merge", default="1")
    return parser.parse_args()


def _repo_display_name(repo_root: Path, workspace_root: Path) -> str:
    return workspace_root.name if repo_root == workspace_root else repo_root.name


def _has_changes(repo_root: Path) -> bool:
    return bool(run_capture(["git", "status", "--porcelain"], cwd=repo_root).strip())


def _current_branch(repo_root: Path) -> str:
    return run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)


def _checkout_branch(repo_root: Path, branch: str) -> None:
    if not branch:
        return
    current = _current_branch(repo_root)
    if current == branch:
        return
    checkout = subprocess.run(
        ["git", "checkout", branch],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if checkout.returncode == 0:
        return
    detail = (checkout.stderr or checkout.stdout).lower()
    if "local changes" in detail or "would be overwritten" in detail:
        run_checked(["git", "checkout", "-B", branch], cwd=repo_root)
        return
    fetch = subprocess.run(
        ["git", "fetch", "origin", branch],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if fetch.returncode == 0:
        run_checked(
            ["git", "checkout", "-B", branch, f"origin/{branch}"], cwd=repo_root
        )
    else:
        run_checked(["git", "checkout", "-B", branch], cwd=repo_root)


def _checkpoint(repo_root: Path, branch: str) -> None:
    if not _has_changes(repo_root):
        return
    run_checked(["git", "add", "-A"], cwd=repo_root)
    staged = run_capture(["git", "diff", "--cached", "--name-only"], cwd=repo_root)
    if not staged.strip():
        return
    run_checked(
        ["git", "commit", "-m", "chore: checkpoint pending 0.11.0-dev changes"],
        cwd=repo_root,
    )
    push_cmd = ["git", "push", "-u", "origin", branch] if branch else ["git", "push"]
    push = subprocess.run(
        push_cmd, cwd=repo_root, check=False, capture_output=True, text=True
    )
    if push.returncode == 0:
        return
    if branch:
        run_checked(["git", "pull", "--rebase", "origin", branch], cwd=repo_root)
    else:
        run_checked(["git", "pull", "--rebase"], cwd=repo_root)
    run_checked(push_cmd, cwd=repo_root)


def _run_pr(repo_root: Path, workspace_root: Path, args: argparse.Namespace) -> int:
    report_dir = workspace_root / ".reports" / "workspace" / "pr"
    report_dir.mkdir(parents=True, exist_ok=True)
    display = _repo_display_name(repo_root, workspace_root)
    log_path = report_dir / f"{display}.log"
    if repo_root == workspace_root:
        command = [
            "python",
            "scripts/github/pr_manager.py",
            "--repo-root",
            str(repo_root),
            "--action",
            args.pr_action,
            "--base",
            args.pr_base,
            "--draft",
            args.pr_draft,
            "--merge-method",
            args.pr_merge_method,
            "--auto",
            args.pr_auto,
            "--delete-branch",
            args.pr_delete_branch,
            "--checks-strict",
            args.pr_checks_strict,
            "--release-on-merge",
            args.pr_release_on_merge,
        ]
        if args.pr_head:
            command.extend(["--head", args.pr_head])
        if args.pr_number:
            command.extend(["--number", args.pr_number])
        if args.pr_title:
            command.extend(["--title", args.pr_title])
        if args.pr_body:
            command.extend(["--body", args.pr_body])
    else:
        command = [
            "make",
            "-C",
            str(repo_root),
            "pr",
            f"PR_ACTION={args.pr_action}",
            f"PR_BASE={args.pr_base}",
            f"PR_DRAFT={args.pr_draft}",
            f"PR_MERGE_METHOD={args.pr_merge_method}",
            f"PR_AUTO={args.pr_auto}",
            f"PR_DELETE_BRANCH={args.pr_delete_branch}",
            f"PR_CHECKS_STRICT={args.pr_checks_strict}",
            f"PR_RELEASE_ON_MERGE={args.pr_release_on_merge}",
        ]
        if args.pr_head:
            command.append(f"PR_HEAD={args.pr_head}")
        if args.pr_number:
            command.append(f"PR_NUMBER={args.pr_number}")
        if args.pr_title:
            command.append(f"PR_TITLE={args.pr_title}")
        if args.pr_body:
            command.append(f"PR_BODY={args.pr_body}")

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    elapsed = int(time.monotonic() - started)
    status = "OK" if result.returncode == 0 else "FAIL"
    print(
        f"[{status}] {display} pr ({elapsed}s) exit={result.returncode} log={log_path}"
    )
    return result.returncode


def main() -> int:
    args = _parse_args()
    workspace_root = args.workspace_root.resolve()
    projects = resolve_projects(workspace_root, list(args.project))
    repos = [project.path for project in projects]
    if args.include_root == 1:
        repos.append(workspace_root)

    failures = 0
    for repo_root in repos:
        display = _repo_display_name(repo_root, workspace_root)
        print(f"[RUN] {display}", flush=True)
        _checkout_branch(repo_root, args.branch)
        if args.checkpoint == 1:
            _checkpoint(repo_root, args.branch)
        exit_code = _run_pr(repo_root, workspace_root, args)
        if exit_code != 0:
            failures += 1
            if args.fail_fast == 1:
                break

    total = len(repos)
    success = total - failures
    print(f"summary total={total} success={success} fail={failures} skip=0")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
