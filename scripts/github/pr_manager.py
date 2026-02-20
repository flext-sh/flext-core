#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _run_capture(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        )
    return result.stdout.strip()


def _run_stream(command: list[str], cwd: Path) -> int:
    result = subprocess.run(command, cwd=cwd, check=False)
    return result.returncode


def _current_branch(repo_root: Path) -> str:
    return _run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)


def _open_pr_for_head(repo_root: Path, head: str) -> dict[str, object] | None:
    raw = _run_capture(
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
        repo_root,
    )
    payload = json.loads(raw)
    if not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        return None
    return first


def _print_status(repo_root: Path, base: str, head: str) -> int:
    pr = _open_pr_for_head(repo_root, head)
    print(f"repo={repo_root}")
    print(f"base={base}")
    print(f"head={head}")
    if pr is None:
        print("status=no-open-pr")
        return 0
    print("status=open")
    print(f"pr_number={pr.get('number')}")
    print(f"pr_title={pr.get('title')}")
    print(f"pr_url={pr.get('url')}")
    print(f"pr_state={pr.get('state')}")
    print(f"pr_draft={pr.get('isDraft')}")
    return 0


def _selector(pr_number: str, head: str) -> str:
    return pr_number if pr_number else head


def _create_pr(
    repo_root: Path,
    base: str,
    head: str,
    title: str,
    body: str,
    draft: int,
) -> int:
    existing = _open_pr_for_head(repo_root, head)
    if existing is not None:
        print(f"status=already-open")
        print(f"pr_url={existing.get('url')}")
        return 0

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
    if draft == 1:
        command.append("--draft")

    created = _run_capture(command, repo_root)
    print("status=created")
    print(f"pr_url={created}")
    return 0


def _merge_pr(
    repo_root: Path,
    selector: str,
    method: str,
    auto: int,
    delete_branch: int,
) -> int:
    command = ["gh", "pr", "merge", selector]
    merge_flag = {
        "merge": "--merge",
        "rebase": "--rebase",
        "squash": "--squash",
    }.get(method, "--squash")
    command.append(merge_flag)
    if auto == 1:
        command.append("--auto")
    if delete_branch == 1:
        command.append("--delete-branch")
    exit_code = _run_stream(command, repo_root)
    if exit_code == 0:
        print("status=merged")
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--repo-root", type=Path, default=Path("."))
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    head = args.head or _current_branch(repo_root)
    base = args.base
    selector = _selector(args.number, head)

    if args.action == "status":
        return _print_status(repo_root, base, head)

    if args.action == "create":
        title = args.title or f"chore: sync {head}"
        body = args.body or "Automated PR managed by scripts/github/pr_manager.py"
        return _create_pr(repo_root, base, head, title, body, args.draft)

    if args.action == "view":
        return _run_stream(["gh", "pr", "view", selector], repo_root)

    if args.action == "checks":
        return _run_stream(["gh", "pr", "checks", selector], repo_root)

    if args.action == "merge":
        return _merge_pr(
            repo_root,
            selector,
            args.merge_method,
            args.auto,
            args.delete_branch,
        )

    if args.action == "close":
        return _run_stream(["gh", "pr", "close", selector], repo_root)

    raise RuntimeError(f"unknown action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
