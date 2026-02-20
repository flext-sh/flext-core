#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.versioning import release_tag_from_branch


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
        msg = f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        raise RuntimeError(msg)
    return result.stdout.strip()


def _run_stream(command: list[str], cwd: Path) -> int:
    result = subprocess.run(command, cwd=cwd, check=False)
    return result.returncode


def _run_stream_with_output(command: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    output_parts = [
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    ]
    output = "\n".join(output_parts)
    if output:
        print(output)
    return result.returncode, output


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
    return pr_number or head


def _release_tag_from_head(head: str) -> str | None:
    return release_tag_from_branch(head)


def _is_workspace_release_repo(repo_root: Path) -> bool:
    return (repo_root / ".github" / "workflows" / "release.yml").exists()


def _trigger_release_if_needed(repo_root: Path, head: str) -> None:
    if not _is_workspace_release_repo(repo_root):
        return
    tag = _release_tag_from_head(head)
    if tag is None:
        return

    if _run_stream(["gh", "release", "view", tag], repo_root) == 0:
        print(f"status=release-exists tag={tag}")
        return

    run_code = _run_stream(
        ["gh", "workflow", "run", "release.yml", "-f", f"tag={tag}"],
        repo_root,
    )
    if run_code == 0:
        print(f"status=release-dispatched tag={tag}")
    else:
        print(f"status=release-dispatch-failed tag={tag} exit={run_code}")


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
        print("status=already-open")
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
    head: str,
    method: str,
    auto: int,
    delete_branch: int,
    release_on_merge: int,
) -> int:
    if selector == head and _open_pr_for_head(repo_root, head) is None:
        print("status=no-open-pr")
        return 0

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
    exit_code, output = _run_stream_with_output(command, repo_root)
    if exit_code != 0 and "not mergeable" in output:
        update_code, _ = _run_stream_with_output(
            ["gh", "pr", "update-branch", selector, "--rebase"],
            repo_root,
        )
        if update_code == 0:
            exit_code, _ = _run_stream_with_output(command, repo_root)
    if exit_code == 0:
        print("status=merged")
        if release_on_merge == 1:
            _trigger_release_if_needed(repo_root, head)
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        exit_code = _run_stream(["gh", "pr", "checks", selector], repo_root)
        if exit_code != 0 and args.checks_strict == 0:
            print("status=checks-nonblocking")
            return 0
        return exit_code

    if args.action == "merge":
        return _merge_pr(
            repo_root,
            selector,
            head,
            args.merge_method,
            args.auto,
            args.delete_branch,
            args.release_on_merge,
        )

    if args.action == "close":
        return _run_stream(["gh", "pr", "close", selector], repo_root)

    msg = f"unknown action: {args.action}"
    raise RuntimeError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
