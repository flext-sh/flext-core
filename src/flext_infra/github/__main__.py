"""CLI entry point for GitHub integration services.

Usage:
    python -m flext_infra github workflows --workspace-root PATH [--apply] [--prune] [--report PATH]
    python -m flext_infra github lint --root PATH [--report PATH] [--strict]
    python -m flext_infra github pr --repo-root PATH --action ACTION [--base BRANCH] ...
    python -m flext_infra github pr-workspace --workspace-root PATH [--pr-action ACTION] ...

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from flext_core import FlextRuntime

from flext_infra.github.linter import WorkflowLinter
from flext_infra.github.pr import main as pr_main
from flext_infra.github.pr_workspace import PrWorkspaceManager
from flext_infra.github.workflows import WorkflowSyncer
from flext_infra.output import output

_MIN_ARGV = 2

_Handler = Callable[[list[str]], int]


def _run_workflows(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="flext-infra github workflows")
    _ = parser.add_argument("--workspace-root", type=Path, required=True)
    _ = parser.add_argument("--apply", action="store_true", default=False)
    _ = parser.add_argument("--prune", action="store_true", default=False)
    _ = parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    syncer = WorkflowSyncer()
    result = syncer.sync_workspace(
        workspace_root=args.workspace_root.resolve(),
        apply=args.apply,
        prune=args.prune,
        report_path=args.report,
    )
    if result.is_failure:
        output.error(result.error or "workflow sync failed")
        return 1

    for _op in result.value:
        pass
    return 0


def _run_lint(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="flext-infra github lint")
    _ = parser.add_argument("--root", type=Path, required=True)
    _ = parser.add_argument("--report", type=Path, default=None)
    _ = parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    linter = WorkflowLinter()
    result = linter.lint(
        root=args.root.resolve(),
        report_path=args.report,
        strict=args.strict,
    )
    if result.is_failure:
        output.error(result.error or "lint failed")
        return 1

    result.value.get("status", "unknown")
    return 0


def _run_pr(argv: list[str]) -> int:
    sys.argv = ["flext-infra github pr"] + argv
    return pr_main()


def _run_pr_workspace(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="flext-infra github pr-workspace")
    _ = parser.add_argument("--workspace-root", type=Path, required=True)
    _ = parser.add_argument("--project", action="append", default=[])
    _ = parser.add_argument("--include-root", type=int, default=1)
    _ = parser.add_argument("--branch", default="")
    _ = parser.add_argument("--checkpoint", type=int, default=1)
    _ = parser.add_argument("--fail-fast", type=int, default=0)
    _ = parser.add_argument("--pr-action", default="status")
    _ = parser.add_argument("--pr-base", default="main")
    _ = parser.add_argument("--pr-head", default="")
    _ = parser.add_argument("--pr-number", default="")
    _ = parser.add_argument("--pr-title", default="")
    _ = parser.add_argument("--pr-body", default="")
    _ = parser.add_argument("--pr-draft", type=int, default=0)
    _ = parser.add_argument("--pr-merge-method", default="squash")
    _ = parser.add_argument("--pr-auto", type=int, default=0)
    _ = parser.add_argument("--pr-delete-branch", type=int, default=0)
    _ = parser.add_argument("--pr-checks-strict", type=int, default=0)
    _ = parser.add_argument("--pr-release-on-merge", type=int, default=1)
    args = parser.parse_args(argv)

    pr_args: Mapping[str, str] = {
        "action": args.pr_action,
        "base": args.pr_base,
        "head": args.pr_head,
        "number": args.pr_number,
        "title": args.pr_title,
        "body": args.pr_body,
        "draft": str(args.pr_draft),
        "merge_method": args.pr_merge_method,
        "auto": str(args.pr_auto),
        "delete_branch": str(args.pr_delete_branch),
        "checks_strict": str(args.pr_checks_strict),
        "release_on_merge": str(args.pr_release_on_merge),
    }

    manager = PrWorkspaceManager()
    result = manager.orchestrate(
        workspace_root=args.workspace_root.resolve(),
        projects=args.project or None,
        include_root=args.include_root == 1,
        branch=args.branch,
        checkpoint=args.checkpoint == 1,
        fail_fast=args.fail_fast == 1,
        pr_args=pr_args,
    )
    if result.is_failure:
        output.error(result.error or "pr-workspace failed")
        return 1

    data = result.value
    return 1 if data.get("fail", 0) else 0


_SUBCOMMANDS: Mapping[str, _Handler] = {
    "lint": _run_lint,
    "pr": _run_pr,
    "pr-workspace": _run_pr_workspace,
    "workflows": _run_workflows,
}


def main() -> int:
    """Dispatch to the appropriate github subcommand."""
    FlextRuntime.ensure_structlog_configured()
    if len(sys.argv) < _MIN_ARGV or sys.argv[1] in {"-h", "--help"}:
        for _name in sorted(_SUBCOMMANDS):
            pass
        return (
            0 if len(sys.argv) >= _MIN_ARGV and sys.argv[1] in {"-h", "--help"} else 1
        )

    subcommand = sys.argv[1]
    handler = _SUBCOMMANDS.get(subcommand)
    if handler is None:
        output.error(f"unknown subcommand '{subcommand}'")
        return 1

    remaining = sys.argv[2:]
    return handler(remaining)


if __name__ == "__main__":
    sys.exit(main())
