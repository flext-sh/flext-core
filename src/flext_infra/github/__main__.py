"""CLI entry point for GitHub integration services."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Mapping
from pathlib import Path

from flext_core import r
from flext_infra import c, m, u
from flext_infra.github.linter import FlextInfraWorkflowLinter
from flext_infra.github.pr import main as pr_main
from flext_infra.github.pr_workspace import FlextInfraPrWorkspaceManager
from flext_infra.github.workflows import FlextInfraWorkflowSyncer, SyncOperation


def _run_workflows(
    cli: u.Infra.CliArgs,
    *,
    prune: bool,
    report: Path | None,
) -> int:
    syncer = FlextInfraWorkflowSyncer()
    result: r[list[SyncOperation]] = syncer.sync_workspace(
        workspace_root=cli.workspace,
        apply=cli.apply,
        prune=prune,
        report_path=report,
    )
    if result.is_failure:
        return u.Infra.exit_code(result, failure_msg="workflow sync failed")
    for _op in result.value:
        pass
    return 0


def _run_lint(
    cli: u.Infra.CliArgs,
    *,
    report: Path | None,
    strict: bool,
) -> int:
    linter = FlextInfraWorkflowLinter()
    lint_result: r[m.Infra.Github.WorkflowLintResult] = linter.lint(
        workspace_root=cli.workspace,
        report_path=report,
        strict=strict,
    )
    if lint_result.is_failure:
        return u.Infra.exit_code(lint_result, failure_msg="lint failed")
    _ = lint_result.value.status
    return 0


def _run_pr(argv: list[str]) -> int:
    sys.argv = ["flext-infra github pr"] + argv
    return pr_main()


def _run_pr_workspace(
    cli: u.Infra.CliArgs,
    *,
    include_root: int,
    branch: str,
    checkpoint: int,
    fail_fast: int,
    pr_action: str,
    pr_base: str,
    pr_head: str,
    pr_number: str,
    pr_title: str,
    pr_body: str,
    pr_draft: int,
    pr_merge_method: str,
    pr_auto: int,
    pr_delete_branch: int,
    pr_checks_strict: int,
    pr_release_on_merge: int,
) -> int:
    pr_args: Mapping[str, str] = {
        c.Infra.ReportKeys.ACTION: pr_action,
        "base": pr_base,
        "head": pr_head,
        "number": pr_number,
        "title": pr_title,
        "body": pr_body,
        "draft": str(pr_draft),
        "merge_method": pr_merge_method,
        "auto": str(pr_auto),
        "delete_branch": str(pr_delete_branch),
        "checks_strict": str(pr_checks_strict),
        "release_on_merge": str(pr_release_on_merge),
    }
    manager = FlextInfraPrWorkspaceManager()
    orch_result = manager.orchestrate(
        workspace_root=cli.workspace,
        projects=cli.project_names(),
        include_root=include_root == 1,
        branch=branch,
        checkpoint=checkpoint == 1,
        fail_fast=fail_fast == 1,
        pr_args=pr_args,
    )
    if orch_result.is_failure:
        return u.Infra.exit_code(orch_result, failure_msg="pr-workspace failed")
    data = orch_result.value
    return 1 if data.fail else 0


def _main_impl(argv: list[str] | None = None) -> int:
    """Dispatch to the appropriate github subcommand."""
    if argv is not None:
        sys.argv = ["flext-infra github"] + argv
    parser, subs = u.Infra.create_subcommand_parser(
        "flext-infra github",
        "GitHub integration services",
        subcommands={
            "workflows": "Sync GitHub workflow files across workspace",
            c.Infra.Toml.LINT_SECTION: "Lint GitHub workflow files",
            c.Infra.Cli.GhCmd.PR: "Manage pull requests for a single project",
            "pr-workspace": "Manage pull requests across workspace projects",
        },
        include_apply=True,
        include_project=True,
    )
    _configure_workflows_parser(subs["workflows"])
    _configure_lint_parser(subs[c.Infra.Toml.LINT_SECTION])
    _configure_pr_workspace_parser(subs["pr-workspace"])

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    cli = u.Infra.resolve(args)
    if args.command == "workflows":
        return _run_workflows(cli, prune=args.prune, report=args.report)
    if args.command == c.Infra.Toml.LINT_SECTION:
        return _run_lint(cli, report=args.report, strict=args.strict)
    if args.command == c.Infra.Cli.GhCmd.PR:
        return _run_pr(sys.argv[2:])
    if args.command == "pr-workspace":
        return _run_pr_workspace(
            cli,
            include_root=args.include_root,
            branch=args.branch,
            checkpoint=args.checkpoint,
            fail_fast=args.fail_fast,
            pr_action=args.pr_action,
            pr_base=args.pr_base,
            pr_head=args.pr_head,
            pr_number=args.pr_number,
            pr_title=args.pr_title,
            pr_body=args.pr_body,
            pr_draft=args.pr_draft,
            pr_merge_method=args.pr_merge_method,
            pr_auto=args.pr_auto,
            pr_delete_branch=args.pr_delete_branch,
            pr_checks_strict=args.pr_checks_strict,
            pr_release_on_merge=args.pr_release_on_merge,
        )
    parser.print_help()
    return 1


def _configure_workflows_parser(parser: ArgumentParser) -> None:
    _ = parser.add_argument("--prune", action="store_true", default=False)
    _ = parser.add_argument("--report", type=Path, default=None)


def _configure_lint_parser(parser: ArgumentParser) -> None:
    _ = parser.add_argument("--report", type=Path, default=None)
    _ = parser.add_argument("--strict", action="store_true", default=False)


def _configure_pr_workspace_parser(parser: ArgumentParser) -> None:
    _ = parser.add_argument("--include-root", type=int, default=1)
    _ = parser.add_argument("--branch", default="")
    _ = parser.add_argument("--checkpoint", type=int, default=1)
    _ = parser.add_argument("--fail-fast", type=int, default=0)
    _ = parser.add_argument("--pr-action", default=c.Infra.ReportKeys.STATUS)
    _ = parser.add_argument("--pr-base", default=c.Infra.Git.MAIN)
    _ = parser.add_argument("--pr-head", default="")
    _ = parser.add_argument("--pr-number", default="")
    _ = parser.add_argument("--pr-title", default="")
    _ = parser.add_argument("--pr-body", default="")
    _ = parser.add_argument("--pr-draft", type=int, default=0)
    _ = parser.add_argument("--pr-merge-method", default=c.Infra.Cli.GhCmd.SQUASH)
    _ = parser.add_argument("--pr-auto", type=int, default=0)
    _ = parser.add_argument("--pr-delete-branch", type=int, default=0)
    _ = parser.add_argument("--pr-checks-strict", type=int, default=0)
    _ = parser.add_argument("--pr-release-on-merge", type=int, default=1)


def main() -> int:
    """CLI entry point wrapped with centralized helpers."""
    return u.Infra.run_cli(_main_impl)


if __name__ == "__main__":
    sys.exit(main())
