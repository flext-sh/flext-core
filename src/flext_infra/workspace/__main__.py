"""CLI entry point for workspace utilities.

Usage:
    python -m flext_infra workspace detect [--workspace PATH]
    python -m flext_infra workspace sync [--workspace PATH]
    python -m flext_infra workspace orchestrate --verb <verb> [--fail-fast] [projects...]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

from flext_core import FlextRuntime
from flext_infra import u
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector
from flext_infra.workspace.migrator import FlextInfraProjectMigrator
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService
from flext_infra.workspace.sync import FlextInfraSyncService


def _run_detect(cli: u.Infra.CliArgs) -> int:
    """Execute workspace detection."""
    detector = FlextInfraWorkspaceDetector()
    result = detector.detect(cli.workspace)
    if result.is_success:
        return 0
    u.Infra.error(result.error or "detection failed")
    return 1


def _run_sync(cli: u.Infra.CliArgs, canonical_root) -> int:
    """Execute base.mk sync."""
    service = FlextInfraSyncService(canonical_root=canonical_root)
    result = service.sync(project_root=cli.workspace, canonical_root=canonical_root)
    if result.is_success:
        return 0
    u.Infra.error(result.error or "sync failed")
    return 1


def _run_orchestrate(
    projects: list[str], verb: str, fail_fast: bool, make_args: list[str]
) -> int:
    """Execute multi-project orchestration."""
    projects = [p for p in projects if p]
    if not projects:
        u.Infra.error("no projects specified")
        return 1
    service = FlextInfraOrchestratorService()
    result = service.orchestrate(
        projects=projects,
        verb=verb,
        fail_fast=fail_fast,
        make_args=make_args,
    )
    if result.is_success:
        outputs = result.value
        failures = [o for o in outputs if o.exit_code != 0]
        return max((o.exit_code for o in failures), default=0)
    u.Infra.error(result.error or "orchestration failed")
    return 1


def _run_migrate(cli: u.Infra.CliArgs) -> int:
    service = FlextInfraProjectMigrator()
    result = service.migrate(workspace_root=cli.workspace, dry_run=not cli.apply)
    if result.is_failure:
        u.Infra.error(result.error or "migration failed")
        return 1
    failed_projects = 0
    for migration in result.value:
        u.Infra.info(f"{migration.project}:")
        for change in migration.changes:
            u.Infra.info(f"  + {change}")
        for err in migration.errors:
            u.Infra.warning(f"  ! {err}")
        if migration.errors:
            failed_projects += 1
    total_changes = sum(len(m.changes) for m in result.value)
    total_errors = sum(len(m.errors) for m in result.value)
    u.Infra.info(
        f"Total: {total_changes} change(s), {total_errors} error(s) across {len(result.value)} project(s)",
    )
    if not cli.apply:
        u.Infra.info("(dry-run — no files modified)")
    return 1 if failed_projects else 0


def main(argv: list[str] | None = None) -> int:
    """Run workspace utilities: detect mode, sync base.mk, orchestrate projects."""
    FlextRuntime.ensure_structlog_configured()
    parser = u.Infra.create_parser(
        "flext_infra workspace",
        "Workspace management utilities",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "detect",
        help="Detect workspace or standalone mode",
    )

    sync_parser = subparsers.add_parser("sync", help="Sync base.mk to project root")
    _ = sync_parser.add_argument(
        "--canonical-root",
        type=str,
        default=None,
        help="Canonical workspace root",
    )

    orch_parser = subparsers.add_parser(
        "orchestrate",
        help="Run make verb across projects",
    )
    _ = orch_parser.add_argument("--verb", required=True, help="Make verb to execute")
    _ = orch_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )
    _ = orch_parser.add_argument(
        "--make-arg",
        action="append",
        default=[],
        help="Additional make arguments",
    )
    _ = orch_parser.add_argument(
        "projects",
        nargs="*",
        help="Project directories to orchestrate",
    )

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate workspace projects to flext_infra tooling",
    )
    _ = migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration changes without writing files",
    )

    args = parser.parse_args(argv)
    cli = u.Infra.resolve(args)

    if args.command == "detect":
        return _run_detect(cli)
    if args.command == "sync":
        return _run_sync(cli, getattr(args, "canonical_root", None))
    if args.command == "orchestrate":
        return _run_orchestrate(
            args.projects,
            args.verb,
            args.fail_fast,
            args.make_arg,
        )
    if args.command == "migrate":
        # For migrate, we need to handle dry_run from args since it's not in CliArgs
        dry_run = getattr(args, "dry_run", False)
        # Create a modified cli with apply=False if dry_run is True
        if dry_run:
            cli = u.Infra.CliArgs(
                workspace=cli.workspace,
                apply=False,
                output_format=cli.output_format,
                check=cli.check,
                project=cli.project,
                projects=cli.projects,
            )
        return _run_migrate(cli)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
