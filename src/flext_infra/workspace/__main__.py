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
from pathlib import Path

from flext_core import r
from flext_infra import u
from flext_infra.models import m
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector
from flext_infra.workspace.migrator import FlextInfraProjectMigrator
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService
from flext_infra.workspace.sync import FlextInfraSyncService


def _run_detect(cli: u.Infra.CliArgs) -> int:
    """Execute workspace detection."""
    detector = FlextInfraWorkspaceDetector()
    result = detector.detect(cli.workspace)
    return u.Infra.exit_code(result, failure_msg="detection failed")


def _run_sync(cli: u.Infra.CliArgs, canonical_root: str | None) -> int:
    """Execute base.mk sync."""
    canonical_path = Path(canonical_root) if canonical_root else None
    service = FlextInfraSyncService(canonical_root=canonical_path)
    result = service.sync(workspace_root=cli.workspace, canonical_root=canonical_path)
    return u.Infra.exit_code(result, failure_msg="sync failed")


def _run_orchestrate(
    projects: list[str], verb: str, *, fail_fast: bool, make_args: list[str]
) -> int:
    """Execute multi-project orchestration."""
    projects = [p for p in projects if p]
    if not projects:
        return u.Infra.exit_code(
            r[str].fail("no projects specified"),
            failure_msg="no projects specified",
        )
    service = FlextInfraOrchestratorService()
    result = service.orchestrate(
        projects=projects,
        verb=verb,
        fail_fast=fail_fast,
        make_args=make_args,
    )
    if result.is_success:
        outputs: list[m.Infra.Core.CommandOutput] = result.value
        failures = [o for o in outputs if o.exit_code != 0]
        return max((o.exit_code for o in failures), default=0)
    return u.Infra.exit_code(result, failure_msg="orchestration failed")


def _run_migrate(cli: u.Infra.CliArgs) -> int:
    service = FlextInfraProjectMigrator()
    result = service.migrate(workspace_root=cli.workspace, dry_run=cli.dry_run)
    if result.is_failure:
        return u.Infra.exit_code(result, failure_msg="migration failed")
    failed_projects = 0
    migrations: list[m.Infra.Workspace.MigrationResult] = result.value
    for migration in migrations:
        u.Infra.info(f"{migration.project}:")
        for change in migration.changes:
            u.Infra.info(f"  + {change}")
        for err in migration.errors:
            u.Infra.warning(f"  ! {err}")
        if migration.errors:
            failed_projects += 1
    total_changes = sum(len(mg.changes) for mg in migrations)
    total_errors = sum(len(mg.errors) for mg in migrations)
    u.Infra.info(
        f"Total: {total_changes} change(s), {total_errors} error(s) across {len(migrations)} project(s)",
    )
    if cli.dry_run:
        u.Infra.info("(dry-run — no files modified)")
    return 1 if failed_projects else 0


def main(argv: list[str] | None = None) -> int:
    """Run workspace utilities: detect mode, sync base.mk, orchestrate projects."""
    parser = u.Infra.create_parser(
        "flext_infra workspace",
        "Workspace management utilities",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect workspace or standalone mode",
    )
    _ = detect_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )

    sync_parser = subparsers.add_parser("sync", help="Sync base.mk to project root")
    _ = sync_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
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
    _ = orch_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
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
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
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
            fail_fast=args.fail_fast,
            make_args=args.make_arg,
        )
    if args.command == "migrate":
        return _run_migrate(cli)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(u.Infra.run_cli(main))
