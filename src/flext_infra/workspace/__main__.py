"""CLI entry point for workspace utilities.

Usage:
    python -m flext_infra.workspace detect [--project-root PATH]
    python -m flext_infra.workspace sync [--project-root PATH]
    python -m flext_infra.workspace orchestrate --verb <verb> [--fail-fast] [projects...]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flext_infra.workspace.detector import WorkspaceDetector
from flext_infra.workspace.migrator import ProjectMigrator
from flext_infra.workspace.orchestrator import OrchestratorService
from flext_infra.workspace.sync import SyncService


def _run_detect(args: argparse.Namespace) -> int:
    """Execute workspace detection."""
    detector = WorkspaceDetector()
    result = detector.detect(args.project_root)

    if result.is_success:
        print(result.value.value)
        return 0
    print(f"Error: {result.error}", file=sys.stderr)
    return 1


def _run_sync(args: argparse.Namespace) -> int:
    """Execute base.mk sync."""
    service = SyncService()
    result = service.sync(project_root=args.project_root)

    if result.is_success:
        print(f"files_changed={result.value.files_changed}")
        return 0
    print(f"Error: {result.error}", file=sys.stderr)
    return 1


def _run_orchestrate(args: argparse.Namespace) -> int:
    """Execute multi-project orchestration."""
    projects = [p for p in args.projects if p]
    if not projects:
        print("Error: no projects specified", file=sys.stderr)
        return 1

    service = OrchestratorService()
    result = service.orchestrate(
        projects=projects,
        verb=args.verb,
        fail_fast=args.fail_fast,
        make_args=args.make_arg,
    )

    if result.is_success:
        outputs = result.value
        failures = [o for o in outputs if o.exit_code != 0]
        return max((o.exit_code for o in failures), default=0)
    print(f"Error: {result.error}", file=sys.stderr)
    return 1


def _run_migrate(args: argparse.Namespace) -> int:
    service = ProjectMigrator()
    result = service.migrate(
        workspace_root=args.workspace_root,
        dry_run=args.dry_run,
    )

    if result.is_failure:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1

    failed_projects = 0
    for migration in result.value:
        print(f"project={migration.project}")
        for change in migration.changes:
            print(f"  - {change}")
        for error in migration.errors:
            print(f"  ! {error}")
        if migration.errors:
            failed_projects += 1

    print(
        f"summary total={len(result.value)} failed={failed_projects} dry_run={str(args.dry_run).lower()}"
    )
    return 1 if failed_projects else 0


def main() -> int:
    """Run workspace utilities: detect mode, sync base.mk, orchestrate projects."""
    parser = argparse.ArgumentParser(description="Workspace management utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # detect subcommand
    detect_parser = subparsers.add_parser(
        "detect", help="Detect workspace or standalone mode"
    )
    _ = detect_parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Path to the project directory",
    )

    # sync subcommand
    sync_parser = subparsers.add_parser(
        "sync", help="Sync base.mk to project root (no scripts/ sync)"
    )
    _ = sync_parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Path to the project directory",
    )
    _ = sync_parser.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help="Canonical workspace root (accepted for compatibility)",
    )

    # orchestrate subcommand
    orch_parser = subparsers.add_parser(
        "orchestrate", help="Run make verb across projects"
    )
    _ = orch_parser.add_argument("--verb", required=True, help="Make verb to execute")
    _ = orch_parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )
    _ = orch_parser.add_argument(
        "--make-arg", action="append", default=[], help="Additional make arguments"
    )
    _ = orch_parser.add_argument(
        "projects", nargs="*", help="Project directories to orchestrate"
    )

    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate workspace projects to flext_infra tooling"
    )
    _ = migrate_parser.add_argument(
        "--workspace-root",
        required=True,
        type=Path,
        help="Workspace root directory containing subprojects",
    )
    _ = migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration changes without writing files",
    )

    args = parser.parse_args()

    if args.command == "detect":
        return _run_detect(args)
    if args.command == "sync":
        return _run_sync(args)
    if args.command == "orchestrate":
        return _run_orchestrate(args)
    if args.command == "migrate":
        return _run_migrate(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
