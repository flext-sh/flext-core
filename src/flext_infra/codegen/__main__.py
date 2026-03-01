"""CLI entry point for code generation services.

Usage:
    python -m flext_infra codegen lazy-init [--check] [--root PATH]
    python -m flext_infra codegen census [--workspace PATH] [--format json|text]
    python -m flext_infra codegen scaffold [--workspace PATH] [--dry-run]
    python -m flext_infra codegen auto-fix [--workspace PATH] [--dry-run]
    python -m flext_infra codegen pipeline [--workspace PATH] [--dry-run] [--format json|text]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flext_core import FlextRuntime

from flext_infra.codegen.census import FlextInfraCodegenCensus
from flext_infra.codegen.fixer import FlextInfraCodegenFixer
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_infra.codegen.scaffolder import FlextInfraCodegenScaffolder
from flext_infra.output import output


def main(argv: list[str] | None = None) -> int:
    """Run codegen service CLI."""
    FlextRuntime.ensure_structlog_configured()

    parser = argparse.ArgumentParser(
        description="Code generation tools for workspace standardization",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- lazy-init subcommand ------------------------------------------------
    lazy_parser = subparsers.add_parser(
        "lazy-init",
        help="Generate/refresh PEP 562 lazy-import __init__.py files",
    )
    _ = lazy_parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode — report unmapped exports without writing files",
    )
    _ = lazy_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )

    # -- census subcommand --------------------------------------------------
    census_parser = subparsers.add_parser(
        "census",
        help="Count namespace violations across workspace projects",
    )
    _ = census_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    _ = census_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )

    # -- scaffold subcommand ------------------------------------------------
    scaffold_parser = subparsers.add_parser(
        "scaffold",
        help="Generate missing base modules in src/ and tests/",
    )
    _ = scaffold_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    _ = scaffold_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be created without writing files",
    )

    # -- auto-fix subcommand ------------------------------------------------
    fix_parser = subparsers.add_parser(
        "auto-fix",
        help="Auto-fix namespace violations (move Finals/TypeVars)",
    )
    _ = fix_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    _ = fix_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be fixed without modifying files",
    )

    # -- pipeline subcommand ------------------------------------------------
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full codegen pipeline: census → scaffold → auto-fix → lazy-init → census",
    )
    _ = pipeline_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    _ = pipeline_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying files",
    )
    _ = pipeline_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )

    args = parser.parse_args(argv)

    if args.command == "lazy-init":
        return _handle_lazy_init(args)
    if args.command == "census":
        return _handle_census(args)
    if args.command == "scaffold":
        return _handle_scaffold(args)
    if args.command == "auto-fix":
        return _handle_auto_fix(args)
    if args.command == "pipeline":
        return _handle_pipeline(args)

    output.error(f"unknown command: {args.command}")
    return 1


def _handle_lazy_init(args: argparse.Namespace) -> int:
    """Handle the ``lazy-init`` subcommand."""
    root = args.root.resolve()
    generator = FlextInfraCodegenLazyInit(workspace_root=root)
    unmapped = generator.run(check_only=args.check, scan_tests=False)
    if args.check and unmapped > 0:
        output.warning(f"{unmapped} files have unmapped exports")
    return 0


def _handle_census(args: argparse.Namespace) -> int:
    """Handle the ``census`` subcommand."""
    census = FlextInfraCodegenCensus(workspace_root=args.workspace.resolve())
    reports = census.run()

    if args.output_format == "json":
        data = {
            "projects": [rpt.model_dump() for rpt in reports],
            "total_violations": sum(rpt.total for rpt in reports),
            "total_fixable": sum(rpt.fixable for rpt in reports),
        }
        print(json.dumps(data, indent=2))  # noqa: T201
    else:
        total_v = sum(rpt.total for rpt in reports)
        total_f = sum(rpt.fixable for rpt in reports)
        for rpt in reports:
            if rpt.total > 0:
                output.info(
                    f"  {rpt.project}: {rpt.total} violations ({rpt.fixable} fixable)",
                )
        output.info(
            f"Total: {total_v} violations ({total_f} fixable) across {len(reports)} projects",
        )
    return 0


def _handle_scaffold(args: argparse.Namespace) -> int:
    """Handle the ``scaffold`` subcommand."""
    scaffolder = FlextInfraCodegenScaffolder(workspace_root=args.workspace.resolve())
    if args.dry_run:
        output.info("Dry-run mode: no files will be created")
    results = scaffolder.run()
    total_created = sum(len(res.files_created) for res in results)
    total_skipped = sum(len(res.files_skipped) for res in results)
    for res in results:
        if res.files_created:
            output.info(f"  {res.project}: created {len(res.files_created)} files")
    output.info(
        f"Scaffold: {total_created} created, {total_skipped} skipped across {len(results)} projects"
    )
    return 0


def _handle_auto_fix(args: argparse.Namespace) -> int:
    """Handle the ``auto-fix`` subcommand."""
    fixer = FlextInfraCodegenFixer(workspace_root=args.workspace.resolve())
    if args.dry_run:
        output.info("Dry-run mode: no files will be modified")
    results = fixer.run()
    total_fixed = sum(len(res.violations_fixed) for res in results)
    total_skipped = sum(len(res.violations_skipped) for res in results)
    for res in results:
        if res.violations_fixed:
            output.info(
                f"  {res.project}: fixed {len(res.violations_fixed)} violations"
            )
    output.info(
        f"Auto-fix: {total_fixed} fixed, {total_skipped} skipped across {len(results)} projects"
    )
    return 0


def _handle_pipeline(args: argparse.Namespace) -> int:
    """Handle the ``pipeline`` subcommand (full codegen cycle)."""
    workspace = args.workspace.resolve()

    # Phase 1: Census before
    census = FlextInfraCodegenCensus(workspace_root=workspace)
    reports_before = census.run()

    # Phase 2: Scaffold
    scaffolder = FlextInfraCodegenScaffolder(workspace_root=workspace)
    scaffold_results = scaffolder.run()

    # Phase 3: Auto-fix
    fixer = FlextInfraCodegenFixer(workspace_root=workspace)
    fix_results = fixer.run()

    # Phase 4: Lazy-init (with tests)
    generator = FlextInfraCodegenLazyInit(workspace_root=workspace)
    generator.run(check_only=args.dry_run, scan_tests=True)

    # Phase 5: Census after
    reports_after = census.run()

    if args.output_format == "json":
        data = {
            "census_before": {
                "total_violations": sum(r.total for r in reports_before),
                "total_fixable": sum(r.fixable for r in reports_before),
            },
            "scaffold": {
                "total_created": sum(len(r.files_created) for r in scaffold_results),
                "total_skipped": sum(len(r.files_skipped) for r in scaffold_results),
            },
            "auto_fix": {
                "total_fixed": sum(len(r.violations_fixed) for r in fix_results),
                "total_skipped": sum(len(r.violations_skipped) for r in fix_results),
            },
            "census_after": {
                "total_violations": sum(r.total for r in reports_after),
                "total_fixable": sum(r.fixable for r in reports_after),
            },
        }
        print(json.dumps(data, indent=2))  # noqa: T201
    else:
        before_v = sum(r.total for r in reports_before)
        after_v = sum(r.total for r in reports_after)
        output.info(f"Census before: {before_v} violations")
        output.info(
            f"Scaffold: {sum(len(r.files_created) for r in scaffold_results)} files created",
        )
        output.info(
            f"Auto-fix: {sum(len(r.violations_fixed) for r in fix_results)} violations fixed",
        )
        output.info(f"Census after: {after_v} violations")
        output.info(f"Improvement: {before_v - after_v} violations resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
