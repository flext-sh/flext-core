"""CLI entry point for code generation services.

Usage:
    python -m flext_infra codegen lazy-init [--check] [--root PATH]
    python -m flext_infra codegen census [--workspace PATH] [--format json|text]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

from flext_core import FlextRuntime

from flext_infra.codegen.lazy_init import FlextInfraLazyInitGenerator
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

    args = parser.parse_args(argv)

    if args.command == "lazy-init":
        return _handle_lazy_init(args)
    if args.command == "census":
        return _handle_census(args)

    output.error(f"unknown command: {args.command}")
    return 1


def _handle_lazy_init(args: argparse.Namespace) -> int:
    """Handle the ``lazy-init`` subcommand."""
    root = args.root.resolve()
    generator = FlextInfraLazyInitGenerator(workspace_root=root)
    unmapped = generator.run(check_only=args.check)
    if args.check and unmapped > 0:
        output.warning(f"{unmapped} files have unmapped exports")
    return 0


def _handle_census(args: argparse.Namespace) -> int:
    """Handle the ``census`` subcommand."""
    from flext_infra.codegen.census import FlextInfraCodegenCensus

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


if __name__ == "__main__":
    sys.exit(main())
