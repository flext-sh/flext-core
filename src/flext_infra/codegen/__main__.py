"""CLI entry point for code generation services.

Usage:
    python -m flext_infra codegen lazy-init [--check] [--root PATH]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys
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

    args = parser.parse_args(argv)

    if args.command == "lazy-init":
        return _handle_lazy_init(args)

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


if __name__ == "__main__":
    sys.exit(main())
