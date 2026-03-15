"""CLI entry point for maintenance services.

Usage:
    python -m flext_infra maintenance [--workspace PATH] [--verbose]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

from flext_core import r
from flext_infra import u
from flext_infra._utilities.output import output
from flext_infra.maintenance.python_version import FlextInfraPythonVersionEnforcer


def main(argv: list[str] | None = None) -> int:
    """Run maintenance service CLI."""
    parser = u.Infra.create_parser(
        prog="maintenance",
        description="Enforce Python version constraints via pyproject.toml",
        include_check=True,
    )
    _ = parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args(argv)
    cli = u.Infra.resolve(args)
    service = FlextInfraPythonVersionEnforcer(
        config_type=None,
        config_overrides=None,
        initial_context=None,
        subproject=None,
        services=None,
        factories=None,
        resources=None,
        container_overrides=None,
        wire_modules=None,
        wire_packages=None,
        wire_classes=None,
    )
    result: r[int] = service.execute(check_only=cli.check, verbose=args.verbose)
    if result.is_success:
        return result.unwrap()
    output.error(result.error or "maintenance failed")
    return 1


if __name__ == "__main__":
    sys.exit(u.Infra.run_cli(main))
