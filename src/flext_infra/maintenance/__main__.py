"""CLI entry point for maintenance services.

Usage:
    python -m flext_infra maintenance [--check] [--verbose]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys

from flext_core.runtime import FlextRuntime

from flext_infra.maintenance.python_version import PythonVersionEnforcer
from flext_infra.output import output


def main(argv: list[str] | None = None) -> int:
    """Run maintenance service CLI."""
    FlextRuntime.ensure_structlog_configured()

    parser = argparse.ArgumentParser(
        description="Enforce Python version constraints via pyproject.toml",
    )
    _ = parser.add_argument(
        "--check", action="store_true", help="Check mode (no writes)",
    )
    _ = parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output",
    )
    args = parser.parse_args(argv)

    service = PythonVersionEnforcer()
    result = service.execute(check_only=args.check, verbose=args.verbose)

    if result.is_success:
        return int(result.value) if result.value is not None else 0
    output.error(result.error or "maintenance failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
