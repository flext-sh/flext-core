"""CLI entry point for check module."""

from __future__ import annotations

import sys

from flext_infra.check.services import run_cli


def main() -> int:
    """Execute the check CLI and return exit code."""
    return run_cli()


if __name__ == "__main__":
    sys.exit(main())
