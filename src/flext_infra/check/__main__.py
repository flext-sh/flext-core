"""CLI entry point for check module."""

from __future__ import annotations

import sys

from flext_infra import u
from flext_infra.check.workspace_check import run_cli


def main() -> int:
    """Execute the check CLI and return exit code."""
    return u.Infra.run_cli(run_cli)


if __name__ == "__main__":
    sys.exit(main())
