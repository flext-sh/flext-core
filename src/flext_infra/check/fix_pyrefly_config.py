"""CLI tool to fix Pyrefly configurations across projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse

from flext_infra.check.services import PyreflyConfigFixer
from flext_infra.output import output


def main(argv: list[str] | None = None) -> int:
    """Run the Pyrefly configuration fixer CLI."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("projects", nargs="*")
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    fixer = PyreflyConfigFixer()
    result = fixer.run(
        projects=args.projects,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    if result.is_failure:
        output.error(result.error or "pyrefly config fix failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
