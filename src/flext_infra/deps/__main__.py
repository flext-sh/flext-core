"""CLI entry point for dependency management services.

Usage:
    python -m flext_infra deps detect [-q] [--no-fail] [--typings]
    python -m flext_infra deps internal-sync --project-root PATH
    python -m flext_infra deps modernize [--skip-check] [--audit]
    python -m flext_infra deps path-sync --mode MODE
    python -m flext_infra deps extra-paths [--project ROOT]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys

_MIN_ARGV = 2

_SUBCOMMANDS: dict[str, str] = {
    "detect": "flext_infra.deps.detector",
    "extra-paths": "flext_infra.deps.extra_paths",
    "internal-sync": "flext_infra.deps.internal_sync",
    "modernize": "flext_infra.deps.modernizer",
    "path-sync": "flext_infra.deps.path_sync",
}


def main() -> int:
    """Dispatch to the appropriate deps subcommand."""
    if len(sys.argv) < _MIN_ARGV or sys.argv[1] in {"-h", "--help"}:
        _ = sys.stdout.write("Usage: flext-infra deps <subcommand> [args...]\n\n")
        _ = sys.stdout.write("Subcommands:\n")
        for name in sorted(_SUBCOMMANDS):
            _ = sys.stdout.write(f"  {name}\n")
        return 0 if len(sys.argv) >= _MIN_ARGV and sys.argv[1] in {"-h", "--help"} else 1

    subcommand = sys.argv[1]
    if subcommand not in _SUBCOMMANDS:
        _ = sys.stderr.write(f"flext-infra deps: unknown subcommand '{subcommand}'\n")
        return 1

    sys.argv = [f"flext-infra deps {subcommand}"] + sys.argv[2:]
    module = importlib.import_module(_SUBCOMMANDS[subcommand])
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
