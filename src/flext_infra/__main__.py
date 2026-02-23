"""Unified CLI entry point for flext-infra.

Usage:
    python -m flext_infra <group> [subcommand] [args...]

Groups:
    basemk        Base.mk template generation
    check         Lint gates and pyrefly config management
    core          Infrastructure validators and diagnostics
    deps          Dependency detection, sync, and modernization
    docs          Documentation audit, fix, build, generate, validate
    github        GitHub workflows, linting, and PR automation
    maintenance   Python version enforcement
    release       Release orchestration
    workspace     Workspace detection, sync, orchestration, migration

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys

_GROUPS: dict[str, str] = {
    "basemk": "flext_infra.basemk.__main__",
    "check": "flext_infra.check.__main__",
    "core": "flext_infra.core.__main__",
    "deps": "flext_infra.deps.__main__",
    "docs": "flext_infra.docs.__main__",
    "github": "flext_infra.github.__main__",
    "maintenance": "flext_infra.maintenance.__main__",
    "release": "flext_infra.release.__main__",
    "workspace": "flext_infra.workspace.__main__",
}


def _print_help() -> None:
    _ = sys.stdout.write(
        "Usage: python -m flext_infra <group> [subcommand] [args...]\n\n"
    )
    _ = sys.stdout.write("Groups:\n")
    descriptions: dict[str, str] = {
        "basemk": "Base.mk template generation",
        "check": "Lint gates and pyrefly config management",
        "core": "Infrastructure validators and diagnostics",
        "deps": "Dependency detection, sync, and modernization",
        "docs": "Documentation audit, fix, build, generate, validate",
        "github": "GitHub workflows, linting, and PR automation",
        "maintenance": "Python version enforcement",
        "release": "Release orchestration",
        "workspace": "Workspace detection, sync, orchestration, migration",
    }
    for group in sorted(_GROUPS):
        _ = sys.stdout.write(f"  {group:<16}{descriptions.get(group, '')}\n")


def main() -> int:
    """Dispatch to the appropriate group CLI."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        return 0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1

    group = sys.argv[1]
    if group not in _GROUPS:
        _ = sys.stderr.write(f"flext-infra: unknown group '{group}'\n")
        _print_help()
        return 1

    # Rewrite argv so each group's argparse sees the correct prog name
    sys.argv = [f"flext-infra {group}"] + sys.argv[2:]

    module = importlib.import_module(_GROUPS[group])
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
