"""CLI entry point for dependency management services.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from types import MappingProxyType

from flext_infra import output, u

_SUBCOMMAND_MODULES: Mapping[str, str] = MappingProxyType({
    "detect": "flext_infra.deps.detector",
    "extra-paths": "flext_infra.deps.extra_paths",
    "internal-sync": "flext_infra.deps.internal_sync",
    "modernize": "flext_infra.deps.modernizer",
    "path-sync": "flext_infra.deps.path_sync",
})


def _main_impl(_argv: list[str] | None = None) -> int:
    """Dispatch to the appropriate deps subcommand."""
    parser, _ = u.Infra.create_subcommand_parser(
        "flext-infra deps",
        "Dependency management services",
        subcommands={
            "detect": "Detect runtime vs dev dependencies",
            "extra-paths": "Synchronize pyright/mypy extraPaths",
            "internal-sync": "Synchronize internal FLEXT dependencies",
            "modernize": "Modernize workspace pyproject files",
            "path-sync": "Rewrite internal FLEXT dependency paths",
        },
        include_apply=True,
        include_project=True,
    )
    min_deps_argv = 3
    if len(sys.argv) < min_deps_argv or sys.argv[1] in {"-h", "--help"}:
        parser.parse_args()
        return 0
    subcommand = sys.argv[1]
    if subcommand not in _SUBCOMMAND_MODULES:
        output.error(f"flext-infra deps: unknown subcommand '{subcommand}'")
        parser.print_help()
        return 1
    sys.argv = [f"flext-infra deps {subcommand}"] + sys.argv[2:]
    module = importlib.import_module(_SUBCOMMAND_MODULES[subcommand])
    exit_code = module.main()
    return int(exit_code) if exit_code is not None else 0


def main() -> int:
    """Dispatch to the appropriate deps subcommand."""
    return u.Infra.run_cli(_main_impl)


if __name__ == "__main__":
    sys.exit(main())
