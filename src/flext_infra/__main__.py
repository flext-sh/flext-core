"""Unified CLI entry point for flext-infra.

Usage:
    python -m flext_infra <group> [subcommand] [args...]

Groups:
    basemk        Base.mk template generation
    check         Lint gates and pyrefly config management
    codegen       Code generation (lazy-init, standardization)
    core          Infrastructure validators and diagnostics
    deps          Dependency detection, sync, and modernization
    docs          Documentation audit, fix, build, generate, validate
    github        GitHub workflows, linting, and PR automation
    maintenance   Python version enforcement
    refactor      Declarative code refactoring (libcst + YAML rules)
    release       Release orchestration
    workspace     Workspace detection, sync, orchestration, migration

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, override

from flext_core import r, s
from flext_infra import c, output, u


class FlextInfraMainCLI(s[int]):
    """Unified CLI dispatcher for flext-infra groups.

    Encapsulates group routing, help display, and argv rewriting
    for the top-level ``python -m flext_infra`` entry point.
    """

    GROUPS: ClassVar[Mapping[str, str]] = MappingProxyType({
        "basemk": "flext_infra.basemk.__main__",
        c.Infra.Verbs.CHECK: "flext_infra.check.__main__",
        "codegen": "flext_infra.codegen.__main__",
        "core": "flext_infra.core.__main__",
        "deps": "flext_infra.deps.__main__",
        c.Infra.Directories.DOCS: "flext_infra.docs.__main__",
        "github": "flext_infra.github.__main__",
        "maintenance": "flext_infra.maintenance.__main__",
        "refactor": "flext_infra.refactor.__main__",
        c.Infra.ReportKeys.RELEASE: "flext_infra.release.__main__",
        c.Infra.ReportKeys.WORKSPACE: "flext_infra.workspace.__main__",
    })
    DESCRIPTIONS: ClassVar[Mapping[str, str]] = MappingProxyType({
        "basemk": "Base.mk template generation",
        c.Infra.Verbs.CHECK: "Lint gates and pyrefly config management",
        "codegen": "Code generation (lazy-init, standardization)",
        "core": "Infrastructure validators and diagnostics",
        "deps": "Dependency detection, sync, and modernization",
        c.Infra.Directories.DOCS: "Documentation audit, fix, build, generate, validate",
        "github": "GitHub workflows, linting, and PR automation",
        "maintenance": "Python version enforcement",
        "refactor": "Declarative code refactoring (libcst + YAML rules)",
        c.Infra.ReportKeys.RELEASE: "Release orchestration",
        c.Infra.ReportKeys.WORKSPACE: "Workspace detection, sync, orchestration, migration",
    })

    @override
    def execute(self) -> r[int]:
        """Execute the CLI dispatcher and return exit code."""
        return r[int].ok(main_inner())

    @staticmethod
    def print_help() -> None:
        """Display available groups and their descriptions."""
        output.info("Usage: python -m flext_infra <group> [subcommand] [args...]")
        output.header("Groups")
        for group in sorted(FlextInfraMainCLI.GROUPS):
            output.info(f"  {group:<16}{FlextInfraMainCLI.DESCRIPTIONS.get(group, '')}")


def main_inner(argv: list[str] | None = None) -> int:
    """Dispatch to the appropriate group CLI."""
    args = argv if argv is not None else sys.argv
    if len(args) < c.Infra.MIN_ARGV or args[1] in {"-h", "--help"}:
        FlextInfraMainCLI.print_help()
        return 0 if len(args) >= c.Infra.MIN_ARGV and args[1] in {"-h", "--help"} else 1
    group = args[1]
    if group not in FlextInfraMainCLI.GROUPS:
        output.error(f"unknown group '{group}'")
        FlextInfraMainCLI.print_help()
        return 1
    sys.argv = [f"flext-infra {group}"] + args[2:]
    module = importlib.import_module(FlextInfraMainCLI.GROUPS[group])
    exit_code = module.main()
    return int(exit_code) if exit_code is not None else 0


def main(argv: list[str] | None = None) -> int:
    """Run the top-level flext-infra CLI dispatcher."""
    return u.Infra.run_cli(main_inner, argv)


if __name__ == "__main__":
    sys.exit(main())
