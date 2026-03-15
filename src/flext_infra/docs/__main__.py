"""CLI entry point for documentation services.

Usage:
    python -m flext_infra docs audit --workspace flext-core
    python -m flext_infra docs fix --workspace flext-core --apply
    python -m flext_infra docs build --workspace flext-core
    python -m flext_infra docs generate --workspace flext-core --apply
    python -m flext_infra docs validate --workspace flext-core

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser

from flext_infra import (
    FlextInfraDocAuditor,
    FlextInfraDocBuilder,
    FlextInfraDocFixer,
    FlextInfraDocGenerator,
    FlextInfraDocValidator,
    c,
    u,
)


def _run_audit(
    cli: u.Infra.CliArgs,
    *,
    check: bool = False,
    strict: bool = False,
    output_dir: str = "",
) -> int:
    """Execute documentation audit."""
    auditor = FlextInfraDocAuditor()
    result = auditor.audit(
        workspace_root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
        check="all" if check else "",
        strict=strict,
    )
    if result.is_failure:
        return u.Infra.exit_code(result, failure_msg="audit failed")
    failures = sum(1 for report in result.value if not report.passed)
    return 1 if failures else 0


def _run_fix(cli: u.Infra.CliArgs, *, output_dir: str = "") -> int:
    """Execute documentation fix."""
    fixer = FlextInfraDocFixer()
    result = fixer.fix(
        workspace_root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
        apply=cli.apply,
    )
    if result.is_failure:
        return u.Infra.exit_code(result, failure_msg="fix failed")
    return 0


def _run_build(cli: u.Infra.CliArgs, *, output_dir: str = "") -> int:
    """Execute documentation build."""
    builder = FlextInfraDocBuilder()
    result = builder.build(
        workspace_root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
    )
    if result.is_failure:
        return u.Infra.exit_code(result, failure_msg="build failed")
    failures = sum(1 for report in result.value if report.result == c.Infra.Status.FAIL)
    return 1 if failures else 0


def _run_generate(cli: u.Infra.CliArgs, *, output_dir: str = "") -> int:
    """Execute documentation generation."""
    generator = FlextInfraDocGenerator()
    result = generator.generate(
        workspace_root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
        apply=cli.apply,
    )
    if result.is_failure:
        return u.Infra.exit_code(result, failure_msg="generate failed")
    return 0


def _run_validate(
    cli: u.Infra.CliArgs,
    *,
    check: bool = False,
    output_dir: str = "",
) -> int:
    """Execute documentation validation."""
    validator = FlextInfraDocValidator()
    result = validator.validate(
        workspace_root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
        check="all" if check else "",
        apply=cli.apply,
    )
    if result.is_failure:
        return u.Infra.exit_code(result, failure_msg="validate failed")
    failures = sum(1 for report in result.value if report.result == c.Infra.Status.FAIL)
    return 1 if failures else 0


def _main_inner(argv: list[str] | None = None) -> int:
    """Run documentation services: audit, fix, build, generate, validate."""
    parser_subs: tuple[ArgumentParser, dict[str, ArgumentParser]] = (
        u.Infra.create_subcommand_parser(
            "flext-infra docs",
            "Documentation management services",
            subcommands={
                "audit": "Audit documentation for broken links and forbidden terms",
                "fix": "Fix documentation issues",
                c.Infra.Directories.BUILD: "Build MkDocs sites",
                "generate": "Generate project docs",
                c.Infra.Verbs.VALIDATE: "Validate documentation",
            },
            include_apply=True,
            include_project=True,
            include_check=True,
        )
    )
    parser, subs = parser_subs

    _ = subs["audit"].add_argument("--strict", action="store_true", help="Strict mode")
    _ = subs["audit"].add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    _ = subs["fix"].add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    _ = subs[c.Infra.Directories.BUILD].add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    _ = subs["generate"].add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    _ = subs[c.Infra.Verbs.VALIDATE].add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    args = parser.parse_args(argv)
    cli = u.Infra.resolve(args)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "audit":
        return _run_audit(
            cli,
            check=cli.check,
            strict=bool(getattr(args, "strict", False)),
            output_dir=str(getattr(args, "output_dir", "")),
        )
    if args.command == "fix":
        return _run_fix(
            cli,
            output_dir=str(getattr(args, "output_dir", "")),
        )
    if args.command == c.Infra.Directories.BUILD:
        return _run_build(
            cli,
            output_dir=str(getattr(args, "output_dir", "")),
        )
    if args.command == "generate":
        return _run_generate(
            cli,
            output_dir=str(getattr(args, "output_dir", "")),
        )
    if args.command == c.Infra.Verbs.VALIDATE:
        return _run_validate(
            cli,
            check=cli.check,
            output_dir=str(getattr(args, "output_dir", "")),
        )
    parser.print_help()
    return 1


def main() -> int:
    """Entry point for documentation CLI."""
    return u.Infra.run_cli(_main_inner)


if __name__ == "__main__":
    sys.exit(main())
