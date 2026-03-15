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

from flext_core import FlextRuntime
from flext_infra import c, output, u
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_infra.docs.generator import FlextInfraDocGenerator
from flext_infra.docs.validator import FlextInfraDocValidator


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
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
        check="all" if check else "",
        strict=strict,
    )
    if result.is_failure:
        output.error(result.error or "audit failed")
        return 1
    failures = sum(1 for report in result.value if not report.passed)
    return 1 if failures else 0


def _run_fix(cli: u.Infra.CliArgs, *, output_dir: str = "") -> int:
    """Execute documentation fix."""
    fixer = FlextInfraDocFixer()
    result = fixer.fix(
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
        apply=cli.apply,
    )
    if result.is_failure:
        output.error(result.error or "fix failed")
        return 1
    return 0


def _run_build(cli: u.Infra.CliArgs, *, output_dir: str = "") -> int:
    """Execute documentation build."""
    builder = FlextInfraDocBuilder()
    result = builder.build(
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
    )
    if result.is_failure:
        output.error(result.error or "build failed")
        return 1
    failures = sum(1 for report in result.value if report.result == c.Infra.Status.FAIL)
    return 1 if failures else 0


def _run_generate(cli: u.Infra.CliArgs, *, output_dir: str = "") -> int:
    """Execute documentation generation."""
    generator = FlextInfraDocGenerator()
    result = generator.generate(
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
        apply=cli.apply,
    )
    if result.is_failure:
        output.error(result.error or "generate failed")
        return 1
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
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=output_dir,
        check="all" if check else "",
        apply=cli.apply,
    )
    if result.is_failure:
        output.error(result.error or "validate failed")
        return 1
    failures = sum(1 for report in result.value if report.result == c.Infra.Status.FAIL)
    return 1 if failures else 0


def main() -> int:
    """Run documentation services: audit, fix, build, generate, validate."""
    FlextRuntime.ensure_structlog_configured()
    parser = u.Infra.create_parser(
        "flext-infra docs",
        "Documentation management services",
        include_apply=False,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    audit_base = u.Infra.create_parser(
        "flext-infra docs audit",
        "Audit documentation",
        include_apply=True,
        include_project=True,
        include_check=True,
    )
    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit documentation",
        parents=[audit_base],
        add_help=False,
    )
    _ = audit_parser.add_argument("--strict", action="store_true", help="Strict mode")
    _ = audit_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    fix_base = u.Infra.create_parser(
        "flext-infra docs fix",
        "Fix documentation issues",
        include_apply=True,
        include_project=True,
    )
    fix_parser = subparsers.add_parser(
        "fix",
        help="Fix documentation issues",
        parents=[fix_base],
        add_help=False,
    )
    _ = fix_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    build_base = u.Infra.create_parser(
        f"flext-infra docs {c.Infra.Directories.BUILD}",
        "Build MkDocs sites",
        include_apply=True,
        include_project=True,
    )
    build_parser = subparsers.add_parser(
        c.Infra.Directories.BUILD,
        help="Build MkDocs sites",
        parents=[build_base],
        add_help=False,
    )
    _ = build_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    generate_base = u.Infra.create_parser(
        "flext-infra docs generate",
        "Generate project docs",
        include_apply=True,
        include_project=True,
    )
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate project docs",
        parents=[generate_base],
        add_help=False,
    )
    _ = gen_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    validate_base = u.Infra.create_parser(
        f"flext-infra docs {c.Infra.Verbs.VALIDATE}",
        "Validate documentation",
        include_apply=True,
        include_project=True,
        include_check=True,
    )
    val_parser = subparsers.add_parser(
        c.Infra.Verbs.VALIDATE,
        help="Validate documentation",
        parents=[validate_base],
        add_help=False,
    )
    _ = val_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    args = parser.parse_args()
    cli = u.Infra.resolve(args)

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


if __name__ == "__main__":
    sys.exit(main())
