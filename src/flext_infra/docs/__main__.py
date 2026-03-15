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

import argparse
import sys
from pathlib import Path

from flext_core import FlextRuntime
from flext_infra import c, output, u
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_infra.docs.generator import FlextInfraDocGenerator
from flext_infra.docs.validator import FlextInfraDocValidator


def _run_audit(cli: u.Infra.CliArgs, args: argparse.Namespace) -> int:
    """Execute documentation audit."""
    auditor = FlextInfraDocAuditor()
    result = auditor.audit(
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=args.output_dir,
        check=args.check,
        strict=bool(args.strict),
    )
    if result.is_failure:
        output.error(result.error or "audit failed")
        return 1
    failures = sum(1 for report in result.value if not report.passed)
    return 1 if failures else 0


def _run_fix(cli: u.Infra.CliArgs, args: argparse.Namespace) -> int:
    """Execute documentation fix."""
    fixer = FlextInfraDocFixer()
    result = fixer.fix(
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=args.output_dir,
        apply=cli.apply,
    )
    if result.is_failure:
        output.error(result.error or "fix failed")
        return 1
    return 0


def _run_build(cli: u.Infra.CliArgs, args: argparse.Namespace) -> int:
    """Execute documentation build."""
    builder = FlextInfraDocBuilder()
    result = builder.build(
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=args.output_dir,
    )
    if result.is_failure:
        output.error(result.error or "build failed")
        return 1
    failures = sum(1 for report in result.value if report.result == c.Infra.Status.FAIL)
    return 1 if failures else 0


def _run_generate(cli: u.Infra.CliArgs, args: argparse.Namespace) -> int:
    """Execute documentation generation."""
    generator = FlextInfraDocGenerator()
    result = generator.generate(
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=args.output_dir,
        apply=cli.apply,
    )
    if result.is_failure:
        output.error(result.error or "generate failed")
        return 1
    return 0


def _run_validate(cli: u.Infra.CliArgs, args: argparse.Namespace) -> int:
    """Execute documentation validation."""
    validator = FlextInfraDocValidator()
    result = validator.validate(
        root=cli.workspace,
        project=cli.project,
        projects=cli.projects,
        output_dir=args.output_dir,
        check=args.check,
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
    parser = argparse.ArgumentParser(description="Documentation management services")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    audit_parser = subparsers.add_parser("audit", help="Audit documentation")
    _ = audit_parser.add_argument(
        "--workspace", type=Path, default=Path.cwd(), help="Workspace root"
    )
    _ = audit_parser.add_argument("--project")
    _ = audit_parser.add_argument("--projects")
    _ = audit_parser.add_argument("--dry-run", action="store_true")
    _ = audit_parser.add_argument("--apply", action="store_true")
    _ = audit_parser.add_argument("--check", default="all")
    _ = audit_parser.add_argument("--strict", type=int, default=1)
    _ = audit_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    fix_parser = subparsers.add_parser("fix", help="Fix documentation issues")
    _ = fix_parser.add_argument(
        "--workspace", type=Path, default=Path.cwd(), help="Workspace root"
    )
    _ = fix_parser.add_argument("--project")
    _ = fix_parser.add_argument("--projects")
    _ = fix_parser.add_argument("--dry-run", action="store_true")
    _ = fix_parser.add_argument("--apply", action="store_true")
    _ = fix_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    build_parser = subparsers.add_parser(
        c.Infra.Directories.BUILD, help="Build MkDocs sites"
    )
    _ = build_parser.add_argument(
        "--workspace", type=Path, default=Path.cwd(), help="Workspace root"
    )
    _ = build_parser.add_argument("--project")
    _ = build_parser.add_argument("--projects")
    _ = build_parser.add_argument("--dry-run", action="store_true")
    _ = build_parser.add_argument("--apply", action="store_true")
    _ = build_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    gen_parser = subparsers.add_parser("generate", help="Generate project docs")
    _ = gen_parser.add_argument(
        "--workspace", type=Path, default=Path.cwd(), help="Workspace root"
    )
    _ = gen_parser.add_argument("--project")
    _ = gen_parser.add_argument("--projects")
    _ = gen_parser.add_argument("--dry-run", action="store_true")
    _ = gen_parser.add_argument("--apply", action="store_true")
    _ = gen_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    val_parser = subparsers.add_parser(
        c.Infra.Verbs.VALIDATE, help="Validate documentation"
    )
    _ = val_parser.add_argument(
        "--workspace", type=Path, default=Path.cwd(), help="Workspace root"
    )
    _ = val_parser.add_argument("--project")
    _ = val_parser.add_argument("--projects")
    _ = val_parser.add_argument("--dry-run", action="store_true")
    _ = val_parser.add_argument("--apply", action="store_true")
    _ = val_parser.add_argument("--check", default="all")
    _ = val_parser.add_argument(
        "--output-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/docs",
    )

    args = parser.parse_args()
    cli = u.Infra.resolve(args)

    handlers = {
        "audit": _run_audit,
        "fix": _run_fix,
        c.Infra.Directories.BUILD: _run_build,
        "generate": _run_generate,
        c.Infra.Verbs.VALIDATE: _run_validate,
    }
    handler = handlers.get(args.command)
    if handler:
        return handler(cli, args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
