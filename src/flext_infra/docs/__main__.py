"""CLI entry point for documentation services.

Usage:
    python -m flext_infra docs audit --root flext-core
    python -m flext_infra docs fix --root flext-core --apply
    python -m flext_infra docs build --root flext-core
    python -m flext_infra docs generate --root flext-core --apply
    python -m flext_infra docs validate --root flext-core

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flext_core import FlextRuntime

from flext_infra import REPORTS_DIR_NAME, c, output
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_infra.docs.generator import FlextInfraDocGenerator
from flext_infra.docs.validator import FlextInfraDocValidator


def _run_audit(args: argparse.Namespace) -> int:
    """Execute documentation audit."""
    auditor = FlextInfraDocAuditor()
    result = auditor.audit(
        root=Path(args.root).resolve(),
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
        check=args.check,
        strict=bool(args.strict),
    )
    if result.is_failure:
        output.error(result.error or "audit failed")
        return 1
    failures = sum(1 for report in result.value if not report.passed)
    return 1 if failures else 0


def _run_fix(args: argparse.Namespace) -> int:
    """Execute documentation fix."""
    fixer = FlextInfraDocFixer()
    result = fixer.fix(
        root=Path(args.root).resolve(),
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
        apply=args.apply,
    )
    if result.is_failure:
        output.error(result.error or "fix failed")
        return 1
    return 0


def _run_build(args: argparse.Namespace) -> int:
    """Execute documentation build."""
    builder = FlextInfraDocBuilder()
    result = builder.build(
        root=Path(args.root).resolve(),
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
    )
    if result.is_failure:
        output.error(result.error or "build failed")
        return 1

    failures = sum(1 for report in result.value if report.result == c.Status.FAIL)
    return 1 if failures else 0


def _run_generate(args: argparse.Namespace) -> int:
    """Execute documentation generation."""
    generator = FlextInfraDocGenerator()
    result = generator.generate(
        root=Path(args.root).resolve(),
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
        apply=args.apply,
    )
    if result.is_failure:
        output.error(result.error or "generate failed")
        return 1
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    """Execute documentation validation."""
    validator = FlextInfraDocValidator()
    result = validator.validate(
        root=Path(args.root).resolve(),
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
        check=args.check,
        apply=args.apply,
    )
    if result.is_failure:
        output.error(result.error or "validate failed")
        return 1

    failures = sum(1 for report in result.value if report.result == c.Status.FAIL)
    return 1 if failures else 0


def main() -> int:
    """Run documentation services: audit, fix, build, generate, validate."""
    FlextRuntime.ensure_structlog_configured()
    parser = argparse.ArgumentParser(description="Documentation management services")
    """Run documentation services: audit, fix, build, generate, validate."""
    parser = argparse.ArgumentParser(description="Documentation management services")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments helper
    def _add_common_args(sub: argparse.ArgumentParser) -> None:
        _ = sub.add_argument("--root", default=".")
        _ = sub.add_argument("--project")
        _ = sub.add_argument("--projects")
        _ = sub.add_argument("--output-dir", default=f"{REPORTS_DIR_NAME}/docs")

    # audit
    audit_parser = subparsers.add_parser("audit", help="Audit documentation")
    _add_common_args(audit_parser)
    _ = audit_parser.add_argument("--check", default="all")
    _ = audit_parser.add_argument("--strict", type=int, default=1)

    # fix
    fix_parser = subparsers.add_parser("fix", help="Fix documentation issues")
    _add_common_args(fix_parser)
    _ = fix_parser.add_argument("--apply", action="store_true")

    # build
    build_parser = subparsers.add_parser("build", help="Build MkDocs sites")
    _add_common_args(build_parser)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate project docs")
    _add_common_args(gen_parser)
    _ = gen_parser.add_argument("--apply", action="store_true")

    # validate
    val_parser = subparsers.add_parser("validate", help="Validate documentation")
    _add_common_args(val_parser)
    _ = val_parser.add_argument("--check", default="all")
    _ = val_parser.add_argument("--apply", action="store_true")

    args = parser.parse_args()

    handlers = {
        "audit": _run_audit,
        "fix": _run_fix,
        "build": _run_build,
        "generate": _run_generate,
        "validate": _run_validate,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
