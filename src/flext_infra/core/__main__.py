"""CLI entry point for core infrastructure services.

Usage:
    python -m flext_infra core basemk-validate [--root PATH]
    python -m flext_infra core inventory [--root PATH] [--output-dir PATH]
    python -m flext_infra core pytest-diag --junit PATH --log PATH
    python -m flext_infra core scan --root PATH --pattern REGEX --include GLOB [--exclude GLOB] [--match present|absent]
    python -m flext_infra core skill-validate --skill NAME [--root PATH] [--mode baseline|strict]
    python -m flext_infra core stub-validate [--root PATH]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flext_infra.core.basemk_validator import BaseMkValidator
from flext_infra.core.inventory import InventoryService
from flext_infra.core.pytest_diag import PytestDiagExtractor
from flext_infra.core.scanner import TextPatternScanner
from flext_infra.core.skill_validator import SkillValidator
from flext_infra.core.stub_chain import StubSupplyChain


def _run_basemk_validate(args: argparse.Namespace) -> int:
    """Execute base.mk sync validation."""
    validator = BaseMkValidator()
    result = validator.validate(Path(args.root).resolve())

    if result.is_success:
        report = result.value
        _ = sys.stdout.write(f"{report.summary}\n")
        for v in report.violations:
            _ = sys.stdout.write(f"  {v}\n")
        return 0 if report.passed else 1
    _ = sys.stderr.write(f"Error: {result.error}\n")
    return 1


def _run_inventory(args: argparse.Namespace) -> int:
    """Execute scripts inventory generation."""
    service = InventoryService()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    result = service.generate(Path(args.root).resolve(), output_dir=output_dir)

    if result.is_success:
        data = result.value
        _ = sys.stdout.write(f"total_scripts={data.get('total_scripts', 0)}\n")
        written = data.get("reports_written", [])
        if isinstance(written, list):
            for path in written:
                _ = sys.stdout.write(f"Wrote: {path}\n")
        return 0
    _ = sys.stderr.write(f"Error: {result.error}\n")
    return 1


def _run_pytest_diag(args: argparse.Namespace) -> int:
    """Execute pytest diagnostics extraction."""
    extractor = PytestDiagExtractor()
    result = extractor.extract(Path(args.junit), Path(args.log))

    if result.is_success:
        data = result.value

        failed_cases_raw = data.get("failed_cases")
        if args.failed and isinstance(failed_cases_raw, list):
            failed_cases = [str(item) for item in failed_cases_raw]
            Path(args.failed).write_text(
                "\n\n".join(failed_cases) + "\n",
                encoding="utf-8",
            )
        error_traces_raw = data.get("error_traces")
        if args.errors and isinstance(error_traces_raw, list):
            error_traces = [str(item) for item in error_traces_raw]
            Path(args.errors).write_text(
                "\n\n".join(error_traces) + "\n",
                encoding="utf-8",
            )
        warning_lines_raw = data.get("warning_lines")
        if args.warnings and isinstance(warning_lines_raw, list):
            warning_lines = [str(item) for item in warning_lines_raw]
            Path(args.warnings).write_text(
                "\n".join(warning_lines) + "\n",
                encoding="utf-8",
            )
        slow_entries_raw = data.get("slow_entries")
        if args.slowest and isinstance(slow_entries_raw, list):
            slow_entries = [str(item) for item in slow_entries_raw]
            Path(args.slowest).write_text(
                "\n".join(slow_entries) + "\n",
                encoding="utf-8",
            )
        skip_cases_raw = data.get("skip_cases")
        if args.skips and isinstance(skip_cases_raw, list):
            skip_cases = [str(item) for item in skip_cases_raw]
            Path(args.skips).write_text("\n".join(skip_cases) + "\n", encoding="utf-8")

        _ = sys.stdout.write(f"failed_count={data.get('failed_count', 0)}\n")
        _ = sys.stdout.write(f"error_count={data.get('error_count', 0)}\n")
        _ = sys.stdout.write(f"warning_count={data.get('warning_count', 0)}\n")
        _ = sys.stdout.write(f"skipped_count={data.get('skipped_count', 0)}\n")
        return 0
    _ = sys.stderr.write(f"Error: {result.error}\n")
    return 1


def _run_scan(args: argparse.Namespace) -> int:
    """Execute text pattern scanning."""
    scanner = TextPatternScanner()
    result = scanner.scan(
        Path(args.root).resolve(),
        args.pattern,
        includes=args.include or [],
        excludes=args.exclude or [],
        match_mode=args.match,
    )

    if result.is_success:
        import json

        data = result.value
        _ = sys.stdout.write(
            f"{json.dumps({'violation_count': data.get('violation_count', 0)})}\n"
        )
        violation_count = data.get("violation_count", 0)
        return 1 if isinstance(violation_count, int) and violation_count > 0 else 0
    _ = sys.stderr.write(f"Error: {result.error}\n")
    return 1


def _run_skill_validate(args: argparse.Namespace) -> int:
    """Execute skill validation."""
    validator = SkillValidator()
    result = validator.validate(
        Path(args.root).resolve(),
        args.skill,
        mode=args.mode,
    )

    if result.is_success:
        report = result.value
        _ = sys.stdout.write(f"{report.summary}\n")
        for v in report.violations:
            _ = sys.stdout.write(f"  {v}\n")
        return 0 if report.passed else 1
    _ = sys.stderr.write(f"Error: {result.error}\n")
    return 1


def _run_stub_validate(args: argparse.Namespace) -> int:
    """Execute stub supply chain validation."""
    chain = StubSupplyChain()
    root = Path(args.root).resolve()
    project_dirs: list[Path] | None = None
    if hasattr(args, "project") and args.project:
        project_dirs = [root / p for p in args.project]
    result = chain.validate(root, project_dirs=project_dirs)

    if result.is_success:
        report = result.value
        _ = sys.stdout.write(f"{report.summary}\n")
        for v in report.violations:
            _ = sys.stdout.write(f"  {v}\n")
        return 0 if report.passed else 1
    _ = sys.stderr.write(f"Error: {result.error}\n")
    return 1


def main() -> int:
    """Run core infrastructure services."""
    parser = argparse.ArgumentParser(description="Core infrastructure services")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # basemk-validate
    bm = subparsers.add_parser("basemk-validate", help="Validate base.mk sync")
    bm.add_argument("--root", default=".", help="Workspace root")

    # inventory
    inv = subparsers.add_parser("inventory", help="Generate scripts inventory")
    inv.add_argument("--root", default=".", help="Workspace root")
    inv.add_argument("--output-dir", default=None, help="Output directory")

    # pytest-diag
    pd = subparsers.add_parser("pytest-diag", help="Extract pytest diagnostics")
    pd.add_argument("--junit", required=True, help="JUnit XML path")
    pd.add_argument("--log", required=True, help="Pytest log path")
    pd.add_argument("--failed", help="Path to write failed cases")
    pd.add_argument("--errors", help="Path to write error traces")
    pd.add_argument("--warnings", help="Path to write warnings")
    pd.add_argument("--slowest", help="Path to write slowest entries")
    pd.add_argument("--skips", help="Path to write skipped cases")

    # scan
    sc = subparsers.add_parser("scan", help="Scan text files for patterns")
    sc.add_argument("--root", required=True, help="Directory to scan")
    sc.add_argument("--pattern", required=True, help="Regex pattern")
    sc.add_argument("--include", action="append", required=True, help="Include glob")
    sc.add_argument("--exclude", action="append", default=[], help="Exclude glob")
    sc.add_argument(
        "--match",
        choices=("present", "absent"),
        default="present",
        help="Violation mode",
    )

    # skill-validate
    sv = subparsers.add_parser("skill-validate", help="Validate a skill")
    sv.add_argument("--skill", required=True, help="Skill folder name")
    sv.add_argument("--root", default=".", help="Workspace root")
    sv.add_argument(
        "--mode",
        choices=("baseline", "strict"),
        default="baseline",
    )

    # stub-validate
    stv = subparsers.add_parser("stub-validate", help="Validate stub supply chain")
    stv.add_argument("--root", default=".", help="Workspace root")
    stv.add_argument("--project", action="append", help="Project to validate")
    stv.add_argument("--all", action="store_true", help="Validate all projects")

    args = parser.parse_args()

    commands = {
        "basemk-validate": _run_basemk_validate,
        "inventory": _run_inventory,
        "pytest-diag": _run_pytest_diag,
        "scan": _run_scan,
        "skill-validate": _run_skill_validate,
        "stub-validate": _run_stub_validate,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
