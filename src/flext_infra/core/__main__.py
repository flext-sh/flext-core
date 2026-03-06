"""CLI entry point for core infrastructure services.

Usage:
    python -m flext_infra core basemk-validate [--root PATH]
    python -m flext_infra core inventory [--root PATH] [--output-dir PATH]
    python -m flext_infra core pytest-diag --junit PATH --log PATH
    python -m flext_infra core scan --root PATH --pattern REGEX
        --include GLOB [--exclude GLOB] [--match present|absent]
    python -m flext_infra core skill-validate --skill NAME [--root PATH]
        [--mode baseline|strict]
    python -m flext_infra core stub-validate [--root PATH]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from flext_core import FlextRuntime, t
from flext_infra import c, m, output
from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator
from flext_infra.core.inventory import FlextInfraInventoryService
from flext_infra.core.pytest_diag import FlextInfraPytestDiagExtractor
from flext_infra.core.scanner import FlextInfraTextPatternScanner
from flext_infra.core.skill_validator import FlextInfraSkillValidator
from flext_infra.core.stub_chain import FlextInfraStubSupplyChain


def _extract_reports_written(
    payload: m.Infra.Core.InventoryReport | Mapping[str, t.ContainerValue],
) -> list[str]:
    if isinstance(payload, Mapping):
        raw = payload.get("reports_written", [])
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, str)]
        return []
    return payload.reports_written


def _extract_diag_entries(
    payload: m.Infra.Core.PytestDiagnostics | Mapping[str, t.ContainerValue],
    key: str,
) -> list[str]:
    if isinstance(payload, Mapping):
        raw = payload.get(key, [])
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, str)]
        return []
    if key == "failed_cases":
        return payload.failed_cases
    if key == "error_traces":
        return payload.error_traces
    if key == "warning_lines":
        return payload.warning_lines
    if key == "slow_entries":
        return payload.slow_entries
    if key == "skip_cases":
        return payload.skip_cases
    return []


def _run_basemk_validate(args: argparse.Namespace) -> int:
    """Execute base.mk sync validation."""
    validator = FlextInfraBaseMkValidator()
    result = validator.validate(Path(args.root).resolve())

    if result.is_success:
        report: m.Infra.Core.ValidationReport = result.value
        output.info(report.summary)
        for v in report.violations:
            output.warning(v)
        return 0 if report.passed else 1
    output.error(result.error or "unknown error")
    return 1


def _run_inventory(args: argparse.Namespace) -> int:
    """Execute scripts inventory generation."""
    service = FlextInfraInventoryService()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    result = service.generate(Path(args.root).resolve(), output_dir=output_dir)

    if result.is_success:
        data = result.value
        for path in _extract_reports_written(data):
            output.info(f"Wrote: {path}")
        return 0
    output.error(result.error or "unknown error")
    return 1


def _run_pytest_diag(args: argparse.Namespace) -> int:
    """Execute pytest diagnostics extraction."""
    extractor = FlextInfraPytestDiagExtractor()
    result = extractor.extract(Path(args.junit), Path(args.log))

    if result.is_success:
        data = result.value

        if args.failed:
            failed_cases = _extract_diag_entries(data, "failed_cases")
            Path(args.failed).write_text(
                "\n\n".join(failed_cases) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        if args.errors:
            error_traces = _extract_diag_entries(data, "error_traces")
            Path(args.errors).write_text(
                "\n\n".join(error_traces) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        if args.warnings:
            warning_lines = _extract_diag_entries(data, "warning_lines")
            Path(args.warnings).write_text(
                "\n".join(warning_lines) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        if args.slowest:
            slow_entries = _extract_diag_entries(data, "slow_entries")
            Path(args.slowest).write_text(
                "\n".join(slow_entries) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        if args.skips:
            skip_cases = _extract_diag_entries(data, "skip_cases")
            Path(args.skips).write_text(
                "\n".join(skip_cases) + "\n", encoding=c.Infra.Encoding.DEFAULT
            )

        return 0
    output.error(result.error or "unknown error")
    return 1


def _run_scan(args: argparse.Namespace) -> int:
    """Execute text pattern scanning."""
    scanner = FlextInfraTextPatternScanner()
    result = scanner.scan(
        Path(args.root).resolve(),
        args.pattern,
        includes=args.include or [],
        excludes=args.exclude or [],
        match_mode=args.match,
    )

    if result.is_success:
        data: Mapping[str, t.ContainerValue] = cast(
            "Mapping[str, t.ContainerValue]",
            result.value,
        )
        violation_count = data.get("violation_count", 0)
        return 1 if isinstance(violation_count, int) and violation_count > 0 else 0
    output.error(result.error or "unknown error")
    return 1


def _run_skill_validate(args: argparse.Namespace) -> int:
    """Execute skill validation."""
    validator = FlextInfraSkillValidator()
    result = validator.validate(
        Path(args.root).resolve(),
        args.skill,
        mode=args.mode,
    )

    if result.is_success:
        report: m.Infra.Core.ValidationReport = result.value
        output.info(report.summary)
        for v in report.violations:
            output.warning(v)
        return 0 if report.passed else 1
    output.error(result.error or "unknown error")
    return 1


def _run_stub_validate(args: argparse.Namespace) -> int:
    """Execute stub supply chain validation."""
    chain = FlextInfraStubSupplyChain()
    root = Path(args.root).resolve()
    project_dirs: list[Path] | None = None
    if hasattr(args, c.Infra.Toml.PROJECT) and args.project:
        project_dirs = [root / p for p in args.project]
    result = chain.validate(root, project_dirs=project_dirs)

    if result.is_success:
        report: m.Infra.Core.ValidationReport = result.value
        output.info(report.summary)
        for v in report.violations:
            output.warning(v)
        return 0 if report.passed else 1
    output.error(result.error or "unknown error")
    return 1


def main() -> int:
    """Run core infrastructure services."""
    FlextRuntime.ensure_structlog_configured()
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
        default=c.Infra.MatchModes.PRESENT,
        help="Violation mode",
    )

    # skill-validate
    sv = subparsers.add_parser("skill-validate", help="Validate a skill")
    sv.add_argument("--skill", required=True, help="Skill folder name")
    sv.add_argument("--root", default=".", help="Workspace root")
    sv.add_argument(
        "--mode",
        choices=("baseline", "strict"),
        default=c.Infra.Modes.BASELINE,
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
