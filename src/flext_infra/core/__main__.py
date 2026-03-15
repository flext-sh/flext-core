"""CLI entry point for core infrastructure services.

Usage:
    python -m flext_infra core basemk-validate [--workspace PATH]
    python -m flext_infra core inventory [--workspace PATH] [--output-dir PATH]
    python -m flext_infra core pytest-diag --junit PATH --log PATH
    python -m flext_infra core scan --workspace PATH --pattern REGEX
        --include GLOB [--exclude GLOB] [--match present|absent]
    python -m flext_infra core skill-validate --skill NAME [--workspace PATH]
        [--mode baseline|strict]
    python -m flext_infra core stub-validate [--workspace PATH]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from flext_infra import c, m, output, t, u
from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator
from flext_infra.core.inventory import FlextInfraInventoryService
from flext_infra.core.pytest_diag import FlextInfraPytestDiagExtractor
from flext_infra.core.scanner import FlextInfraTextPatternScanner
from flext_infra.core.skill_validator import FlextInfraSkillValidator
from flext_infra.core.stub_chain import FlextInfraStubSupplyChain


def _extract_reports_written(
    payload: m.Infra.Core.InventoryReport | Mapping[str, JsonValue],
) -> list[str]:
    if isinstance(payload, Mapping):
        raw = payload.get("reports_written", [])
        if isinstance(raw, list):
            try:
                typed_items: list[JsonValue] = TypeAdapter(
                    list[JsonValue]
                ).validate_python(raw)
            except ValidationError:
                return []
            return [item for item in typed_items if isinstance(item, str)]
        return []
    return payload.reports_written


def _extract_diag_entries(
    payload: m.Infra.Core.PytestDiagnostics | Mapping[str, JsonValue],
    key: str,
) -> list[str]:
    if isinstance(payload, Mapping):
        raw = payload.get(key, [])
        if isinstance(raw, list):
            try:
                typed_items: list[JsonValue] = TypeAdapter(
                    list[JsonValue]
                ).validate_python(raw)
            except ValidationError:
                return []
            return [item for item in typed_items if isinstance(item, str)]
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


def _run_basemk_validate(cli: u.Infra.CliArgs) -> int:
    """Execute base.mk sync validation."""
    validator = FlextInfraBaseMkValidator()
    result = validator.validate(cli.workspace)
    if result.is_success:
        report: m.Infra.Core.ValidationReport = result.value
        output.info(report.summary)
        for v in report.violations:
            output.warning(v)
        return 0 if report.passed else 1
    output.error(result.error or "unknown error")
    return 1


def _run_inventory(cli: u.Infra.CliArgs, output_dir: str | None) -> int:
    """Execute scripts inventory generation."""
    service = FlextInfraInventoryService()
    output_dir_path = Path(output_dir).resolve() if output_dir else None
    result = service.generate(cli.workspace, output_dir=output_dir_path)
    if result.is_success:
        data = result.value
        for path in _extract_reports_written(data):
            output.info(f"Wrote: {path}")
        return 0
    output.error(result.error or "unknown error")
    return 1


def _run_pytest_diag(
    junit: str,
    log: str,
    failed: str | None,
    errors: str | None,
    warnings: str | None,
    slowest: str | None,
    skips: str | None,
) -> int:
    """Execute pytest diagnostics extraction."""
    extractor = FlextInfraPytestDiagExtractor()
    result = extractor.extract(Path(junit), Path(log))
    if result.is_success:
        data = result.value
        if failed:
            failed_cases = _extract_diag_entries(data, "failed_cases")
            u.write_file(
                Path(failed),
                "\n\n".join(failed_cases) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        if errors:
            error_traces = _extract_diag_entries(data, "error_traces")
            u.write_file(
                Path(errors),
                "\n\n".join(error_traces) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        if warnings:
            warning_lines = _extract_diag_entries(data, "warning_lines")
            u.write_file(
                Path(warnings),
                "\n".join(warning_lines) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        if slowest:
            slow_entries = _extract_diag_entries(data, "slow_entries")
            u.write_file(
                Path(slowest),
                "\n".join(slow_entries) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        if skips:
            skip_cases = _extract_diag_entries(data, "skip_cases")
            u.write_file(
                Path(skips),
                "\n".join(skip_cases) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
        return 0
    output.error(result.error or "unknown error")
    return 1


def _run_scan(
    cli: u.Infra.CliArgs,
    pattern: str,
    include: list[str],
    exclude: list[str],
    match: str,
) -> int:
    """Execute text pattern scanning."""
    scanner = FlextInfraTextPatternScanner()
    result = scanner.scan(
        cli.workspace,
        pattern,
        includes=include or [],
        excludes=exclude or [],
        match_mode=match,
    )
    if result.is_success:
        data: Mapping[str, t.Scalar] = result.value
        violation_count = data.get("violation_count", 0)
        return 1 if isinstance(violation_count, int) and violation_count > 0 else 0
    output.error(result.error or "unknown error")
    return 1


def _run_skill_validate(cli: u.Infra.CliArgs, skill: str, mode: str) -> int:
    """Execute skill validation."""
    validator = FlextInfraSkillValidator()
    result = validator.validate(cli.workspace, skill, mode=mode)
    if result.is_success:
        report: m.Infra.Core.ValidationReport = result.value
        output.info(report.summary)
        for v in report.violations:
            output.warning(v)
        return 0 if report.passed else 1
    output.error(result.error or "unknown error")
    return 1


def _run_stub_validate(cli: u.Infra.CliArgs, project: list[str] | None) -> int:
    """Execute stub supply chain validation."""
    chain = FlextInfraStubSupplyChain()
    project_dirs: list[Path] | None = None
    if project:
        project_dirs = [cli.workspace / p for p in project]
    result = chain.validate(cli.workspace, project_dirs=project_dirs)
    if result.is_success:
        report: m.Infra.Core.ValidationReport = result.value
        output.info(report.summary)
        for v in report.violations:
            output.warning(v)
        return 0 if report.passed else 1
    output.error(result.error or "unknown error")
    return 1


def _main_inner(argv: list[str] | None = None) -> int:
    """Run core infrastructure services."""
    parser = u.Infra.create_parser(
        prog="flext_infra core",
        description="Core infrastructure services",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    bm = subparsers.add_parser("basemk-validate", help="Validate base.mk sync")
    _ = bm.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )

    inv = subparsers.add_parser("inventory", help="Generate scripts inventory")
    _ = inv.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    inv.add_argument("--output-dir", default=None, help="Output directory")

    pd = subparsers.add_parser("pytest-diag", help="Extract pytest diagnostics")
    pd.add_argument("--junit", required=True, help="JUnit XML path")
    pd.add_argument("--log", required=True, help="Pytest log path")
    pd.add_argument("--failed", help="Path to write failed cases")
    pd.add_argument("--errors", help="Path to write error traces")
    pd.add_argument("--warnings", help="Path to write warnings")
    pd.add_argument("--slowest", help="Path to write slowest entries")
    pd.add_argument("--skips", help="Path to write skipped cases")

    sc = subparsers.add_parser("scan", help="Scan text files for patterns")
    _ = sc.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    sc.add_argument("--pattern", required=True, help="Regex pattern")
    sc.add_argument("--include", action="append", required=True, help="Include glob")
    sc.add_argument("--exclude", action="append", default=[], help="Exclude glob")
    sc.add_argument(
        "--match",
        choices=("present", "absent"),
        default=c.Infra.MatchModes.PRESENT,
        help="Violation mode",
    )

    sv = subparsers.add_parser("skill-validate", help="Validate a skill")
    _ = sv.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    sv.add_argument("--skill", required=True, help="Skill folder name")
    sv.add_argument(
        "--mode",
        choices=("baseline", "strict"),
        default=c.Infra.Modes.BASELINE,
    )

    stv = subparsers.add_parser("stub-validate", help="Validate stub supply chain")
    _ = stv.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    stv.add_argument("--project", action="append", help="Project to validate")
    _ = stv.add_argument("--all", action="store_true", help="Validate all projects")

    args = parser.parse_args(argv)
    cli = u.Infra.resolve(args)

    commands = {
        "basemk-validate": lambda: _run_basemk_validate(cli),
        "inventory": lambda: _run_inventory(cli, getattr(args, "output_dir", None)),
        "pytest-diag": lambda: _run_pytest_diag(
            getattr(args, "junit", ""),
            getattr(args, "log", ""),
            getattr(args, "failed", None),
            getattr(args, "errors", None),
            getattr(args, "warnings", None),
            getattr(args, "slowest", None),
            getattr(args, "skips", None),
        ),
        "scan": lambda: _run_scan(
            cli,
            getattr(args, "pattern", ""),
            getattr(args, "include", None) or [],
            getattr(args, "exclude", None) or [],
            getattr(args, "match", "present"),
        ),
        "skill-validate": lambda: _run_skill_validate(
            cli, getattr(args, "skill", ""), getattr(args, "mode", "baseline")
        ),
        "stub-validate": lambda: _run_stub_validate(
            cli, getattr(args, "project", None)
        ),
    }
    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler()


def main() -> int:
    """Run core infrastructure services."""
    return u.Infra.run_cli(_main_inner)


if __name__ == "__main__":
    sys.exit(main())
