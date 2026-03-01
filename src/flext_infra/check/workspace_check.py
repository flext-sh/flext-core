"""Workspace check CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from flext_infra import c, output
from flext_infra.check.services import FlextInfraWorkspaceChecker


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run workspace checks for specified projects."""
    parser = argparse.ArgumentParser(description="FLEXT Workspace Check")
    _ = parser.add_argument("projects", nargs="*")
    _ = parser.add_argument("--gates", default=c.Gates.DEFAULT_CSV)
    _ = parser.add_argument("--reports-dir", default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/check")
    _ = parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    if not args.projects:
        output.error("no projects specified")
        return 1

    checker = FlextInfraWorkspaceChecker()
    gates = FlextInfraWorkspaceChecker.parse_gate_csv(args.gates)

    reports_dir = Path(args.reports_dir).expanduser()
    if not reports_dir.is_absolute():
        reports_dir = (Path.cwd() / reports_dir).resolve()

    result = checker.run_projects(
        projects=args.projects,
        gates=gates,
        reports_dir=reports_dir,
        fail_fast=args.fail_fast,
    )
    if result.is_failure:
        output.error(result.error or "workspace check failed")
        return 2

    failed_projects = [project for project in result.value if not project.passed]
    return 1 if failed_projects else 0


if __name__ == "__main__":
    raise SystemExit(main())
