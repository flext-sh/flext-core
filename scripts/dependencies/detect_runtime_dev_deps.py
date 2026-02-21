#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-dependencies/SKILL.md
"""Detect runtime vs dev dependencies using deptry and pip check.

Runs deptry per project (DEP001 missing, DEP002 unused, DEP003 transitive, DEP004 dev in runtime)
and optionally pip check for the workspace env. Writes a machine-readable report to
.reports/dependencies/ for use by setup/upgrade automation or CI.

Usage:
    python scripts/dependencies/detect_runtime_dev_deps.py [--project NAME] [--no-pip-check] [--dry-run]
    python scripts/dependencies/detect_runtime_dev_deps.py --json  # print report JSON to stdout only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from dependencies import dependency_detection as dd  # noqa: E402

VENV_BIN = ROOT / ".venv" / "bin"
REPORTS_DIR = ROOT / ".reports" / "dependencies"
LIMITS_PATH = Path(__file__).resolve().parent / "dependency_limits.toml"


def main() -> int:
    """Run dependency detection and emit workspace report."""
    parser = argparse.ArgumentParser(
        description="Detect runtime vs dev dependencies (deptry + pip check).",
    )
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="Run only for this project (directory name).",
    )
    parser.add_argument(
        "--projects",
        metavar="NAMES",
        help="Comma-separated list of project names.",
    )
    parser.add_argument(
        "--no-pip-check",
        action="store_true",
        help="Skip pip check (workspace-level).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write report files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_stdout",
        help="Print full report JSON to stdout only (no file write).",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write report to this path (default: .reports/dependencies/detect-runtime-dev-<timestamp>.json).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal stdout (summary only).",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 (report only); default is to exit 1 when issues or pip check fail.",
    )
    parser.add_argument(
        "--typings",
        action="store_true",
        help="Detect required typing libraries (types-*) per project via mypy stub hints and dependency_limits.toml.",
    )
    parser.add_argument(
        "--apply-typings",
        action="store_true",
        help="Add missing typings to each project (poetry add --group typings). Implies --typings. Use --dry-run to only report.",
    )
    parser.add_argument(
        "--limits",
        metavar="FILE",
        default=str(LIMITS_PATH),
        help="Path to dependency_limits.toml (default: scripts/dependencies/dependency_limits.toml).",
    )
    args = parser.parse_args()

    limits_path = Path(args.limits) if args.limits else LIMITS_PATH
    apply_typings = getattr(args, "apply_typings", False)
    do_typings = getattr(args, "typings", False) or apply_typings

    projects_filter: list[str] | None = None
    if args.project:
        projects_filter = [args.project]
    elif args.projects:
        projects_filter = [p.strip() for p in args.projects.split(",") if p.strip()]

    projects = dd.discover_projects(ROOT, projects_filter=projects_filter)
    if not projects:
        print("No projects found.", file=sys.stderr)
        return 2

    if not (VENV_BIN / "deptry").exists():
        print("deptry not found in .venv. Run make setup first.", file=sys.stderr)
        return 3

    projects_report: dict[str, dict[str, object]] = {}
    report: dict[str, object] = {
        "workspace": str(ROOT),
        "projects": projects_report,
        "pip_check": None,
        "dependency_limits": None,
    }
    if do_typings:
        limits_data = dd.load_dependency_limits(limits_path)
        if isinstance(limits_data, dict) and limits_data:
            python_cfg = limits_data.get("python")
            report["dependency_limits"] = {
                "python_version": (
                    python_cfg.get("version") if isinstance(python_cfg, dict) else None
                ),
                "limits_path": str(limits_path),
            }

    for proj_path in projects:
        name = proj_path.name
        if not args.quiet:
            print(f"Running deptry for {name}...", file=sys.stderr)
        issues, _ = dd.run_deptry(proj_path, VENV_BIN)
        project_payload = dd.build_project_report(name, issues)
        projects_report[name] = dict(project_payload)

        if do_typings and (proj_path / "src").is_dir():
            if not args.quiet:
                print(f"Detecting typings for {name}...", file=sys.stderr)
            typings_report = dd.get_required_typings(
                proj_path, VENV_BIN, limits_path=limits_path
            )
            projects_report[name]["typings"] = typings_report
            to_add_obj = typings_report.get("to_add")
            to_add = to_add_obj if isinstance(to_add_obj, list) else []
            if apply_typings and to_add and not args.dry_run:
                env = {
                    **os.environ,
                    "VIRTUAL_ENV": str(VENV_BIN.parent),
                    "PATH": f"{VENV_BIN}:{os.environ.get('PATH', '')}",
                }
                for pkg in to_add:
                    if not isinstance(pkg, str):
                        continue
                    rc = subprocess.run(
                        ["poetry", "add", "--group", "typings", pkg],  # noqa: S607
                        cwd=proj_path,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        env=env,
                        check=False,
                    )
                    if rc.returncode != 0 and not args.quiet:
                        print(f"  add {pkg}: failed", file=sys.stderr)

    if not args.no_pip_check:
        if not args.quiet:
            print("Running pip check (workspace)...", file=sys.stderr)
        pip_lines, pip_exit = dd.run_pip_check(ROOT, VENV_BIN)
        report["pip_check"] = {"ok": pip_exit == 0, "lines": pip_lines}

    if args.json_stdout:
        json.dump(report, sys.stdout, indent=2)
        return 0

    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output)
    elif not args.dry_run:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / "detect-runtime-dev-latest.json"

    if out_path and not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"Report written to {out_path}", file=sys.stderr)

    # Summary
    total_issues = 0
    for payload in projects_report.values():
        deptry_obj = payload.get("deptry")
        if isinstance(deptry_obj, dict):
            raw_count = deptry_obj.get("raw_count", 0)
            if isinstance(raw_count, int):
                total_issues += raw_count

    pip_check_obj = report.get("pip_check")
    if pip_check_obj is None:
        pip_ok = True
    elif isinstance(pip_check_obj, dict):
        pip_ok = bool(pip_check_obj.get("ok", True))
    else:
        pip_ok = True
    if not args.quiet:
        print(
            f"Projects: {len(projects)} | Deptry issues: {total_issues} | Pip check: {'ok' if pip_ok else 'FAIL'}",
            file=sys.stderr,
        )
    if args.no_fail:
        return 0
    return 0 if total_issues == 0 and pip_ok else 1


if __name__ == "__main__":
    sys.exit(main())
