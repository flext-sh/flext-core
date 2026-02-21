#!/usr/bin/env python3
"""Script to build multiple projects within the workspace using Make."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# pylint: disable=wrong-import-position
from release.shared import resolve_projects, workspace_root  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the build script."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", type=Path, default=Path())
    _ = parser.add_argument("--version", required=True)
    _ = parser.add_argument("--output-dir", type=Path, required=True)
    _ = parser.add_argument("--projects", nargs="*", default=[])
    return parser.parse_args()


def _run_make(project_path: Path, verb: str) -> tuple[int, str]:
    """Execute a make command for a specific project.

    Args:
        project_path: The path to the project directory.
        verb: The make target to run (e.g., "build").

    Returns:
        A tuple of (return_code, combined_output).

    """
    command = ["make", "-C", str(project_path), verb]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode, output


def main() -> int:
    """Execute the multi-project build process.

    Returns:
        0 if all projects built successfully, 1 otherwise.

    """
    args = _parse_args()
    root = workspace_root(args.root)
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "build-report.json"

    projects = resolve_projects(root, args.projects)
    targets = [
        ("root", root),
        *[(project.name, project.path) for project in projects],
    ]

    seen: set[str] = set()
    unique_targets: list[tuple[str, Path]] = []
    for name, path in targets:
        if name in seen:
            continue
        seen.add(name)
        if not path.exists():
            continue
        unique_targets.append((name, path))

    records: list[dict[str, str | int]] = []
    failures = 0
    for name, path in unique_targets:
        code, output = _run_make(path, "build")
        if code != 0:
            failures += 1
        log = output_dir / f"build-{name}.log"
        log.write_text(output + "\n", encoding="utf-8")
        records.append({
            "project": name,
            "path": str(path),
            "exit_code": code,
            "log": str(log),
        })
        _ = print(f"[{name}] build exit={code}")

    report = {
        "version": args.version,
        "total": len(records),
        "failures": failures,
        "records": records,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _ = print(f"report: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
