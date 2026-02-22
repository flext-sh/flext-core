#!/usr/bin/env python3
"""Run actionlint and persist workflow lint reports."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from scripts.libs.json_io import write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", type=Path, default=Path())
    _ = parser.add_argument(
        "--report",
        type=Path,
        default=Path(".reports/workflows/actionlint.json"),
    )
    _ = parser.add_argument("--strict", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    """Execute workflow linting and write JSON result."""
    args = _parse_args()
    root = args.root.resolve()
    report = args.report if args.report.is_absolute() else root / args.report

    actionlint = shutil.which("actionlint")
    if actionlint is None:
        payload_skipped: dict[str, object] = {
            "status": "skipped",
            "reason": "actionlint not installed",
        }
        write_json(report, payload_skipped, sort_keys=True)
        _ = print(f"wrote: {report}")
        return 0

    result = subprocess.run(
        [actionlint],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload: dict[str, object] = {
        "status": "ok" if result.returncode == 0 else "fail",
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    write_json(report, payload, sort_keys=True)
    _ = print(f"wrote: {report}")
    if result.returncode != 0:
        _ = print(result.stdout)
        _ = print(result.stderr)
    if args.strict == 1:
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
