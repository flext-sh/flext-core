#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


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
    args = _parse_args()
    root = args.root.resolve()
    report = args.report if args.report.is_absolute() else root / args.report
    report.parent.mkdir(parents=True, exist_ok=True)

    actionlint = shutil.which("actionlint")
    if actionlint is None:
        payload = {
            "status": "skipped",
            "reason": "actionlint not installed",
        }
        report.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _ = print(f"wrote: {report}")
        return 0

    result = subprocess.run(
        [actionlint],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "status": "ok" if result.returncode == 0 else "fail",
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _ = print(f"wrote: {report}")
    if result.returncode != 0:
        _ = print(result.stdout)
        _ = print(result.stderr)
    if args.strict == 1:
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
