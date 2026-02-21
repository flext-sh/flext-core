#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Run make verbs across projects with per-project logs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run(
    project: str,
    verb: str,
    index: int,
    _total: int,
    *,
    fail_fast: bool,
    make_args: list[str],
) -> tuple[int, bool]:
    reports_dir = Path(".reports") / "workspace" / verb
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / f"{project}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            ["make", "-C", project, verb, *make_args],  # noqa: S607
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = int(time.monotonic() - started)
    status = "OK" if proc.returncode == 0 else "FAIL"
    print(
        f"{index:02d} [{status}] {project} {verb} ({elapsed}s) exit={proc.returncode} log={log_path}"
    )
    if proc.returncode != 0 and fail_fast:
        return proc.returncode, True
    return proc.returncode, False


def main() -> int:
    """Execute make verb across projects with optional fail-fast."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--verb", required=True)
    _ = parser.add_argument("--fail-fast", action="store_true")
    _ = parser.add_argument("--make-arg", action="append", default=[])
    _ = parser.add_argument("projects", nargs="*")
    args = parser.parse_args()

    projects = [project for project in args.projects if project]
    success = 0
    failed = 0
    skipped = 0
    failures: list[int] = []
    total = len(projects)
    for idx, project in enumerate(projects, start=1):
        if skipped:
            print(f"{idx:02d} [SKIP] {project} {args.verb} (0s) exit=0")
            continue
        code, stop = _run(
            project,
            args.verb,
            idx,
            total,
            fail_fast=args.fail_fast,
            make_args=args.make_arg,
        )
        if code == 0:
            success += 1
        else:
            failed += 1
            failures.append(code)
        if stop:
            skipped = total - idx

    print(f"summary total={total} success={success} fail={failed} skip={skipped}")
    return max(failures) if failures else 0


if __name__ == "__main__":
    sys.exit(main())
