#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-infra/SKILL.md
"""Scan text files with regex and report violation counts as JSON."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Project root directory")
    parser.add_argument("--pattern", required=True, help="Regex pattern")
    parser.add_argument(
        "--match",
        choices=("present", "absent"),
        default="present",
        help="Violation mode",
    )
    parser.add_argument(
        "--include",
        action="append",
        required=True,
        help="Glob pattern to include (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude (repeatable)",
    )
    return parser.parse_args()


def collect_files(root: Path, includes: list[str], excludes: list[str]) -> list[Path]:
    selected: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in includes):
            if any(fnmatch.fnmatch(rel, pattern) for pattern in excludes):
                continue
            selected.append(path)
    return selected


def count_matches(files: list[Path], regex: re.Pattern[str]) -> int:
    total = 0
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        total += sum(1 for _ in regex.finditer(text))
    return total


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    pattern = str(args.pattern)
    match_mode = str(args.match)
    includes = [str(item) for item in (args.include or []) if str(item).strip()]
    excludes = [str(item) for item in (args.exclude or []) if str(item).strip()]

    if not root.exists() or not root.is_dir():
        raise SystemExit(2)
    if not includes:
        raise SystemExit(2)

    regex = re.compile(pattern, flags=re.MULTILINE)
    files = collect_files(root, includes, excludes)
    matches = count_matches(files, regex)

    violation_count = matches if match_mode == "present" else 0 if matches > 0 else 1

    print(json.dumps({"violation_count": violation_count}))
    return 1 if violation_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
