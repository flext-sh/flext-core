#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def _repo_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path if parsed.path else url
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def detect_mode(project_root: Path) -> str:
    parent = project_root.resolve().parent
    git_marker = parent / ".git"
    if not git_marker.exists():
        return "standalone"
    result = subprocess.run(
        ["git", "-C", str(parent), "config", "--get", "remote.origin.url"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "standalone"
    origin = result.stdout.strip()
    if not origin:
        return "standalone"
    return "workspace" if _repo_name_from_url(origin) == "flext" else "standalone"


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    print(detect_mode(args.project_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
