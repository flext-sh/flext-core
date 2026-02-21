#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Detect whether a project runs in standalone or workspace mode."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def _repo_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or url
    name = path.rsplit("/", 1)[-1]
    return name.removesuffix(".git")


def detect_mode(project_root: Path) -> str:
    """Detect mode by inspecting parent repository origin URL."""
    parent = project_root.resolve().parent
    git_marker = parent / ".git"
    if not git_marker.exists():
        return "standalone"
    result = subprocess.run(
        ["git", "-C", str(parent), "config", "--get", "remote.origin.url"],  # noqa: S607
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
    """Print detected execution mode for a project path."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    print(detect_mode(args.project_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
