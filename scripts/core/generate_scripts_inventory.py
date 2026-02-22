#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-infra/SKILL.md
"""Generate script inventory artifacts for workspace governance."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.libs.config import MAKEFILE_FILENAME
from scripts.libs.json_io import write_json
from scripts.libs.reporting import reports_root


def _artifact_path(slug: str) -> Path:
    return reports_root(Path()) / f"scripts-infra--json--{slug}.json"


def main() -> int:
    """Build and write scripts inventory reports."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scripts = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "scripts").rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    )
    inventory = {
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "total_scripts": len(scripts),
        "scripts": scripts,
    }
    wiring = {
        "generated_at": datetime.now(UTC).isoformat(),
        "root_makefile": [MAKEFILE_FILENAME],
        "unwired_scripts": [],
    }
    external = {"generated_at": datetime.now(UTC).isoformat(), "candidates": []}

    outputs = {
        _artifact_path("scripts-inventory"): inventory,
        _artifact_path("scripts-wiring"): wiring,
        _artifact_path("external-scripts-candidates"): external,
    }
    for path, payload in outputs.items():
        write_json(path, payload, sort_keys=True)
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
