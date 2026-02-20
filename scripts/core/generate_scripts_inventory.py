#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-infra/SKILL.md
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def _artifact_path(slug: str) -> Path:
    return Path(".reports") / f"scripts-infra--json--{slug}.json"


def main() -> int:
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
        "root_makefile": ["Makefile"],
        "unwired_scripts": [],
    }
    external = {"generated_at": datetime.now(UTC).isoformat(), "candidates": []}

    outputs = {
        _artifact_path("scripts-inventory"): inventory,
        _artifact_path("scripts-wiring"): wiring,
        _artifact_path("external-scripts-candidates"): external,
    }
    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
