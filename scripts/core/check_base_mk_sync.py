#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-infra/SKILL.md
"""Check that vendored project base.mk files match root base.mk."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from scripts.libs.paths import workspace_root_from_file


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Validate base.mk copies and report drift."""
    root = workspace_root_from_file(__file__)
    source = root / "base.mk"
    if not source.exists():
        print("[base-mk-sync] missing root base.mk", file=sys.stderr)
        return 1
    source_hash = _sha256(source)
    mismatched: list[Path] = []
    checked = 0
    for pyproject in sorted(root.glob("*/pyproject.toml")):
        local_base = pyproject.parent / "base.mk"
        if not local_base.exists():
            continue
        checked += 1
        if _sha256(local_base) != source_hash:
            mismatched.append(local_base.relative_to(root))
    if mismatched:
        for path in mismatched:
            print(f"[base-mk-sync] drift: {path}")
        return 1
    print(f"[base-mk-sync] all vendored base.mk copies are in sync ({checked} checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
