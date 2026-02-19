#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    script = root / "archive" / "legacy" / "documentation" / "generate.py"
    return subprocess.run(
        [sys.executable, str(script), *sys.argv[1:]], check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
