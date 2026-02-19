#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-infra/SKILL.md
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_python_legacy(relative_path: str, args: list[str]) -> int:
    root = Path(__file__).resolve().parent
    script_path = root / "archive" / "legacy" / relative_path
    if not script_path.exists():
        print(f"ERROR: legacy script not found: {script_path}", file=sys.stderr)
        return 3
    process = subprocess.run([sys.executable, str(script_path), *args], check=False)
    return process.returncode
