#!/usr/bin/env python3
# Owner-Skill: .claude/skills/workspace-maintenance/SKILL.md
"""Enforce Python version constraints across all workspace projects.

Creates ``.python-version`` files and verifies ``requires-python`` in
each project's ``pyproject.toml``.

Runtime version checking is handled automatically by
``flext_core._python_version_guard`` (imported on ``from flext_core import …``).
This script only manages the static ``.python-version`` files used by
pyenv / asdf / mise for interpreter selection.

Usage::

    python scripts/maintenance/enforce_python_version.py [--check] [--verbose]

Modes:
    (default)  Apply: create / fix .python-version files
    --check    Verify: exit non-zero if any project is missing files
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from scripts.libs.selection import resolve_projects

ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ────────────────────────────────────────────────────────


def _read_required_minor(workspace_root: Path) -> int:
    """Read the required Python minor version from the workspace ``pyproject.toml``.

    Falls back to ``13`` if the field cannot be parsed.
    """
    pyproject = workspace_root / "pyproject.toml"
    if not pyproject.is_file():
        return 13
    content = pyproject.read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"[>!=]*(\d+)\.(\d+)', content)
    if match is None:
        return 13
    return int(match.group(2))


def _discover_projects(workspace_root: Path) -> list[Path]:
    return [
        project.path
        for project in resolve_projects(workspace_root, names=[])
        if (project.path / "pyproject.toml").exists()
    ]


def _ensure_python_version_file(
    project: Path, *, required_minor: int, check_only: bool, verbose: bool
) -> bool:
    """Ensure ``pyproject.toml`` requires-python matches the required minor version and runtime."""
    local_minor = _read_required_minor(project)

    # 1. Validate pyproject.toml
    if local_minor != required_minor:
        if check_only:
            print(
                f"  ✗ pyproject.toml requires-python WRONG ({local_minor}): {project.name}"
            )
        else:
            print(
                f"  ✗ pyproject.toml requires-python MISMATCH ({local_minor} != {required_minor}): {project.name}"
            )
            print(
                f"    Please manually update requires-python in {project.name}/pyproject.toml"
            )
        return False

    # 2. Validate current runtime
    runtime_minor = sys.version_info.minor
    if runtime_minor != required_minor:
        print(
            f"  ✗ Runtime Python mismatch (3.{runtime_minor} != 3.{required_minor}): {project.name}"
        )
        return False

    if verbose:
        print(
            f"  ✓ Python 3.{required_minor} validated for runtime and pyproject.toml: {project.name}"
        )

    return True


# ── Main ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Run enforcement."""
    parser = argparse.ArgumentParser(
        description="Enforce Python version constraints via pyproject.toml"
    )
    _ = parser.add_argument(
        "--check", action="store_true", help="Check mode (no writes)"
    )
    _ = parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    args = parser.parse_args(argv)

    required_minor = _read_required_minor(ROOT)
    projects = _discover_projects(ROOT)
    mode = "Checking" if args.check else "Enforcing"

    print(f"{mode} Python 3.{required_minor} for {len(projects)} projects...")

    # Workspace root pyproject.toml
    if not _ensure_python_version_file(
        ROOT,
        required_minor=required_minor,
        check_only=args.check,
        verbose=args.verbose,
    ):
        print(f"✗ Failed: Missing Python 3.{required_minor} enforcement")
        return 1

    for project in projects:
        if not _ensure_python_version_file(
            project,
            required_minor=required_minor,
            check_only=args.check,
            verbose=args.verbose,
        ):
            print(f"✗ Failed: Missing Python 3.{required_minor} enforcement")
            return 1

    print(f"✓ All {len(projects)} projects enforce Python 3.{required_minor}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
