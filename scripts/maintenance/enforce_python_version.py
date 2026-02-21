#!/usr/bin/env python3
# Owner-Skill: .claude/skills/workspace-maintenance/SKILL.md
"""Enforce Python version constraints across all workspace projects.

Creates .python-version files and injects conftest.py version guards
to prevent venv creation with wrong Python interpreter.

Usage::

    python scripts/maintenance/enforce_python_version.py [--check] [--verbose]

Modes:
    (default)  Apply: create .python-version, inject conftest guards
    --check    Verify: exit non-zero if any project is missing guards
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SCRIPTS_ROOT = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from libs.selection import resolve_projects  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_MINOR = 13
PYTHON_VERSION_CONTENT = f"3.{REQUIRED_MINOR}\n"

# Marker comment used to detect if the guard is already injected
GUARD_MARKER = "# PYTHON_VERSION_GUARD"

# The guard block injected into conftest.py files
GUARD_BLOCK = f"""\
{GUARD_MARKER} — Do not remove. Managed by scripts/maintenance/enforce_python_version.py
import sys as _sys

if _sys.version_info[:2] != (3, {REQUIRED_MINOR}):
    _v = f"{{_sys.version_info.major}}.{{_sys.version_info.minor}}.{{_sys.version_info.micro}}"
    raise RuntimeError(
        f"\\n{{'=' * 72}}\\n"
        f"FATAL: Python {{_v}} detected — this project requires Python 3.{REQUIRED_MINOR}.\\n"
        f"\\n"
        f"The virtual environment was created with the WRONG Python interpreter.\\n"
        f"\\n"
        f"Fix:\\n"
        f"  1. rm -rf .venv\\n"
        f"  2. poetry env use python3.{REQUIRED_MINOR}\\n"
        f"  3. poetry install\\n"
        f"\\n"
        f"Or use the workspace Makefile:\\n"
        f"  make setup PROJECT=<project-name>\\n"
        f"{{'=' * 72}}\\n"
    )
del _sys
{GUARD_MARKER}_END
"""


def _discover_projects(workspace_root: Path) -> list[Path]:
    return [
        project.path
        for project in resolve_projects(workspace_root, names=[])
        if (project.path / "pyproject.toml").exists()
    ]


def _ensure_python_version_file(
    project: Path, *, check_only: bool, verbose: bool
) -> bool:
    """Ensure .python-version exists with correct content."""
    pv_file = project / ".python-version"
    if pv_file.exists():
        content = pv_file.read_text(encoding="utf-8").strip()
        if content == f"3.{REQUIRED_MINOR}":
            if verbose:
                print(f"  ✓ .python-version OK: {project.name}")
            return True
        if check_only:
            print(f"  ✗ .python-version WRONG ({content}): {project.name}")
            return False
        if verbose:
            print(
                f"  ↻ .python-version FIXED ({content} → 3.{REQUIRED_MINOR}): {project.name}"
            )
    else:
        if check_only:
            print(f"  ✗ .python-version MISSING: {project.name}")
            return False
        if verbose:
            print(f"  + .python-version CREATED: {project.name}")

    _ = pv_file.write_text(PYTHON_VERSION_CONTENT, encoding="utf-8")
    return True


def _has_guard(content: str) -> bool:
    """Check if conftest.py already has the version guard."""
    return GUARD_MARKER in content


def _remove_existing_guard(content: str) -> str:
    """Remove existing guard block (for replacement)."""
    pattern = re.compile(
        rf"^{re.escape(GUARD_MARKER)}.*?^{re.escape(GUARD_MARKER)}_END\n?",
        re.MULTILINE | re.DOTALL,
    )
    return pattern.sub("", content)


def _inject_guard(content: str) -> str:
    """Inject version guard after the module docstring, before other imports."""
    # Remove any existing guard first
    content = _remove_existing_guard(content)

    # Find insertion point: after module docstring, before first import
    # Strategy: find the end of the docstring block, insert guard there
    lines = content.split("\n")
    insert_idx = 0

    # Skip shebang
    if lines and lines[0].startswith("#!"):
        insert_idx = 1

    # Skip leading comments
    while insert_idx < len(lines) and lines[insert_idx].startswith("#"):
        insert_idx += 1

    # Skip blank lines
    while insert_idx < len(lines) and not lines[insert_idx].strip():
        insert_idx += 1

    # Skip docstring (triple-quoted)
    if insert_idx < len(lines):
        line = lines[insert_idx].strip()
        if line.startswith(('"""', "'''")):
            quote = line[:3]
            # Check if single-line docstring
            if line.count(quote) >= 2 and len(line) > 3:
                insert_idx += 1
            else:
                # Multi-line docstring — find closing quotes
                insert_idx += 1
                while insert_idx < len(lines) and quote not in lines[insert_idx]:
                    insert_idx += 1
                if insert_idx < len(lines):
                    insert_idx += 1

    # Skip blank lines after docstring
    while insert_idx < len(lines) and not lines[insert_idx].strip():
        insert_idx += 1

    # Skip __future__ imports (must come before guard)
    while insert_idx < len(lines) and lines[insert_idx].strip().startswith(
        "from __future__"
    ):
        insert_idx += 1

    # Skip blank lines after __future__
    while insert_idx < len(lines) and not lines[insert_idx].strip():
        insert_idx += 1

    # Insert guard
    before = "\n".join(lines[:insert_idx])
    after = "\n".join(lines[insert_idx:])

    if before and not before.endswith("\n"):
        before += "\n"

    return f"{before}{GUARD_BLOCK}\n{after}"


def _ensure_conftest_guard(project: Path, *, check_only: bool, verbose: bool) -> bool:
    """Ensure tests/conftest.py has the Python version guard."""
    conftest = project / "tests" / "conftest.py"

    if not conftest.exists():
        if verbose:
            print(f"  - No tests/conftest.py: {project.name} (skipped)")
        return True  # Not a failure — project might not have tests

    content = conftest.read_text(encoding="utf-8")

    if _has_guard(content):
        if verbose:
            print(f"  ✓ conftest.py guard OK: {project.name}")
        return True

    if check_only:
        print(f"  ✗ conftest.py guard MISSING: {project.name}")
        return False

    new_content = _inject_guard(content)
    _ = conftest.write_text(new_content, encoding="utf-8")
    if verbose:
        print(f"  + conftest.py guard INJECTED: {project.name}")
    return True


def main(argv: list[str] | None = None) -> int:
    """Run enforcement."""
    parser = argparse.ArgumentParser(description="Enforce Python version constraints")
    _ = parser.add_argument(
        "--check", action="store_true", help="Check mode (no writes)"
    )
    _ = parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    args = parser.parse_args(argv)

    projects = _discover_projects(ROOT)
    all_ok = True
    mode = "Checking" if args.check else "Enforcing"

    print(f"{mode} Python 3.{REQUIRED_MINOR} for {len(projects)} projects...")

    # Workspace root .python-version
    if not _ensure_python_version_file(
        ROOT, check_only=args.check, verbose=args.verbose
    ):
        all_ok = False

    for project in projects:
        if not _ensure_python_version_file(
            project, check_only=args.check, verbose=args.verbose
        ):
            all_ok = False
        if not _ensure_conftest_guard(
            project, check_only=args.check, verbose=args.verbose
        ):
            all_ok = False

    if all_ok:
        print(f"✓ All {len(projects)} projects enforce Python 3.{REQUIRED_MINOR}")
        return 0

    if args.check:
        print(f"✗ Some projects missing Python 3.{REQUIRED_MINOR} enforcement")
        print("  Run: python scripts/maintenance/enforce_python_version.py")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
