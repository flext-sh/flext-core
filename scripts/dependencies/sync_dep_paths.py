#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-dependencies/SKILL.md
"""Rewrite internal FLEXT dependency paths for workspace or standalone mode.

In **workspace** mode, dependencies point directly to sibling projects::

    # Root:    flext-core @ ./flext-core
    # Poetry:  path = "flext-core"
    # Sub:     flext-core @ ../flext-core
    # Poetry:  path = "../flext-core"

In **standalone** mode, dependencies use the ``.flext-deps/`` staging directory::

    # Root:    flext-core @ ./.flext-deps/flext-core
    # Poetry:  path = ".flext-deps/flext-core"

Usage::

    .venv/bin/python -m scripts.dependencies.sync_dep_paths --mode workspace [--dry-run]
    .venv/bin/python -m scripts.dependencies.sync_dep_paths --mode standalone [--dry-run]
    .venv/bin/python -m scripts.dependencies.sync_dep_paths --mode auto [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tomlkit.toml_document import TOMLDocument

from scripts.libs.config import PYPROJECT_FILENAME
from scripts.libs.discovery import discover_projects
from scripts.libs.paths import workspace_root_from_file
from scripts.libs.toml_io import read_toml_document, write_toml_document

__all__ = [
    "FLEXT_DEPS_DIR",
    "extract_dep_name",
    "rewrite_dep_paths",
]

ROOT = workspace_root_from_file(__file__)

FLEXT_DEPS_DIR = ".flext-deps"
"""Name of the staging directory for standalone-mode dependencies."""

_PEP621_PATH_DEP_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)\s*@\s*(?:file:)?(?P<path>.+)$"
)


def _detect_mode(project_root: Path) -> str:
    """Auto-detect workspace or standalone mode.

    Workspace mode is detected when the project lives inside a directory
    that contains ``.gitmodules`` (i.e., the monorepo root).
    """
    for candidate in (project_root, *project_root.parents):
        if (candidate / ".gitmodules").exists():
            return "workspace"
    return "standalone"


def extract_dep_name(raw_path: str) -> str:
    """Extract the bare project name from any path format.

    Handles all variants::

        .flext-deps/flext-core  →  flext-core
        ../flext-core           →  flext-core
        ./flext-core            →  flext-core
        flext-core              →  flext-core
    """
    p = raw_path.strip().removeprefix("./")
    for prefix in (f"{FLEXT_DEPS_DIR}/", "../"):
        p = p.removeprefix(prefix)
    return p


def _target_path(dep_name: str, *, is_root: bool, mode: str) -> str:
    """Construct the dependency path for a given mode and project position.

    Args:
        dep_name: Bare project name (e.g., ``flext-core``).
        is_root: True when processing the workspace root pyproject.toml.
        mode: ``"workspace"`` or ``"standalone"``.

    Returns:
        The relative path string for the dependency declaration.
    """
    if mode == "workspace":
        return dep_name if is_root else f"../{dep_name}"
    return f"{FLEXT_DEPS_DIR}/{dep_name}"


def _rewrite_pep621(doc: TOMLDocument, *, is_root: bool, mode: str) -> list[str]:
    """Rewrite PEP 621 ``[project.dependencies]`` path deps.

    Returns list of change descriptions.
    """
    project = doc.get("project")
    if not project or not isinstance(project, dict):
        return []
    deps = project.get("dependencies")
    if not isinstance(deps, list):
        return []

    changes: list[str] = []
    for i, item in enumerate(deps):
        if not isinstance(item, str) or " @ " not in item:
            continue
        match = _PEP621_PATH_DEP_RE.match(item)
        if not match:
            continue
        name = match.group("name")
        raw_path = match.group("path").strip()
        dep_name = extract_dep_name(raw_path)
        new_path = _target_path(dep_name, is_root=is_root, mode=mode)
        path_prefix = "./" if is_root else ""
        new_entry = f"{name} @ {path_prefix}{new_path}"
        if deps[i] != new_entry:
            changes.append(f"  PEP621: {deps[i]} → {new_entry}")
            deps[i] = new_entry
    return changes


def _rewrite_poetry(doc: TOMLDocument, *, is_root: bool, mode: str) -> list[str]:
    """Rewrite ``[tool.poetry].dependencies.*.path`` values.

    Returns list of change descriptions.
    """
    tool = doc.get("tool")
    if not isinstance(tool, dict):
        return []
    poetry = tool.get("poetry")
    if not isinstance(poetry, dict):
        return []
    deps = poetry.get("dependencies")
    if not isinstance(deps, dict):
        return []

    changes: list[str] = []
    for dep_key, val in deps.items():
        if not isinstance(val, dict) or "path" not in val:
            continue
        raw_path = val["path"]
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        dep_name = extract_dep_name(raw_path)
        new_path = _target_path(dep_name, is_root=is_root, mode=mode)
        if raw_path != new_path:
            changes.append(f"  Poetry: {dep_key}.path = {raw_path!r} → {new_path!r}")
            val["path"] = new_path
    return changes


def rewrite_dep_paths(
    pyproject_path: Path,
    *,
    mode: str,
    is_root: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Rewrite dependency paths in a single pyproject.toml.

    Args:
        pyproject_path: Path to the pyproject.toml file.
        mode: ``"workspace"`` or ``"standalone"``.
        is_root: True if this is the workspace root pyproject.toml.
        dry_run: If True, don't write changes to disk.

    Returns:
        List of change descriptions (empty if nothing changed).
    """
    doc = read_toml_document(pyproject_path)
    if doc is None:
        return []

    changes = _rewrite_pep621(doc, is_root=is_root, mode=mode)
    changes += _rewrite_poetry(doc, is_root=is_root, mode=mode)

    if changes and not dry_run:
        write_toml_document(pyproject_path, doc)

    return changes


def _main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Rewrite internal FLEXT dependency paths for workspace/standalone mode."
    )
    _ = parser.add_argument(
        "--mode",
        choices=["workspace", "standalone", "auto"],
        default="auto",
        help="Target mode (default: auto-detect)",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing",
    )
    _ = parser.add_argument(
        "--project",
        action="append",
        dest="projects",
        metavar="DIR",
        help="Specific project(s) to process (default: all)",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = _detect_mode(ROOT)
        print(f"[sync-dep-paths] auto-detected mode: {mode}")

    total_changes = 0

    if not args.projects:
        root_pyproject = ROOT / PYPROJECT_FILENAME
        if root_pyproject.exists():
            changes = rewrite_dep_paths(
                root_pyproject, mode=mode, is_root=True, dry_run=args.dry_run
            )
            if changes:
                prefix = "[DRY-RUN] " if args.dry_run else ""
                print(f"{prefix}{root_pyproject}:")
                for c in changes:
                    print(c)
                total_changes += len(changes)

    if args.projects:
        project_dirs = [ROOT / p for p in args.projects]
    else:
        project_dirs = [pi.path for pi in discover_projects(ROOT)]

    for project_dir in sorted(project_dirs):
        pyproject = project_dir / PYPROJECT_FILENAME
        if not pyproject.exists():
            continue
        changes = rewrite_dep_paths(
            pyproject, mode=mode, is_root=False, dry_run=args.dry_run
        )
        if changes:
            prefix = "[DRY-RUN] " if args.dry_run else ""
            print(f"{prefix}{pyproject}:")
            for c in changes:
                print(c)
            total_changes += len(changes)

    if total_changes == 0:
        print(
            "[sync-dep-paths] No changes needed — all paths already match target mode."
        )
    else:
        action = "would change" if args.dry_run else "changed"
        print(f"\n[sync-dep-paths] {action} {total_changes} path(s).")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
