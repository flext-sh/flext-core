#!/usr/bin/env python3
"""Sync [tool.pyright].extraPaths and [tool.mypy].mypy_path from [project.dependencies].

Reads path dependencies (e.g. flext-core @ ./.flext-deps/flext-core) from the
workspace root pyproject.toml and updates extraPaths/mypy_path with <name>/src
for each, so type checkers resolve imports without manual list maintenance.

Usage:
    .venv/bin/python scripts/dependencies/sync_extra_paths_from_deps.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

import tomlkit
from tomlkit.toml_document import TOMLDocument

from scripts.libs.config import PYPROJECT_FILENAME
from scripts.libs.paths import workspace_root_from_file

ROOT = workspace_root_from_file(__file__)
PYPROJECT = ROOT / PYPROJECT_FILENAME

# Base paths kept before any dependency-derived paths (pyright/mypy).
PYRIGHT_BASE_PATHS = ["scripts", "src", "../typings", "../typings/generated"]
MYPY_BASE_PATHS = ["../typings", "../typings/generated", "src"]


def _path_dep_names(doc: TOMLDocument) -> list[str]:
    """Extract package names from [project.dependencies] that are path deps."""
    project = doc.get("project")
    if not project or not isinstance(project, dict):
        return []
    deps = project.get("dependencies")
    if not isinstance(deps, list):
        return []
    names: list[str] = []
    for item in deps:
        if not isinstance(item, str) or " @ " not in item:
            continue
        name = item.split(" @ ", 1)[0].strip()
        if name:
            names.append(name)
    return sorted(names)


def _sync_extra_paths(*, dry_run: bool = False) -> int:
    """Update pyproject.toml extraPaths and mypy_path from path dependencies."""
    if not PYPROJECT.exists():
        print(f"Missing {PYPROJECT}", file=sys.stderr)
        return 1
    text = PYPROJECT.read_text(encoding="utf-8")
    doc = tomlkit.parse(text)
    names = _path_dep_names(doc)
    dep_paths = [f"{name}/src" for name in names]

    pyright_extra = PYRIGHT_BASE_PATHS + dep_paths
    mypy_path = MYPY_BASE_PATHS + dep_paths

    tool = doc.get("tool")
    if not isinstance(tool, dict):
        print("No [tool] in pyproject.toml", file=sys.stderr)
        return 1
    pyright = tool.get("pyright")
    mypy = tool.get("mypy")
    if not isinstance(pyright, dict):
        print("No [tool.pyright] in pyproject.toml", file=sys.stderr)
        return 1

    changed = False
    current_pyright = pyright.get("extraPaths", [])
    if list(current_pyright) != pyright_extra:
        arr = tomlkit.array()
        for p in pyright_extra:
            arr.append(p)
        pyright["extraPaths"] = arr
        changed = True
    if isinstance(mypy, dict):
        current_mypy = mypy.get("mypy_path", [])
        if list(current_mypy) != mypy_path:
            arr = tomlkit.array()
            for p in mypy_path:
                arr.append(p)
            mypy["mypy_path"] = arr
            changed = True

    if dry_run:
        print("Would set [tool.pyright].extraPaths to:", pyright_extra[:5], "...")
        print("Would set [tool.mypy].mypy_path to:", mypy_path[:5], "...")
        return 0
    if changed:
        _ = PYPROJECT.write_text(tomlkit.dumps(doc), encoding="utf-8")
        print("Updated extraPaths and mypy_path from path dependencies.")
    return 0


def main() -> int:
    """Entry point."""
    doc_desc = (__doc__ or "").split("\n\n")[1] if "\n\n" in (__doc__ or "") else ""
    parser = argparse.ArgumentParser(description=doc_desc)
    _ = parser.add_argument(
        "--dry-run", action="store_true", help="Print would-be changes only"
    )
    args = parser.parse_args()
    return _sync_extra_paths(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
