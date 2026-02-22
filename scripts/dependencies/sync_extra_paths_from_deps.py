#!/usr/bin/env python3
"""Sync [tool.pyright].extraPaths and [tool.mypy].mypy_path from path dependencies.

Reads path dependencies from [project.dependencies] (PEP 621) and/or
[tool.poetry].dependencies and sets extraPaths/mypy_path to <path>/src so
type checkers resolve imports. Supports workspace root or --project <dir>.

Usage:
    .venv/bin/python scripts/dependencies/sync_extra_paths_from_deps.py [--dry-run]
    .venv/bin/python scripts/dependencies/sync_extra_paths_from_deps.py --project flext-ldif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tomlkit
from tomlkit.toml_document import TOMLDocument

from scripts.libs.config import PYPROJECT_FILENAME
from scripts.libs.paths import workspace_root_from_file
from scripts.libs.toml_io import read_toml_document, write_toml_document

ROOT = workspace_root_from_file(__file__)

# Base paths for root (relative to root).
PYRIGHT_BASE_ROOT = ["scripts", "src", "typings", "typings/generated"]
MYPY_BASE_ROOT = ["typings", "typings/generated", "src"]

# Base paths for subprojects (relative to project dir). ".." = workspace root for scripts.libs.
# Exported for modernize_pyproject SSOT so mass sync does not overwrite per-project paths.
PYRIGHT_BASE_PROJECT = [
    "..",
    "src",
    "tests",
    "examples",
    "scripts",
    "../typings",
    "../typings/generated",
]
MYPY_BASE_PROJECT = ["..", "../typings", "../typings/generated", "src"]


def _path_dep_paths_pep621(doc: TOMLDocument) -> list[str]:
    """Extract path part from [project.dependencies] path deps (name @ path)."""
    project = doc.get("project")
    if not project or not isinstance(project, dict):
        return []
    deps = project.get("dependencies")
    if not isinstance(deps, list):
        return []
    paths: list[str] = []
    for item in deps:
        if not isinstance(item, str) or " @ " not in item:
            continue
        _name, path_part = item.split(" @ ", 1)
        path_part = path_part.strip()
        if path_part.startswith("file:"):
            path_part = path_part[5:].strip()
        if path_part.startswith("./"):
            path_part = path_part[2:].strip()
        if path_part:
            paths.append(path_part)
    return sorted(set(paths))


def _path_dep_paths_poetry(doc: TOMLDocument) -> list[str]:
    """Extract path from [tool.poetry].dependencies.*.path."""
    tool = doc.get("tool")
    if not isinstance(tool, dict):
        return []
    poetry = tool.get("poetry")
    if not isinstance(poetry, dict):
        return []
    deps = poetry.get("dependencies")
    if not isinstance(deps, dict):
        return []
    paths: list[str] = []
    for val in deps.values():
        if isinstance(val, dict) and "path" in val:
            p = val["path"]
            if isinstance(p, str) and p:
                p = p.strip()
                if p.startswith("./"):
                    p = p[2:].strip()
                if p:
                    paths.append(p)
    return sorted(set(paths))


def _path_dep_paths(doc: TOMLDocument) -> list[str]:
    """All path-dependency paths (relative) from PEP 621 and Poetry."""
    paths = _path_dep_paths_pep621(doc) + _path_dep_paths_poetry(doc)
    return sorted(set(paths))


def get_dep_paths(doc: TOMLDocument, *, is_root: bool = False) -> list[str]:
    """Return type-checker src paths for path deps (SSOT for modernize_pyproject).

    Resolves ``.flext-deps/X`` → actual workspace project ``X`` so paths
    reference real sibling projects instead of the symlink staging directory.
    """
    raw_paths = _path_dep_paths(doc)
    resolved: list[str] = []
    for p in raw_paths:
        if not p:
            continue
        name = p.removeprefix(".flext-deps/")
        if is_root:
            resolved.append(f"{name}/src")
        else:
            resolved.append(f"../{name}/src")
    return resolved


def _sync_one(
    pyproject_path: Path,
    *,
    dry_run: bool = False,
    is_root: bool = False,
) -> bool:
    """Update one pyproject.toml; return True if changed."""
    if not pyproject_path.exists():
        return False
    doc = read_toml_document(pyproject_path)
    if doc is None:
        return False
    dep_paths = get_dep_paths(doc, is_root=is_root)
    if is_root:
        pyright_extra = PYRIGHT_BASE_ROOT + dep_paths
        mypy_path = MYPY_BASE_ROOT + dep_paths
    else:
        pyright_extra = PYRIGHT_BASE_PROJECT + dep_paths
        mypy_path = MYPY_BASE_PROJECT + dep_paths

    tool = doc.get("tool")
    if not isinstance(tool, dict):
        return False
    pyright = tool.get("pyright")
    mypy = tool.get("mypy")
    if not isinstance(pyright, dict):
        return False

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
    if not is_root:
        pyrefly = tool.get("pyrefly")
        if isinstance(pyrefly, dict):
            # Subproject: search-path = .. (workspace root) + dep src paths + existing entries (no duplicate ..)
            base_sp = [".."] + dep_paths
            current_sp = list(pyrefly.get("search-path", []))
            # Keep existing entries that are not already in base_sp
            seen = set(base_sp)
            for p in current_sp:
                if p not in seen:
                    base_sp.append(p)
                    seen.add(p)
            if base_sp != current_sp:
                arr = tomlkit.array()
                for p in base_sp:
                    arr.append(p)
                pyrefly["search-path"] = arr
                changed = True

    if not dry_run and changed:
        write_toml_document(pyproject_path, doc)
    return changed


def _sync_extra_paths(
    *,
    dry_run: bool = False,
    project_dirs: list[Path] | None = None,
) -> int:
    """Update pyproject.toml(s) extraPaths and mypy_path from path dependencies."""
    if project_dirs:
        updated = 0
        for d in project_dirs:
            pyproject = d / PYPROJECT_FILENAME
            if _sync_one(pyproject, dry_run=dry_run, is_root=(d == ROOT)):
                updated += 1
                if not dry_run:
                    print(f"Updated {pyproject}")
        return 0
    pyproject = ROOT / PYPROJECT_FILENAME
    if not pyproject.exists():
        print(f"Missing {pyproject}", file=sys.stderr)
        return 1
    if _sync_one(pyproject, dry_run=dry_run, is_root=True) and not dry_run:
        print("Updated extraPaths and mypy_path from path dependencies.")
    return 0


def main() -> int:
    """Entry point."""
    doc_desc = (__doc__ or "").split("\n\n")[1] if "\n\n" in (__doc__ or "") else ""
    parser = argparse.ArgumentParser(description=doc_desc)
    _ = parser.add_argument(
        "--dry-run", action="store_true", help="Print would-be changes only"
    )
    _ = parser.add_argument(
        "--project",
        action="append",
        dest="projects",
        metavar="DIR",
        help="Project directory to sync (can be repeated); default is workspace root only",
    )
    args = parser.parse_args()
    project_dirs: list[Path] | None = None
    if args.projects:
        project_dirs = [ROOT / p for p in args.projects]
    return _sync_extra_paths(dry_run=args.dry_run, project_dirs=project_dirs)


if __name__ == "__main__":
    sys.exit(main())
