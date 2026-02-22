from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tomlkit
from flext_core import r
from tomlkit.toml_document import TOMLDocument

from flext_infra.constants import ic
from flext_infra.deps.path_sync import extract_dep_name
from flext_infra.paths import PathResolver
from flext_infra.toml_io import TomlService

_resolver = PathResolver()
_root_result = _resolver.workspace_root_from_file(__file__)
ROOT = (
    _root_result.value
    if _root_result.is_success
    else Path(__file__).resolve().parents[4]
)

PYRIGHT_BASE_ROOT = ["scripts", "src", "typings", "typings/generated"]
MYPY_BASE_ROOT = ["typings", "typings/generated", "src"]

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
            dep_path = val["path"]
            if isinstance(dep_path, str) and dep_path:
                dep_path = dep_path.strip()
                if dep_path.startswith("./"):
                    dep_path = dep_path[2:].strip()
                if dep_path:
                    paths.append(dep_path)
    return sorted(set(paths))


def _path_dep_paths(doc: TOMLDocument) -> list[str]:
    return sorted(set(_path_dep_paths_pep621(doc) + _path_dep_paths_poetry(doc)))


def get_dep_paths(doc: TOMLDocument, *, is_root: bool = False) -> list[str]:
    raw_paths = _path_dep_paths(doc)
    resolved: list[str] = []
    for path_value in raw_paths:
        if not path_value:
            continue
        name = extract_dep_name(path_value)
        if is_root:
            resolved.append(f"{name}/src")
        else:
            resolved.append(f"../{name}/src")
    return resolved


def sync_one(
    pyproject_path: Path,
    *,
    dry_run: bool = False,
    is_root: bool = False,
) -> r[bool]:
    if not pyproject_path.exists():
        return r[bool].ok(False)

    toml_service = TomlService()
    doc_result = toml_service.read_document(pyproject_path)
    if doc_result.is_failure:
        return r[bool].fail(doc_result.error or f"failed to read {pyproject_path}")
    doc = doc_result.value

    dep_paths = get_dep_paths(doc, is_root=is_root)
    pyright_extra = (
        PYRIGHT_BASE_ROOT + dep_paths if is_root else PYRIGHT_BASE_PROJECT + dep_paths
    )
    mypy_path = MYPY_BASE_ROOT + dep_paths if is_root else MYPY_BASE_PROJECT + dep_paths

    tool = doc.get("tool")
    if not isinstance(tool, dict):
        return r[bool].ok(False)
    pyright = tool.get("pyright")
    mypy = tool.get("mypy")
    if not isinstance(pyright, dict):
        return r[bool].ok(False)

    changed = False
    current_pyright = pyright.get("extraPaths", [])
    if list(current_pyright) != pyright_extra:
        arr = tomlkit.array()
        for path_value in pyright_extra:
            arr.append(path_value)
        pyright["extraPaths"] = arr
        changed = True

    if isinstance(mypy, dict):
        current_mypy = mypy.get("mypy_path", [])
        if list(current_mypy) != mypy_path:
            arr = tomlkit.array()
            for path_value in mypy_path:
                arr.append(path_value)
            mypy["mypy_path"] = arr
            changed = True

    if not is_root:
        pyrefly = tool.get("pyrefly")
        if isinstance(pyrefly, dict):
            base_search = [".."] + dep_paths
            current_search = list(pyrefly.get("search-path", []))
            seen = set(base_search)
            for path_value in current_search:
                if path_value not in seen:
                    base_search.append(path_value)
                    seen.add(path_value)
            if base_search != current_search:
                arr = tomlkit.array()
                for path_value in base_search:
                    arr.append(path_value)
                pyrefly["search-path"] = arr
                changed = True

    if changed and not dry_run:
        write_result = toml_service.write_document(pyproject_path, doc)
        if write_result.is_failure:
            return r[bool].fail(
                write_result.error or f"failed to write {pyproject_path}"
            )

    return r[bool].ok(changed)


def sync_extra_paths(
    *,
    dry_run: bool = False,
    project_dirs: list[Path] | None = None,
) -> r[int]:
    if project_dirs:
        for project_dir in project_dirs:
            pyproject = project_dir / ic.Files.PYPROJECT_FILENAME
            sync_result = sync_one(
                pyproject, dry_run=dry_run, is_root=(project_dir == ROOT)
            )
            if sync_result.is_failure:
                return r[int].fail(sync_result.error or f"sync failed for {pyproject}")
            if sync_result.value and not dry_run:
                print(f"Updated {pyproject}")
        return r[int].ok(0)

    pyproject = ROOT / ic.Files.PYPROJECT_FILENAME
    if not pyproject.exists():
        return r[int].fail(f"Missing {pyproject}")
    sync_result = sync_one(pyproject, dry_run=dry_run, is_root=True)
    if sync_result.is_failure:
        return r[int].fail(sync_result.error or f"sync failed for {pyproject}")
    if sync_result.value and not dry_run:
        print("Updated extraPaths and mypy_path from path dependencies.")
    return r[int].ok(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print would-be changes only",
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
        project_dirs = [ROOT / project for project in args.projects]

    result = sync_extra_paths(dry_run=args.dry_run, project_dirs=project_dirs)
    if result.is_success:
        return result.value
    _ = sys.stderr.write(f"{result.error}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "MYPY_BASE_PROJECT",
    "MYPY_BASE_ROOT",
    "PYRIGHT_BASE_PROJECT",
    "PYRIGHT_BASE_ROOT",
    "ROOT",
    "get_dep_paths",
    "main",
    "sync_extra_paths",
    "sync_one",
]
