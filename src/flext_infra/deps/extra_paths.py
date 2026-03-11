"""Synchronize pyright and mypy extraPaths from path dependencies."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from tomlkit.toml_document import TOMLDocument

from flext_core import r
from flext_infra import FlextInfraUtilitiesPaths, FlextInfraUtilitiesToml, c, output, t
from flext_infra.deps.path_sync import extract_dep_name

_resolver = FlextInfraUtilitiesPaths()
_root_result = _resolver.workspace_root_from_file(__file__)
ROOT = _root_result.value_or(Path(__file__).resolve().parents[4])


class FlextInfraExtraPathsManager:
    """Manager for synchronizing pyright and mypy extraPaths from path dependencies."""

    def __init__(self) -> None:
        """Initialize the extra paths manager with path resolver and TOML service."""
        super().__init__()
        self.resolver = FlextInfraUtilitiesPaths()
        self.toml = FlextInfraUtilitiesToml()


_DICT_ADAPTER = TypeAdapter(dict[str, t.ContainerValue])
_LIST_ADAPTER = TypeAdapter(list[t.ContainerValue])


def _as_container_dict(value: t.ContainerValue | None) -> dict[str, t.ContainerValue]:
    """Validate and normalize a mapping-like value to dict[str, ContainerValue]."""
    if value is None:
        return {}
    try:
        return _DICT_ADAPTER.validate_python(value)
    except ValidationError:
        return {}


def _as_container_list(value: t.ContainerValue | None) -> list[t.ContainerValue]:
    """Validate and normalize a list-like value to list[ContainerValue]."""
    if value is None:
        return []
    try:
        return _LIST_ADAPTER.validate_python(value)
    except ValidationError:
        return []


def _as_string_list(value: t.ContainerValue | None) -> list[str]:
    """Normalize sequence-like values to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return []
    if isinstance(value, Sequence) and (not isinstance(value, Mapping)):
        return [str(item) for item in value]
    return [str(item) for item in _as_container_list(value)]


def _doc_as_container_dict(doc: TOMLDocument) -> dict[str, t.ContainerValue]:
    """Unwrap and validate TOML document as container dict."""
    try:
        return _DICT_ADAPTER.validate_python(doc.unwrap())
    except ValidationError:
        return {}


def path_dep_paths_pep621(doc: TOMLDocument) -> list[str]:
    """Extract path dependency paths from PEP 621 project.dependencies."""
    doc_dict = _doc_as_container_dict(doc)
    project_dict = _as_container_dict(doc_dict.get(c.Infra.Toml.PROJECT))
    if not project_dict:
        return []
    deps_list = _as_container_list(project_dict.get(c.Infra.Toml.DEPENDENCIES))
    paths: list[str] = []
    for item in deps_list:
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


def path_dep_paths_poetry(doc: TOMLDocument) -> list[str]:
    """Extract path dependency paths from Poetry tool.poetry.dependencies."""
    doc_dict = _doc_as_container_dict(doc)
    tool_dict = _as_container_dict(doc_dict.get(c.Infra.Toml.TOOL))
    if not tool_dict:
        return []
    poetry_dict = _as_container_dict(tool_dict.get(c.Infra.Toml.POETRY))
    if not poetry_dict:
        return []
    deps_dict = _as_container_dict(poetry_dict.get(c.Infra.Toml.DEPENDENCIES))
    if not deps_dict:
        return []
    paths: list[str] = []
    for val in deps_dict.values():
        if isinstance(val, Mapping) and c.Infra.Toml.PATH in val:
            val_dict = _as_container_dict(val)
            dep_path = val_dict[c.Infra.Toml.PATH]
            if isinstance(dep_path, str) and dep_path:
                dep_path = dep_path.strip()
                if dep_path.startswith("./"):
                    dep_path = dep_path[2:].strip()
                if dep_path:
                    paths.append(dep_path)
    return sorted(set(paths))


def path_dep_paths(doc: TOMLDocument) -> list[str]:
    """Combine PEP 621 and Poetry path dependencies."""
    return sorted(set(path_dep_paths_pep621(doc) + path_dep_paths_poetry(doc)))


def get_dep_paths(doc: TOMLDocument, *, is_root: bool = False) -> list[str]:
    """Resolve path dependencies to src directory paths."""
    raw_paths = path_dep_paths(doc)
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
    """Synchronize pyright and mypy paths for single pyproject.toml."""
    if not pyproject_path.exists():
        return r[bool].ok(False)
    toml_service = FlextInfraUtilitiesToml()
    doc_result = toml_service.read_document(pyproject_path)
    if doc_result.is_failure:
        return r[bool].fail(doc_result.error or f"failed to read {pyproject_path}")
    doc = doc_result.value
    dep_paths = get_dep_paths(doc, is_root=is_root)
    pyright_extra = (
        c.Infra.Deps.PYRIGHT_BASE_ROOT + dep_paths
        if is_root
        else c.Infra.Deps.PYRIGHT_BASE_PROJECT + dep_paths
    )
    mypy_path = (
        c.Infra.Deps.MYPY_BASE_ROOT + dep_paths
        if is_root
        else c.Infra.Deps.MYPY_BASE_PROJECT + dep_paths
    )
    doc_dict = _doc_as_container_dict(doc)
    tool_dict = _as_container_dict(doc_dict.get(c.Infra.Toml.TOOL))
    if not tool_dict:
        return r[bool].ok(False)
    pyright_dict = _as_container_dict(tool_dict.get(c.Infra.Toml.PYRIGHT))
    if not pyright_dict:
        return r[bool].ok(False)
    mypy_dict = _as_container_dict(tool_dict.get(c.Infra.Toml.MYPY))
    pyrefly_dict = _as_container_dict(tool_dict.get(c.Infra.Toml.PYREFLY))
    changed = False
    current_pyright = _as_string_list(pyright_dict.get("extraPaths"))
    if current_pyright != pyright_extra:
        pyright_dict["extraPaths"] = pyright_extra
        changed = True
    if mypy_dict:
        current_mypy = _as_string_list(mypy_dict.get("mypy_path"))
        if current_mypy != mypy_path:
            mypy_dict["mypy_path"] = mypy_path
            tool_dict[c.Infra.Toml.MYPY] = mypy_dict
            changed = True
    if not is_root and pyrefly_dict:
        base_search: list[str] = ["."] + dep_paths
        current_search = _as_string_list(pyrefly_dict.get(c.Infra.Toml.SEARCH_PATH))
        seen: set[str] = set(base_search)
        for path_value in current_search:
            if path_value not in seen:
                base_search.append(path_value)
                seen.add(path_value)
        if base_search != current_search:
            pyrefly_dict[c.Infra.Toml.SEARCH_PATH] = base_search
            tool_dict[c.Infra.Toml.PYREFLY] = pyrefly_dict
            changed = True
    tool_dict[c.Infra.Toml.PYRIGHT] = pyright_dict
    doc[c.Infra.Toml.TOOL] = tool_dict
    if changed and (not dry_run):
        write_result = toml_service.write_document(pyproject_path, doc)
        if write_result.is_failure:
            return r[bool].fail(
                write_result.error or f"failed to write {pyproject_path}",
            )
    return r[bool].ok(changed)


def sync_extra_paths(
    *,
    dry_run: bool = False,
    project_dirs: list[Path] | None = None,
) -> r[int]:
    """Synchronize extraPaths and mypy_path across projects."""
    if project_dirs:
        for project_dir in project_dirs:
            pyproject = project_dir / c.Infra.Files.PYPROJECT_FILENAME
            sync_result = sync_one(
                pyproject,
                dry_run=dry_run,
                is_root=project_dir == ROOT,
            )
            if sync_result.is_failure:
                return r[int].fail(sync_result.error or f"sync failed for {pyproject}")
            if sync_result.value and (not dry_run):
                output.info(f"Updated {pyproject}")
        return r[int].ok(0)
    pyproject = ROOT / c.Infra.Files.PYPROJECT_FILENAME
    if not pyproject.exists():
        return r[int].fail(f"Missing {pyproject}")
    sync_result = sync_one(pyproject, dry_run=dry_run, is_root=True)
    if sync_result.is_failure:
        return r[int].fail(sync_result.error or f"sync failed for {pyproject}")
    if sync_result.value and (not dry_run):
        output.info("Updated extraPaths and mypy_path from path dependencies.")
    return r[int].ok(0)


def main() -> int:
    """Execute extra paths synchronization from command line."""
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
    output.error(result.error or "sync failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
__all__ = [
    "ROOT",
    "get_dep_paths",
    "main",
    "path_dep_paths",
    "path_dep_paths_pep621",
    "path_dep_paths_poetry",
    "sync_extra_paths",
    "sync_one",
]
