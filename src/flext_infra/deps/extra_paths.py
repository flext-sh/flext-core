"""Synchronize pyright and mypy extraPaths from path dependencies."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from tomlkit.items import Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import r
from flext_infra import (
    FlextInfraUtilitiesPaths,
    FlextInfraUtilitiesToml,
    c,
    u,
)
from flext_infra.deps.path_sync import FlextInfraDependencyPathSync


class FlextInfraExtraPathsManager:
    """Manager for synchronizing pyright and mypy extraPaths from path dependencies."""

    ROOT = u.Infra.resolve_workspace_root(__file__)
    _STRING_LIST_ADAPTER: TypeAdapter[list[str]] = TypeAdapter(list[str])

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize the extra paths manager with path resolver and TOML service."""
        super().__init__()
        self.root = workspace_root or self.ROOT
        self.resolver = FlextInfraUtilitiesPaths()
        self.toml = FlextInfraUtilitiesToml()

    @staticmethod
    def _table_get(container: TOMLDocument | Table, key: str) -> Item | None:
        if key not in container:
            return None
        raw_item = container[key]
        if isinstance(raw_item, Item):
            return raw_item
        return None

    @staticmethod
    def _as_table(value: Item | None) -> Table | None:
        if not isinstance(value, Table):
            return None
        return value

    @staticmethod
    def _as_string_list(value: Item | None) -> list[str]:
        """Normalize sequence-like values to a list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return []
        try:
            return FlextInfraExtraPathsManager._STRING_LIST_ADAPTER.validate_python(
                value
            )
        except ValidationError:
            return []

    def path_dep_paths_pep621(self, doc: TOMLDocument) -> list[str]:
        """Extract path dependency paths from PEP 621 project.dependencies."""
        project_table = self._as_table(self._table_get(doc, c.Infra.Toml.PROJECT))
        if project_table is None:
            return []
        deps_list = self._table_get(project_table, c.Infra.Toml.DEPENDENCIES)
        deps_items = self._as_string_list(deps_list)
        paths: list[str] = []
        for item in deps_items:
            if " @ " not in item:
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

    def path_dep_paths_poetry(self, doc: TOMLDocument) -> list[str]:
        """Extract path dependency paths from Poetry tool.poetry.dependencies."""
        tool_table = self._as_table(self._table_get(doc, c.Infra.Toml.TOOL))
        if tool_table is None:
            return []
        poetry_table = self._as_table(self._table_get(tool_table, c.Infra.Toml.POETRY))
        if poetry_table is None:
            return []
        deps_table = self._as_table(
            self._table_get(poetry_table, c.Infra.Toml.DEPENDENCIES)
        )
        if deps_table is None:
            return []
        paths: list[str] = []
        for dep_key in deps_table:
            dep_table = self._as_table(self._table_get(deps_table, dep_key))
            if dep_table is None:
                continue
            dep_path = self._table_get(dep_table, c.Infra.Toml.PATH)
            if isinstance(dep_path, str) and dep_path:
                dep_path = dep_path.strip()
                if dep_path.startswith("./"):
                    dep_path = dep_path[2:].strip()
                if dep_path:
                    paths.append(dep_path)
        return sorted(set(paths))

    def path_dep_paths(self, doc: TOMLDocument) -> list[str]:
        """Combine PEP 621 and Poetry path dependencies."""
        return sorted(
            set(self.path_dep_paths_pep621(doc) + self.path_dep_paths_poetry(doc))
        )

    def get_dep_paths(self, doc: TOMLDocument, *, is_root: bool = False) -> list[str]:
        """Resolve path dependencies to src directory paths."""
        raw_paths = self.path_dep_paths(doc)
        resolved: list[str] = []
        for path_value in raw_paths:
            if not path_value:
                continue
            name = FlextInfraDependencyPathSync.extract_dep_name(path_value)
            if is_root:
                resolved.append(f"{name}/src")
            else:
                resolved.append(f"../{name}/src")
        return resolved

    def sync_one(
        self,
        pyproject_path: Path,
        *,
        dry_run: bool = False,
        is_root: bool = False,
    ) -> r[bool]:
        """Synchronize pyright and mypy paths for single pyproject.toml."""
        if not pyproject_path.exists():
            return r[bool].fail(f"pyproject not found: {pyproject_path}")
        doc_result = self.toml.read_document(pyproject_path)
        if doc_result.is_failure:
            return r[bool].fail(doc_result.error or f"failed to read {pyproject_path}")
        doc: TOMLDocument = doc_result.value
        dep_paths = self.get_dep_paths(doc, is_root=is_root)
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
        tool_table = self._as_table(self._table_get(doc, c.Infra.Toml.TOOL))
        if tool_table is None:
            return r[bool].fail(f"no [tool] section in {pyproject_path}")
        pyright_table = self._as_table(
            self._table_get(tool_table, c.Infra.Toml.PYRIGHT)
        )
        if pyright_table is None:
            return r[bool].fail(f"no [tool.pyright] section in {pyproject_path}")
        mypy_table = self._as_table(self._table_get(tool_table, c.Infra.Toml.MYPY))
        pyrefly_table = self._as_table(
            self._table_get(tool_table, c.Infra.Toml.PYREFLY)
        )
        changed = False
        current_pyright = self._as_string_list(
            self._table_get(pyright_table, "extraPaths")
        )
        if current_pyright != pyright_extra:
            pyright_table["extraPaths"] = pyright_extra
            changed = True
        if mypy_table is not None:
            current_mypy = self._as_string_list(
                self._table_get(mypy_table, "mypy_path")
            )
            if current_mypy != mypy_path:
                mypy_table["mypy_path"] = mypy_path
                tool_table[c.Infra.Toml.MYPY] = mypy_table
                changed = True
        if not is_root and pyrefly_table is not None:
            base_search: list[str] = ["."] + dep_paths
            current_search = self._as_string_list(
                self._table_get(pyrefly_table, c.Infra.Toml.SEARCH_PATH)
            )
            seen: set[str] = set(base_search)
            for path_value in current_search:
                if path_value not in seen:
                    base_search.append(path_value)
                    seen.add(path_value)
            if base_search != current_search:
                pyrefly_table[c.Infra.Toml.SEARCH_PATH] = base_search
                tool_table[c.Infra.Toml.PYREFLY] = pyrefly_table
                changed = True
        tool_table[c.Infra.Toml.PYRIGHT] = pyright_table
        doc[c.Infra.Toml.TOOL] = tool_table
        if changed and (not dry_run):
            write_result = self.toml.write_document(pyproject_path, doc)
            if write_result.is_failure:
                return r[bool].fail(
                    write_result.error or f"failed to write {pyproject_path}",
                )
        return r[bool].ok(changed)

    @staticmethod
    def sync_one_path(
        pyproject_path: Path,
        *,
        dry_run: bool = False,
        is_root: bool = False,
    ) -> r[bool]:
        """Synchronize a single pyproject.toml path without instance wiring."""
        manager = FlextInfraExtraPathsManager()
        return manager.sync_one(pyproject_path, dry_run=dry_run, is_root=is_root)

    def sync_extra_paths(
        self,
        *,
        dry_run: bool = False,
        project_dirs: list[Path] | None = None,
    ) -> r[int]:
        """Synchronize extraPaths and mypy_path across projects."""
        if project_dirs:
            for project_dir in project_dirs:
                pyproject = project_dir / c.Infra.Files.PYPROJECT_FILENAME
                sync_result = self.sync_one(
                    pyproject,
                    dry_run=dry_run,
                    is_root=project_dir == self.root,
                )
                if sync_result.is_failure:
                    return r[int].fail(
                        sync_result.error or f"sync failed for {pyproject}"
                    )
                if sync_result.value and (not dry_run):
                    u.Infra.info(f"Updated {pyproject}")
            return r[int].ok(0)
        pyproject = self.root / c.Infra.Files.PYPROJECT_FILENAME
        if not pyproject.exists():
            return r[int].fail(f"Missing {pyproject}")
        sync_result = self.sync_one(pyproject, dry_run=dry_run, is_root=True)
        if sync_result.is_failure:
            return r[int].fail(sync_result.error or f"sync failed for {pyproject}")
        if sync_result.value and (not dry_run):
            u.Infra.info("Updated extraPaths and mypy_path from path dependencies.")
        return r[int].ok(0)


def _resolve_project_dirs(
    cli: u.Infra.CliArgs,
) -> list[Path] | None:
    return cli.project_dirs()


def main(argv: list[str] | None = None) -> int:
    """Execute extra paths synchronization from command line."""
    parser = u.Infra.create_parser(
        "flext-infra deps extra-paths",
        "Synchronize pyright and mypy extraPaths from path dependencies",
        include_apply=True,
        include_project=True,
    )
    args = parser.parse_args(argv)
    cli = u.Infra.resolve(args)
    manager = FlextInfraExtraPathsManager(workspace_root=cli.workspace)
    project_dirs = _resolve_project_dirs(cli)
    result = manager.sync_extra_paths(dry_run=cli.dry_run, project_dirs=project_dirs)
    if result.is_success:
        return result.value
    u.Infra.error(result.error or "sync failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "FlextInfraExtraPathsManager",
]
