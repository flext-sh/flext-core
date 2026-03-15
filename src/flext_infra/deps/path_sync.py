"""Rewrite internal FLEXT dependency paths for workspace or standalone mode.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue
from tomlkit.items import Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import FlextLogger, r
from flext_infra import (
    FlextInfraUtilitiesDiscovery,
    FlextInfraUtilitiesToml,
    c,
    m,
    u,
)


class _FlextInfraOutput:
    """Output handler for dependency path sync operations."""

    def info(self, message: str) -> None:
        """Log an info message using the infrastructure logger.

        Args:
            message: The message to log.

        """
        u.Infra.info(message)


output: _FlextInfraOutput = _FlextInfraOutput()


class FlextInfraDependencyPathSync:
    """Rewrite internal FLEXT dependency paths for workspace or standalone mode."""

    ROOT = u.Infra.resolve_workspace_root(__file__)
    _log = FlextLogger.create_module_logger(__name__)

    def __init__(self) -> None:
        """Initialize the dependency path sync service with TOML service."""
        self._root = self.ROOT
        self._discovery = FlextInfraUtilitiesDiscovery()
        self._toml = FlextInfraUtilitiesToml()

    def set_workspace_root(self, workspace_root: Path) -> None:
        """Configure workspace root for path resolution."""
        self._root = workspace_root

    def _discover_projects(self) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        try:
            return self._discovery.discover_projects(self._root)
        except TypeError:
            return FlextInfraUtilitiesDiscovery.discover_projects(self._root)

    def _read_document(self, path: Path) -> r[TOMLDocument]:
        try:
            return self._toml.read_document(path)
        except TypeError:
            return FlextInfraUtilitiesToml.read_document(path)

    def _write_document(self, path: Path, doc: TOMLDocument) -> r[bool]:
        try:
            return self._toml.write_document(path, doc)
        except TypeError:
            return FlextInfraUtilitiesToml.write_document(path, doc)

    @staticmethod
    def detect_mode(project_root: Path) -> str:
        """Detect workspace or standalone mode from project structure."""
        for candidate in (project_root, *project_root.parents):
            if (candidate / c.Infra.Files.GITMODULES).exists():
                return c.Infra.ReportKeys.WORKSPACE
        return "standalone"

    @staticmethod
    def extract_dep_name(raw_path: str) -> str:
        """Extract dependency name from path string."""
        normalized = raw_path.strip().lstrip("/").removeprefix("./")
        for prefix in (f"{c.Infra.Deps.FLEXT_DEPS_DIR}/", "../"):
            normalized = normalized.removeprefix(prefix)
        return normalized

    @staticmethod
    def _target_path(dep_name: str, *, is_root: bool, mode: str) -> str:
        """Compute target path for dependency based on mode and location."""
        if mode == c.Infra.ReportKeys.WORKSPACE:
            return dep_name if is_root else f"../{dep_name}"
        return f"{c.Infra.Deps.FLEXT_DEPS_DIR}/{dep_name}"

    @staticmethod
    def _mapping_str_value(
        mapping: Table | Mapping[str, JsonValue], key: str
    ) -> str | None:
        if key not in mapping:
            return None
        value = mapping[key]
        if isinstance(value, str) and value:
            return value
        return None

    @staticmethod
    def _extract_requirement_name(entry: str) -> str | None:
        """Extract requirement name from PEP 621 dependency entry."""
        if " @ " in entry:
            match = c.Infra.Deps.PEP621_PATH_DEP_RE.match(entry)
            if match:
                return match.group("name")
        match = c.Infra.Deps.PEP621_NAME_RE.match(entry)
        if not match:
            return None
        return match.group("name")

    @staticmethod
    def _table_get(
        container: TOMLDocument | Table,
        key: str,
    ) -> Item | None:
        if key not in container:
            return None
        item = container[key]
        return item if isinstance(item, Item) else None

    def _rewrite_pep621(
        self,
        doc: TOMLDocument,
        *,
        is_root: bool,
        mode: str,
        internal_names: set[str],
    ) -> list[str]:
        project_raw = self._table_get(doc, c.Infra.Toml.PROJECT)
        if not isinstance(project_raw, Table):
            return []
        project_section: Table = project_raw
        deps: list[str] = u.Infra.as_string_list(
            self._table_get(project_section, c.Infra.Toml.DEPENDENCIES)
        )
        if not deps:
            return []
        changes: list[str] = []
        updated_deps: list[str] = []
        for item_raw in deps:
            item = item_raw
            marker = ""
            requirement_part = item
            if ";" in item:
                requirement_part, marker_part = item.split(";", 1)
                marker = f" ;{marker_part}"
            dep_name = self._extract_requirement_name(requirement_part)
            if not dep_name or dep_name not in internal_names:
                updated_deps.append(item)
                continue
            if " @ " in requirement_part:
                match = c.Infra.Deps.PEP621_PATH_DEP_RE.match(requirement_part)
                if not match:
                    updated_deps.append(item)
                    continue
                raw_path = match.group("path").strip()
                dep_name = self.extract_dep_name(raw_path)
            new_path = self._target_path(dep_name, is_root=is_root, mode=mode)
            path_prefix = "./" if is_root else ""
            new_entry = f"{dep_name} @ file:{path_prefix}{new_path}{marker}"
            if item != new_entry:
                changes.append(f"  PEP621: {item} -> {new_entry}")
                updated_deps.append(new_entry)
            else:
                updated_deps.append(item)
        if changes:
            project_section[c.Infra.Toml.DEPENDENCIES] = updated_deps
        return changes

    @staticmethod
    def _rewrite_poetry(doc: TOMLDocument, *, is_root: bool, mode: str) -> list[str]:
        tool_raw = FlextInfraDependencyPathSync._table_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool_raw, Table):
            return []
        tool_section: Table = tool_raw
        poetry_raw = FlextInfraDependencyPathSync._table_get(
            tool_section, c.Infra.Toml.POETRY
        )
        if not isinstance(poetry_raw, Table):
            return []
        poetry_section: Table = poetry_raw
        deps_raw = FlextInfraDependencyPathSync._table_get(
            poetry_section, c.Infra.Toml.DEPENDENCIES
        )
        if not isinstance(deps_raw, Table):
            return []
        deps: Table = deps_raw
        changes: list[str] = []
        for dep_key_raw in deps:
            dep_key = dep_key_raw
            value = deps[dep_key_raw]
            if not isinstance(value, Table) or c.Infra.Toml.PATH not in value:
                continue
            value_map: Table = value
            raw_path = value_map[c.Infra.Toml.PATH]
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            dep_name = FlextInfraDependencyPathSync.extract_dep_name(raw_path)
            new_path = FlextInfraDependencyPathSync._target_path(
                dep_name, is_root=is_root, mode=mode
            )
            if raw_path != new_path:
                changes.append(
                    f"  Poetry: {dep_key}.path = {raw_path!r} -> {new_path!r}"
                )
                value_map[c.Infra.Toml.PATH] = new_path
        return changes

    def rewrite_dep_paths(
        self,
        pyproject_path: Path,
        *,
        mode: str,
        internal_names: set[str],
        is_root: bool = False,
        dry_run: bool = False,
    ) -> r[list[str]]:
        """Rewrite PEP 621 and Poetry dependency paths."""
        doc_result = self._read_document(pyproject_path)
        if doc_result.is_failure:
            return r[list[str]].fail(doc_result.error or "failed to read TOML document")
        doc: TOMLDocument = doc_result.value
        changes = self._rewrite_pep621(
            doc,
            is_root=is_root,
            mode=mode,
            internal_names=internal_names,
        )
        changes += self._rewrite_poetry(doc, is_root=is_root, mode=mode)
        if changes and (not dry_run):
            write_result = self._write_document(pyproject_path, doc)
            if write_result.is_failure:
                return r[list[str]].fail(write_result.error or "failed to write TOML")
        return r[list[str]].ok(changes)

    def run(self, *, cli: u.Infra.CliArgs, mode: str) -> int:
        """Execute path synchronization for the given CLI arguments."""
        self.set_workspace_root(cli.workspace)
        dry_run = cli.dry_run
        selected_projects: list[str] = cli.project_names() or []

        if mode == "auto":
            mode = self.detect_mode(self._root)
            output.info(f"[sync-dep-paths] auto-detected mode: {mode}")

        total_changes = 0
        internal_names: set[str] = set()
        root_pyproject = self._root / c.Infra.Files.PYPROJECT_FILENAME

        if root_pyproject.exists():
            root_data_result = self._read_document(root_pyproject)
            if root_data_result.is_success:
                root_data: TOMLDocument = root_data_result.value
                root_project = self._table_get(root_data, c.Infra.Toml.PROJECT)
                if isinstance(root_project, Mapping):
                    root_name = self._mapping_str_value(root_project, c.Infra.Toml.NAME)
                    if root_name is not None:
                        internal_names.add(root_name)

        if not selected_projects and root_pyproject.exists():
            changes_result = self.rewrite_dep_paths(
                root_pyproject,
                mode=mode,
                internal_names=internal_names,
                is_root=True,
                dry_run=dry_run,
            )
            if changes_result.is_failure:
                root_error = changes_result.error or "sync_dep_paths_root_failed"
                self._log.error(
                    "sync_dep_paths_root_failed",
                    pyproject=str(root_pyproject),
                    error=root_error,
                )
                return 1
            changes: list[str] = changes_result.value
            if changes:
                prefix = "[DRY-RUN] " if dry_run else ""
                output.info(f"{prefix}{root_pyproject}:")
                for change in changes:
                    output.info(change)
                total_changes += len(changes)

        discover_result = self._discover_projects()
        if discover_result.is_failure:
            discovery_error = discover_result.error or "sync_dep_paths_discovery_failed"
            self._log.error(
                "sync_dep_paths_discovery_failed",
                root=str(self._root),
                error=discovery_error,
            )
            return 1

        projects_list: list[m.Infra.Workspace.ProjectInfo] = discover_result.value
        all_project_dirs = [project.path for project in projects_list]
        if selected_projects:
            project_dirs = [self._root / project for project in selected_projects]
        else:
            project_dirs = all_project_dirs

        for project_dir in all_project_dirs:
            pyproject = project_dir / c.Infra.Files.PYPROJECT_FILENAME
            if not pyproject.exists():
                continue
            data_result = self._read_document(pyproject)
            if data_result.is_failure:
                continue
            project_data: TOMLDocument = data_result.value
            project_obj = self._table_get(project_data, c.Infra.Toml.PROJECT)
            if not isinstance(project_obj, Mapping):
                continue
            project_name = self._mapping_str_value(project_obj, c.Infra.Toml.NAME)
            if project_name is not None:
                internal_names.add(project_name)

        for project_dir in sorted(project_dirs):
            pyproject = project_dir / c.Infra.Files.PYPROJECT_FILENAME
            if not pyproject.exists():
                continue
            changes_result = self.rewrite_dep_paths(
                pyproject,
                mode=mode,
                internal_names=internal_names,
                is_root=False,
                dry_run=dry_run,
            )
            if changes_result.is_failure:
                project_error = changes_result.error or "sync_dep_paths_project_failed"
                self._log.error(
                    "sync_dep_paths_project_failed",
                    pyproject=str(pyproject),
                    error=project_error,
                )
                return 1
            project_changes: list[str] = changes_result.value
            if project_changes:
                prefix = "[DRY-RUN] " if dry_run else ""
                output.info(f"{prefix}{pyproject}:")
                for change in project_changes:
                    output.info(change)
                total_changes += len(project_changes)

        if total_changes == 0:
            output.info(
                "[sync-dep-paths] No changes needed - all paths already match target mode."
            )
        else:
            action = "would change" if dry_run else "changed"
            output.info(f"[sync-dep-paths] {action} {total_changes} path(s).")
        return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for path sync CLI."""
    parser = u.Infra.create_parser(
        "flext-infra deps path-sync",
        "Rewrite internal FLEXT dependency paths for workspace/standalone mode.",
        include_apply=True,
        include_project=True,
    )
    _ = parser.add_argument(
        "--mode",
        choices=["workspace", "standalone", "auto"],
        default="auto",
        help="Target mode (default: auto-detect)",
    )
    args = parser.parse_args(argv)
    cli = u.Infra.resolve(args)
    return FlextInfraDependencyPathSync().run(cli=cli, mode=args.mode)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["FlextInfraDependencyPathSync"]
