"""Rewrite internal FLEXT dependency paths for workspace or standalone mode.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from tomlkit.items import Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import FlextLogger, r
from flext_infra import (
    FlextInfraUtilitiesDiscovery,
    FlextInfraUtilitiesPaths,
    FlextInfraUtilitiesToml,
    c,
    m,
    t,
    u,
)

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraDependencyPathSync:
    """Rewrite internal FLEXT dependency paths for workspace or standalone mode."""

    _resolver = FlextInfraUtilitiesPaths()
    _root_result = _resolver.workspace_root_from_file(__file__)
    ROOT: Path = (
        _root_result.value
        if _root_result.is_success
        else Path(__file__).resolve().parents[4]
    )

    def __init__(self) -> None:
        """Initialize the dependency path sync service with TOML service."""
        self._toml = FlextInfraUtilitiesToml()

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

    def _target_path(self, dep_name: str, *, is_root: bool, mode: str) -> str:
        """Compute target path for dependency based on mode and location."""
        if mode == c.Infra.ReportKeys.WORKSPACE:
            return dep_name if is_root else f"../{dep_name}"
        return f"{c.Infra.Deps.FLEXT_DEPS_DIR}/{dep_name}"

    @staticmethod
    def _mapping_str_value(
        mapping: Table | t.ConfigurationMapping, key: str
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
    ) -> Item | t.ContainerValue | None:
        if key not in container:
            return None
        return container[key]

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
        deps_raw = self._table_get(project_section, c.Infra.Toml.DEPENDENCIES)
        if not isinstance(deps_raw, list):
            return []
        deps_values: list[t.ContainerValue] = deps_raw
        deps_filtered: list[str] = [
            entry for entry in deps_values if isinstance(entry, str)
        ]
        deps: list[str] = deps_filtered
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
                continue
            if " @ " in requirement_part:
                match = c.Infra.Deps.PEP621_PATH_DEP_RE.match(requirement_part)
                if not match:
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

    def _rewrite_poetry(
        self, doc: TOMLDocument, *, is_root: bool, mode: str
    ) -> list[str]:
        tool_raw = self._table_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool_raw, Table):
            return []
        tool_section: Table = tool_raw
        poetry_raw = self._table_get(tool_section, c.Infra.Toml.POETRY)
        if not isinstance(poetry_raw, Table):
            return []
        poetry_section: Table = poetry_raw
        deps_raw = self._table_get(poetry_section, c.Infra.Toml.DEPENDENCIES)
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
            dep_name = self.extract_dep_name(raw_path)
            new_path = self._target_path(dep_name, is_root=is_root, mode=mode)
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
        doc_result = self._toml.read_document(pyproject_path)
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
            write_result = self._toml.write_document(pyproject_path, doc)
            if write_result.is_failure:
                return r[list[str]].fail(write_result.error or "failed to write TOML")
        return r[list[str]].ok(changes)

    def run(self, argv: list[str] | None = None) -> int:
        """Execute dependency path rewriting from command line."""
        parser = argparse.ArgumentParser(
            description="Rewrite internal FLEXT dependency paths for workspace/standalone mode.",
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
        args = parser.parse_args(argv)
        mode = args.mode

        if mode == "auto":
            mode = self.detect_mode(self.ROOT)
            u.Infra.info(f"[sync-dep-paths] auto-detected mode: {mode}")

        total_changes = 0
        internal_names: set[str] = set()
        root_pyproject = self.ROOT / c.Infra.Files.PYPROJECT_FILENAME

        if root_pyproject.exists():
            root_data_result = self._toml.read_document(root_pyproject)
            if root_data_result.is_success:
                root_data: TOMLDocument = root_data_result.value
                root_project = self._table_get(root_data, c.Infra.Toml.PROJECT)
                if isinstance(root_project, Mapping):
                    root_name = self._mapping_str_value(root_project, c.Infra.Toml.NAME)
                    if root_name is not None:
                        internal_names.add(root_name)

        if not args.projects and root_pyproject.exists():
            changes_result = self.rewrite_dep_paths(
                root_pyproject,
                mode=mode,
                internal_names=internal_names,
                is_root=True,
                dry_run=args.dry_run,
            )
            if changes_result.is_failure:
                logger.error(
                    "sync_dep_paths_root_failed",
                    pyproject=str(root_pyproject),
                    error=changes_result.error,
                )
                return 1
            changes: list[str] = changes_result.value
            if changes:
                prefix = "[DRY-RUN] " if args.dry_run else ""
                u.Infra.info(f"{prefix}{root_pyproject}:")
                for change in changes:
                    u.Infra.info(change)
                total_changes += len(changes)

        discover_result = FlextInfraUtilitiesDiscovery().discover_projects(self.ROOT)
        if discover_result.is_failure:
            logger.error(
                "sync_dep_paths_discovery_failed",
                root=str(self.ROOT),
                error=discover_result.error,
            )
            return 1

        projects_list: list[m.Infra.Workspace.ProjectInfo] = discover_result.value
        all_project_dirs = [project.path for project in projects_list]
        if args.projects:
            project_dirs = [self.ROOT / project for project in args.projects]
        else:
            project_dirs = all_project_dirs

        for project_dir in all_project_dirs:
            pyproject = project_dir / c.Infra.Files.PYPROJECT_FILENAME
            if not pyproject.exists():
                continue
            data_result = self._toml.read_document(pyproject)
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
                dry_run=args.dry_run,
            )
            if changes_result.is_failure:
                logger.error(
                    "sync_dep_paths_project_failed",
                    pyproject=str(pyproject),
                    error=changes_result.error,
                )
                continue
            project_changes: list[str] = changes_result.value
            if project_changes:
                prefix = "[DRY-RUN] " if args.dry_run else ""
                u.Infra.info(f"{prefix}{pyproject}:")
                for change in project_changes:
                    u.Infra.info(change)
                total_changes += len(project_changes)

        if total_changes == 0:
            u.Infra.info(
                "[sync-dep-paths] No changes needed - all paths already match target mode."
            )
        else:
            action = "would change" if args.dry_run else "changed"
            u.Infra.info(f"[sync-dep-paths] {action} {total_changes} path(s).")
        return 0


def main() -> int:
    """Entry point for path sync CLI."""
    return FlextInfraDependencyPathSync().run()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FlextInfraDependencyPathSync", "main"]
