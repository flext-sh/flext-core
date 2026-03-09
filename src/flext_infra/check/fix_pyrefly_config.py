"""CLI tool to fix Pyrefly configurations across projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import override

import tomlkit
from pydantic import TypeAdapter
from tomlkit import items

from flext_core import FlextLogger, r, s
from flext_infra import (
    FlextInfraDiscoveryService,
    FlextInfraJsonService,
    FlextInfraPathResolver,
    c,
    t,
)
from flext_infra.output import output

_logger = FlextLogger.create_module_logger(__name__)


class FlextInfraConfigFixer(s[list[str]]):
    """Fix pyrefly configuration across workspace projects."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize pyrefly config fixer."""
        super().__init__()
        self._path_resolver = FlextInfraPathResolver()
        self._discovery = FlextInfraDiscoveryService()
        self._workspace_root = self._resolve_workspace_root(workspace_root)

    @staticmethod
    def _to_array(items_list: list[str]) -> items.Array:
        serialized_result = FlextInfraJsonService().serialize(items_list)
        if serialized_result.is_failure:
            return tomlkit.array()
        inline_doc = tomlkit.parse(f"items = {serialized_result.value}\n")
        arr_raw = inline_doc["items"]
        if not isinstance(arr_raw, items.Array):
            return tomlkit.array()
        arr = arr_raw
        if len(items_list) > 1:
            arr.multiline(True)
        return arr

    @override
    def execute(self) -> r[list[str]]:
        return r[list[str]].fail("Use run() directly")

    def find_pyproject_files(
        self,
        project_paths: list[Path] | None = None,
    ) -> r[list[Path]]:
        """Find pyproject.toml files for selected projects."""
        return self._discovery.find_all_pyproject_files(
            self._workspace_root,
            project_paths=project_paths,
        )

    def process_file(self, path: Path, *, dry_run: bool = False) -> r[list[str]]:
        """Process one pyproject.toml file and apply fixes."""
        try:
            text = path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            doc = tomlkit.parse(text)
            doc_data = doc.unwrap()
        except OSError as exc:
            return r[list[str]].fail(f"failed to read {path}: {exc}")
        except Exception as exc:
            return r[list[str]].fail(f"failed to parse {path}: {exc}")
        tool_data = doc_data.get(c.Infra.Toml.TOOL)
        if not isinstance(tool_data, dict):
            return r[list[str]].ok([])
        typed_tool_data = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
            tool_data,
        )
        pyrefly_data = typed_tool_data.get(c.Infra.Toml.PYREFLY)
        if not isinstance(pyrefly_data, Mapping):
            return r[list[str]].ok([])
        pyrefly: MutableMapping[str, t.ContainerValue] = dict(pyrefly_data)
        all_fixes: list[str] = []
        fixes = self._fix_search_paths_tk(pyrefly, path.parent)
        all_fixes.extend(fixes)
        fixes = self._remove_ignore_sub_config_tk(pyrefly)
        all_fixes.extend(fixes)
        if (
            any("removed ignore" in item for item in all_fixes)
            or path.parent == self._workspace_root
        ):
            fixes = self._ensure_project_excludes_tk(pyrefly)
            all_fixes.extend(fixes)
        if all_fixes and (not dry_run):
            try:
                typed_tool_data[c.Infra.Toml.PYREFLY] = dict(pyrefly)
                doc_data[c.Infra.Toml.TOOL] = typed_tool_data
                new_doc = tomlkit.document()
                for key, value in doc_data.items():
                    new_doc[str(key)] = value
                new_text = new_doc.as_string()
                _ = path.write_text(new_text, encoding=c.Infra.Encoding.DEFAULT)
            except OSError as exc:
                return r[list[str]].fail(f"failed to write {path}: {exc}")
        return r[list[str]].ok(all_fixes)

    def run(
        self,
        projects: Sequence[str],
        *,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> r[list[str]]:
        """Run pyrefly configuration fixes for selected projects."""
        project_paths = [self._resolve_project_path(project) for project in projects]
        files_result = self.find_pyproject_files(project_paths or None)
        if files_result.is_failure:
            return r[list[str]].fail(
                files_result.error or "failed to find pyproject files",
            )
        messages: list[str] = []
        total_fixes = 0
        pyproject_files: list[Path] = files_result.value
        for path in pyproject_files:
            fixes_result = self.process_file(path, dry_run=dry_run)
            if fixes_result.is_failure:
                return r[list[str]].fail(
                    fixes_result.error or f"failed to process {path}",
                )
            fixes: list[str] = fixes_result.value
            if not fixes:
                continue
            total_fixes += len(fixes)
            if verbose:
                try:
                    rel = path.relative_to(self._workspace_root)
                except ValueError:
                    rel = path
                for fix in fixes:
                    line = f"  {('(dry)' if dry_run else '✓')} {rel}: {fix}"
                    _logger.info("pyrefly_config_fix", detail=line)
                    messages.append(line)
        if verbose and total_fixes == 0:
            _logger.info("pyrefly_configs_clean")
        return r[list[str]].ok(messages)

    def _ensure_project_excludes_tk(
        self,
        pyrefly: MutableMapping[str, t.ContainerValue],
    ) -> list[str]:
        fixes: list[str] = []
        excludes = pyrefly.get(c.Infra.Toml.PROJECT_EXCLUDES)
        current: list[str] = []
        if isinstance(excludes, list):
            current = [str(x) for x in excludes]
        stripped_to_add: list[str] = []
        for glob in c.Infra.Check.REQUIRED_EXCLUDES:
            clean_glob = glob.strip('"').strip("'")
            if clean_glob not in current and glob not in current:
                stripped_to_add.append(clean_glob)
        if stripped_to_add:
            updated = sorted(set(current) | set(stripped_to_add))
            pyrefly[c.Infra.Toml.PROJECT_EXCLUDES] = self._to_array(updated)
            fixes.append(f"added {', '.join(stripped_to_add)} to project-excludes")
        return fixes

    def _fix_search_paths_tk(
        self,
        pyrefly: MutableMapping[str, t.ContainerValue],
        project_dir: Path,
    ) -> list[str]:
        fixes: list[str] = []
        search_path = pyrefly.get(c.Infra.Toml.SEARCH_PATH)
        if not isinstance(search_path, list):
            return []
        if project_dir == self._workspace_root:
            new_paths: list[str] = []
            for path_item in search_path:
                if not isinstance(path_item, str):
                    continue
                if path_item == "../typings/generated":
                    new_paths.append("typings/generated")
                    fixes.append(
                        "search-path ../typings/generated -> typings/generated",
                    )
                elif path_item == "../typings":
                    new_paths.append(c.Infra.Directories.TYPINGS)
                    fixes.append("search-path ../typings -> typings")
                else:
                    new_paths.append(path_item)
            if fixes:
                pyrefly[c.Infra.Toml.SEARCH_PATH] = self._to_array(new_paths)
        search_raw = pyrefly.get(c.Infra.Toml.SEARCH_PATH)
        current_paths: list[t.ContainerValue] = (
            list(search_raw) if isinstance(search_raw, list) else []
        )
        nonexistent = [
            path_item
            for path_item in current_paths
            if isinstance(path_item, str) and (not (project_dir / path_item).exists())
        ]
        if nonexistent:
            remaining: list[str] = [
                str(path_item)
                for path_item in current_paths
                if isinstance(path_item, str) and path_item not in nonexistent
            ]
            pyrefly[c.Infra.Toml.SEARCH_PATH] = self._to_array(remaining)
            fixes.append(f"removed nonexistent search-path: {', '.join(nonexistent)}")
        return fixes

    def _remove_ignore_sub_config_tk(
        self,
        pyrefly: MutableMapping[str, t.ContainerValue],
    ) -> list[str]:
        fixes: list[str] = []
        sub_configs = pyrefly.get(c.Infra.Toml.SUB_CONFIG)
        if not isinstance(sub_configs, list):
            return []
        new_configs: list[t.ContainerValue] = []
        for conf in sub_configs:
            if isinstance(conf, Mapping) and conf.get(c.Infra.Toml.IGNORE) is True:
                matches = conf.get("matches", c.Infra.Defaults.UNKNOWN)
                fixes.append(f"removed ignore=true sub-config for '{matches}'")
                continue
            new_configs.append(conf)
        if len(new_configs) != len(sub_configs):
            pyrefly[c.Infra.Toml.SUB_CONFIG] = new_configs
        return fixes

    def _resolve_project_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self._workspace_root / path
        return path.resolve()

    def _resolve_workspace_root(self, workspace_root: Path | None) -> Path:
        if workspace_root is not None:
            return workspace_root.resolve()
        result = self._path_resolver.workspace_root()
        return result.value if result.is_success else Path.cwd().resolve()


def main(argv: list[str] | None = None) -> int:
    """Run the pyrefly configuration fixer CLI."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("projects", nargs="*")
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    fixer = FlextInfraConfigFixer()
    result = fixer.run(
        projects=args.projects,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    if result.is_failure:
        output.error(result.error or "pyrefly config fix failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FlextInfraConfigFixer", "main"]
