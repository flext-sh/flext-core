"""Modernize workspace pyproject.toml files to standardized format."""

from __future__ import annotations

import argparse
from pathlib import Path

from tomlkit.items import Table

from flext_infra import FlextInfraUtilitiesSubprocess, c, p, u
from flext_infra.deps._phases import (
    ConsolidateGroupsPhase,
    EnsureFormattingToolingPhase,
    EnsureMypyConfigPhase,
    EnsureNamespaceToolingPhase,
    EnsurePydanticMypyConfigPhase,
    EnsurePyreflyConfigPhase,
    EnsurePyrightConfigPhase,
    EnsurePytestConfigPhase,
    EnsureRuffConfigPhase,
    InjectCommentsPhase,
)
from flext_infra.deps.tool_config import load_tool_config

canonical_dev_dependencies = u.Infra.canonical_dev_dependencies
dedupe_specs = u.Infra.dedupe_specs
dep_name = u.Infra.dep_name
project_dev_groups = u.Infra.project_dev_groups

array_parser = u.Infra.array
as_string_list_parser = u.Infra.as_string_list
canonical_dev_dependencies_parser = u.Infra.canonical_dev_dependencies
dedupe_specs_parser = u.Infra.dedupe_specs
dep_name_parser = u.Infra.dep_name
ensure_table_parser = u.Infra.ensure_table
project_dev_groups_parser = u.Infra.project_dev_groups
read_doc_parser = u.Infra.read
table_string_keys_parser = u.Infra.table_string_keys
toml_get_parser = u.Infra.get
unwrap_item_parser = u.Infra.unwrap_item


def workspace_root(start: Path) -> Path:
    """Detect workspace root by searching for .gitmodules or .git with pyproject.toml."""
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / c.Infra.Files.GITMODULES).exists() and (
            parent / c.Infra.Files.PYPROJECT_FILENAME
        ).exists():
            return parent
    for parent in (current, *current.parents):
        if (parent / c.Infra.Git.DIR).exists() and (
            parent / c.Infra.Files.PYPROJECT_FILENAME
        ).exists():
            return parent
    return start.resolve().parents[4]


ROOT = workspace_root(Path(__file__))


class FlextInfraPyprojectModernizer:
    """Modernize all workspace pyproject.toml files."""

    def __init__(self, root: Path | None = None) -> None:
        """Initialize pyproject modernizer."""
        super().__init__()
        self.root = root or ROOT
        self._runner: p.Infra.CommandRunner = FlextInfraUtilitiesSubprocess()
        tool_config_result = load_tool_config()
        if tool_config_result.is_failure:
            msg = tool_config_result.error or "failed to load deps tool config"
            raise ValueError(msg)
        self._tool_config = tool_config_result.value

    def find_pyproject_files(self) -> list[Path]:
        """Find all workspace pyproject.toml files."""
        result = u.Infra.find_all_pyproject_files(
            self.root,
            skip_dirs=c.Infra.Deps.SKIP_DIRS,
        )
        return result.fold(
            on_failure=lambda _: [],
            on_success=lambda v: sorted(v),
        )

    def process_file(
        self,
        path: Path,
        *,
        canonical_dev: list[str],
        dry_run: bool,
        skip_comments: bool,
    ) -> list[str]:
        """Process one pyproject.toml file and collect changes."""
        doc = u.Infra.read(path)
        if doc is None:
            return ["invalid TOML"]
        is_root = path.parent.resolve() == self.root.resolve()
        changes: list[str] = []
        tool: object | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if isinstance(tool, Table):
            poetry: object | None = None
            if c.Infra.Toml.POETRY in tool:
                poetry = tool[c.Infra.Toml.POETRY]
            if isinstance(poetry, Table):
                group: object | None = None
                if c.Infra.Toml.GROUP in poetry:
                    group = poetry[c.Infra.Toml.GROUP]
                if isinstance(group, Table):
                    empty_groups: list[str] = []
                    for name in u.Infra.table_string_keys(group):
                        group_item: object | None = None
                        if name in group:
                            group_item = group[name]
                        if isinstance(group_item, Table):
                            deps: object | None = None
                            if c.Infra.Toml.DEPENDENCIES in group_item:
                                deps = group_item[c.Infra.Toml.DEPENDENCIES]
                            if isinstance(deps, Table) and len(deps) == 0:
                                empty_groups.append(name)
                    for name in empty_groups:
                        del group[name]
                        changes.append(f"removed empty poetry group '{name}'")
                    if len(group) == 0:
                        del poetry[c.Infra.Toml.GROUP]
                        changes.append("removed empty poetry group container")
        changes.extend(ConsolidateGroupsPhase().apply(doc, canonical_dev))
        changes.extend(EnsurePytestConfigPhase(self._tool_config).apply(doc))
        changes.extend(
            EnsurePyreflyConfigPhase(self._tool_config).apply(doc, is_root=is_root)
        )
        changes.extend(EnsureMypyConfigPhase(self._tool_config).apply(doc))
        changes.extend(EnsurePydanticMypyConfigPhase(self._tool_config).apply(doc))
        changes.extend(EnsureFormattingToolingPhase(self._tool_config).apply(doc))
        changes.extend(EnsureNamespaceToolingPhase().apply(doc, path=path))
        changes.extend(
            EnsureRuffConfigPhase(self._tool_config).apply(
                doc,
                path=path,
                workspace_root=self.root,
            ),
        )
        changes.extend(
            EnsurePyrightConfigPhase(self._tool_config).apply(
                doc,
                is_root=is_root,
                workspace_root=self.root,
            )
        )
        rendered = doc.as_string()
        if not skip_comments:
            rendered, comment_changes = InjectCommentsPhase().apply(rendered)
            changes.extend(comment_changes)
        if changes and (not dry_run):
            _ = path.write_text(rendered, encoding=c.Infra.Encoding.DEFAULT)
        return changes

    def run(self, args: argparse.Namespace) -> int:
        """Run pyproject modernization for the workspace."""
        dry_run = bool(args.dry_run or args.audit)
        files = self.find_pyproject_files()
        root_doc = u.Infra.read(self.root / c.Infra.Files.PYPROJECT_FILENAME)
        if root_doc is None:
            return 2
        canonical_dev = u.Infra.canonical_dev_dependencies(root_doc)
        violations: dict[str, list[str]] = {}
        total = 0
        for file_path in files:
            changes = self.process_file(
                file_path,
                canonical_dev=canonical_dev,
                dry_run=dry_run,
                skip_comments=bool(args.skip_comments),
            )
            if not changes:
                continue
            rel = str(file_path.relative_to(self.root))
            violations[rel] = changes
            total += len(changes)
        if violations:
            for rel_path, changes in violations.items():
                u.Infra.info(f"{rel_path}:")
                for change in changes:
                    u.Infra.info(f"  - {change}")
            u.Infra.info(
                f"Total: {total} change(s) across {len(violations)} file(s)",
            )
            if dry_run:
                u.Infra.info("(dry-run — no files modified)")
        if args.audit and total > 0:
            return 1
        if not dry_run and (not args.skip_check):
            return self._run_poetry_check(files)
        return 0

    def _run_poetry_check(self, files: list[Path]) -> int:
        has_warning = False
        for path in files:
            project_dir = path.parent
            result = self._runner.run_raw(
                [c.Infra.Cli.POETRY, c.Infra.Cli.PoetryCmd.CHECK],
                cwd=project_dir,
            )
            if result.is_failure:
                has_warning = True
                continue
            if result.value.exit_code != 0:
                has_warning = True
        return 1 if has_warning else 0


def parser() -> argparse.ArgumentParser:
    """Create CLI parser for pyproject modernizer options."""
    parser = argparse.ArgumentParser(description="Modernize workspace pyproject files")
    _ = parser.add_argument("--audit", action="store_true")
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--skip-comments", action="store_true")
    _ = parser.add_argument("--skip-check", action="store_true")
    return parser


def main() -> int:
    """Run the pyproject modernizer CLI."""
    parser_model = parser()
    args = parser_model.parse_args()
    return FlextInfraPyprojectModernizer().run(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConsolidateGroupsPhase",
    "EnsurePyreflyConfigPhase",
    "EnsurePyrightConfigPhase",
    "EnsurePytestConfigPhase",
    "FlextInfraPyprojectModernizer",
    "InjectCommentsPhase",
    "array_parser",
    "as_string_list_parser",
    "canonical_dev_dependencies_parser",
    "dedupe_specs_parser",
    "dep_name_parser",
    "ensure_table_parser",
    "main",
    "parser",
    "project_dev_groups_parser",
    "read_doc_parser",
    "unwrap_item_parser",
    "workspace_root",
]
