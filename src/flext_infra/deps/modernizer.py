"""Modernize workspace pyproject.toml files to standardized format."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomlkit
from tomlkit.items import Table

from flext_core import r
from flext_infra import FlextInfraUtilitiesSubprocess, ProjectClassifier, c, p, u
from flext_infra.deps._phases import (
    ConsolidateGroupsPhase,
    EnsureCoverageConfigPhase,
    EnsureExtraPathsPhase,
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
from flext_infra.deps.tool_config import FlextInfraDependencyToolConfig


class FlextInfraPyprojectModernizer:
    """Modernize all workspace pyproject.toml files."""

    ROOT = u.Infra.resolve_workspace_root(__file__)

    def __init__(self, root: Path | None = None) -> None:
        """Initialize pyproject modernizer."""
        self.root = root or self.ROOT
        self._runner: p.Infra.CommandRunner = FlextInfraUtilitiesSubprocess()
        tool_config_result = FlextInfraDependencyToolConfig.load_tool_config()
        if tool_config_result.is_failure:
            msg = tool_config_result.error or "failed to load deps tool config"
            raise ValueError(msg)
        self._tool_config = tool_config_result.value

    @staticmethod
    def _table_child(parent: tomlkit.TOMLDocument | Table, key: str) -> Table | None:
        if key not in parent:
            return None
        child_value = parent[key]
        if isinstance(child_value, Table):
            return child_value
        return None

    def _classify_project(self, project_dir: Path) -> r[str]:
        """Classify project kind for pyright/coverage config selection."""
        kind = ProjectClassifier(project_dir).classify().project_kind
        return r[str].ok(kind)

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
        project_kind = "core"
        if not is_root:
            kind_result = self._classify_project(path.parent)
            if kind_result.is_success:
                project_kind = kind_result.value
        changes: list[str] = []
        tool_item = self._table_child(doc, c.Infra.Toml.TOOL)
        if tool_item is None:
            tool_item = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool_item
        poetry_item = self._table_child(tool_item, c.Infra.Toml.POETRY)
        if poetry_item is not None:
            group_item = self._table_child(poetry_item, c.Infra.Toml.GROUP)
            if group_item is not None:
                empty_groups: list[str] = []
                for name in u.Infra.table_string_keys(group_item):
                    group_dep_item = self._table_child(group_item, name)
                    if group_dep_item is None:
                        continue
                    deps_item = self._table_child(
                        group_dep_item,
                        c.Infra.Toml.DEPENDENCIES,
                    )
                    if deps_item is not None and len(deps_item) == 0:
                        empty_groups.append(name)
                for name in empty_groups:
                    del group_item[name]
                    changes.append(f"removed empty poetry group '{name}'")
                if len(group_item) == 0:
                    del poetry_item[c.Infra.Toml.GROUP]
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
                project_kind=project_kind,
            )
        )
        changes.extend(
            EnsureCoverageConfigPhase(self._tool_config).apply(
                doc,
                project_kind=project_kind,
            )
        )
        changes.extend(
            EnsureExtraPathsPhase().apply(
                doc,
                path=path,
                is_root=is_root,
                dry_run=dry_run,
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


def main(argv: list[str] | None = None) -> int:
    """Run the pyproject modernizer CLI."""
    parser = argparse.ArgumentParser(description="Modernize workspace pyproject files")
    _ = parser.add_argument("--audit", action="store_true")
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--skip-comments", action="store_true")
    _ = parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args(argv)
    return FlextInfraPyprojectModernizer().run(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FlextInfraPyprojectModernizer", "main", "u"]
