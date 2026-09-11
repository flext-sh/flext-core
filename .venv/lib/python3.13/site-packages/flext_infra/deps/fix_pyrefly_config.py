"""CLI tool to fix Pyrefly configurations across projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import override

from flext_infra import c, m, p, r, s, t, u
from flext_infra.deps._pyrefly_fix_steps import FlextInfraConfigFixerSteps

logger = u.fetch_logger(__name__)


class FlextInfraConfigFixer(FlextInfraConfigFixerSteps, s[bool]):
    """Fix pyrefly configuration across workspace projects."""

    _repository_root: Path

    def __init__(
        self, repository_root: Path | None = None, *, workspace: Path | None = None
    ) -> None:
        """Initialize pyrefly settings fixer."""
        resolved_workspace = u.Infra.resolve_repository_root_or_cwd(
            repository_root or workspace
        )
        super().__init__(repository_root=resolved_workspace)
        self._repository_root = self.repository_root

    @override
    def execute(self) -> p.Result[bool]:
        """Execute."""
        return r[bool].fail("Use execute_command() directly")

    @classmethod
    def execute_payload(cls, params: m.Infra.FixPyreflyConfigCommand) -> p.Result[bool]:
        """Execute pyrefly config repair from the canonical check command payload."""
        fixer = cls(repository_root=params.workspace_path)
        fix_result = fixer.run(
            projects=params.project_names or [],
            dry_run=params.dry_run,
            verbose=params.verbose,
        )
        if fix_result.failure:
            return r[bool].fail(fix_result.error or "pyrefly config fix failed")
        return r[bool].ok(True)

    def process_file(
        self, path: Path, *, dry_run: bool = False
    ) -> p.Result[t.StrSequence]:
        """Process one pyproject.toml file and apply fixes."""
        document_result = u.Cli.toml_read_document(path)
        if document_result.failure:
            return r[t.StrSequence].fail(
                document_result.error or f"failed to read {path}"
            )
        doc = document_result.value
        doc_data = doc.unwrap()
        tool_data = doc_data.get(c.Infra.TOOL)
        if not isinstance(tool_data, Mapping):
            return r[t.StrSequence].ok(())
        typed_tool_data: MutableMapping[str, t.Infra.InfraValue] = (
            t.Infra.MUTABLE_INFRA_MAPPING_ADAPTER.validate_python(tool_data)
        )
        pyrefly_data = typed_tool_data.get(c.Infra.PYREFLY)
        if not isinstance(pyrefly_data, Mapping):
            return r[t.StrSequence].ok(())
        try:
            pyrefly: MutableMapping[str, t.Infra.InfraValue] = (
                t.Infra.MUTABLE_INFRA_MAPPING_ADAPTER.validate_python(pyrefly_data)
            )
        except c.ValidationError as err:
            return r[t.StrSequence].fail_op(f"validate {path} [tool.pyrefly]", err)
        original_pyrefly: t.JsonMapping = dict(pyrefly)
        all_fixes: t.MutableSequenceOf[str] = []
        project_dir = path.parent
        is_root = project_dir == self._repository_root
        search_result = self._sync_search_path(pyrefly, project_dir, is_root=is_root)
        if search_result.failure:
            return search_result
        all_fixes.extend(search_result.value)
        includes_result = self._sync_project_includes(
            pyrefly, project_dir, is_root=is_root
        )
        if includes_result.failure:
            return includes_result
        all_fixes.extend(includes_result.value)
        sub_result = self._strip_ignored_sub_configs(pyrefly)
        if sub_result.failure:
            return r[t.StrSequence].fail(sub_result.error or "validate-sub-configs")
        sub_fixes, removed_ignore = sub_result.value
        all_fixes.extend(sub_fixes)
        if removed_ignore or is_root:
            excludes_result = self._sync_project_excludes(pyrefly)
            if excludes_result.failure:
                return excludes_result
            all_fixes.extend(excludes_result.value)
        if all_fixes and (not dry_run):
            tool_table = doc[c.Infra.TOOL]
            if not isinstance(tool_table, MutableMapping):
                return r[t.StrSequence].fail(f"invalid {path} [tool] table")
            pyrefly_table = tool_table[c.Infra.PYREFLY]
            if not isinstance(pyrefly_table, MutableMapping):
                return r[t.StrSequence].fail(f"invalid {path} [tool.pyrefly] table")
            # flext-wkii.17 (codex): reassign only changed keys so an untouched
            # nested table retains adjacent managed comments and TOML trivia.
            for key, value in pyrefly.items():
                if key in original_pyrefly and original_pyrefly[key] == value:
                    continue
                pyrefly_table[key] = value
            write_result = u.Cli.toml_write_document(path, doc)
            if write_result.failure:
                return r[t.StrSequence].fail(
                    write_result.error or f"failed to write {path}"
                )
        return r[t.StrSequence].ok(all_fixes)

    def run(
        self, projects: t.StrSequence, *, dry_run: bool = False, verbose: bool = False
    ) -> p.Result[t.StrSequence]:
        """Run pyrefly configuration fixes for selected projects."""
        project_paths = [
            (
                project_path
                if project_path.is_absolute()
                else (self._repository_root / project_path)
            ).resolve()
            for project in projects
            for project_path in [Path(project)]
        ]
        files_result = u.Infra.find_all_pyproject_files(
            self._repository_root, project_paths=project_paths or None
        )
        if files_result.failure:
            return r[t.StrSequence].fail(
                files_result.error or "failed to find pyproject files"
            )
        messages: t.MutableSequenceOf[str] = []
        total_fixes = 0
        pyproject_files: t.SequenceOf[Path] = files_result.value
        for path in pyproject_files:
            fixes_result = self.process_file(path, dry_run=dry_run)
            if fixes_result.failure:
                return r[t.StrSequence].fail(
                    fixes_result.error or f"failed to process {path}"
                )
            fixes: t.StrSequence = fixes_result.value
            if not fixes:
                continue
            total_fixes += len(fixes)
            if verbose:
                try:
                    rel = path.relative_to(self._repository_root)
                except ValueError:
                    rel = path
                for fix in fixes:
                    line = f"  {('(dry)' if dry_run else '✓')} {rel}: {fix}"
                    logger.info("pyrefly_config_fix", detail=line)
                    messages.append(line)
        if verbose and total_fixes == 0:
            logger.info("pyrefly_configs_clean")
        return r[t.StrSequence].ok(messages)


if __name__ == "__main__":
    raise SystemExit(0)


__all__: list[str] = ["FlextInfraConfigFixer"]
