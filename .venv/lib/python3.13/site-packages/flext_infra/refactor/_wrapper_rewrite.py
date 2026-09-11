"""Wrapper-root per-file AST rewrite helpers."""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, ClassVar

from flext_infra import c, m, u
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from flext_infra import t


class FlextInfraWrapperRootNamespaceRewriteMixin:
    """Detect ``Core.Tests`` chain rewrites + import candidates per file.

    Composed into FlextInfraWrapperRootNamespaceRefactor via inheritance;
    borrows repository_root + the include-init / dry-run flags + the wrapper
    package set from the facade via FLEXT. ``module_ast`` is typed ``object`` to
    mirror the rope-AST abstraction (FlextInfraUtilitiesRopeAnalysis), which
    deliberately avoids ``import ast`` at the consumer layer (tracked: flext-6flt).
    """

    if TYPE_CHECKING:
        repository_root: Path
        include_init: bool
        _WRAPPER_PACKAGES: ClassVar[t.StrSequence]

        @property
        def effective_dry_run(self) -> bool: ...

    def _process_wrapper_file(
        self,
        file_path: Path,
        *,
        accumulator: m.Infra.WrapperRewriteAccumulator,
        project_runtime_aliases: Mapping[str, frozenset[str]],
        wrapper_submodules: frozenset[str],
        metadata_runtime_aliases: frozenset[str],
    ) -> None:
        """Detect Core.Tests chain rewrites and import candidates for one file."""
        try:
            rel = file_path.relative_to(self.repository_root)
        except ValueError:
            rel = file_path
        project_name = rel.parts[0] if rel.parts else "."
        runtime_aliases = project_runtime_aliases.get(
            project_name, metadata_runtime_aliases
        )
        if not any(
            part in c.Infra.ROOT_WRAPPER_SEGMENTS and part != c.Infra.DEFAULT_SRC_DIR
            for part in rel.parts
        ):
            return
        if file_path.name == c.Infra.INIT_PY and (not self.include_init):
            return
        source = u.Cli.files_read_text(file_path).unwrap()
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return
        module_ast = pymodule.get_ast()
        line_offsets = self._build_line_offsets(source)
        core_rewrites = self._collect_core_test_rewrites(
            module_ast, line_offsets=line_offsets, runtime_aliases=runtime_aliases
        )
        has_import_candidate = self._has_wrapper_import_candidate(
            module_ast,
            wrapper_submodules=wrapper_submodules,
            runtime_aliases=runtime_aliases,
        )
        replacements = len(core_rewrites) + (1 if has_import_candidate else 0)
        if replacements == 0:
            return
        core_updated = self._apply_byte_rewrites(source, core_rewrites)
        accumulator.changed_files.append(str(file_path))
        accumulator.total_replacements += replacements
        accumulator.total_core_replacements += len(core_rewrites)
        if has_import_candidate:
            accumulator.import_rewrite_candidates += 1
            accumulator.wrapper_candidates.append(file_path)
        accumulator.per_project_changes[project_name] += 1
        accumulator.per_project_replacements[project_name] += replacements
        if not self.effective_dry_run and core_updated != source:
            accumulator.updates[file_path] = core_updated

    @staticmethod
    def _build_line_offsets(source: str) -> list[int]:
        """Return cumulative line-start byte offsets for ``source``."""
        line_offsets = [0]
        for line_text in source.splitlines(keepends=True):
            line_offsets.append(line_offsets[-1] + len(line_text))
        return line_offsets

    def _collect_core_test_rewrites(
        self,
        module_ast: object,
        *,
        line_offsets: list[int],
        runtime_aliases: frozenset[str],
    ) -> list[tuple[int, int, str]]:
        """Find every ``<alias>.Core.Tests`` chain and emit ``(start, end, repl)``."""
        rewrites: list[tuple[int, int, str]] = []
        for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(module_ast):
            if (
                FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "Attribute"
                or getattr(node, "attr", "") != "Tests"
            ):
                continue
            parent_attr = getattr(node, "value", None)
            if (
                parent_attr is None
                or FlextInfraUtilitiesRopeAnalysis.node_kind(parent_attr) != "Attribute"
                or getattr(parent_attr, "attr", "") != "Core"
            ):
                continue
            base_name = getattr(parent_attr, "value", None)
            if (
                base_name is None
                or FlextInfraUtilitiesRopeAnalysis.node_kind(base_name) != "Name"
            ):
                continue
            base_id = getattr(base_name, "id", "")
            if base_id not in runtime_aliases:
                continue
            line_col = FlextInfraUtilitiesRopeAnalysis.line_col_range(node)
            if line_col is None:
                continue
            lineno, col_offset, end_lineno, end_col_offset = line_col
            start = line_offsets[lineno - 1] + col_offset
            end = line_offsets[end_lineno - 1] + end_col_offset
            rewrites.append((start, end, f"{base_id}.Tests"))
        return rewrites

    def _has_wrapper_import_candidate(
        self,
        module_ast: object,
        *,
        wrapper_submodules: frozenset[str],
        runtime_aliases: frozenset[str],
    ) -> bool:
        """Return whether any ``from wrapper.<sub> import <alias>`` exists."""
        for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(module_ast):
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "ImportFrom":
                continue
            module_name = getattr(node, "module", "") or ""
            parent, dot, child = module_name.partition(".")
            if not (
                dot and parent in self._WRAPPER_PACKAGES and child in wrapper_submodules
            ):
                continue
            names = getattr(node, "names", []) or []
            if any(
                (getattr(alias, "asname", None) or getattr(alias, "name", ""))
                in runtime_aliases
                for alias in names
            ):
                return True
        return False

    @staticmethod
    def _apply_byte_rewrites(
        source: str, rewrites: t.SequenceOf[tuple[int, int, str]]
    ) -> str:
        """Apply ``(start, end, replacement)`` triples to ``source`` (right-to-left)."""
        updated = source
        for start, end, replacement in sorted(
            rewrites, key=itemgetter(0), reverse=True
        ):
            updated = updated[:start] + replacement + updated[end:]
        return updated


__all__: list[str] = ["FlextInfraWrapperRootNamespaceRewriteMixin"]
