"""Accessor token rewriting + manual-warning detection — extracted concern."""

from __future__ import annotations

import io
from operator import itemgetter
from tokenize import NAME, generate_tokens
from typing import TYPE_CHECKING, ClassVar

from flext_infra import c, m, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraAccessorMigrationRewriteMixin:
    """Token-level get_/set_/is_ rewriting plus public-accessor warning scan.

    Composed into FlextInfraAccessorMigrationOrchestrator via inheritance; owns
    the automated-rename catalog and the idempotent token rewrite + the manual
    review-warning detection over loose public accessors.
    """

    # Rename rules sourced from c.ENFORCEMENT_ACCESSOR_RENAMES (flext-core SSOT).
    # All entries target flext-core surface (origin="flext_core"); adding a
    # rename = one entry in flext-core's enforcement constant, never duplicated
    # here. Token-level rename is idempotent — once source_name has been
    # renamed, subsequent passes find zero matching tokens.
    _AUTOMATED_RULES: ClassVar[tuple[m.Infra.AccessorMigrationRule, ...]] = tuple(
        m.Infra.AccessorMigrationRule(
            source_name=src, replacement_name=repl, reason=reason, origin="flext_core"
        )
        for src, (repl, reason) in c.ENFORCEMENT_ACCESSOR_RENAMES.items()
    )
    _AUTOMATED_NAMES: ClassVar[frozenset[str]] = frozenset(
        c.ENFORCEMENT_ACCESSOR_RENAMES
    )
    _MANUAL_WARNING_REASON: ClassVar[str] = (
        "Public {prefix}-prefixed accessor: rename to canonical verb "
        "(drop the prefix or use resolve_/fetch_/build_/etc.)"
    )

    def _apply_automated_rewrites(
        self, rope_project: t.Infra.RopeProject, py_file: Path, source: str
    ) -> tuple[str, t.SequenceOf[m.Infra.AccessorMigrationChange]]:
        """Apply automated rewrites."""
        resource = u.Infra.get_resource_from_path(rope_project, py_file)
        if resource is None:
            return source, ()
        updated_source = source
        changes: t.MutableSequenceOf[m.Infra.AccessorMigrationChange] = []
        for rule in self._AUTOMATED_RULES:
            updated_source, rule_changes = self._rename_symbol_tokens(
                rope_project,
                resource,
                updated_source,
                source_name=rule.source_name,
                replacement_name=rule.replacement_name,
                reason=rule.reason,
                file_path=py_file,
            )
            changes.extend(rule_changes)
        return updated_source, changes

    @staticmethod
    def _rename_symbol_tokens(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        source: str,
        *,
        source_name: str,
        replacement_name: str,
        reason: str,
        file_path: Path,
    ) -> tuple[str, t.SequenceOf[m.Infra.AccessorMigrationChange]]:
        """Rename symbol tokens."""
        token_lines: t.MutableSequenceOf[m.Infra.AccessorMigrationChange] = []
        rewrite_ranges: t.MutableSequenceOf[tuple[int, int, str]] = []
        for token in generate_tokens(io.StringIO(source).readline):
            if token.type != NAME or token.string != source_name:
                continue
            line, column = token.start
            start = FlextInfraAccessorMigrationRewriteMixin._offset_from_position(
                source, line, column
            )
            end = start + len(source_name)
            rewrite_ranges.append((start, end, replacement_name))
            token_lines.append(
                m.Infra.AccessorMigrationChange(
                    file=str(file_path),
                    line=line,
                    original_name=source_name,
                    replacement_name=replacement_name,
                    automated=True,
                    reason=reason,
                )
            )
        if not rewrite_ranges:
            return source, ()
        del rope_project
        del resource
        updated_source = source
        for start, end, replacement in sorted(
            rewrite_ranges, key=itemgetter(0), reverse=True
        ):
            updated_source = updated_source[:start] + replacement + updated_source[end:]
        return updated_source, tuple(token_lines)

    @staticmethod
    def _offset_from_position(source: str, line: int, column: int) -> int:
        """Offset from position."""
        source_lines = source.splitlines(keepends=True)
        line_offset = sum(len(item) for item in source_lines[: line - 1])
        return line_offset + column

    def _collect_manual_warnings(
        self, py_file: Path, source: str
    ) -> t.SequenceOf[m.Infra.AccessorMigrationChange]:
        """Collect manual warnings."""
        lines = source.splitlines()
        warnings: t.MutableSequenceOf[m.Infra.AccessorMigrationChange] = []
        scope_stack: t.MutableSequenceOf[tuple[str, int]] = []
        for line_index, line_text in enumerate(lines, start=1):
            stripped = line_text.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line_text) - len(stripped)
            while scope_stack and indent <= scope_stack[-1][1]:
                scope_stack.pop()
            if stripped.startswith("class "):
                class_name = (
                    stripped
                    .split("class ", maxsplit=1)[1]
                    .split("(", maxsplit=1)[0]
                    .split(":", maxsplit=1)[0]
                    .strip()
                )
                scope_stack.append((f"class:{class_name}", indent))
                continue
            function_prefix = ""
            if stripped.startswith("def "):
                function_prefix = "def "
            elif stripped.startswith("async def "):
                function_prefix = "async def "
            if not function_prefix:
                continue
            function_name = (
                stripped
                .split(function_prefix, maxsplit=1)[1]
                .split("(", maxsplit=1)[0]
                .strip()
            )
            parent_scope = scope_stack[-1][0] if scope_stack else "module"
            scope_stack.append((f"def:{function_name}", indent))
            if parent_scope.startswith("def:"):
                continue
            if function_name.startswith("_") or function_name in self._AUTOMATED_NAMES:
                continue
            matched_prefix = next(
                (
                    p
                    for p in c.Infra.ACCESSOR_WARNING_PREFIXES
                    if function_name.startswith(p)
                ),
                None,
            )
            if matched_prefix is None:
                continue
            warnings.append(
                m.Infra.AccessorMigrationChange(
                    file=str(py_file),
                    line=line_index,
                    original_name=function_name,
                    replacement_name=function_name[len(matched_prefix) :],
                    automated=False,
                    reason=self._MANUAL_WARNING_REASON.format(
                        prefix=matched_prefix.rstrip("_")
                    ),
                )
            )
        return warnings


__all__: list[str] = ["FlextInfraAccessorMigrationRewriteMixin"]
