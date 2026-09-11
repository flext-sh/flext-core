"""Signature propagation transformer — rope-driven call-site keyword rewrite."""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

from flext_infra import c, u
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra.transformers.base import FlextInfraChangeTrackingTransformer

if TYPE_CHECKING:
    from flext_infra import m, t


class FlextInfraRefactorSignaturePropagator(FlextInfraChangeTrackingTransformer):
    """Apply declarative signature migrations to call sites via rope + regex.

    Uses rope's ``parse_string_module`` to locate ``Call`` nodes by name,
    then slices each call's byte range from the original source and
    applies keyword renames/removals/additions inside that one slice
    (regex helpers from ``c.Infra``). No ast import is needed.
    """

    def __init__(
        self,
        *,
        migrations: t.SequenceOf[m.Infra.SignatureMigration],
        on_change: t.Infra.ChangeCallback = None,
    ) -> None:
        """Initialize transformer state for declarative signature migrations."""
        super().__init__(on_change=on_change)
        self._migrations = migrations

    def apply_to_source(self, source: str) -> str:
        """Apply all migrations to source text and return transformed source."""
        result = source
        for migration in self._migrations:
            result = self._apply_migration(result, migration)
        return result

    def _apply_migration(
        self, source: str, migration: m.Infra.SignatureMigration
    ) -> str:
        """Apply a single migration to source text."""
        keyword_renames = dict(migration.keyword_renames)
        remove_keywords = set(migration.remove_keywords)
        add_keywords = dict(migration.add_keywords)
        if not keyword_renames and not remove_keywords and not add_keywords:
            return source
        targets = set(migration.target_simple_names) | set(
            migration.target_qualified_names
        )
        for target in targets:
            simple_name = target.rsplit(".", 1)[-1] if "." in target else target
            source = self._rewrite_calls(
                source,
                simple_name=simple_name,
                migration_id=migration.id,
                keyword_renames=keyword_renames,
                remove_keywords=remove_keywords,
                add_keywords=add_keywords,
            )
        return source

    def _rewrite_calls(
        self,
        source: str,
        *,
        simple_name: str,
        migration_id: str,
        keyword_renames: t.MutableStrMapping,
        remove_keywords: t.Infra.StrSet,
        add_keywords: t.MutableStrMapping,
    ) -> str:
        """Rewrite keyword arguments in calls to ``simple_name`` via rope-located ranges."""
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return source
        line_offsets = self._line_offsets(source)
        edits: list[tuple[int, int, str]] = []
        for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(pymodule.get_ast()):
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "Call":
                continue
            if (
                FlextInfraUtilitiesRopeAnalysis.name_of(getattr(node, "func", None))
                != simple_name
            ):
                continue
            span = FlextInfraUtilitiesRopeAnalysis.line_col_range(node)
            if span is None:
                continue
            lineno, col_offset, end_lineno, end_col_offset = span
            start = self._offset(line_offsets, lineno, col_offset)
            end = self._offset(line_offsets, end_lineno, end_col_offset)
            replacement, changed = self._rewrite_call_text(
                source[start:end],
                keyword_renames=keyword_renames,
                remove_keywords=remove_keywords,
                add_keywords=add_keywords,
            )
            if not changed:
                continue
            edits.append((start, end, replacement))
            self._record_change(f"[{migration_id}] Updated call: {simple_name}(...)")
        updated = source
        for start, end, replacement in sorted(edits, key=itemgetter(0), reverse=True):
            updated = updated[:start] + replacement + updated[end:]
        return updated

    @staticmethod
    def _rewrite_call_text(
        call_text: str,
        *,
        keyword_renames: t.MutableStrMapping,
        remove_keywords: t.Infra.StrSet,
        add_keywords: t.MutableStrMapping,
    ) -> tuple[str, bool]:
        """Rewrite keywords inside a single call's source slice (regex per-name)."""
        result = call_text
        changed = False
        for old_name, new_name in keyword_renames.items():
            pattern = c.Infra.compile_keyword_argument(old_name)
            new_text, count = pattern.subn(rf"{new_name}\1", result)
            if count:
                changed = True
                result = new_text
        for remove_name in remove_keywords:
            pattern = c.Infra.compile_keyword_argument(remove_name)
            stripped, drops = FlextInfraRefactorSignaturePropagator._drop_keyword(
                result, pattern
            )
            if drops:
                changed = True
                result = stripped
        if add_keywords:
            close = result.rfind(")")
            if close >= 0:
                existing = {
                    match.group(0).split("=", 1)[0].strip()
                    for match in c.Infra.compile_keyword_argument(r"\w+").finditer(
                        result
                    )
                }
                additions = [
                    f"{key}={u.norm_str(value)}"
                    for key, value in add_keywords.items()
                    if key not in existing
                ]
                if additions:
                    inner = result[:close].rstrip()
                    inner_has_args = inner.endswith(",") or inner[-1:] != "("
                    sep = ", " if inner_has_args and not inner.endswith(",") else ""
                    if inner.endswith(","):
                        sep = " "
                    if inner.endswith("("):
                        sep = ""
                    result = f"{inner}{sep}{', '.join(additions)}{result[close:]}"
                    changed = True
        return result, changed

    @staticmethod
    def _drop_keyword(text: str, pattern: t.Infra.RegexPattern) -> tuple[str, int]:
        """Remove ``<name>=<value>[,]?`` occurrences from a call slice."""
        result = text
        drops = 0
        match = pattern.search(result)
        while match is not None:
            start = match.start()
            cursor = match.end()
            depth = 0
            while cursor < len(result):
                ch = result[cursor]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    if depth == 0:
                        break
                    depth -= 1
                elif ch == "," and depth == 0:
                    cursor += 1
                    break
                cursor += 1
            tail = result[cursor:].lstrip()
            head = result[:start].rstrip()
            head = head.removesuffix(",")
            joiner = "" if not head.endswith("(") and tail.startswith(")") else " "
            if head.endswith("(") or tail.startswith(")"):
                joiner = ""
            elif tail and not tail.startswith(","):
                joiner = ", "
            else:
                joiner = ""
            result = f"{head}{joiner}{tail}"
            drops += 1
            match = pattern.search(result)
        return result, drops

    @staticmethod
    def _line_offsets(source: str) -> tuple[int, ...]:
        """Cumulative byte offsets at each line start."""
        offsets = [0]
        for line in source.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return tuple(offsets)

    @staticmethod
    def _offset(line_offsets: tuple[int, ...], line: int, column: int) -> int:
        """Convert ``(line, column)`` to a byte offset."""
        return line_offsets[line - 1] + column


__all__: list[str] = ["FlextInfraRefactorSignaturePropagator"]
