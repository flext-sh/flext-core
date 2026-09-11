"""Lazy import fixer transformer — rope-based implementation.

Hoists function-local imports to module top level, deduplicating against
existing imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c, u
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorLazyImportFixer(FlextInfraRopeTransformer):
    """Hoist function-local imports to module top while preserving ordering."""

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Hoist function-local imports to module level."""
        lines = source.splitlines(keepends=True)
        kept_lines, hoisted = self.scan_lines_for_hoist(lines)
        if not hoisted:
            no_changes: list[str] = []
            return source, no_changes
        insert_idx = u.Infra.find_import_insert_position(kept_lines)
        new_lines = kept_lines[:insert_idx] + hoisted + kept_lines[insert_idx:]
        new_source = "".join(new_lines)
        return new_source, list(self.changes)

    def scan_lines_for_hoist(self, lines: list[str]) -> tuple[list[str], list[str]]:
        """Scan source lines, hoist body-local imports, and keep remaining lines."""
        existing_imports: set[str] = set()
        hoisted: list[str] = []
        kept_lines: list[str] = []
        in_body = False
        indent_depth = 0
        for line in lines:
            stripped = line.strip()
            if self.starts_body_scope(in_body=in_body, stripped_line=stripped):
                in_body = True
                indent_depth = self.indent_of(line)
                kept_lines.append(line)
                continue
            if in_body:
                consumed, in_body = self.consume_body_line(
                    stripped_line=stripped,
                    line=line,
                    body_indent=indent_depth,
                    scan_state=(existing_imports, hoisted),
                )
                if consumed:
                    continue
            if not in_body and self._is_import_line(stripped):
                existing_imports.add(stripped)
            kept_lines.append(line)
        return kept_lines, hoisted

    def consume_body_line(
        self,
        *,
        stripped_line: str,
        line: str,
        body_indent: int,
        scan_state: tuple[set[str], list[str]],
    ) -> tuple[bool, bool]:
        """Process one line inside a body and return (consumed, still_in_body)."""
        existing_imports, hoisted = scan_state
        cur_indent = self.body_line_indent(
            line=line, stripped_line=stripped_line, body_indent=body_indent
        )
        if self.ends_body_scope(
            stripped_line=stripped_line,
            current_indent=cur_indent,
            body_indent=body_indent,
        ):
            return False, False
        if stripped_line and self._is_import_line(stripped_line):
            if stripped_line not in existing_imports:
                hoisted.append(stripped_line + "\n")
                existing_imports.add(stripped_line)
                self._record_change(f"Hoisted lazy import: {stripped_line}")
            return True, True
        return False, True

    @staticmethod
    def _is_import_line(stripped: str) -> bool:
        """Is import line."""
        return stripped.startswith(("from ", "import "))

    @classmethod
    def starts_body_scope(cls, *, in_body: bool, stripped_line: str) -> bool:
        """Return whether the current line starts a def/class body scope."""
        return (not in_body) and c.Infra.DEF_ASYNC_CLASS_RE.match(
            stripped_line
        ) is not None

    @staticmethod
    def indent_of(line: str) -> int:
        """Return indentation width for one source line."""
        return len(line) - len(line.lstrip())

    @classmethod
    def body_line_indent(
        cls, *, line: str, stripped_line: str, body_indent: int
    ) -> int:
        """Compute effective indentation for one line while scanning a body."""
        return cls.indent_of(line) if stripped_line else body_indent + 4

    @staticmethod
    def ends_body_scope(
        *, stripped_line: str, current_indent: int, body_indent: int
    ) -> bool:
        """Return whether scanner should leave the current def/class body."""
        return bool(stripped_line) and current_indent <= body_indent


__all__: list[str] = ["FlextInfraRefactorLazyImportFixer"]
