"""Shared source-rewrite primitives for AST-based transformers.

Provides a small, dependency-free helper layer for transformers that need to
record byte-range replacements in a source string and apply them safely from
bottom-right to top-left so earlier offsets remain valid.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FlextInfraSourceRewrite:
    """One source rewrite: replace ``source[start:end]`` with ``text``."""

    start: int
    end: int
    text: str

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, FlextInfraSourceRewrite):
            return NotImplemented
        return (self.start, self.end) < (other.start, other.end)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, FlextInfraSourceRewrite):
            return NotImplemented
        return (self.start, self.end) > (other.start, other.end)


class FlextInfraSourceRewriter(ast.NodeVisitor):
    """Base AST visitor that records source rewrites by byte range.

    Subclasses implement ``visit_*`` methods and call
    :meth:`append_rewrite` to register replacements.  When traversal is
    complete, :func:`apply_rewrites` can materialize the updated source
    text.
    """

    def __init__(self, source: str) -> None:
        """Initialize rewriter with the original source text."""
        super().__init__()
        self._source = source
        self.rewrites: list[FlextInfraSourceRewrite] = []
        self.changes: list[str] = []

    def _offset(self, lineno: int, col_offset: int) -> int:
        """Convert (1-based line, 0-based column) to byte offset."""
        lines = self._source.splitlines(keepends=True)
        return sum(len(lines[i]) for i in range(lineno - 1)) + col_offset

    def node_offset(self, node: ast.AST, *, start: bool) -> int:
        """Return byte offset for a node's start or end position."""
        if start:
            lineno = getattr(node, "lineno", 1)
            col_offset = getattr(node, "col_offset", 0)
        else:
            lineno = getattr(node, "end_lineno", getattr(node, "lineno", 1))
            col_offset = getattr(node, "end_col_offset", 0)
        return self._offset(lineno, col_offset)

    def node_text(self, node: ast.AST) -> str:
        """Return source text for a node."""
        start = self.node_offset(node, start=True)
        end = self.node_offset(node, start=False)
        return self._source[start:end]

    def append_rewrite(self, node: ast.AST, text: str, change: str) -> None:
        """Record a rewrite spanning a node's source range."""
        start = self.node_offset(node, start=True)
        end = self.node_offset(node, start=False)
        self.rewrites.append(FlextInfraSourceRewrite(start, end, text))
        self.changes.append(change)

    @staticmethod
    def apply_rewrites(source: str, rewrites: list[FlextInfraSourceRewrite]) -> str:
        """Apply rewrites from bottom-right to top-left to preserve offsets."""
        result = source
        for rewrite in sorted(rewrites, reverse=True):
            result = result[: rewrite.start] + rewrite.text + result[rewrite.end :]
        return result
