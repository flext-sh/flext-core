"""Logging modernizer transformer — migrate ``logging`` usage to FLEXT ``u.fetch_logger``.

Conservative AST rewrites:

- ``import logging`` → removed when only ``logging.getLogger`` was used.
- ``from logging import getLogger`` → removed.
- ``from logging import getLogger, ...`` → ``getLogger`` dropped.
- ``logging.getLogger(...)`` → ``u.fetch_logger(...)``.
- ``logger = logging.getLogger(...)`` → ``logger = u.fetch_logger(...)``.

The transformer only rewrites when the result is unambiguous and records every
change.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING, override

from flext_infra.transformers._rewrite import (
    FlextInfraSourceRewrite,
    FlextInfraSourceRewriter,
)
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorLoggingModernizer(FlextInfraRopeTransformer):
    """AST-driven transformer for logging anti-patterns."""

    _description = "migrate logging usage to u.fetch_logger"

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply logging modernizations to source text."""
        self.changes.clear()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, list(self.changes)

        visitor = self._LoggingVisitor(source)
        visitor.visit(tree)

        rewrites = list(visitor.rewrites)

        if visitor.logging_import_node is not None and not visitor.other_logging_usage:
            import_node = visitor.logging_import_node
            start = visitor.node_offset(import_node, start=True)
            end = visitor.node_offset(import_node, start=False)
            rewrites.append(FlextInfraSourceRewrite(start, end, "pass"))
            visitor.changes.append("Removed unused import logging")

        for from_import_node in visitor.logging_from_import_nodes:
            new_text, change = self._rewrite_from_import_node(from_import_node)
            if new_text is not None:
                start = visitor.node_offset(from_import_node, start=True)
                end = visitor.node_offset(from_import_node, start=False)
                rewrites.append(FlextInfraSourceRewrite(start, end, new_text))
                visitor.changes.append(change)

        if not rewrites:
            return source, list(self.changes)

        updated = FlextInfraSourceRewriter.apply_rewrites(source, rewrites)

        if visitor.needs_u_import:
            updated = self._ensure_u_import(updated)

        for change in visitor.changes:
            self._record_change(change)

        return updated, list(self.changes)

    @staticmethod
    def _rewrite_from_import_node(node: ast.ImportFrom) -> tuple[str | None, str]:
        """Drop ``getLogger`` from a ``from logging import ...`` statement."""
        names_to_keep: list[str] = []
        removed_get_logger = False
        for alias in node.names:
            if alias.name == "getLogger" and alias.asname is None:
                removed_get_logger = True
                continue
            names_to_keep.append(
                alias.asname if alias.asname is not None else alias.name
            )
        if not removed_get_logger:
            return None, ""
        if not names_to_keep:
            return "pass", "Removed from logging import getLogger"
        return (
            f"from logging import {', '.join(names_to_keep)}",
            "Removed getLogger from logging imports",
        )

    class _LoggingVisitor(FlextInfraSourceRewriter):
        """Collect rewrites for logging anti-patterns."""

        def __init__(self, source: str) -> None:
            super().__init__(source)
            self.logging_import_node: ast.Import | None = None
            self.logging_from_import_nodes: list[ast.ImportFrom] = []
            self.other_logging_usage = False
            self.needs_u_import = False
            self._logging_function_names: set[str] = set()

        @override
        def visit_Import(self, node: ast.Import) -> None:
            """Track bare ``import logging`` candidates for removal."""
            for alias in node.names:
                if alias.name == "logging" and alias.asname is None:
                    self.logging_import_node = node
                    break
            self.generic_visit(node)

        @override
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            """Track ``from logging import ...`` candidates for cleanup."""
            if node.module == "logging":
                self.logging_from_import_nodes.append(node)
                for alias in node.names:
                    if alias.name == "getLogger":
                        self._logging_function_names.add(
                            alias.asname if alias.asname is not None else alias.name
                        )
            self.generic_visit(node)

        @override
        def visit_Attribute(self, node: ast.Attribute) -> None:
            """Detect ``logging.*`` usage that is not ``getLogger``."""
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "logging"
                and node.attr != "getLogger"
            ):
                self.other_logging_usage = True
            self.generic_visit(node)

        @override
        def visit_Call(self, node: ast.Call) -> None:
            """Rewrite ``logging.getLogger(...)`` and bare ``getLogger(...)`` calls."""
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logging"
                and node.func.attr == "getLogger"
            ):
                call_text = self.node_text(node)
                new_call = call_text.replace("logging.getLogger", "u.fetch_logger", 1)
                self.append_rewrite(
                    node,
                    new_call,
                    "Replaced logging.getLogger(...) with u.fetch_logger(...)",
                )
                self.needs_u_import = True
                return
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in self._logging_function_names
            ):
                call_text = self.node_text(node)
                new_call = re.sub(
                    rf"\b{re.escape(node.func.id)}\b",
                    "u.fetch_logger",
                    call_text,
                    count=1,
                )
                self.append_rewrite(
                    node, new_call, "Replaced getLogger(...) with u.fetch_logger(...)"
                )
                self.needs_u_import = True
                return
            self.generic_visit(node)


__all__: list[str] = ["FlextInfraRefactorLoggingModernizer"]
