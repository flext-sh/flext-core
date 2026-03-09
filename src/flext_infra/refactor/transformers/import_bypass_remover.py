"""Import bypass remover transformer for legacy import fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import libcst as cst


class FlextInfraRefactorImportBypassRemover(cst.CSTTransformer):
    """Replace import bypass try/except blocks with the primary import."""

    def __init__(self, on_change: Callable[[str], None] | None = None) -> None:
        """Initialize optional callback for emitted change messages."""
        self._on_change = on_change
        self.changes: list[str] = []

    @override
    def leave_Try(
        self,
        original_node: cst.Try,
        updated_node: cst.Try,
    ) -> cst.BaseStatement:
        del original_node
        if len(updated_node.body.body) != 1:
            return updated_node
        if len(updated_node.handlers) != 1:
            return updated_node
        body_stmt = updated_node.body.body[0]
        handler = updated_node.handlers[0]
        if not isinstance(handler.body, cst.IndentedBlock):
            return updated_node
        if len(handler.body.body) != 1:
            return updated_node
        fallback_stmt = handler.body.body[0]
        if not (
            isinstance(body_stmt, cst.SimpleStatementLine)
            and isinstance(fallback_stmt, cst.SimpleStatementLine)
        ):
            return updated_node
        if len(body_stmt.body) != 1 or len(fallback_stmt.body) != 1:
            return updated_node
        primary_import = body_stmt.body[0]
        fallback_import = fallback_stmt.body[0]
        if not (
            isinstance(primary_import, cst.ImportFrom)
            and isinstance(fallback_import, cst.ImportFrom)
        ):
            return updated_node
        handler_type = handler.type
        if not isinstance(handler_type, cst.Name):
            return updated_node
        if handler_type.value != "ImportError":
            return updated_node
        message = "Removed import bypass fallback"
        self.changes.append(message)
        if self._on_change is not None:
            self._on_change(message)
        return body_stmt
