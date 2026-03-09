"""MRO redeclaration remover transformer for nested classes."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import libcst as cst


class FlextInfraRefactorMRORemover(cst.CSTTransformer):
    """Remove nested class bases that redundantly reference the parent class."""

    def __init__(self, on_change: Callable[[str], None] | None = None) -> None:
        """Initialize optional callback for emitted change messages."""
        self._on_change = on_change
        self.changes: list[str] = []

    @override
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        del original_node
        if not isinstance(updated_node.body, cst.IndentedBlock):
            return updated_node
        new_body: list[cst.BaseStatement] = []
        for stmt in updated_node.body.body:
            if isinstance(stmt, cst.ClassDef) and stmt.bases:
                for base in stmt.bases:
                    if not isinstance(base.value, cst.Attribute):
                        continue
                    root_name = self._attribute_root_name(base.value)
                    if root_name != updated_node.name.value:
                        continue
                    message = f"Fixed MRO redeclaration: {stmt.name.value}"
                    self.changes.append(message)
                    if self._on_change is not None:
                        self._on_change(message)
                    stmt = stmt.with_changes(bases=(), lpar=(), rpar=())
                    break
            new_body.append(stmt)
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body),
        )

    def _attribute_root_name(self, expr: cst.BaseExpression) -> str:
        current = expr
        while isinstance(current, cst.Attribute):
            current = current.value
        if isinstance(current, cst.Name):
            return current.value
        return ""
