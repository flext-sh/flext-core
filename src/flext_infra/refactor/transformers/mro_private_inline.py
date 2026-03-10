"""CST transformer for private-symbol inlining in MRO migration."""

from __future__ import annotations

from typing import override

import libcst as cst


class FlextInfraRefactorMROPrivateInlineTransformer(cst.CSTTransformer):
    """Inline configured private-name values after migration."""

    def __init__(self, *, replacement_values: dict[str, cst.BaseExpression]) -> None:
        """Initialize with symbol-to-value mapping for private constant inlining."""
        self.replacement_values = replacement_values

    @override
    def leave_Name(
        self,
        original_node: cst.Name,
        updated_node: cst.Name,
    ) -> cst.BaseExpression:
        if original_node.value in self.replacement_values:
            return self.replacement_values[original_node.value]
        return updated_node


class FlextInfraRefactorMROQualifiedReferenceTransformer(cst.CSTTransformer):
    """Replace bare name references with qualified facade paths after migration.

    Skips definition positions (TypeAlias.name, AnnAssign.target, Assign.target)
    so only reference occurrences are renamed.
    """

    def __init__(self, *, renames: dict[str, cst.BaseExpression]) -> None:
        """Initialize with symbol-to-qualified-expression rename mapping."""
        self._renames = renames
        self._defining: set[str] = set()

    @override
    def visit_TypeAlias(self, node: cst.TypeAlias) -> bool:
        self._defining.add(node.name.value)
        return True

    @override
    def leave_TypeAlias(
        self,
        original_node: cst.TypeAlias,
        updated_node: cst.TypeAlias,
    ) -> cst.TypeAlias:
        self._defining.discard(original_node.name.value)
        return updated_node

    @override
    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        if isinstance(node.target, cst.Name):
            self._defining.add(node.target.value)
        return True

    @override
    def leave_AnnAssign(
        self,
        original_node: cst.AnnAssign,
        updated_node: cst.AnnAssign,
    ) -> cst.AnnAssign:
        if isinstance(original_node.target, cst.Name):
            self._defining.discard(original_node.target.value)
        return updated_node

    @override
    def leave_Name(
        self,
        original_node: cst.Name,
        updated_node: cst.Name,
    ) -> cst.BaseExpression:
        if original_node.value in self._defining:
            return updated_node
        replacement = self._renames.get(original_node.value)
        if replacement is not None:
            return replacement
        return updated_node


__all__ = [
    "FlextInfraRefactorMROPrivateInlineTransformer",
    "FlextInfraRefactorMROQualifiedReferenceTransformer",
]
