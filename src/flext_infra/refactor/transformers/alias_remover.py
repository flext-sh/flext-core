"""Alias remover transformer for legacy compatibility cleanup."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import libcst as cst


class FlextInfraRefactorAliasRemover(cst.CSTTransformer):
    """Remove module-level ``Name = Name`` aliases with allowlist support."""

    def __init__(
        self,
        allow_aliases: set[str],
        allow_target_suffixes: tuple[str, ...],
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize alias remover with allow-list configuration."""
        self._scope_depth = 0
        self._allow_aliases = allow_aliases
        self._allow_target_suffixes = allow_target_suffixes
        self._on_change = on_change
        self.changes: list[str] = []

    @override
    def leave_Assign(
        self,
        original_node: cst.Assign,
        updated_node: cst.Assign,
    ) -> cst.BaseSmallStatement | cst.RemovalSentinel:
        if self._scope_depth > 0:
            return updated_node
        if (
            len(original_node.targets) == 1
            and isinstance(original_node.targets[0].target, cst.Name)
            and isinstance(original_node.value, cst.Name)
        ):
            target = original_node.targets[0].target.value
            value = original_node.value.value
            if target in self._allow_aliases:
                return updated_node
            if self._allow_target_suffixes and value.endswith(
                self._allow_target_suffixes,
            ):
                return updated_node
            if target != value and target not in {"__version__", "__all__"}:
                message = f"Removed alias: {target} = {value}"
                self.changes.append(message)
                if self._on_change is not None:
                    self._on_change(message)
                return cst.RemovalSentinel.REMOVE
        return updated_node

    @override
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        del original_node
        self._scope_depth -= 1
        return updated_node

    @override
    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        del original_node
        self._scope_depth -= 1
        return updated_node

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        del node
        self._scope_depth += 1

    @override
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        del node
        self._scope_depth += 1
